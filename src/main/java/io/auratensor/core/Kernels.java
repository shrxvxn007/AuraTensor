package io.auratensor.core;

import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShuffle;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
/**
 * SIMD-accelerated math kernels for the transformer forward pass.
 *
 * <p>All kernels operate directly on the raw {@link MemorySegment} of a
 * tensor in FP32 layout, fusing all arithmetic into {@link FloatVector}
 * primitives (FMA, lanewise mul/add, reductions).
 *
 * <p>A note on shape conventions: 2D tensors for matmul use row-major
 * {@code [M, K]} and {@code [K, N]}, matching the GGUF weight layout used by
 * every modern llama.cpp export.
 */
public final class Kernels {

    private static final VectorShuffle SWAP_ADJ;
    /**
     * Static mask of length {@code SPECIES.length()} baked once, alternating
     * {-1, +1} for the RoPE sign-flip step. Because the Vector API requires
     * fully-populated lane buffers, this lives in static memory.
     */
    private static final FloatVector SIGN_MASK;

    static {
        int lanes = Simd.SPECIES.length();
        int[] idx = new int[lanes];
        for (int i = 0; i < lanes; i += 2) {
            idx[i] = i + 1;
            idx[i + 1] = i;
        }
        SWAP_ADJ = VectorShuffle.fromValues(Simd.SPECIES, idx);
        float[] sgnBuf = new float[lanes];
        for (int i = 0; i < lanes; i += 2) {
            sgnBuf[i]     = -1.0f;
            sgnBuf[i + 1] =  1.0f;
        }
        SIGN_MASK = FloatVector.fromArray(Simd.SPECIES, sgnBuf, 0);
    }

    private Kernels() {}

    // ---------------------------------------------------------------------
    // SiLU:  x * sigmoid(x) = x / (1 + e^-x)
    // ---------------------------------------------------------------------

    /** In-place SiLU activation on tensor {@code x}. */
    public static void siluInPlace(Tensor x) {
        if (x.dtype() != DType.FP32) {
            throw new IllegalArgumentException("siluInPlace requires FP32 tensor");
        }
        siluInPlaceSegment(x.data(), x.numElements());
    }

    public static void siluInPlaceSegment(MemorySegment seg, long n) {
        var species = Simd.SPECIES;
        int upper = species.loopBound((int) n);
        long endByte = n * 4L;

        if (n > Integer.MAX_VALUE / 4L) {
            throw new IllegalArgumentException("siluInPlaceSegment: n too large");
        }
        for (int i = 0; i < upper; i += species.length()) {
            long off = (long) i * 4L;
            FloatVector v = FloatVector.fromMemorySegment(species, seg, off, ByteOrder.nativeOrder());
            // sigmoid(x) = 1 / (1 + exp(-x)); the Vector API does not yet
            // expose a lanewise exp, so we go through a stack-allocated
            // float[] for the exp call.
            float[] arr = v.toArray();
            for (int j = 0; j < arr.length; j++) {
                arr[j] = arr[j] / (1.0f + (float) Math.exp(-arr[j]));
            }
            FloatVector r = FloatVector.fromArray(species, arr, 0);
            r.intoMemorySegment(seg, off, ByteOrder.nativeOrder());
        }
        // Scalar tail
        for (int i = upper; i < n; i++) {
            long off = (long) i * 4L;
            float xv = seg.get(ValueLayout.JAVA_FLOAT, off);
            float sigmoid = 1.0f / (1.0f + (float) Math.exp(-xv));
            seg.set(ValueLayout.JAVA_FLOAT, off, xv * sigmoid);
        }
        if (endByte < 0) throw new AssertionError("overflow");  // explicit
    }

    // ---------------------------------------------------------------------
    // RMSNorm:  x / sqrt(mean(x^2) + eps) * gamma
    // ---------------------------------------------------------------------

    /**
     * Computes RMSNorm in-place on {@code x}, scaled by per-row {@code weight}
     * (the gamma). Both tensors must be FP32. {@code weight.length == x.lastDim}.
     */
    public static void rmsNormInPlace(Tensor x, Tensor weight, float eps) {
        if (x.dtype() != DType.FP32 || weight.dtype() != DType.FP32) {
            throw new IllegalArgumentException("rmsNorm requires FP32 tensors");
        }
        int[] shape = x.shape();
        int rank = x.rank();
        if (rank != 1 && rank != 2) {
            throw new IllegalArgumentException("rmsNorm requires rank 1 or 2 tensors");
        }
        int d = shape[rank - 1];
        if (weight.numElements() != d) {
            throw new IllegalArgumentException("weight length " + weight.numElements()
                + " != last dim " + d);
        }

        int leading = (int) (x.numElements() / d);
        var species = Simd.SPECIES;
        int upper = species.loopBound(d);

        MemorySegment xseg = x.data();
        MemorySegment wseg = weight.data();

        for (int row = 0; row < leading; row++) {
            long rowOff = (long) row * d * 4L;

            // 1) sum of squares
            FloatVector acc = FloatVector.zero(species);
            for (int i = 0; i < upper; i += species.length()) {
                FloatVector v = FloatVector.fromMemorySegment(
                    species, xseg, rowOff + (long) i * 4L, ByteOrder.nativeOrder());
                acc = v.fma(v, acc);
            }
            float sumSq = acc.reduceLanes(VectorOperators.ADD);
            for (int i = upper; i < d; i++) {
                float xv = xseg.get(ValueLayout.JAVA_FLOAT, rowOff + (long) i * 4L);
                sumSq += xv * xv;
            }
            float invRms = (float) (1.0 / Math.sqrt(sumSq / (float) d + eps));

            // 2) scale: x * invRms * gamma
            for (int i = 0; i < upper; i += species.length()) {
                FloatVector v = FloatVector.fromMemorySegment(
                    species, xseg, rowOff + (long) i * 4L, ByteOrder.nativeOrder());
                FloatVector w = FloatVector.fromMemorySegment(
                    species, wseg, (long) i * 4L, ByteOrder.nativeOrder());
                FloatVector r = v.mul(invRms).mul(w);
                r.intoMemorySegment(xseg, rowOff + (long) i * 4L, ByteOrder.nativeOrder());
            }
            for (int i = upper; i < d; i++) {
                float xv = xseg.get(ValueLayout.JAVA_FLOAT, rowOff + (long) i * 4L);
                float wv = wseg.get(ValueLayout.JAVA_FLOAT, (long) i * 4L);
                xseg.set(ValueLayout.JAVA_FLOAT, rowOff + (long) i * 4L, xv * invRms * wv);
            }
        }
    }

    // ---------------------------------------------------------------------
    // Softmax (numerically stable)
    // ---------------------------------------------------------------------

    /** In-place stable softmax along the last dimension of a rank-1 or rank-2 tensor. */
    public static void softmaxInPlace(Tensor x) {
        if (x.dtype() != DType.FP32) {
            throw new IllegalArgumentException("softmax requires FP32 tensor");
        }
        int[] shape = x.shape();
        int rank = x.rank();
        int d = shape[rank - 1];
        int leading = (int) (x.numElements() / d);
        MemorySegment seg = x.data();
        var species = Simd.SPECIES;
        int upper = species.loopBound(d);

        for (int row = 0; row < leading; row++) {
            long rowOff = (long) row * d * 4L;

            // 1) find max
            FloatVector vmax = FloatVector.broadcast(species, Float.NEGATIVE_INFINITY);
            for (int i = 0; i < upper; i += species.length()) {
                FloatVector v = FloatVector.fromMemorySegment(
                    species, seg, rowOff + (long) i * 4L, ByteOrder.nativeOrder());
                vmax = vmax.max(v);
            }
            float rowMax = vmax.reduceLanes(VectorOperators.MAX);
            for (int i = upper; i < d; i++) {
                float xv = seg.get(ValueLayout.JAVA_FLOAT, rowOff + (long) i * 4L);
                if (xv > rowMax) rowMax = xv;
            }

            // 2) exp(x - max) and sum
            FloatVector vsum = FloatVector.zero(species);
            for (int i = 0; i < upper; i += species.length()) {
                FloatVector v = FloatVector.fromMemorySegment(
                    species, seg, rowOff + (long) i * 4L, ByteOrder.nativeOrder());
                v = v.sub(rowMax);
                // exp via array is regrettable — see siluInPlace for context.
                float[] arr = v.toArray();
                for (int j = 0; j < arr.length; j++) arr[j] = (float) Math.exp(arr[j]);
                FloatVector ev = FloatVector.fromArray(species, arr, 0);
                ev.intoMemorySegment(seg, rowOff + (long) i * 4L, ByteOrder.nativeOrder());
                vsum = ev.add(vsum);
            }
            float rowSum = vsum.reduceLanes(VectorOperators.ADD);
            for (int i = upper; i < d; i++) {
                float ev = (float) Math.exp(
                    seg.get(ValueLayout.JAVA_FLOAT, rowOff + (long) i * 4L) - rowMax);
                seg.set(ValueLayout.JAVA_FLOAT, rowOff + (long) i * 4L, ev);
                rowSum += ev;
            }

            // 3) normalize
            float inv = 1.0f / rowSum;
            FloatVector vinv = FloatVector.broadcast(species, inv);
            for (int i = 0; i < upper; i += species.length()) {
                FloatVector v = FloatVector.fromMemorySegment(
                    species, seg, rowOff + (long) i * 4L, ByteOrder.nativeOrder());
                v.mul(inv).intoMemorySegment(
                    seg, rowOff + (long) i * 4L, ByteOrder.nativeOrder());
            }
            for (int i = upper; i < d; i++) {
                seg.set(ValueLayout.JAVA_FLOAT, rowOff + (long) i * 4L, seg.get(ValueLayout.JAVA_FLOAT, rowOff + (long) i * 4L) * inv);
            }
        }
    }

    // ---------------------------------------------------------------------
    // RoPE (in-place on a flat head_dim vector, per head)
    // ---------------------------------------------------------------------

    /**
     * Apply Rotary Position Embedding in place to Q or K vectors of shape
     * {@code [numHeads * headDim]}. Implementation uses the standard Llama
     * formulation: for each adjacent pair (a, b) at index 2i and 2i+1 within
     * {@code headDim}:
     *
     *   freq_i = base^(-2i / headDim)
     *   angle = position * freq_i
     *   a' = a * cos(angle) - b * sin(angle)
     *   b' = a * sin(angle) + b * cos(angle)
     *
     * Caller pre-supplies cos/sin arrays of size {@code headDim}.
     */
    public static void ropeInPlace(Tensor qk, Tensor cos, Tensor sin, int headDim) {
        if (qk.dtype() != DType.FP32) {
            throw new IllegalArgumentException("ropeInPlace requires FP32 tensors");
        }
        int heads = (int) (qk.numElements() / headDim);
        MemorySegment qseg = qk.data();
        MemorySegment cseg = cos.data();
        MemorySegment sseg = sin.data();

        var species = Simd.SPECIES;
        // V = [a_0, b_0, a_1, b_1, ...]            (qk segment stride-1)
        // C = [c_0, c_0, c_1, c_1, ...]            (cos segment stride-1)
        // S = [s_0, s_0, s_1, s_1, ...]            (sin segment stride-1)
        // For the rotation:
        //   a'_i = a_i * c_i - b_i * s_i
        //   b'_i = a_i * s_i + b_i * c_i
        // Vectorizing:  V_swap = [b_i, a_i, ...].  To get -b_i*s_i we need
        // a swapped sin vector whose lanes for the (a,b) pair are (s, -s).
        int upper = species.loopBound(headDim);
        FloatVector signMask = SIGN_MASK;  // [-1, +1, -1, +1, ...] baked once.

        for (int h = 0; h < heads; h++) {
            long off = (long) h * headDim * 4L;

            for (int i = 0; i < upper; i += species.length()) {
                long bo = off + (long) i * 4L;

                FloatVector V  = FloatVector.fromMemorySegment(species, qseg, bo, ByteOrder.nativeOrder());
                FloatVector Cv = FloatVector.fromMemorySegment(species, cseg, (long) i * 4L, ByteOrder.nativeOrder());
                FloatVector Sv = FloatVector.fromMemorySegment(species, sseg, (long) i * 4L, ByteOrder.nativeOrder());

                // Signed sin: multiply by [-1, +1, -1, +1] so lane l of pair p
                // becomes (s = sin_p, -s = -sin_p).
                FloatVector svSigned = Sv.mul(signMask);
                // V * C contributes a*c on lane 0, b*c on lane 1, ...
                // V_swap contributes -b*s on lane 0, +a*s on lane 1, ...
                FloatVector out = V.fma(Cv, V.rearrange(SWAP_ADJ).mul(svSigned));
                out.intoMemorySegment(qseg, bo, ByteOrder.nativeOrder());
            }
            for (int i = upper; i < headDim; i += 2) {
                long bo = off + (long) i * 4L;
                float a = qseg.get(ValueLayout.JAVA_FLOAT, bo);
                float b = qseg.get(ValueLayout.JAVA_FLOAT, bo + 4);
                float c = cseg.get(ValueLayout.JAVA_FLOAT, (long) i * 4L);
                float s = sseg.get(ValueLayout.JAVA_FLOAT, (long) i * 4L);
                qseg.set(ValueLayout.JAVA_FLOAT, bo, a * c - b * s);
                qseg.set(ValueLayout.JAVA_FLOAT, bo + 4, a * s + b * c);
            }
        }
    }

    // ---------------------------------------------------------------------
    // GEMV: y[M] = A[M,K] * x[K]
    // ---------------------------------------------------------------------

    /**
     * Compute {@code y = A @ x} for matrices stored in row-major FP32.
     * Uses Vector API with K-loop inside, accumulating into vector lanes.
     */
    public static void sgemv(Tensor A, Tensor x, Tensor y) {
        if (A.rank() != 2 || x.rank() != 1 || y.rank() != 1) {
            throw new IllegalArgumentException("sgemv: rank mismatch A=" + A.rank()
                + " x=" + x.rank() + " y=" + y.rank());
        }
        int M = A.shape()[0];
        int K = A.shape()[1];
        if (x.numElements() != K || y.numElements() != M) {
            throw new IllegalArgumentException("sgemv: dim mismatch");
        }
        MemorySegment aseg = A.data();
        MemorySegment xseg = x.data();
        MemorySegment yseg = y.data();
        var species = Simd.SPECIES;
        int upper = species.loopBound(K);

        for (int i = 0; i < M; i++) {
            long rowOff = (long) i * K * 4L;
            FloatVector acc = FloatVector.zero(species);
            for (int k = 0; k < upper; k += species.length()) {
                FloatVector av = FloatVector.fromMemorySegment(
                    species, aseg, rowOff + (long) k * 4L, ByteOrder.nativeOrder());
                FloatVector xv = FloatVector.fromMemorySegment(
                    species, xseg, (long) k * 4L, ByteOrder.nativeOrder());
                acc = av.fma(xv, acc);
            }
            float sum = acc.reduceLanes(VectorOperators.ADD);
            for (int k = upper; k < K; k++) {
                sum += aseg.get(ValueLayout.JAVA_FLOAT, rowOff + (long) k * 4L)
                     * xseg.get(ValueLayout.JAVA_FLOAT, (long) k * 4L);
            }
            yseg.set(ValueLayout.JAVA_FLOAT, (long) i * 4L, sum);
        }
    }

    // ---------------------------------------------------------------------
    // GEMM: C[M,N] = A[M,K] * B[K,N]   (FP32 row-major)
    // ---------------------------------------------------------------------

    /**
     * Naive SIMD-accelerated SGEMM. For LLM inference workloads (M=token
     * count, often 1 during decode) the inner loop is bounded by K and we
     * rely on cache reuse rather than register tiling.
     */
    public static void sgemm(Tensor A, Tensor B, Tensor C) {
        if (A.rank() != 2 || B.rank() != 2 || C.rank() != 2) {
            throw new IllegalArgumentException("sgemm: rank mismatch");
        }
        int M = A.shape()[0];
        int K = A.shape()[1];
        int N = B.shape()[1];
        if (B.shape()[0] != K || C.shape()[0] != M || C.shape()[1] != N) {
            throw new IllegalArgumentException("sgemm: dim mismatch");
        }
        sgemm(A.data(), B.data(), C.data(), M, K, N);
    }

    public static void sgemm(MemorySegment aSeg, MemorySegment bSeg, MemorySegment cSeg,
                             int M, int K, int N) {        // The original SIMD inner loop used a wrong strided read of B in
        // row-major memory. For correctness, delegate to the scalar baseline.
        // A register-tiled SIMD implementation is tracked separately.
        sgemmScalar(aSeg, bSeg, cSeg, M, K, N);
    }

    /**
     * Scalar (non-vectorized) GEMM baseline, used for JMH comparisons.
     */
    public static void sgemmScalar(MemorySegment aSeg, MemorySegment bSeg, MemorySegment cSeg,
                                   int M, int K, int N) {
        for (int i = 0; i < M; i++) {
            long rowA = (long) i * K * 4L;
            long rowC = (long) i * N * 4L;
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += aSeg.get(ValueLayout.JAVA_FLOAT, rowA + (long) k * 4L)
                         * bSeg.get(ValueLayout.JAVA_FLOAT, (long) k * N * 4L + (long) j * 4L);
                }
                cSeg.set(ValueLayout.JAVA_FLOAT, rowC + (long) j * 4L, sum);
            }
        }
    }
}
