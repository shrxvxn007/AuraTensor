package io.auratensor.inference;

import io.auratensor.core.Tensor;
import io.auratensor.core.DType;
import io.auratensor.core.Kernels;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Precomputed Rotary Position Embedding frequency tables.
 *
 * <p>Layout: for each position p and each pair index i in {@code [0, headDim/2)},
 * we duplicate the value across both elements of the pair, so the cache is laid
 * out as {@code [c_0, c_0, c_1, c_1, ...]} for length {@code headDim}. This
 * matches the dense stride-1 access pattern the SIMD {@link Kernels#ropeInPlace}
 * expects (lane i of the QK vector pairs with cos[i]).
 *
 * <p>Llama 3 uses base = 500_000; Llama 2 and Mistral use base = 10_000.
 */
public final class RopeCache {

    private final Tensor cos;
    private final Tensor sin;

    public RopeCache(long headDim, float base, long maxContext) {
        long half = headDim / 2;
        long strideElems = headDim;
        // One row per position, length = headDim (cos duplicated per element).
        long n = maxContext * strideElems;
        Arena arena = Arena.ofConfined();
        MemorySegment cosSeg = arena.allocate(n * 4L, 16);
        MemorySegment sinSeg = arena.allocate(n * 4L, 16);
        long rowBytes = strideElems * 4L;
        for (long pos = 0; pos < maxContext; pos++) {
            long rowBase = pos * rowBytes;
            for (long i = 0; i < half; i++) {
                float freq = (float) Math.pow(base, -2.0 * i / (double) headDim);
                float angle = pos * freq;
                float c = (float) Math.cos(angle);
                float s = (float) Math.sin(angle);
                // Duplicate per element of the pair: lanes (2i, 2i+1) both see (c, s).
                cosSeg.set(ValueLayout.JAVA_FLOAT, rowBase + (2L * i) * 4L, c);
                cosSeg.set(ValueLayout.JAVA_FLOAT, rowBase + (2L * i + 1L) * 4L, c);
                sinSeg.set(ValueLayout.JAVA_FLOAT, rowBase + (2L * i) * 4L, s);
                sinSeg.set(ValueLayout.JAVA_FLOAT, rowBase + (2L * i + 1L) * 4L, s);
            }
        }
        this.cos = Tensor.wrap(cosSeg, arena, DType.FP32,
                               new int[]{ (int) (maxContext * strideElems) });
        this.sin = Tensor.wrap(sinSeg, Arena.ofConfined(), DType.FP32,
                               new int[]{ (int) (maxContext * strideElems) });
    }

    public Tensor cosFor(long pos, long headDim) {
        return slice(cos, pos * headDim, headDim);
    }

    public Tensor sinFor(long pos, long headDim) {
        return slice(sin, pos * headDim, headDim);
    }

    private static Tensor slice(Tensor t, long start, long len) {
        return Tensor.wrap(t.data().asSlice(start * 4L, len * 4L),
                           Arena.ofConfined(), DType.FP32, new int[]{ (int) len });
    }
}
