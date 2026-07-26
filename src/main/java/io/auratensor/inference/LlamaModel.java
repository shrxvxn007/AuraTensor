package io.auratensor.inference;

import io.auratensor.core.Kernels;
import io.auratensor.core.Tensor;
import io.auratensor.core.DType;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Llama 3 / Mistral transformer forward pass with off-heap KV-cache and
 * virtual-thread-friendly single-batch state.
 *
 * <p>Layout follows the standard Llama recipe:
 * <pre>
 *   x = tokenEmbeddings[tokenId]
 *   for each layer:
 *     x' = rmsNorm(x, attnNorm)
 *     q, k, v = projections(x')          [sgemv on pre-transposed weights]
 *     apply RoPE to q, k with current position
 *     write k, v into KV cache
 *     attn = softmax(Q @ K^T / sqrt(headDim)) @ V   [sgemv + softmax + sgemm]
 *     x = x + attn_output(attn)          [sgemv on pre-transposed attnOut]
 *     x' = rmsNorm(x, ffnNorm)
 *     h = silu(ffn_gate(x')) * ffn_up(x') [sgemv + silu + elementwise mul]
 *     x = x + ffn_down(h)                [sgemv on ffnDown]
 *   x = rmsNorm(x, outputNorm)
 *   logits = output_weight @ x
 * </pre>
 *
 * <p>All per-layer projections go through {@link Kernels#sgemv} so the SIMD
 * Vector API is on the inference hot path. Attention reduction uses
 * {@link Kernels#sgemv} (Q @ K-cache row), {@link Kernels#softmaxInPlaceSegment},
 * and {@link Kernels#sgemm} (softmaxed @ V-cache row). The Q/K/V and
 * attn-output weight matrices are pre-transposed into SIMD-friendly 2D
 * row-major form at construction time (one-time cost at model load); FFN
 * gate/up/down already arrive in row-major [M, K] form from the GGUF
 * exporter so no transpose is required for them.
 */
public final class LlamaModel {

    private final LlamaConfig cfg;
    private final Weights weights;
    private final Tokenizer tokenizer;
    private final KVCache kvCache;
    private final RopeCache ropeCache;

    /** Pre-transposed per-layer attention projection weights.
     *  Layouts (row-major FP32):
     *  <ul>
     *    <li>{@code attnQT[L]} — shape {@code [nHeads * headDim, embDim]} —
     *        Q projection collapsed from {@code [emb, nHeads, headDim]}.</li>
     *    <li>{@code attnKT[L], attnVT[L]} — same for K, V with nHeadsKv.</li>
     *    <li>{@code attnOutT[L]} — shape {@code [embDim, nHeads * headDim]} —
     *        output projection collapsed from {@code [nHeads, headDim, emb]}.</li>
     *  </ul> */
    private final Tensor[] attnQT;
    private final Tensor[] attnKT;
    private final Tensor[] attnVT;
    private final Tensor[] attnOutT;

    /** Per-decode-step scratch buffers — allocated once, reused across all steps. */
    private final Tensor hiddenState;          // rank-1 [embDim]
    private final Tensor qScratch;             // rank-1 [nHeads    * headDim]
    private final Tensor kScratch;             // rank-1 [nHeadsKv  * headDim]
    private final Tensor vScratch;             // rank-1 [nHeadsKv  * headDim]
    private final Tensor attnOutScratch;       // rank-1 [nHeads    * headDim]
    private final Tensor attnLogitsScratch;    // rank-1 [cfg.contextLength()]  (also softmax output)
    private final Tensor projOutScratch;       // rank-1 [embDim]
    private final Tensor gateScratch;          // rank-1 [feedForwardLength]
    private final Tensor upScratch;            // rank-1 [feedForwardLength]
    private final Tensor ffnOutScratch;        // rank-1 [embDim]

    /** Off-heap logits buffer; the per-decode {@code float[vocab]} is wrapped here. */
    private final MemorySegment logitsSeg;     // raw [vocab] FP32
    private final float[] logitsArray;         // heap view (returned to callers)

    public LlamaModel(LlamaConfig cfg, Weights weights, Tokenizer tokenizer) {
        this.cfg = cfg;
        this.weights = weights;
        this.tokenizer = tokenizer;
        this.kvCache = new KVCache(cfg.blockCount(), cfg.headCountKv(), cfg.headDim(),
                                   cfg.contextLength());
        this.ropeCache = new RopeCache(cfg.headDim(), cfg.ropeFrequencyBase(),
                                       cfg.contextLength());

        int layers     = (int) cfg.blockCount();
        int embDim     = (int) cfg.embeddingLength();
        int headDim    = (int) cfg.headDim();
        int nHeads     = (int) cfg.headCount();
        int nHeadsKv   = (int) cfg.headCountKv();
        int ffnDim     = (int) cfg.feedForwardLength();
        int vocab      = (int) cfg.vocabSize();
        int ctx        = (int) cfg.contextLength();

        // One-time load-time transposes for attention projections. Done once
        // per layer so the inner sgemv inner loop sees stride-1 weight reads.
        this.attnQT   = new Tensor[layers];
        this.attnKT   = new Tensor[layers];
        this.attnVT   = new Tensor[layers];
        this.attnOutT = new Tensor[layers];
        for (int L = 0; L < layers; L++) {
            Weights.Layer wl = weights.layers[L];
            this.attnQT[L]   = transposeAttnProj(wl.attnQ,   embDim, nHeads,    headDim);
            this.attnKT[L]   = transposeAttnProj(wl.attnK,   embDim, nHeadsKv,  headDim);
            this.attnVT[L]   = transposeAttnProj(wl.attnV,   embDim, nHeadsKv,  headDim);
            this.attnOutT[L] = transposeAttnOut(  wl.attnOut, nHeads, headDim,  embDim);
        }

        // Scratch buffers — single allocation, reused across every decode step
        // and the synthetic-prefix prefill. No per-step Arena.ofConfined cost.
        this.hiddenState        = Tensor.allocate1D(DType.FP32, embDim);
        this.qScratch           = Tensor.allocate1D(DType.FP32, nHeads    * headDim);
        this.kScratch           = Tensor.allocate1D(DType.FP32, nHeadsKv  * headDim);
        this.vScratch           = Tensor.allocate1D(DType.FP32, nHeadsKv  * headDim);
        this.attnOutScratch     = Tensor.allocate1D(DType.FP32, nHeads    * headDim);
        this.attnLogitsScratch  = Tensor.allocate1D(DType.FP32, ctx);
        this.projOutScratch     = Tensor.allocate1D(DType.FP32, embDim);
        this.gateScratch        = Tensor.allocate1D(DType.FP32, ffnDim);
        this.upScratch          = Tensor.allocate1D(DType.FP32, ffnDim);
        this.ffnOutScratch      = Tensor.allocate1D(DType.FP32, embDim);

        // Logits buffer: heap-backed so the same bytes alias both the
        // raw sgemv output (MemorySegment) and the heap float[] returned to
        // callers. No per-step Arena allocation, no MemorySegment.copy.
        this.logitsArray = new float[vocab];
        this.logitsSeg   = MemorySegment.ofArray(logitsArray);
    }

    public LlamaConfig config() { return cfg; }
    public Weights weights() { return weights; }
    public Tokenizer tokenizer() { return tokenizer; }
    public KVCache kvCache() { return kvCache; }
    public RopeCache ropeCache() { return ropeCache; }

    /**
     * Process a single token at {@code position} (a single decode step).
     * Returns the raw logits over the vocabulary.
     */
    public float[] forwardStep(int tokenId, long position) {
        writeTokenEmbedding(tokenId, hiddenState.data());

        for (int layer = 0; layer < cfg.blockCount(); layer++) {
            layerStep((int) layer, position);
        }

        Kernels.rmsNormInPlace(hiddenState, weights.outputNorm, cfg.rmsNormEpsilon());

        // logits = outputWeight @ x — outputWeight is row-major [vocab, embDim]
        // so a single sgemv yields all vocab logits at once. logitsSeg is
        // heap-backed (see ctor) and aliases logitsArray's bytes, so this
        // single SIMD call writes the result directly into the returned
        // float[] with no per-step Arena allocation and no MemorySegment.copy.
        Kernels.sgemv(weights.outputWeight.data(), hiddenState.data(),
                      logitsSeg, (int) cfg.vocabSize(), (int) cfg.embeddingLength());
        return logitsArray;
    }

    /**
     * Process an entire prompt. After completion the KV cache contains
     * {@code promptTokenIds.length} tokens; the caller should call
     * {@link #forwardStep(int, long)} with {@code position = promptTokenIds.length}
     * for the first generated token.
     */
    public float[] forwardPrompt(int[] promptTokenIds) {
        if (promptTokenIds.length >= cfg.contextLength()) {
            throw new IllegalArgumentException(
                "Prompt length " + promptTokenIds.length + " exceeds contextLength " + cfg.contextLength());
        }
        for (int i = 0; i < promptTokenIds.length; i++) {
            forwardStep(promptTokenIds[i], i);
        }
        return null;
    }

    // ---------------------------------------------------------------------
    // Internals
    // ---------------------------------------------------------------------

    private void writeTokenEmbedding(int tokenId, MemorySegment dst) {
        long dim = cfg.embeddingLength();
        MemorySegment emb = weights.tokenEmbeddings.data();
        long rowOff = (long) tokenId * dim * 4L;
        MemorySegment.copy(emb, rowOff, dst, 0L, dim * 4L);
    }

    private void layerStep(int layer, long position) {
        Weights.Layer L = weights.layers[layer];

        // Pre-attention RMSNorm (in place).
        Kernels.rmsNormInPlace(hiddenState, L.attnNorm, cfg.rmsNormEpsilon());

        // Q/K/V projections via sgemv on pre-transposed weights. The
        // transposed layout makes the inner SIMD K-loop stride-1 in W.
        Kernels.sgemv(attnQT[layer], hiddenState, qScratch);
        Kernels.sgemv(attnKT[layer], hiddenState, kScratch);
        Kernels.sgemv(attnVT[layer], hiddenState, vScratch);

        // Apply RoPE to q and k. ropeCache still allocates a Tensor.wrap per
        // call (pre-existing allocation churn); see WeightsCache refactor for
        // a future optimisation.
        Tensor ropeCos = ropeCache.cosFor(position, cfg.headDim());
        Tensor ropeSin = ropeCache.sinFor(position, cfg.headDim());
        int headDim = (int) cfg.headDim();
        Kernels.ropeInPlace(qScratch, ropeCos, ropeSin, headDim);
        Kernels.ropeInPlace(kScratch, ropeCos, ropeSin, headDim);

        // Write k, v into KV cache. The KV cache is flat [B, H, ctx, headDim];
        // kScratch / vScratch hold nHeadsKv * headDim contiguous floats.
        kvCache.append(layer, position, kScratch.data(), vScratch.data());

        // Inner attention: per-head (Q[h] @ K-cache row) softmax
        // (softmaxed @ V-cache row).
        attention(layer, position, headDim, (int) cfg.headCount(),
                  (int) cfg.headCountKv(), (int) cfg.contextLength());

        // Output projection: attnOutT is pre-transposed [embDim, nHeads*headDim].
        Kernels.sgemv(attnOutT[layer], attnOutScratch, projOutScratch);
        residualAddIntoHidden(projOutScratch.data());

        // FFN: rmsNorm → gate, up → silu(gate) * up → down → residual.
        Kernels.rmsNormInPlace(hiddenState, L.ffnNorm, cfg.rmsNormEpsilon());
        Kernels.sgemv(L.ffnGate, hiddenState, gateScratch);
        Kernels.sgemv(L.ffnUp,   hiddenState, upScratch);
        Kernels.siluInPlace(gateScratch);
        elementwiseMul(gateScratch, upScratch, (int) cfg.feedForwardLength());
        Kernels.sgemv(L.ffnDown, gateScratch, ffnOutScratch);
        residualAddIntoHidden(ffnOutScratch.data());
    }

    /**
     * Inner attention reduction for a single layer at the given position.
     *
     * <p>For each query head {@code h}, with KV-head mapping
     * {@code kvHead = h * nHeadsKv / nHeads} (GQA), the routine:
     * <ol>
     *   <li>Computes {@code logits[p] = sum_d Q[h,d] * K_cache[L,kvHead,p,d] / sqrt(headDim)}
     *       for {@code p in 0..position}, via {@link Kernels#sgemv} on the
     *       KV-cache row slice.</li>
     *   <li>Softmaxes {@code logits} in place via {@link Kernels#softmaxInPlaceSegment}.</li>
     *   <li>Computes {@code out[h,d] = sum_p softmaxed[p] * V_cache[L,kvHead,p,d]}
     *       for {@code d in 0..headDim}, via {@link Kernels#sgemm} with M=1,
     *       K=position+1, N=headDim.</li>
     * </ol>
     *
     * <p>The KV cache is sliced via {@link MemorySegment#asSlice(long, long)}
     * so no per-step Tensor wrap allocation occurs.
     */
    private void attention(int layer, long position, int headDim, int nHeads,
                           int nHeadsKv, int ctx) {
        long kvCacheHeadRowBytes = (long) ctx * headDim * 4L;

        MemorySegment kCache     = kvCache.keys();
        MemorySegment vCache     = kvCache.values();
        MemorySegment qSeg       = qScratch.data();
        MemorySegment attnOutSeg = attnOutScratch.data();
        MemorySegment logitsSeg  = attnLogitsScratch.data();
        float scale = (float) (1.0 / Math.sqrt((double) headDim));

        int M = (int) (position + 1);

        for (int h = 0; h < nHeads; h++) {
            int kvHead = (nHeadsKv == nHeads) ? h : (int) ((long) h * nHeadsKv / nHeads);
            long qOff = (long) h * headDim * 4L;
            long headBaseLocal = ((long) layer * nHeadsKv + kvHead) * kvCacheHeadRowBytes;

            // Step 1: logits[M] = K_row[M, headDim] @ Q[h, headDim].
            // K-row is a slice of the flat KV cache at the (layer, kvHead)
            // base, viewed as row-major [ctx, headDim]; we ask sgemv for
            // M = position + 1 rows so only the prefix is materialised.
            MemorySegment kRow = kCache.asSlice(headBaseLocal, kvCacheHeadRowBytes);
            MemorySegment qH   = qSeg.asSlice(qOff, (long) headDim * 4L);
            Kernels.sgemv(kRow, qH, logitsSeg, M, headDim);

            // Apply the 1/sqrt(headDim) scale to the first M slots.
            for (int p = 0; p < M; p++) {
                logitsSeg.set(ValueLayout.JAVA_FLOAT, (long) p * 4L,
                              logitsSeg.get(ValueLayout.JAVA_FLOAT, (long) p * 4L) * scale);
            }

            // Step 2: softmax in place on the prefix.
            Kernels.softmaxInPlaceSegment(logitsSeg, M);

            // Step 3: out[h, headDim] = softmaxed[1, M] @ V_row[M, headDim]
            // via raw sgemm with M=1, K=M, N=headDim.
            MemorySegment softmaxedRow = logitsSeg.asSlice(0L, (long) M * 4L);
            MemorySegment vRow         = vCache.asSlice(headBaseLocal, (long) M * headDim * 4L);
            MemorySegment outRow       = attnOutSeg.asSlice(qOff, (long) headDim * 4L);
            Kernels.sgemm(softmaxedRow, vRow, outRow, 1, M, headDim);
        }
    }

    /** hiddenState += add (in-place residual). */
    private void residualAddIntoHidden(MemorySegment add) {
        MemorySegment x = hiddenState.data();
        int n = (int) cfg.embeddingLength();
        for (int i = 0; i < n; i++) {
            long off = (long) i * 4L;
            x.set(ValueLayout.JAVA_FLOAT, off,
                  x.get(ValueLayout.JAVA_FLOAT, off) + add.get(ValueLayout.JAVA_FLOAT, off));
        }
    }

    // ---------------------------------------------------------------------
    // Load-time weight transposes
    // ---------------------------------------------------------------------

    /**
     * Transpose a rank-3 GGUF attention projection weight
     * {@code [embDim, nHeads{,Kv}, headDim]} into SIMD-friendly 2D form
     * {@code [nHeads * headDim, embDim]}. After the transpose, the new
     * layout has {@code W'[h*headDim + d, e] = W[e, h, d]}, which is the
     * exact ordering {@link Kernels#sgemv} expects for the matvec
     * {@code y[h*headDim + d] = sum_e W'[h*headDim + d, e] * x[e]}.
     */
    private static Tensor transposeAttnProj(Tensor src3D, int embDim, int nHeads, int headDim) {
        Tensor dst = Tensor.allocate2D(DType.FP32, nHeads * headDim, embDim);
        MemorySegment src = src3D.data();
        MemorySegment d   = dst.data();
        long srcStrideE = (long) nHeads * headDim * 4L;
        long dstStrideR = (long) embDim * 4L;
        for (int e = 0; e < embDim; e++) {
            long srcRowBase = (long) e * srcStrideE;
            for (int h = 0; h < nHeads; h++) {
                long srcHeadBase = srcRowBase + (long) h * headDim * 4L;
                long dstRowBase  = (long) (h * headDim) * dstStrideR + (long) e * 4L;
                for (int dd = 0; dd < headDim; dd++) {
                    d.set(ValueLayout.JAVA_FLOAT, dstRowBase + (long) dd * dstStrideR,
                          src.get(ValueLayout.JAVA_FLOAT, srcHeadBase + (long) dd * 4L));
                }
            }
        }
        return dst;
    }

    /**
     * Transpose a rank-3 GGUF attn-output weight
     * {@code [nHeads, headDim, embDim]} into SIMD-friendly 2D form
     * {@code [embDim, nHeads * headDim]}. After the transpose, the new
     * layout has {@code W'[e, h*headDim + d] = W[h, d, e]}, which is the
     * exact ordering {@link Kernels#sgemv} expects for the matvec
     * {@code y[e] = sum_k W'[e, k] * x[k]}.
     */
    private static Tensor transposeAttnOut(Tensor src3D, int nHeads, int headDim, int embDim) {
        Tensor dst = Tensor.allocate2D(DType.FP32, embDim, nHeads * headDim);
        MemorySegment src = src3D.data();
        MemorySegment d   = dst.data();
        long srcStrideH = (long) headDim * embDim * 4L;
        long srcStrideD = (long) embDim * 4L;
        long dstStrideR = (long) (nHeads * headDim) * 4L;
        for (int h = 0; h < nHeads; h++) {
            long srcHeadBase = (long) h * srcStrideH;
            for (int dd = 0; dd < headDim; dd++) {
                long srcDimBase = srcHeadBase + (long) dd * srcStrideD;
                long dstDimBase = (long) (h * headDim + dd) * 4L;
                for (int e = 0; e < embDim; e++) {
                    d.set(ValueLayout.JAVA_FLOAT, (long) e * dstStrideR + dstDimBase,
                          src.get(ValueLayout.JAVA_FLOAT, srcDimBase + (long) e * 4L));
                }
            }
        }
        return dst;
    }

    /** a *= b elementwise on two same-length Tensors. */
    private static void elementwiseMul(Tensor a, Tensor b, int dim) {
        MemorySegment aSeg = a.data();
        MemorySegment bSeg = b.data();
        for (int i = 0; i < dim; i++) {
            long off = (long) i * 4L;
            aSeg.set(ValueLayout.JAVA_FLOAT, off,
                     aSeg.get(ValueLayout.JAVA_FLOAT, off) * bSeg.get(ValueLayout.JAVA_FLOAT, off));
        }
    }
}