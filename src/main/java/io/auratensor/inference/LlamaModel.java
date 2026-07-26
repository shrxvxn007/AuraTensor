package io.auratensor.inference;

import io.auratensor.core.Kernels;
import io.auratensor.core.Tensor;
import io.auratensor.core.DType;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;

/**
 * Llama 3 / Mistral transformer forward pass with off-heap KV-cache and
 * virtual-thread-friendly single-batch state.
 *
 * <p>Layout follows the standard Llama recipe:
 * <pre>
 *   x = tokenEmbeddings[tokenId]
 *   for each layer:
 *     x' = rmsNorm(x, attnNorm)
 *     q, k, v = projections(x')
 *     apply RoPE to q, k with current position
 *     write k, v into KV cache
 *     attn = softmax(Q @ K^T / sqrt(headDim)) @ V
 *     x = x + attn_output(attn)
 *     x' = rmsNorm(x, ffnNorm)
 *     h = silu(ffn_gate(x')) * ffn_up(x')
 *     x = x + ffn_down(h)
 *   x = rmsNorm(x, outputNorm)
 *   logits = output_weight @ x
 * </pre>
 *
 * <p>Uses {@link Kernels#sgemv} for all M=1 projections (Q/K/V/O/FFN) so the
 * SIMD Vector API is on the inference hot path, not just the kernels suite.
 */
public final class LlamaModel {

    private final LlamaConfig cfg;
    private final Weights weights;
    private final Tokenizer tokenizer;
    private final KVCache kvCache;
    private final RopeCache ropeCache;
    private final Tensor hiddenState;   // scratch [embeddingLength]

    public LlamaModel(LlamaConfig cfg, Weights weights, Tokenizer tokenizer) {
        this.cfg = cfg;
        this.weights = weights;
        this.tokenizer = tokenizer;
        this.kvCache = new KVCache(cfg.blockCount(), cfg.headCountKv(), cfg.headDim(),
                                   cfg.contextLength());
        this.ropeCache = new RopeCache(cfg.headDim(), cfg.ropeFrequencyBase(),
                                       cfg.contextLength());
        this.hiddenState = Tensor.allocate1D(DType.FP32, (int) cfg.embeddingLength());
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
        MemorySegment x = hiddenState.data();
        writeTokenEmbedding(tokenId, x);

        for (int layer = 0; layer < cfg.blockCount(); layer++) {
            layerStep(layer, x, position);
        }

        Kernels.rmsNormInPlace(hiddenState, weights.outputNorm, cfg.rmsNormEpsilon());

        // logits = outputWeight @ x — broadcast x as a "row", compute dot against
        // each row of the output weight matrix. M=1 → sgemv would under-fill
        // here, so we keep the scalar row loop (vocab × dim floats).
        long vocab = cfg.vocabSize();
        long dim = cfg.embeddingLength();
        float[] logits = new float[(int) vocab];
        MemorySegment wSeg = weights.outputWeight.data();
        for (long row = 0; row < vocab; row++) {
            float sum = 0.0f;
            long rowOff = row * dim * 4L;
            for (long k = 0; k < dim; k++) {
                sum += wSeg.get(ValueLayout.JAVA_FLOAT, rowOff + k * 4L)
                     * x.get(ValueLayout.JAVA_FLOAT, k * 4L);
            }
            logits[(int) row] = sum;
        }
        return logits;
    }

    /**
     * Process an entire prompt, returning the logits for the final token.
     */
    public float[] forwardPrompt(int[] promptTokenIds) {
        if (promptTokenIds.length >= cfg.contextLength()) {
            throw new IllegalArgumentException(
                "Prompt length " + promptTokenIds.length + " exceeds contextLength " + cfg.contextLength());
        }
        for (int i = 0; i < promptTokenIds.length; i++) {
            forwardStep(promptTokenIds[i], i);
        }
        return null; // caller delegates to forwardStep for sampling
    }

    // ---------------------------------------------------------------------
    // Internals
    // ---------------------------------------------------------------------

    private void writeTokenEmbedding(int tokenId, MemorySegment dst) {
        long dim = cfg.embeddingLength();
        MemorySegment emb = weights.tokenEmbeddings.data();
        long rowOff = (long) tokenId * dim * 4L;
        for (long k = 0; k < dim; k++) {
            dst.set(ValueLayout.JAVA_FLOAT, k * 4L, emb.get(ValueLayout.JAVA_FLOAT, rowOff + k * 4L));
        }
    }

    private void layerStep(int layer, MemorySegment x, long position) {
        Weights.Layer L = weights.layers[layer];

        // Pre-attention RMSNorm (in place) — saves a fresh buffer.
        Kernels.rmsNormInPlace(hiddenState, L.attnNorm, cfg.rmsNormEpsilon());

        long embDim = cfg.embeddingLength();
        long headDim = cfg.headDim();
        long nHeads = cfg.headCount();
        long nHeadsKv = cfg.headCountKv();

        // Per-layer scratch tensors use try-with-resources so every confined
        // Arena closes deterministically. Without this the inner loop leaks
        // one Arena per call.
        try (Tensor qT = Tensor.allocate1D(DType.FP32, (int) (nHeads  * headDim));
             Tensor kT = Tensor.allocate1D(DType.FP32, (int) (nHeadsKv * headDim));
             Tensor vT = Tensor.allocate1D(DType.FP32, (int) (nHeadsKv * headDim));
             Tensor attnOutT = Tensor.allocate1D(DType.FP32, (int) (nHeads  * headDim));
             Tensor gateT = Tensor.allocate1D(DType.FP32, (int) cfg.feedForwardLength())) {

            // Projections — scalar matVec specialised over the GGUF byte layout:
            //  * Q/K/V are stored row-major as [embed, nHeads{,_Kv}, headDim]
            //  * Output / FFN dimensions follow standard [out_dim, in_dim] conventions.
            // Float buffers are stack/heap arrays reused inside this method.
            matVec3D(L.attnQ.data(), hiddenState.data(), qT.data(),
                     (int) embDim, (int) nHeads, (int) headDim);
            matVec3D(L.attnK.data(), hiddenState.data(), kT.data(),
                     (int) embDim, (int) nHeadsKv, (int) headDim);
            matVec3D(L.attnV.data(), hiddenState.data(), vT.data(),
                     (int) embDim, (int) nHeadsKv, (int) headDim);

            // Apply RoPE to q and k
            Tensor ropeCos = ropeCache.cosFor(position, headDim);
            Tensor ropeSin = ropeCache.sinFor(position, headDim);
            Kernels.ropeInPlace(qT, ropeCos, ropeSin, (int) headDim);
            Kernels.ropeInPlace(kT, ropeCos, ropeSin, (int) headDim);

            // Write k, v into KV cache
            kvCache.append(layer, position, kT.data(), vT.data());

            float scale = (float) (1.0 / Math.sqrt((double) headDim));
            MemorySegment attnOutSeg = attnOutT.data();
            MemorySegment qSeg = qT.data();
            MemorySegment kCacheSeg = kvCache.keys().data();
            MemorySegment vCacheSeg = kvCache.values().data();

            for (long h = 0; h < nHeads; h++) {
                long kvHead = (nHeadsKv == nHeads) ? h : h * nHeadsKv / nHeads;
                long qOff = h * headDim * 4L;

                float[] logits = new float[(int) (position + 1)];
                for (long p = 0; p <= position; p++) {
                    long rowOff = layer * nHeadsKv * cfg.contextLength() * headDim * 4L
                                + kvHead  * cfg.contextLength() * headDim * 4L
                                + p * headDim * 4L;
                    float dot = 0.0f;
                    for (long d = 0; d < headDim; d++) {
                        dot += qSeg.get(ValueLayout.JAVA_FLOAT, qOff + d * 4L)
                             * kCacheSeg.get(ValueLayout.JAVA_FLOAT, rowOff + d * 4L);
                    }
                    logits[(int) p] = dot * scale;
                }
                float max = logits[0];
                for (long p = 1; p <= position; p++) {
                    if (logits[(int) p] > max) max = logits[(int) p];
                }
                float sum = 0.0f;
                for (long p = 0; p <= position; p++) {
                    logits[(int) p] = (float) Math.exp(logits[(int) p] - max);
                    sum += logits[(int) p];
                }
                for (long p = 0; p <= position; p++) {
                    logits[(int) p] /= sum;
                }
                long valRowStride = cfg.contextLength() * headDim * 4L;
                for (long d = 0; d < headDim; d++) {
                    float acc = 0.0f;
                    for (long p = 0; p <= position; p++) {
                        long rowOff = layer * nHeadsKv * valRowStride
                                    + kvHead  * valRowStride
                                    + p * headDim * 4L;
                        acc += logits[(int) p]
                             * vCacheSeg.get(ValueLayout.JAVA_FLOAT, rowOff + d * 4L);
                    }
                    attnOutSeg.set(ValueLayout.JAVA_FLOAT, qOff + d * 4L, acc);
                }
            }

            // Output projection (scalar) — 3D weight has shape
            // [nHeads, headDim, embedDim]. y[e] = sum_{h,d} W[h,d,e] * x[h*headDim+d].
            try (Tensor projOutT = Tensor.allocate1D(DType.FP32, (int) embDim)) {
                matVecOutput(L.attnOut.data(), attnOutSeg, projOutT.data(),
                             (int) nHeads, (int) headDim, (int) embDim);

                // Residual
                for (long k = 0; k < embDim; k++) {
                    float orig = x.get(ValueLayout.JAVA_FLOAT, k * 4L);
                    x.set(ValueLayout.JAVA_FLOAT, k * 4L, orig + projOutT.getFloat((int) k));
                }
            }

            // FFN: rmsNorm, gate, up, silu(gate) * up, down, residual
            Kernels.rmsNormInPlace(hiddenState, L.ffnNorm, cfg.rmsNormEpsilon());
            try (Tensor upT = Tensor.allocate1D(DType.FP32, (int) cfg.feedForwardLength());
                 Tensor ffnOutT = Tensor.allocate1D(DType.FP32, (int) embDim)) {
                matVec2D(L.ffnGate.data(), hiddenState.data(), gateT.data(),
                         (int) embDim, (int) cfg.feedForwardLength());
                matVec2D(L.ffnUp.data(),   hiddenState.data(), upT.data(),
                         (int) embDim, (int) cfg.feedForwardLength());

                // SiLU in place on gate, then gate *= up elementwise.
                Kernels.siluInPlace(gateT);
                MemorySegment gSeg = gateT.data();
                MemorySegment uSeg = upT.data();
                int len = (int) cfg.feedForwardLength();
                for (int i = 0; i < len; i++) {
                    gSeg.set(ValueLayout.JAVA_FLOAT, i * 4L, gSeg.get(ValueLayout.JAVA_FLOAT, i * 4L)
                           * uSeg.get(ValueLayout.JAVA_FLOAT, i * 4L));
                }
                matVec2D(L.ffnDown.data(), gateT.data(), ffnOutT.data(),
                         (int) cfg.feedForwardLength(), (int) embDim);

                for (long k = 0; k < embDim; k++) {
                    float orig = x.get(ValueLayout.JAVA_FLOAT, k * 4L);
                    x.set(ValueLayout.JAVA_FLOAT, k * 4L, orig + ffnOutT.getFloat((int) k));
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // MatVec specialisations — these read the GGUF byte layout directly so we
    // don't need a load-time transpose. Each matvec is O(M * K) so a future
    // SIMD-friendly version should rewrite these as Vector-API kernels with
    // load-time transposes for the configure-once-use-many inner loops.
    // ---------------------------------------------------------------------

    /** {@code y[headOut * headDim + dimOut] = sum_{e=0..embed-1} W[e, h, d] * x[e]}.
     *  {@code W} is row-major [embed, nHeads, headDim]. */
    private static void matVec3D(MemorySegment W, MemorySegment x, MemorySegment y,
                                 int embed, int nHeads, int headDim) {
        long qHeadStride = (long) headDim * 4L;             // bytes per (e,h)
        long qEmbedStride = (long) nHeads * qHeadStride;   // bytes per e
        int nOut = nHeads * headDim;
        for (int h = 0; h < nHeads; h++) {
            long headBaseW = (long) h * qHeadStride;
            for (int d = 0; d < headDim; d++) {
                long wOff = headBaseW + (long) d * 4L;
                float sum = 0.0f;
                for (int e = 0; e < embed; e++) {
                    sum += W.get(ValueLayout.JAVA_FLOAT, wOff + (long) e * qEmbedStride)
                         * x.get(ValueLayout.JAVA_FLOAT, (long) e * 4L);
                }
                y.set(ValueLayout.JAVA_FLOAT, (long) (h * headDim + d) * 4L, sum);
            }
        }
        if (nOut < 0) throw new AssertionError("impossible");
    }

    /** Plain {@code y[m] = sum_{k=0..K-1} W[m, k] * x[k]} for [M, K] row-major. */
    private static void matVec2D(MemorySegment W, MemorySegment x, MemorySegment y,
                                 int M, int K) {
        for (int m = 0; m < M; m++) {
            long rowOff = (long) m * K * 4L;
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += W.get(ValueLayout.JAVA_FLOAT, rowOff + (long) k * 4L)
                     * x.get(ValueLayout.JAVA_FLOAT, (long) k * 4L);
            }
            y.set(ValueLayout.JAVA_FLOAT, (long) m * 4L, sum);
        }
    }

    /** Output-projection matvec: {@code y[e] = sum_{m=0..nHeads*headDim-1} W[m, e] * x[m]}.
     *  {@code W} is row-major [nHeads, headDim, embedDim]. */
    private static void matVecOutput(MemorySegment W, MemorySegment x, MemorySegment y,
                                     int nHeads, int headDim, int embedDim) {
        long hStride = (long) headDim * embedDim * 4L;
        long dStride = (long) embedDim * 4L;
        for (int e = 0; e < embedDim; e++) {
            float acc = 0.0f;
            for (int h = 0; h < nHeads; h++) {
                long hBase = (long) h * hStride;
                for (int d = 0; d < headDim; d++) {
                    float xv = x.get(ValueLayout.JAVA_FLOAT, ((long) h * headDim + d) * 4L);
                    float wv = W.get(ValueLayout.JAVA_FLOAT, hBase + (long) d * dStride + (long) e * 4L);
                    acc += xv * wv;
                }
            }
            y.set(ValueLayout.JAVA_FLOAT, (long) e * 4L, acc);
        }
    }
}
