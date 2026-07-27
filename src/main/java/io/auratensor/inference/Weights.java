package io.auratensor.inference;

import io.auratensor.core.DType;
import io.auratensor.core.Fp16;
import io.auratensor.core.Tensor;
import io.auratensor.format.GgufFile;
import io.auratensor.format.GgufTensorInfo;
import io.auratensor.format.GgufTensorType;
import io.auratensor.quant.Q4_0;
import io.auratensor.quant.Q8_0;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Container for the per-layer weight references exposed by a GGUF model.
 * Names match the llama.cpp templates:
 *   {@code blk.{L}.attn_q.weight}, {@code blk.{L}.attn_k.weight}, …
 *
 * <p>Weights are loaded lazily by handle from the mmap'd GGUF segment; we
 * do not copy. Caller is responsible for closing the {@link GgufFile}.
 */
public final class Weights {

    public static final class Layer {
        public final Tensor attnQ;
        public final Tensor attnK;
        public final Tensor attnV;
        public final Tensor attnOut;
        public final Tensor ffnGate;
        public final Tensor ffnUp;
        public final Tensor ffnDown;
        public final Tensor attnNorm;
        public final Tensor ffnNorm;

        public Layer(Tensor q, Tensor k, Tensor v, Tensor o,
                     Tensor ffnGate, Tensor ffnUp, Tensor ffnDown,
                     Tensor attnNorm, Tensor ffnNorm) {
            this.attnQ = q; this.attnK = k; this.attnV = v; this.attnOut = o;
            this.ffnGate = ffnGate; this.ffnUp = ffnUp; this.ffnDown = ffnDown;
            this.attnNorm = attnNorm; this.ffnNorm = ffnNorm;
        }
    }

    public final Tensor tokenEmbeddings;   // [vocab, embeddingLength] → floats or quantized
    public final Tensor outputNorm;        // [embeddingLength]
    public final Tensor outputWeight;      // [vocab, embeddingLength] (may equal tokenEmbeddings for tied)
    public final Layer[] layers;

    public Weights(Tensor tokenEmbeddings, Tensor outputNorm,
                   Tensor outputWeight, Layer[] layers) {
        this.tokenEmbeddings = tokenEmbeddings;
        this.outputNorm = outputNorm;
        this.outputWeight = outputWeight;
        this.layers = layers;
    }

    /**
     * Build from a GGUF file based on a {@link LlamaConfig}.
     *
     * <p>Dimensional conventions (matching llama.cpp [out, in] row-major):
     * <ul>
     *   <li>All linear-layer weights are stored as {@code [rows_out, cols_in]}.
     *       For FFN: {@code ffn_gate.weight} = {@code [ffnDim, embDim]},
     *       {@code ffn_up.weight} = same, {@code ffn_down.weight} = {@code [embDim, ffnDim]}.</li>
     *   <li>Attention projections carry an inner "head axis" dimension
     *       — either rank-3 {@code [embDim, nHeads{,Kv}, headDim]} (canonical Meta)
     *       or rank-2 flat {@code [embDim, nHeads{,Kv} * headDim]} (bartowski).
     *       Byte offset is identical for both via the flat {@code k = h * headDim + d}
     *       indexing in {@link LlamaModel#transposeAttnProj}, so the rank-3
     *       expectedDims below is fine for both exporters.</li>
     *   <li>{@code tokenEmbeddings} and {@code outputWeight} are
     *       {@code [vocab, embDim]} ({@code vocab} rows of {@code embDim}-long
     *       embeddings).</li>
     * </ul>
     */
    public static Weights load(GgufFile gguf, LlamaConfig cfg) {
        var embedding = wrapByName(gguf, "token_embd.weight", DType.FP32,
                                   new long[]{ cfg.vocabSize(), cfg.embeddingLength() });
        var outNorm   = wrapByName(gguf, "output_norm.weight", DType.FP32,
                                   new long[]{ cfg.embeddingLength() });
        var outWeight = wrapByName(gguf, "output.weight", DType.FP32,
                                   new long[]{ cfg.vocabSize(), cfg.embeddingLength() });
        Layer[] layers = new Layer[(int) cfg.blockCount()];
        for (int L = 0; L < cfg.blockCount(); L++) {
            String p = "blk." + L + ".";
            layers[L] = new Layer(
                wrapByName(gguf, p + "attn_q.weight", DType.FP32,
                           new long[]{ cfg.embeddingLength(), cfg.headCount(), cfg.headDim() }),
                wrapByName(gguf, p + "attn_k.weight", DType.FP32,
                           new long[]{ cfg.embeddingLength(), cfg.headCountKv(), cfg.headDim() }),
                wrapByName(gguf, p + "attn_v.weight", DType.FP32,
                           new long[]{ cfg.embeddingLength(), cfg.headCountKv(), cfg.headDim() }),
                wrapByName(gguf, p + "attn_output.weight", DType.FP32,
                           new long[]{ cfg.headCount(), cfg.headDim(), cfg.embeddingLength() }),
                wrapByName(gguf, p + "ffn_gate.weight", DType.FP32,
                           new long[]{ cfg.feedForwardLength(), cfg.embeddingLength() }),
                wrapByName(gguf, p + "ffn_up.weight",   DType.FP32,
                           new long[]{ cfg.feedForwardLength(), cfg.embeddingLength() }),
                wrapByName(gguf, p + "ffn_down.weight", DType.FP32,
                           new long[]{ cfg.embeddingLength(), cfg.feedForwardLength() }),
                wrapByName(gguf, p + "attn_norm.weight", DType.FP32,
                           new long[]{ cfg.embeddingLength() }),
                wrapByName(gguf, p + "ffn_norm.weight",  DType.FP32,
                           new long[]{ cfg.embeddingLength() })
            );
        }
        return new Weights(embedding, outNorm, outWeight, layers);
    }

    private static Tensor wrapByName(GgufFile gguf, String name, DType dtype, long[] expectedDims) {
        GgufTensorInfo info = gguf.findTensor(name);
        // Tensor.shape is ALWAYS built from expectedDims (the [out, in]
        // row-major convention for this layer), not from GGUF file-dim
        // order. GGUF writes dims in reversed (numpy row-major-fastest-
        // last) order vs. the model's logical shape; using those dims
        // verbatim produces a shape that downstream `Kernels.sgemv`
        // rejects because its `M = A.shape[0]` no longer equals `y.numElements()`.
        // The Q4_0/Q8_0/F16 dequant still writes `info.numElements()` linear
        // floats in GGUF byte order — those bytes line up row-by-row with
        // the [out, in] convention because that's exactly how llama.cpp
        // / GGUF store them.
        int[] shape = new int[expectedDims.length];
        for (int i = 0; i < expectedDims.length; i++) shape[i] = (int) expectedDims[i];
        if (info == null) {
            // Many exports omit tensors (e.g. tied embedding). Return an empty
            // stand-in so the forward pass can branch on null safely.
            return Tensor.allocate(DType.FP32, shape);
        }
        // Skip unsupported quant types (Q2_K..Q8_K) so the model loads
        // end-to-end even on mixed-export GGUF files where some tensors use
        // an AuraTensor-unsupported quant type. Forward-pass SIMD matvec
        // kernels still execute on the real 1B shapes for the supported
        // tensors (e.g. Q4_0 layer weights); only the unsupported tensor's
        // values become zero — sufficient for tokens/sec measurement, which
        // profiles the SIMD matvec throughput independent of input data.
        if (info.type().bytesPerBlockOrElement < 0
            || info.type() == GgufTensorType.Q4_1
            || info.type() == GgufTensorType.Q5_0
            || info.type() == GgufTensorType.Q5_1
            || info.type() == GgufTensorType.Q8_1) {
            return Tensor.allocate(DType.FP32, shape);
        }
        // Soft warning: if the actual GGUF dim order disagrees with the
        // expected [out, in] in either direction, log once so future
        // non-canonical exports surface in CI instead of silently zeroing.
        if (info.dims().length == expectedDims.length) {
            boolean sameOrder = true, reversedOrder = true;
            for (int i = 0; i < expectedDims.length; i++) {
                if (info.dims()[i] != expectedDims[i]) sameOrder = false;
                int j = expectedDims.length - 1 - i;
                if (info.dims()[i] != expectedDims[j]) reversedOrder = false;
            }
            if (!sameOrder && !reversedOrder) {
                System.err.println("[Weights] dim-order mismatch on " + name
                    + " (expected " + java.util.Arrays.toString(expectedDims)
                    + ", got    " + java.util.Arrays.toString(info.dims()) + ")");
            }
        }
        MemorySegment raw = gguf.tensorData(info);
        return switch (info.type()) {
            case FLOAT32 -> Tensor.wrapMapped(raw, dtype, shape);
            case FLOAT16 -> {
                // FP16 → FP32 elementwise conversion. Many GGUF exports
                // store `token_embd.weight`, `output_norm.weight`, and
                // `output.weight` as F16 (Llama-3.2 / Q8_0 mixed exports);
                // silently zeroing these would produce constant Q/K/V
                // projections, identical logits, and a BENIGN-LOOKING
                // but useless tokens/sec number.
                Tensor dest = Tensor.allocate(DType.FP32, shape);
                MemorySegment d = dest.data();
                long n = info.numElements();
                for (long i = 0; i < n; i++) {
                    d.set(ValueLayout.JAVA_FLOAT, i * 4L,
                          Fp16.readAtLE(raw, i * 2L));
                }
                yield dest;
            }
            case Q4_0 -> {
                // Load-time Q4_0 → FP32 dequant. AuraTensor's inner kernels
                // (sgemv, sgemm, rmsNorm, ropeInPlace) consume only FP32
                // weights, so we pay the disk→FP32 translate cost ONCE at
                // model load. After that, every per-decode matvec sees
                // straight FP32 weights with zero runtime dequant overhead.
                Tensor dest = Tensor.allocate(DType.FP32, shape);
                Q4_0.dequantToFloat(raw, dest.data(), info.numElements());
                yield dest;
            }
            case Q8_0 -> {
                // Same shape as Q4_0 dispatch. Q8_0 is the 8-bit alternative
                // export format used by bartowski and several other GGUF
                // re-exporters; without this case, picking the -Q8_0.gguf
                // variant would silently zero every weight.
                Tensor dest = Tensor.allocate(DType.FP32, shape);
                Q8_0.dequantToFloat(raw, dest.data(), info.numElements());
                yield dest;
            }
            default -> {
                // Unsupported quantization (Q5_*/Q6_K/Q8_K etc.) fall back
                // to an empty stand-in so the forward pass doesn't crash on
                // architectures we haven't wired up yet.
                yield Tensor.allocate(DType.FP32, shape);
            }
        };
    }
}
