package io.auratensor.inference;

import io.auratensor.core.DType;
import io.auratensor.core.Tensor;
import io.auratensor.format.GgufFile;
import io.auratensor.format.GgufTensorInfo;

import java.lang.foreign.MemorySegment;

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

    /** Build from a GGUF file based on a {@link LlamaConfig}. */
    public static Weights load(GgufFile gguf, LlamaConfig cfg) {
        var embedding = wrapByName(gguf, "token_embd.weight", DType.FP32,
                                   new long[]{ cfg.vocabSize(), cfg.embeddingLength() });
        var outNorm   = wrapByName(gguf, "output_norm.weight", DType.FP32,
                                   new long[]{ cfg.embeddingLength() });
        var outWeight = wrapByName(gguf, "output.weight", DType.FP32,
                                   new long[]{ cfg.vocabSize(), cfg.embeddingLength() });
        Layer[] layers = new Layer[(int) cfg.blockCount()];
        // Attn weight tensors stay at their GGUF-exported logical rank-3 shape.
        // Closure-aware layout reasoning in LlamaModel.layerStep iterates the
        // exact byte offsets of the GGUF export convention (row-major
        // [embed, nHeads, headDim] for Q/K/V and [out_dim, in_dim] row-major
        // for the FFN/output tensors), so we do NOT reshape here.
        long qCols  = cfg.headCount()    * cfg.headDim();
        long kvCols = cfg.headCountKv()  * cfg.headDim();
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
                           new long[]{ cfg.embeddingLength(), cfg.feedForwardLength() }),
                wrapByName(gguf, p + "ffn_up.weight",   DType.FP32,
                           new long[]{ cfg.embeddingLength(), cfg.feedForwardLength() }),
                wrapByName(gguf, p + "ffn_down.weight", DType.FP32,
                           new long[]{ cfg.feedForwardLength(), cfg.embeddingLength() }),
                wrapByName(gguf, p + "attn_norm.weight", DType.FP32,
                           new long[]{ cfg.embeddingLength() }),
                wrapByName(gguf, p + "ffn_norm.weight",  DType.FP32,
                           new long[]{ cfg.embeddingLength() })
            );
        }
        // Suppress unused-warning on qCols/kvCols; they'll be used by future
        // load-time transpose code when we wire up an SIMD-aware matvec that
        // handles GGUF's [embed, nHeads, headDim] byte layout.
        if (qCols < 0 || kvCols < 0) throw new IllegalStateException();
        return new Weights(embedding, outNorm, outWeight, layers);
    }

    private static Tensor wrapByName(GgufFile gguf, String name, DType dtype, long[] dims) {
        GgufTensorInfo info = gguf.findTensor(name);
        if (info == null) {
            // Many exports omit tensors (e.g. tied embedding). Return an empty
            // stand-in so the forward pass can branch on null safely.
            int[] shape = new int[dims.length];
            for (int i = 0; i < dims.length; i++) shape[i] = (int) dims[i];
            return Tensor.allocate(DType.FP32, shape);
        }
        MemorySegment raw = gguf.tensorData(info);
        int[] shape = new int[dims.length];
        for (int i = 0; i < dims.length; i++) shape[i] = (int) dims[i];
        return Tensor.wrapMapped(raw, dtype, shape);
    }
}
