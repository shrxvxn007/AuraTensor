package io.auratensor.inference;

import io.auratensor.format.GgufMetadata;

/**
 * Model hyperparameters for a Llama 3 / Mistral transformer, extracted
 * from GGUF metadata key-value pairs.
 *
 * <p>Fields fall back to safe defaults if not present, so partially-populated
 * GGUF exports still load.
 */
public record LlamaConfig(
    String architecture,
    long contextLength,
    long embeddingLength,
    long blockCount,
    long headCount,
    long headCountKv,
    long feedForwardLength,
    float ropeFrequencyBase,
    float rmsNormEpsilon,
    long vocabSize
) {
    /** Derive {@code headDim} from embeddings / heads. */
    public long headDim() {
        if (headCount == 0) throw new IllegalStateException("headCount == 0");
        return embeddingLength / headCount;
    }

    public static LlamaConfig fromGguf(GgufMetadata m) {
        String arch = m.stringOrDefault("general.architecture", "llama");
        long nCtx   = m.longOrDefault(arch + ".context_length", 4096);
        long nEmb   = m.longOrDefault(arch + ".embedding_length", 4096);
        long nBlocks= m.longOrDefault(arch + ".block_count", 32);
        long nHeads = m.longOrDefault(arch + ".attention.head_count", 32);
        long nHeadsKv = m.longOrDefault(arch + ".attention.head_count_kv", nHeads);
        long nFf    = m.longOrDefault(arch + ".feed_forward_length", 11008);

        // RoPE base depends on architecture variant:
        //  - Llama 3: 500000.0 (embed >= 4096 is a reliable proxy)
        //  - Llama 2 / Mistral: 10000.0
        float rope = m.floatOrDefault(arch + ".rope.freq_base",
            (arch.contains("llama") && nEmb >= 4096) ? 500000.0f : 10000.0f);

        float eps = m.floatOrDefault(arch + ".attention.layer_norm_rms_epsilon", 1e-5f);

        // vocab size lives under token_embd in many templates; fall back to known fields.
        long vocab = m.longOrDefault(arch + ".vocab_size",
              m.longOrDefault("tokenizer.ggml.vocab_size", 32000L));

        return new LlamaConfig(arch, nCtx, nEmb, nBlocks, nHeads, nHeadsKv,
                               nFf, rope, eps, vocab);
    }
}
