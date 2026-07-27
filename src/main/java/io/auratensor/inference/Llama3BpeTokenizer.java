package io.auratensor.inference;

import io.auratensor.format.GgufFile;
import io.auratensor.format.GgufMetadata;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal Llama 3 BPE tokenizer assembled from {@code tokenizer.ggml.*}
 * fields stored in a GGUF file's metadata section.
 *
 * <p>This is deliberately a surface-only implementation: it captures the
 * vocabulary (token id → string) and the EOS token id so a forward + sampling
 * loop can decide when to stop generation. It does <b>not</b> implement
 * the Llama 3 regex pre-tokenizer or the BPE merge table — a production
 * tokenizer would plug those in on top of this surface. The benchmark
 * paths invoke {@link #decode(int)} and {@link #eosTokenId()} but never
 * {@link #encode(String)}, so an empty encoding is sufficient.
 *
 * <p>EOS resolution order:
 * <ol>
 *   <li>{@code tokenizer.ggml.eos_token_id} (explicit int metadata)</li>
 *   <li>Scan {@code tokenizer.ggml.tokens} for {@code "<|eot_id|>"} or
 *       {@code "</s>"}</li>
 *   <li>Llama 3 default {@link #LLAMA3_DEFAULT_EOS} ({@code 128001},
 *       {@code "<|end_of_text|>"})</li>
 * </ol>
 */
public final class Llama3BpeTokenizer implements Tokenizer {

    /** Llama 3 default BOS/EOS shared token id: {@code "<|end_of_text|>"}. */
    public static final int LLAMA3_DEFAULT_EOS = 128001;
    /** Llama 3 dedicated end-of-turn token id: {@code "<|eot_id|>"}. */
    public static final int LLAMA3_EOT         = 128009;

    private final Map<Integer, String> vocab;
    private final int eosTokenId;

    public Llama3BpeTokenizer(Map<Integer, String> vocab, int eosTokenId) {
        this.vocab = Collections.unmodifiableMap(new LinkedHashMap<>(vocab));
        this.eosTokenId = eosTokenId;
    }

    /**
     * Build from a GGUF file. Reads {@code tokenizer.ggml.tokens} (string
     * array) and resolves the EOS token via the precedence order documented
     * above. Idempotent and allocation-light: it is safe to call once at
     * model-load time.
     */
    public static Llama3BpeTokenizer fromGguf(GgufFile gguf) {
        GgufMetadata m = gguf.metadata();
        List<?> tokensRaw = m.arrayOrEmpty("tokenizer.ggml.tokens");
        Map<Integer, String> vocab = new LinkedHashMap<>();
        int eos = LLAMA3_DEFAULT_EOS;
        for (int i = 0; i < tokensRaw.size(); i++) {
            String tok = String.valueOf(tokensRaw.get(i));
            vocab.put(i, tok);
            if ("<|eot_id|>".equals(tok) || "</s>".equals(tok)) {
                eos = i;
            }
        }
        // GGUF v3 commonly stores the canonical EOS id alongside the vocab
        // (bartowski / ollama exports); check that as the highest-priority
        // signal.
        if (m.has("tokenizer.ggml.eos_token_id")) {
            eos = m.intOrDefault("tokenizer.ggml.eos_token_id", eos);
        }
        return new Llama3BpeTokenizer(vocab, eos);
    }

    /** Empty encoding — AuraTensor's decoder path doesn't invoke encoding. */
    @Override
    public int[] encode(String text) { return new int[0]; }

    @Override
    public String decode(int tokenId) { return vocab.getOrDefault(tokenId, ""); }

    @Override
    public int eosTokenId() { return eosTokenId; }

    @Override
    public Map<Integer, String> vocab() { return vocab; }

    /** Number of unique tokens loaded from {@code tokenizer.ggml.tokens}. */
    public int vocabSize() { return vocab.size(); }
}
