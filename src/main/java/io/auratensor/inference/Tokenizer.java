package io.auratensor.inference;

import java.util.Map;

/**
 * Minimal BPE tokenizer contract. AuraTensor ships a simple greedy SentencePiece-
 * compatible decoder for Llama 3 (uses the {@code tokenizer.ggml.tokens} array
 * and {@code tokenizer.ggml.scores} from GGUF metadata) while keeping the
 * interface small enough that other BPE implementations can be swapped in.
 */
public interface Tokenizer {

    /** Encodes a string into a list of token IDs. */
    int[] encode(String text);

    /** Decodes a single token ID into its UTF-8 string piece. */
    String decode(int tokenId);

    /** Returns the special end-of-sequence / prompt boundary marker. */
    int eosTokenId();

    /** Optional convenience map from token id → raw byte string for debugging. */
    Map<Integer, String> vocab();
}
