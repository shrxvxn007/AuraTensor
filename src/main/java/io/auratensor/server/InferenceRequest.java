package io.auratensor.server;

/**
 * Parsed OpenAI-compatible chat completion request.
 *
 * <p>We parse only the fields AuraTensor understands. Anything else is ignored
 * so requests don't break between minor OpenAI client revisions.
 */
public record InferenceRequest(
    String model,
    String prompt,                 // joined messages or the raw prompt field
    float temperature,
    int topK,
    float topP,
    float repetitionPenalty,
    int maxTokens,
    boolean stream
) {
    public static InferenceRequest defaults() {
        return new InferenceRequest(
            "auratensor", "", 0.7f, 40, 0.95f, 1.1f, 256, false);
    }
}
