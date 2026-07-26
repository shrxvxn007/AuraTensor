package io.auratensor.server;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Tolerant JSON tokenizer/parser for the request bodies we accept on
 * {@code /v1/chat/completions}. We accept only the subset of fields the
 * server actually consumes and ignore the rest.
 */
public final class RequestParser {

    private RequestParser() {}

    /** Read all bytes from {@code in} into a UTF-8 string. */
    public static String readAll(InputStream in) throws IOException {
        return new String(in.readAllBytes(), StandardCharsets.UTF_8);
    }

    /**
     * Parse the request body for the well-known top-level fields.
     */
    public static InferenceRequest parse(String body) {
        InferenceRequest defaults = InferenceRequest.defaults();

        String model = stringAt(body, "model", defaults.model());
        float temperature = floatAt(body, "temperature", defaults.temperature());
        int topK = intAt(body, "top_k", defaults.topK());
        float topP = floatAt(body, "top_p", defaults.topP());
        float rep = floatAt(body, "repetition_penalty", defaults.repetitionPenalty());
        int maxTokens = intAt(body, "max_tokens", defaults.maxTokens());
        boolean stream = boolAt(body, "stream", defaults.stream());

        String prompt = stringAt(body, "prompt", "");
        String messages = stringAt(body, "messages", "");
        if (!messages.isEmpty()) {
            prompt = extractMessageContent(messages);
        }

        return new InferenceRequest(model, prompt, temperature, topK, topP,
                                    rep, maxTokens, stream);
    }

    // ---------------------------------------------------------------------
    // Tiny field extractors
    // ---------------------------------------------------------------------

    static String stringAt(String body, String key, String def) {
        return scalarString(body, "\"" + key + "\"");
    }

    static float floatAt(String body, String key, float def) {
        String v = scalarAny(body, "\"" + key + "\"");
        if (v == null) return def;
        try { return Float.parseFloat(v); } catch (NumberFormatException e) { return def; }
    }

    static int intAt(String body, String key, int def) {
        String v = scalarAny(body, "\"" + key + "\"");
        if (v == null) return def;
        try { return Integer.parseInt(v); } catch (NumberFormatException e) { return def; }
    }

    static boolean boolAt(String body, String key, boolean def) {
        String v = scalarAny(body, "\"" + key + "\"");
        if (v == null) return def;
        return v.equals("true");
    }

        /**
     * Extract the last `content` field's string value from a messages JSON array.
     * Returns "" if no content field is found.
     */
    private static String extractMessageContent(String messagesArr) {
        int idx = messagesArr.lastIndexOf("\"content\"");
        if (idx < 0) return "";
        int colon = messagesArr.indexOf(':', idx);
        if (colon < 0) return "";
        int strStart = messagesArr.indexOf('"', colon + 1);
        if (strStart < 0) return "";
        int strEnd = messagesArr.indexOf('"', strStart + 1);
        if (strEnd < strStart) return "";
        return messagesArr.substring(strStart + 1, strEnd);
    }

    private static String scalarString(String body, String keyToken) {
        String v = scalarAny(body, keyToken);
        if (v == null) return "";
        if (v.length() >= 2 && v.charAt(0) == '"' && v.charAt(v.length() - 1) == '"') {
            // Strip quotes; the body's escaping already matches JSON.
            return v.substring(1, v.length() - 1);
        }
        return v;
    }

    /** Returns the token following {@code keyToken} up to one of {@code ,} {@code } } { }. */
    private static String scalarAny(String body, String keyToken) {
        int k = body.indexOf(keyToken);
        if (k < 0) return null;
        k += keyToken.length();
        // Skip whitespace and colon
        while (k < body.length() && (body.charAt(k) == ' ' || body.charAt(k) == ':'
                                     || body.charAt(k) == '\n' || body.charAt(k) == '\t')) {
            k++;
        }
        if (k >= body.length()) return null;

        // String
        if (body.charAt(k) == '"') {
            int end = k + 1;
            while (end < body.length()) {
                char c = body.charAt(end);
                if (c == '\\') { end += 2; continue; }
                if (c == '"') return body.substring(k, end + 1);
                end++;
            }
            return body.substring(k);
        }

        // Scalar: read until comma or closing brace (outside arrays)
        int end = k;
        int depth = 0;
        while (end < body.length()) {
            char c = body.charAt(end);
            if (c == '[' || c == '{') depth++;
            else if (c == ']' || c == '}') {
                if (depth == 0) break;
                depth--;
            } else if (c == ',' && depth == 0) {
                break;
            }
            end++;
        }
        return body.substring(k, end).trim();
    }
}
