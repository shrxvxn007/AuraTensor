package io.auratensor.server;

import java.lang.reflect.Field;
import java.util.Map;

/**
 * Minimal allocation-light JSON encoder for the AuraTensor HTTP server.
 *
 * <p>This is intentionally not a general-purpose JSON library. The server
 * only emits a few well-known shapes (OpenAI request/response, status, errors)
 * so we ship a tiny purpose-built encoder that avoids pulling in jackson/gson.
 */
public final class Json {

    private Json() {}

    public static String quote(String s) {
        StringBuilder sb = new StringBuilder(s.length() + 4);
        sb.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"':  sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n");  break;
                case '\r': sb.append("\\r");  break;
                case '\t': sb.append("\\t");  break;
                default:
                    if (c < 0x20) sb.append(String.format("\\u%04x", (int) c));
                    else sb.append(c);
            }
        }
        sb.append('"');
        return sb.toString();
    }

    /** Emit an object string from alternating keys and stringified values. */
    public static String obj(String... kv) {
        if (kv.length % 2 != 0) throw new IllegalArgumentException("kv pairs must be even");
        StringBuilder sb = new StringBuilder("{");
        for (int i = 0; i < kv.length; i += 2) {
            if (i > 0) sb.append(',');
            sb.append(quote(kv[i])).append(':').append(kv[i + 1]);
        }
        sb.append('}');
        return sb.toString();
    }

    /** Wrap a list of strings as a JSON array of strings. */
    public static String strArray(String... items) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < items.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(quote(items[i]));
        }
        sb.append(']');
        return sb.toString();
    }
}
