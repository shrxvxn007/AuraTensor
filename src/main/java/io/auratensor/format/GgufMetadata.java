package io.auratensor.format;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Holder for the metadata KV pairs extracted from a GGUF file header.
 * Exposes strongly-typed accessors for the fields AuraTensor reads.
 */
public final class GgufMetadata {
    private final Map<String, Object> map = new LinkedHashMap<>();

    void put(String key, Object value) {
        map.put(key, value);
    }

    public Map<String, Object> asMap() { return Collections.unmodifiableMap(map); }

    public boolean has(String key) { return map.containsKey(key); }

    public Object get(String key) { return map.get(key); }

    public long longOrDefault(String key, long defaultValue) {
        Object v = map.get(key);
        if (v == null) return defaultValue;
        if (v instanceof Long) return (Long) v;
        if (v instanceof Integer) return ((Integer) v).longValue();
        if (v instanceof Number) return ((Number) v).longValue();
        throw new IllegalStateException("Metadata '" + key + "' is not a number: " + v.getClass());
    }

    public int intOrDefault(String key, int defaultValue) {
        return (int) longOrDefault(key, defaultValue);
    }

    public float floatOrDefault(String key, float defaultValue) {
        Object v = map.get(key);
        if (v == null) return defaultValue;
        if (v instanceof Float) return (Float) v;
        if (v instanceof Number) return ((Number) v).floatValue();
        throw new IllegalStateException("Metadata '" + key + "' is not float-like: " + v.getClass());
    }

    public String stringOrDefault(String key, String defaultValue) {
        Object v = map.get(key);
        return v == null ? defaultValue : v.toString();
    }

    public List<?> arrayOrEmpty(String key) {
        Object v = map.get(key);
        if (v == null) return Collections.emptyList();
        if (v instanceof List<?>) return (List<?>) v;
        throw new IllegalStateException("Metadata '" + key + "' is not an array");
    }
}
