package io.auratensor.format;

/**
 * GGUF metadata value types. Only the codes used by llama.cpp / Llama 3
 * exports are listed exhaustively; unknown codes throw a parse error.
 */
public enum GgufMetadataValueType {
    UINT8   (0),
    INT8    (1),
    UINT16  (2),
    INT16   (3),
    UINT32  (4),
    INT32   (5),
    FLOAT32 (6),
    BOOL    (7),
    STRING  (8),
    ARRAY   (9),
    UINT64  (10),
    INT64   (11),
    FLOAT64 (12);

    public final int code;
    GgufMetadataValueType(int code) { this.code = code; }

    public static GgufMetadataValueType fromCode(int code) {
        for (GgufMetadataValueType t : values()) {
            if (t.code == code) return t;
        }
        throw new IllegalArgumentException("Unknown GGUF metadata value type: " + code);
    }
}
