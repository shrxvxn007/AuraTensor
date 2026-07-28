package io.auratensor.format;

/**
 * GGUF tensor element types as defined by the GGUF v3 spec.
 *
 * <p>Only the types AuraTensor actually implements are listed here. Float16
 * (1) and Float32 (0) are stored as opaque per-element bytes; the
 * quantized block formats (Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7,
 * Q8_0 = 8, Q8_1 = 9, Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13,
 * Q6_K = 14, Q8_K = 15) are decoded by {@link io.auratensor.quant}.
 *
 * <p>This enum intentionally mirrors the numeric ordering of the GGUF
 * specification — never reorder these.
 */
public enum GgufTensorType {
    FLOAT32  (0,  4, "F32"),
    FLOAT16  (1,  2, "F16"),
    Q4_0     (2, 18, "Q4_0"),    // 32 elements per block
    Q4_1     (3, 20, "Q4_1"),
    Q5_0     (6, 22, "Q5_0"),
    Q5_1     (7, 24, "Q5_1"),
    Q8_0     (8, 34, "Q8_0"),    // 32 elements per block
    Q8_1     (9, 36, "Q8_1"),
    Q2_K     (10,  84, "Q2_K"),    // 256 elements per block (16 scales[sc|m nibble] + 64 qs[2-bit] + 2 dmin fp16 + 2 d fp16)
    Q3_K     (11, 110, "Q3_K"),    // 256 elements per block (32 hmask[1-bit] + 64 qs[2-bit] + 12 scales[6-bit packed] + 2 d fp16)
    Q4_K     (12, 148, "Q4_K"),    // 256 elements per block (2 d + 2 dmin + 16 scales[sc|m nibble] + 128 qs[4-bit])
    Q5_K     (13, 180, "Q5_K"),    // 256 elements per block (2 d + 2 dmin + 16 scales + 32 qh[1-bit] + 128 qs[4-bit])
    Q6_K     (14, 210, "Q6_K"),    // 256 elements per block (128 ql + 64 qh + 16 scales + 2 d bytes)
    Q8_K     (15, 260, "Q8_K");    // 256 elements per block (4 d fp32 + 256 qs[i8])

    /** GGUF code (matches the byte in the file). */
    public final int code;
    /** Block size in bytes (for block-quantized types) or per-element bytes. */
    public final int bytesPerBlockOrElement;
    /** Display name. */
    public final String label;

    GgufTensorType(int code, int bytesPerBlockOrElement, String label) {
        this.code = code;
        this.bytesPerBlockOrElement = bytesPerBlockOrElement;
        this.label = label;
    }

    public static GgufTensorType fromCode(int code) {
        for (GgufTensorType t : values()) {
            if (t.code == code) return t;
        }
        throw new IllegalArgumentException("Unknown GGUF tensor type code: " + code);
    }

    public boolean isBlockQuantized() {
        return code >= Q4_0.code && code <= Q8_K.code;
    }
}
