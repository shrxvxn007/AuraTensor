package io.auratensor.quant;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Q8_K (k-quant 8-bit) dequantization, faithful to the canonical
 * llama.cpp {@code ggml/src/ggml-quants.c::dequantize_row_q8_K} reference.
 *
 * <p>Block layout ({@value #BLOCK_BYTES} bytes per {@value #BLOCK_ELEMENTS} elements):
 * <pre>
 *   d      FP32  — super-block scale (4 bytes)
 *   qs[256] int8  — signed quantized values, one byte per element
 * </pre>
 *
 * <p>Per-element formula: {@code x_i = d * qs[i]}. No scale array, no
 * per-element fixup — the byte directly holds the signed [-128, 127]
 * quantized value scaled by the FP32 super-block scale.
 *
 * <p>Q8_K is the highest-fidelity of the k-quants: roughly
 * comparable to Q8_0 on a per-element basis, but with a per-block
 * FP32 scale (no FP16 round-trip on d). It dominates the Q8-tier of
 * bartowski / unsloth GGUF exports when a model needs more accuracy
 * than Q4_K but the file size of Q8_0 is too large.
 */
public final class Q8_K {

    /** Elements per block (QK_K = 256 in llama.cpp convention). */
    public static final int BLOCK_ELEMENTS = 256;
    /** Bytes per block: 4 (d fp32) + 256 (qs int8[]) = 260. */
    public static final int BLOCK_BYTES = 260;

    private Q8_K() {}

    /**
     * Scalar reference dequantization.
     *
     * <p>Linear 0..255 sweep — no 4-way interleaving (unlike Q6_K /
     * Q2_K) because the input/output ratio is 1 byte per element
     * with no per-element sub-block addressing. The FP32 super-block
     * scale is read directly via {@link ValueLayout#JAVA_FLOAT}
     * (no FP16 round-trip, unlike every other k-quant).
     */
    public static void dequantToFloat(MemorySegment src, MemorySegment dst, long numElements) {
        long numBlocks = numElements / BLOCK_ELEMENTS;
        for (long b = 0; b < numBlocks; b++) {
            long blockOff = b * BLOCK_BYTES;
            float d = src.get(ValueLayout.JAVA_FLOAT, blockOff);
            long qsBase = blockOff + 4L;
            long dstBase = b * BLOCK_ELEMENTS * 4L;

            for (int i = 0; i < BLOCK_ELEMENTS; i++) {
                byte q = src.get(ValueLayout.JAVA_BYTE, qsBase + i);
                dst.set(ValueLayout.JAVA_FLOAT, dstBase + (long) i * 4L, d * q);
            }
        }
    }
}
