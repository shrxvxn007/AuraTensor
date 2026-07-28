package io.auratensor.quant;

import io.auratensor.core.Fp16;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Q2_K (k-quant 2-bit) dequantization, faithful to the canonical
 * llama.cpp {@code ggml/src/ggml-quants.c::dequantize_row_q2_K} reference.
 *
 * <p>Block layout ({@value #BLOCK_BYTES} bytes per {@value #BLOCK_ELEMENTS} elements):
 * <pre>
 *   scales[16] uint8  — 4-bit d-scale (low nibble) | 4-bit min-scale (high nibble), 16 sub-blocks of 16 elements each
 *   qs[64]     uint8  — 2-bit quantized values, 4 elements per byte
 *   dmin       FP16   — super-block min scale
 *   d          FP16   — super-block main scale
 * </pre>
 *
 * <p>Per-element formula: {@code x_i = d * (sc_d & 0xF) * q_i - dmin * (sc >> 4)}.
 * Where {@code q_i} is the unsigned 2-bit value [0..3] extracted from
 * {@code qs[(i/128)*64 + (i%128)/4] >> (2*(i%4))}, and the per-16-element
 * sub-block scale byte packs a 4-bit d-scale with a 4-bit min-scale.
 *
 * <p>Q2_K is the most aggressive k-quant (~3.5 bits per weight). It is
 * the format bartowski / unsloth / ollama publish as {@code *Q2_K.gguf}
 * when storage cost is paramount and accuracy loss is acceptable.
 *
 * <p>The 4-way inner-loop pattern (with 2 scale-byte reads per 4-element
 * 2-bit shift step across each 16-element group) matches llama.cpp's
 * SIMD-friendly C reference exactly.
 */
public final class Q2_K {

    /** Elements per block (QK_K = 256 in llama.cpp convention). */
    public static final int BLOCK_ELEMENTS = 256;
    /** Bytes per block: scales[16] + qs[64] + dmin(2 fp16) + d(2 fp16) = 84. */
    public static final int BLOCK_BYTES = 84;

    private Q2_K() {}

    /**
     * Scalar reference dequantization.
     *
     * <p>Byte-exact vs llama.cpp {@code dequantize_row_q2_K}: nested
     * outer n-step (0, 128) → mid j-step (0..3) → inner 16-element
     * l-loops, advancing the 2-bit shift register by 2 per j. Each
     * j-step reads 2 scale bytes (16 elements × 1 byte × 2 batches = 32
     * outputs); each scale byte packs a 4-bit d-scale (low nibble) and
     * a 4-bit min-scale (high nibble).
     */
    public static void dequantToFloat(MemorySegment src, MemorySegment dst, long numElements) {
        long numBlocks = numElements / BLOCK_ELEMENTS;
        for (long b = 0; b < numBlocks; b++) {
            long blockOff = b * BLOCK_BYTES;
            long scalesBase = blockOff;
            float d    = Fp16.readAtLE(src, blockOff + 80L);
            float dmin = Fp16.readAtLE(src, blockOff + 82L);
            long dstBase = b * BLOCK_ELEMENTS * 4L;

            // ys mirrors the C's *y++ pointer walk; local var is clearer
            // than re-deriving IndexOff each iteration.
            long yOff = dstBase;

            for (int n = 0; n < BLOCK_ELEMENTS; n += 128) {
                long qOff = blockOff + 16L + n / 4L;
                int shift = 0;
                for (int j = 0; j < 4; j++) {
                    // First 16-element batch: low scales byte (16 elements)
                    byte sc1 = src.get(ValueLayout.JAVA_BYTE, scalesBase + (n / 16) + (j * 2) + 0);
                    float dl1 = d    * (sc1 & 0x0F);
                    float ml1 = dmin * (sc1 >> 4);
                    for (int l = 0; l < 16; l++) {
                        byte qb = src.get(ValueLayout.JAVA_BYTE, qOff + l);
                        int q = (qb >> shift) & 0x03;
                        dst.set(ValueLayout.JAVA_FLOAT, yOff, dl1 * q - ml1);
                        yOff += 4;
                    }
                    // Second 16-element batch (reads Q from qOff+16..)
                    byte sc2 = src.get(ValueLayout.JAVA_BYTE, scalesBase + (n / 16) + (j * 2) + 1);
                    float dl2 = d    * (sc2 & 0x0F);
                    float ml2 = dmin * (sc2 >> 4);
                    for (int l = 0; l < 16; l++) {
                        byte qb = src.get(ValueLayout.JAVA_BYTE, qOff + 16L + l);
                        int q = (qb >> shift) & 0x03;
                        dst.set(ValueLayout.JAVA_FLOAT, yOff, dl2 * q - ml2);
                        yOff += 4;
                    }
                    shift += 2;
                }
            }
        }
    }
}
