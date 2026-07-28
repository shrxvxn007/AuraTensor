package io.auratensor.quant;

import io.auratensor.core.Fp16;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Q4_K (k-quant 4-bit) dequantization, faithful to the canonical
 * llama.cpp {@code ggml/src/ggml-quants.c::dequantize_row_q4_K} reference.
 *
 * <p>Block layout ({@value #BLOCK_BYTES} bytes per {@value #BLOCK_ELEMENTS} elements):
 * <pre>
 *   d          FP16    — super-block main scale
 *   dmin       FP16    — super-block min scale
 *   scales[16] uint8   — 4-bit d-scale | 4-bit min-scale, packed via get_scale_min_k4
 *   qs[128]    uint8   — 4-bit quantized values, 2 per byte (low/high)
 * </pre>
 *
 * <p>Per-element formula: {@code x_i = d * sc_J * q_i - dmin * m_J} where
 * {@code J = i/32} (so each pair of consecutive 32-element sub-blocks
 * shares one (sc, m) pair, with 8 such pairs covering the 256-element
 * super-block). The {@code sc} byte comes from the canonical
 * {@code get_scale_min_k4} quirked 4-bit packing where the high-2-bit
 * tail of one byte spills into the next.
 *
 * <p>Q4_K is the dominant 4-bit k-quant for Llama-3 / Mid-tier GGUF
 * exports. bartowski's {@code *Q4_K.gguf} files use this for the
 * bulk layer weights.
 */
public final class Q4_K {

    /** Elements per block (QK_K = 256 in llama.cpp convention). */
    public static final int BLOCK_ELEMENTS = 256;
    /** Bytes per block: d(2) + dmin(2) + scales[16] + qs[128] = 148. */
    public static final int BLOCK_BYTES = 148;

    private Q4_K() {}

    /**
     * Scalar reference dequantization.
     *
     * <p>Outer 64-element j-step × 8 scale pairs (sc,m) per
     * super-block × 32 low + 32 high nibble outputs per j-step. The
     * {@code get_scale_min_k4} helper is inlined for clarity; it
     * implements the canonical Q4_K scale-bit-packing quirk
     * (4-bit scale bytes with high-2-bit spillover via {@code q[j-4] >> 6}
     * for j ≥ 4).
     */
    public static void dequantToFloat(MemorySegment src, MemorySegment dst, long numElements) {
        long numBlocks = numElements / BLOCK_ELEMENTS;
        for (long b = 0; b < numBlocks; b++) {
            long blockOff = b * BLOCK_BYTES;
            float d    = Fp16.readAtLE(src, blockOff);
            float dmin = Fp16.readAtLE(src, blockOff + 2L);
            long dstBase = b * BLOCK_ELEMENTS * 4L;

            int is = 0;
            for (int j = 0; j < BLOCK_ELEMENTS; j += 64) {
                long qOff = blockOff + 20L + (j / 2);

                // get_scale_min_k4(is + 0)
                int sc0, m0;
                if (is + 0 < 4) {
                    sc0 = src.get(ValueLayout.JAVA_BYTE, blockOff + 4L + (is + 0)) & 0x3F;
                    m0  = src.get(ValueLayout.JAVA_BYTE, blockOff + 4L + (is + 0 + 4)) & 0x3F;
                } else {
                    sc0 = (src.get(ValueLayout.JAVA_BYTE, blockOff + 4L + (is + 0 + 4)) & 0x0F)
                        | ((src.get(ValueLayout.JAVA_BYTE, blockOff + 4L + (is + 0 - 4)) >> 6) << 4);
                    m0  = (src.get(ValueLayout.JAVA_BYTE, blockOff + 4L + (is + 0 + 4)) >> 4)
                        | ((src.get(ValueLayout.JAVA_BYTE, blockOff + 4L + (is + 0 - 0)) >> 6) << 4);
                }

                // get_scale_min_k4(is + 1)
                int sc1, m1;
                if (is + 1 < 4) {
                    sc1 = src.get(ValueLayout.JAVA_BYTE, blockOff + 4L + (is + 1)) & 0x3F;
                    m1  = src.get(ValueLayout.JAVA_BYTE, blockOff + 4L + (is + 1 + 4)) & 0x3F;
                } else {
                    sc1 = (src.get(ValueLayout.JAVA_BYTE, blockOff + 4L + (is + 1 + 4)) & 0x0F)
                        | ((src.get(ValueLayout.JAVA_BYTE, blockOff + 4L + (is + 1 - 4)) >> 6) << 4);
                    m1  = (src.get(ValueLayout.JAVA_BYTE, blockOff + 4L + (is + 1 + 4)) >> 4)
                        | ((src.get(ValueLayout.JAVA_BYTE, blockOff + 4L + (is + 1 - 0)) >> 6) << 4);
                }

                float d1 = d * sc0; float mm1 = dmin * m0;
                float d2 = d * sc1; float mm2 = dmin * m1;

                // First 32 outputs (qs[l] & 0xF): outputs [j+0..j+31]
                long yOff1 = dstBase + (long) j * 4L;
                for (int l = 0; l < 32; l++) {
                    byte q = src.get(ValueLayout.JAVA_BYTE, qOff + l);
                    dst.set(ValueLayout.JAVA_FLOAT, yOff1 + (long) l * 4L, d1 * (q & 0x0F) - mm1);
                }
                // Second 32 outputs (qs[l] >> 4): outputs [j+32..j+63]
                long yOff2 = dstBase + (long) (j + 32) * 4L;
                for (int l = 0; l < 32; l++) {
                    byte q = src.get(ValueLayout.JAVA_BYTE, qOff + l);
                    dst.set(ValueLayout.JAVA_FLOAT, yOff2 + (long) l * 4L, d2 * ((q >> 4) & 0x0F) - mm2);
                }
                is += 2;
            }
        }
    }
}
