package io.auratensor.quant;

import io.auratensor.core.Fp16;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Q5_K (k-quant 5-bit) dequantization, faithful to the canonical
 * llama.cpp {@code ggml/src/ggml-quants.c::dequantize_row_q5_K} reference.
 *
 * <p>Block layout ({@value #BLOCK_BYTES} bytes per {@value #BLOCK_ELEMENTS} elements):
 * <pre>
 *   d          FP16    — super-block main scale
 *   dmin       FP16    — super-block min scale
 *   scales[16] uint8   — 4-bit d-scale | 4-bit min-scale (same get_scale_min_k4 as Q4_K)
 *   qh[32]     uint8   — high-1-bit of each 5-bit q, packed 8 bits per byte
 *   qs[128]    uint8   — low-4-bit of each 5-bit q, 2 per byte (low/high)
 * </pre>
 *
 * <p>Per-element formula: {@code x_i = d * (q_i - 16) - dmin * m_J} where
 * {@code q_i} is the unsigned 5-bit value assembled from {@code qs} (4 bits
 * — low/high nibble) + {@code qh} bit (1 bit). The {@code q - 16} centering
 * converts the unsigned [0..31] representation to signed [-16..+15].
 *
 * <p>Q5_K is the accuracy tier above Q4_K (~5.5 bits per weight) and
 * below Q6_K (~6.5 bits). It's the format bartowski publishes as
 * {@code *Q5_K.gguf} when Q4_K accuracy is borderline on a
 * quantization-sensitive model.
 *
 * <p>The {@code qh[32]} byte indexing quirk: each {@code qh[l]} byte
 * holds 8 qh bits covering 8 outputs (specifically, the 8 outputs of
 * {@code qs[l] * 2 + (low or high nibble of qs[l+0,l+32,l+64,l+96])}.
 * The bit mask is {@code 1 << (2 * (j/64))} for low-nibble and
 * {@code 1 << (2 * (j/64) + 1)} for high-nibble — emitted by the
 * 4-way interleaved u1/shift mask that increments twice per j-step.
 */
public final class Q5_K {

    /** Elements per block (QK_K = 256 in llama.cpp convention). */
    public static final int BLOCK_ELEMENTS = 256;
    /** Bytes per block: d(2) + dmin(2) + scales[16] + qh[32] + qs[128] = 180. */
    public static final int BLOCK_BYTES = 180;

    private Q5_K() {}

    /**
     * Scalar reference dequantization.
     *
     * <p>Byte-exact vs llama.cpp {@code dequantize_row_q5_K}. Inner
     * loop reads {@code qh[l]} bit positions driven by the u1/u2 mask
     * that increments twice per 64-element j-step, mirroring the
     * canonical C reference exactly.
     */
    public static void dequantToFloat(MemorySegment src, MemorySegment dst, long numElements) {
        long numBlocks = numElements / BLOCK_ELEMENTS;
        for (long b = 0; b < numBlocks; b++) {
            long blockOff  = b * BLOCK_BYTES;
            float d        = Fp16.readAtLE(src, blockOff);
            float dmin     = Fp16.readAtLE(src, blockOff + 2L);
            long qhBase    = blockOff + 20L;
            long qlBase    = blockOff + 52L;
            long dstBase   = b * BLOCK_ELEMENTS * 4L;

            int is = 0;
            for (int j = 0; j < BLOCK_ELEMENTS; j += 64) {
                long qlOff = qlBase + (j / 2);
                int u1 = 1 << (2 * (j / 64));       // bits 0, 2, 4, 6 across j=0,64,128,192
                int u2 = 1 << (2 * (j / 64) + 1);   // bits 1, 3, 5, 7

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

                // First 32 outputs: low nibble of qs[l] + qh[l] bit u1
                long yOff1 = dstBase + (long) j * 4L;
                for (int l = 0; l < 32; l++) {
                    byte qlB = src.get(ValueLayout.JAVA_BYTE, qlOff + l);
                    byte qhB = src.get(ValueLayout.JAVA_BYTE, qhBase + l);
                    int q = (qlB & 0x0F) + (((qhB & u1) != 0) ? 16 : 0);
                    dst.set(ValueLayout.JAVA_FLOAT, yOff1 + (long) l * 4L, d1 * q - mm1);
                }
                // Second 32 outputs: high nibble of qs[l] + qh[l] bit u2
                long yOff2 = dstBase + (long) (j + 32) * 4L;
                for (int l = 0; l < 32; l++) {
                    byte qlB = src.get(ValueLayout.JAVA_BYTE, qlOff + l);
                    byte qhB = src.get(ValueLayout.JAVA_BYTE, qhBase + l);
                    int q = ((qlB >> 4) & 0x0F) + (((qhB & u2) != 0) ? 16 : 0);
                    dst.set(ValueLayout.JAVA_FLOAT, yOff2 + (long) l * 4L, d2 * q - mm2);
                }
                is += 2;
            }
        }
    }
}
