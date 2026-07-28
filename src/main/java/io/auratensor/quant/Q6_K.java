package io.auratensor.quant;

import io.auratensor.core.Fp16;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Q6_K (k-quant 6-bit) dequantization, faithful to the canonical
 * llama.cpp {@code ggml/src/ggml-quants.c::dequantize_row_q6_K} reference.
 *
 * <p>Block layout ({@value #BLOCK_BYTES} bytes per {@value #BLOCK_ELEMENTS} elements):
 * <pre>
 *   ql[128]    uint8 — low 4 bits per element, packed 2 per byte
 *   qh[64]     uint8 — high 2 bits per element, packed 4 per byte
 *   scales[16] int8  — signed scales, 16 elements per scale (16 total)
 *   d          FP16  — super-block scale (2 bytes)
 * </pre>
 *
 * <p>Per-element formula:
 * {@code x_i = d * scale[i/16] * (q_i - 32)} where {@code q_i} is the
 * 6-bit unsigned integer [0..63] reassembled from the ql+qh nibble pair.
 * {@code scales[]} bytes are raw signed int8 (no centering, range −128..127).
 *
 * <p>This is the dominant Llama-3 / Mistral k-quant 6-bit weight format.
 * In bartowski's Llama-3.2-1B-Instruct-Q4_0 GGUF, {@code token_embd.weight}
 * and {@code output.weight} are stored as Q6_K, so closing this format
 * removes the per-tensor FP32 stand-in documented in README footnote ⁴.
 *
 * <p>The 4-way interleaved ql/qh scale access pattern (writing
 * {@code y[l+0]}, {@code y[l+32]}, {@code y[l+64]}, {@code y[l+96]}
 * simultaneously per inner-loop iteration) is intentional and matches
 * llama.cpp's SIMD-friendly C reference loop exactly — re-ordering it
 * to a linear 0..255 sweep would diverge from the byte addressing.
 */
public final class Q6_K {

    /** Elements per block (QK_K = 256 in llama.cpp convention). */
    public static final int BLOCK_ELEMENTS = 256;
    /** Bytes per block: 128 (ql) + 64 (qh) + 16 (scales) + 2 (d). */
    public static final int BLOCK_BYTES = 210;

    private Q6_K() {}

    /**
     * Scalar reference dequantization.
     *
     * <p>Reads the FP16 super-block scale via {@link Fp16#readAtLE}
     * (the same helper Q4_0 and Q8_0 use) so the FP16 → FP32 conversion
     * is centralised. The inner loop is byte-exact against llama.cpp
     * {@code dequantize_row_q6_K} so a byte-level test against the
     * canonical Q6_K reference output produces zero per-element drift.
     */
    public static void dequantToFloat(MemorySegment src, MemorySegment dst, long numElements) {
        long numBlocks = numElements / BLOCK_ELEMENTS;
        for (long b = 0; b < numBlocks; b++) {
            long blockOff = b * BLOCK_BYTES;
            long qlBase   = blockOff;
            long qhBase   = blockOff + 128L;
            long scBase   = blockOff + 192L;
            float d = Fp16.readAtLE(src, blockOff + 208L);
            long dstBase = b * BLOCK_ELEMENTS * 4L;

            // Two half-block passes (n = 0, 128). Each child loop writes
            // 128 output floats via the 4-way interleaved layout:
            //   y[l+0]   ← ql[l+0]   low nibble + qh[l]  >> 0
            //   y[l+32]  ← ql[l+32]  low nibble + qh[l]  >> 2
            //   y[l+64]  ← ql[l+0]   high nibble + qh[l] >> 4
            //   y[l+96]  ← ql[l+32]  high nibble + qh[l] >> 6
            // Within each l-iteration, the 4 scale bytes (sc[is+0/2/4/6]
            // for the first 16-element l group; sc[is+1/3/5/7] for the
            // second) are signed int8 covering 16 elements each.
            for (int n = 0; n < BLOCK_ELEMENTS; n += 128) {
                long qlOff    = qlBase + n / 2L;
                long qhOff    = qhBase + n / 4L;
                long scOff    = scBase + n / 16L;
                long yByteOff = dstBase + n * 4L;

                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    byte qlLow = src.get(ValueLayout.JAVA_BYTE, qlOff + l);
                    byte qlHi  = src.get(ValueLayout.JAVA_BYTE, qlOff + l + 32L);
                    byte qh    = src.get(ValueLayout.JAVA_BYTE, qhOff + l);
                    float s0 = src.get(ValueLayout.JAVA_BYTE, scOff + is);
                    float s1 = src.get(ValueLayout.JAVA_BYTE, scOff + is + 2L);
                    float s2 = src.get(ValueLayout.JAVA_BYTE, scOff + is + 4L);
                    float s3 = src.get(ValueLayout.JAVA_BYTE, scOff + is + 6L);

                    int q1 = ((qlLow & 0x0F) | (((qh >> 0) & 0x03) << 4)) - 32;
                    int q2 = ((qlHi  & 0x0F) | (((qh >> 2) & 0x03) << 4)) - 32;
                    // Java's `byte` is signed; sign-extension on `byte -> int`
                    // promotion turns byte 0xFF into int 0xFFFFFFFF. A bare
                    // `(qlLow >>> 4)` then zero-fills 28 high bits of garbage
                    // (giving 0x0FFFFFFF instead of the unsigned 4-bit nibble
                    // 0x0F). The `>> 4 & 0x0F` mask discards the sign-extension
                    // noise. Same fix for qlHi.
                    int q3 = (((qlLow >> 4) & 0x0F) | (((qh >> 4) & 0x03) << 4)) - 32;
                    int q4 = (((qlHi  >> 4) & 0x0F) | (((qh >> 6) & 0x03) << 4)) - 32;

                    dst.set(ValueLayout.JAVA_FLOAT, yByteOff + (long) l * 4L,        d * s0 * q1);
                    dst.set(ValueLayout.JAVA_FLOAT, yByteOff + (long)(l + 32) * 4L, d * s1 * q2);
                    dst.set(ValueLayout.JAVA_FLOAT, yByteOff + (long)(l + 64) * 4L, d * s2 * q3);
                    dst.set(ValueLayout.JAVA_FLOAT, yByteOff + (long)(l + 96) * 4L, d * s3 * q4);
                }
            }
        }
    }
}
