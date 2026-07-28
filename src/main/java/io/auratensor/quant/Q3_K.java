package io.auratensor.quant;

import io.auratensor.core.Fp16;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Q3_K (k-quant 3-bit) dequantization, faithful to the canonical
 * llama.cpp {@code ggml/src/ggml-quants.c::dequantize_row_q3_K} reference.
 *
 * <p>Block layout ({@value #BLOCK_BYTES} bytes per {@value #BLOCK_ELEMENTS} elements):
 * <pre>
 *   hmask[32]  uint8  — high-1-bit of each 3-bit q (8 outputs per byte)
 *   qs[64]     uint8  — low-2-bit of each 3-bit q, 4 per byte
 *   scales[12] uint8  — 6-bit signed scales, packed via the aux[] nibble-split trick
 *   d          FP16   — super-block scale
 * </pre>
 *
 * <p>Per-element formula: {@code x_i = d * scale_J * (3-bit_q - 4)} where
 * the 3-bit_q is reassembled as low-2-bit {(qs[(i/128)*64 + l/4] >> 2*(l%4)) & 3}
 * plus a high-1-bit from hmask (if 0, subtract 4 to enforce the
 * asymmetric [-4, +3] representation).
 *
 * <p>Q3_K is the ~3.5-bit k-quant with the highest accuracy per bit
 * among the k-quants under 4 bits; the asymmetric [-4, +3] range
 * avoids the obviously-broken -8 case in straight unsigned 4-bit.
 *
 * <p>The 12-byte scales-field packing uses a split-nibble/h-bit trick:
 * bytes 96..103 carry the low-4 bits of scales 0..15 and the low-4
 * bits of scales 8..15 interleaved; bytes 104..107 carry the high-2
 * bits of scales 0..15 in 4 × 2-bit slots per byte. The dequant
 * unpack routine uses the canonical {@code aux[0..3]} 4-uint32 mask
 * trick (kmask1 = 0x0f0f0f0f, kmask2 = 0x03030303) to reconstruct
 * all 16 6-bit signed scales.
 *
 * <p>Inner loop walks {@code yOff += 4} per output (256 contiguous
 * floats per super-block), avoiding the overlap bug where every
 * j-shift would otherwise re-target the same dst[0..15] range.
 */
public final class Q3_K {

    /** Elements per block (QK_K = 256 in llama.cpp convention). */
    public static final int BLOCK_ELEMENTS = 256;
    /** Bytes per block: hmask[32] + qs[64] + scales[12] + d(2 fp16) = 110. */
    public static final int BLOCK_BYTES = 110;

    private Q3_K() {}

    /**
     * Scalar reference dequantization.
     *
     * <p>Byte-exact vs llama.cpp {@code dequantize_row_q3_K}. The
     * 12-byte {@code scales} field is unpacked into 16 int8 scales
     * via the {@code aux[0..3]} uint32-cast trick (low 4 bits of one
     * byte hold the low 4 bits of one scale; high-2 of the same byte
     * are stored as bits 4/5 in {@code aux[2]} and aux[2..3] — see the
     * canonical {@code dequantize_row_q3_K} body for the bit algebra).
     *
     * <p>Destination writes use a contiguous {@code yOff += 4} walk
     * over the 256-output super-block — same pattern as {@link Q2_K}.
     */
    public static void dequantToFloat(MemorySegment src, MemorySegment dst, long numElements) {
        long numBlocks = numElements / BLOCK_ELEMENTS;
        for (long b = 0; b < numBlocks; b++) {
            long blockOff = b * BLOCK_BYTES;
            long hmaskBase = blockOff;
            long qsBase    = blockOff + 32L;

            // Read 12 scales bytes into 4 uint32 (aux[0..3]) and apply
            // the canonical bit-split:
            //   aux[0] = (a0 & 0x0f0f0f0f)        | ((a2     ) & 0x03030303) << 4;
            //   aux[1] = (a1 & 0x0f0f0f0f)        | ((a2 >> 2) & 0x03030303) << 4;
            //   aux[2] = ((a0 >> 4) & 0x0f0f0f0f) | ((a2 >> 4) & 0x03030303) << 4;
            //   aux[3] = ((a1 >> 4) & 0x0f0f0f0f) | ((a2 >> 6) & 0x03030303) << 4;
            // Then aux[0..3] reinterpreted as 16 int8 scales.
            int a0 = src.get(ValueLayout.JAVA_INT, blockOff + 96L);
            int a1 = src.get(ValueLayout.JAVA_INT, blockOff + 100L);
            int a2 = src.get(ValueLayout.JAVA_INT, blockOff + 104L);

            int newAux2 = ((a0 >>> 4) & 0x0f0f0f0f) | (((a2 >>> 4) & 0x03030303) << 4);
            int newAux3 = ((a1 >>> 4) & 0x0f0f0f0f) | (((a2 >>> 6) & 0x03030303) << 4);
            int newAux0 = (a0  & 0x0f0f0f0f)        | (((a2      ) & 0x03030303) << 4);
            int newAux1 = (a1  & 0x0f0f0f0f)        | (((a2 >>> 2) & 0x03030303) << 4);

            byte[] s = new byte[16];
            s[0]  = (byte)  (newAux0         & 0xFF);
            s[1]  = (byte) ((newAux0 >>>  8) & 0xFF);
            s[2]  = (byte) ((newAux0 >>> 16) & 0xFF);
            s[3]  = (byte) ((newAux0 >>> 24) & 0xFF);
            s[4]  = (byte)  (newAux1         & 0xFF);
            s[5]  = (byte) ((newAux1 >>>  8) & 0xFF);
            s[6]  = (byte) ((newAux1 >>> 16) & 0xFF);
            s[7]  = (byte) ((newAux1 >>> 24) & 0xFF);
            s[8]  = (byte)  (newAux2         & 0xFF);
            s[9]  = (byte) ((newAux2 >>>  8) & 0xFF);
            s[10] = (byte) ((newAux2 >>> 16) & 0xFF);
            s[11] = (byte) ((newAux2 >>> 24) & 0xFF);
            s[12] = (byte)  (newAux3         & 0xFF);
            s[13] = (byte) ((newAux3 >>>  8) & 0xFF);
            s[14] = (byte) ((newAux3 >>> 16) & 0xFF);
            s[15] = (byte) ((newAux3 >>> 24) & 0xFF);

            float dAll = Fp16.readAtLE(src, blockOff + 108L);
            long dstBase = b * BLOCK_ELEMENTS * 4L;
            long yOff = dstBase;
            int is = 0;
            for (int n = 0; n < BLOCK_ELEMENTS; n += 128) {
                long qOff = qsBase + n / 4L;
                long hOff = hmaskBase + n / 8L;
                int shift = 0;
                int mBit = 1;
                for (int j = 0; j < 4; j++) {
                    // s[is] is signed int8 (originally byte). Subtract 32
                    // as signed arithmetic — NOT `(s & 0xFF) - 32` — because
                    // Java's `byte` is signed and `& 0xFF` would corrupt the
                    // high-range scales (-32..+31 plus the spillover extension).
                    float dl1 = dAll * (s[is++] - 32);
                    // First 16 contiguous outputs of this 32-output j-range
                    for (int l = 0; l < 16; l++) {
                        byte qb = src.get(ValueLayout.JAVA_BYTE, qOff + l);
                        byte hb = src.get(ValueLayout.JAVA_BYTE, hOff + l);
                        int q = (qb >> shift) & 0x03;
                        if ((hb & mBit) == 0) q -= 4;
                        dst.set(ValueLayout.JAVA_FLOAT, yOff, dl1 * q);
                        yOff += 4;
                    }
                    float dl2 = dAll * (s[is++] - 32);
                    // Second 16 contiguous outputs (immediately follows first 16)
                    for (int l = 0; l < 16; l++) {
                        byte qb = src.get(ValueLayout.JAVA_BYTE, qOff + 16L + l);
                        byte hb = src.get(ValueLayout.JAVA_BYTE, hOff + 16L + l);
                        int q = (qb >> shift) & 0x03;
                        if ((hb & mBit) == 0) q -= 4;
                        dst.set(ValueLayout.JAVA_FLOAT, yOff, dl2 * q);
                        yOff += 4;
                    }
                    shift += 2;
                    mBit <<= 1;
                }
            }
        }
    }
}
