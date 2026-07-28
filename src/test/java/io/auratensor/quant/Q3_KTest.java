package io.auratensor.quant;

import io.auratensor.core.Fp16;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Round-trip and edge-case tests for {@link Q3_K}. The inner loop
 * pattern is byte-exact against llama.cpp's
 * {@code dequantize_row_q3_K}, with destination writes walking
 * contiguously through the 256-output super-block (same pattern
 * as {@link Q2_K}); reference re-derivation uses the same walk.
 */
class Q3_KTest {

    @Test
    void dequantMatchesCanonicalInnerLoop() {
        long numElements = Q3_K.BLOCK_ELEMENTS;
        Arena arena = Arena.ofConfined();
        MemorySegment src = arena.allocate(Q3_K.BLOCK_BYTES, 16);
        MemorySegment dst = arena.allocate(numElements * 4L, 16);

        Random r = new Random(23);

        short d16 = Float.floatToFloat16(0.04f);
        src.set(ValueLayout.JAVA_SHORT, 108L, d16);

        // Build the 12-byte scales field by writing all 16 int8 scale
        // values we want, packed in the canonical GGUF v3 layout that
        // the dequant aux[0..3] mask trick reconstructs byte-exactly:
        //   byte[96+i] for i=0..7: low nibble = scales[i]_lo4,
        //                                  high nibble = scales[8+i]_lo4
        //   byte[104+j] for j=0..3: 4 2-bit slots at bits 0/2/4/6 holding
        //                                  scales[j]_hi2, scales[4+j]_hi2,
        //                                  scales[8+j]_hi2, scales[12+j]_hi2
        byte[] scales = new byte[16];
        for (int i = 0; i < 16; i++) {
            scales[i] = (byte) (r.nextInt(65) - 32);
        }
        for (int i = 0; i < 8; i++) {
            int lo   = scales[i]     & 0x0F;
            int hiLo = scales[8 + i] & 0x0F;
            src.set(ValueLayout.JAVA_BYTE, 96L + i, (byte) ((hiLo << 4) | lo));
        }
        for (int j = 0; j < 4; j++) {
            int packed = (scales[j]      & 0x03)
                       | ((scales[4  + j] & 0x03) << 2)
                       | ((scales[8  + j] & 0x03) << 4)
                       | ((scales[12 + j] & 0x03) << 6);
            src.set(ValueLayout.JAVA_BYTE, 104L + j, (byte) packed);
        }

        // hmask[32]: random byte-packed high bits.
        for (int i = 0; i < 32; i++) {
            src.set(ValueLayout.JAVA_BYTE, i, (byte) r.nextInt(256));
        }
        // qs[64]: each byte holds four 2-bit values.
        for (int i = 0; i < 64; i++) {
            int packed = r.nextInt(4)
                       | (r.nextInt(4) << 2)
                       | (r.nextInt(4) << 4)
                       | (r.nextInt(4) << 6);
            src.set(ValueLayout.JAVA_BYTE, 32L + i, (byte) packed);
        }

        Q3_K.dequantToFloat(src, dst, numElements);

        // Reference: re-build each output via the canonical
        // aux[0..3] mask trick → unpacked 6-bit signed scales →
        // d * (s - 32) * (q - (4 if hmask=0)) formula, with the
        // same contiguous yOffRef walk the impl uses.
        int a0 = src.get(ValueLayout.JAVA_INT, 96L);
        int a1 = src.get(ValueLayout.JAVA_INT, 100L);
        int a2 = src.get(ValueLayout.JAVA_INT, 104L);
        int newAux2 = ((a0 >>> 4) & 0x0f0f0f0f) | (((a2 >>> 4) & 0x03030303) << 4);
        int newAux3 = ((a1 >>> 4) & 0x0f0f0f0f) | (((a2 >>> 6) & 0x03030303) << 4);
        int newAux0 = (a0 & 0x0f0f0f0f)        | (((a2      ) & 0x03030303) << 4);
        int newAux1 = (a1 & 0x0f0f0f0f)        | (((a2 >>> 2) & 0x03030303) << 4);
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

        float dExp = Fp16.readAtLE(src, 108L);
        long yOffRef = 0;
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            long qOff = 32L + n / 4L;
            long hOff = n / 8L;
            int shift = 0;
            int mBit = 1;
            for (int j = 0; j < 4; j++) {
                float dl1 = dExp * (s[is++] - 32);
                for (int l = 0; l < 16; l++) {
                    byte qb = src.get(ValueLayout.JAVA_BYTE, qOff + l);
                    byte hb = src.get(ValueLayout.JAVA_BYTE, hOff + l);
                    int q = (qb >> shift) & 0x03;
                    if ((hb & mBit) == 0) q -= 4;
                    float exp = dl1 * q;
                    assertEquals(exp,
                        dst.get(ValueLayout.JAVA_FLOAT, yOffRef), 0f,
                        "pos#" + (yOffRef / 4L) + " n=" + n + ",j=" + j + ",l=" + l + ",first-batch");
                    yOffRef += 4;
                }
                float dl2 = dExp * (s[is++] - 32);
                for (int l = 0; l < 16; l++) {
                    byte qb = src.get(ValueLayout.JAVA_BYTE, qOff + 16L + l);
                    byte hb = src.get(ValueLayout.JAVA_BYTE, hOff + 16L + l);
                    int q = (qb >> shift) & 0x03;
                    if ((hb & mBit) == 0) q -= 4;
                    float exp = dl2 * q;
                    assertEquals(exp,
                        dst.get(ValueLayout.JAVA_FLOAT, yOffRef), 0f,
                        "pos#" + (yOffRef / 4L) + " n=" + n + ",j=" + j + ",l=" + l + ",second-batch");
                    yOffRef += 4;
                }
                shift += 2;
                mBit <<= 1;
            }
        }
    }

    @Test
    void allZerosProducesAllZero() {
        // d = 0 forces output 0 regardless of scale/q/hmask.
        long numElements = Q3_K.BLOCK_ELEMENTS;
        Arena arena = Arena.ofConfined();
        MemorySegment src = arena.allocate(Q3_K.BLOCK_BYTES, 16);
        MemorySegment dst = arena.allocate(numElements * 4L, 16);

        src.set(ValueLayout.JAVA_SHORT, 108L, (short) 0);
        // hmask/qs/scales default zero (arena.allocate contents undefined
        // but never required once d=0).

        Q3_K.dequantToFloat(src, dst, numElements);

        for (long i = 0; i < numElements; i++) {
            assertEquals(0.0f, dst.get(ValueLayout.JAVA_FLOAT, i * 4L), 0f, "i=" + i);
        }
    }
}
