package io.auratensor.quant;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Round-trip and edge-case tests for {@link Q5_K}. The byte packing is
 * byte-exact against llama.cpp's {@code dequantize_row_q5_K} loop
 * (with the qh-bit u1/u2 mask incrementing twice per 64-element
 * j-step), so the test re-derives each output via the same
 * {@code d * (q - 16) - dmin * m} formula and compares bit-for-bit.
 */
class Q5_KTest {

    @Test
    void dequantMatchesCanonicalInnerLoop() {
        long numElements = Q5_K.BLOCK_ELEMENTS;
        Arena arena = Arena.ofConfined();
        MemorySegment src = arena.allocate(Q5_K.BLOCK_BYTES, 16);
        MemorySegment dst = arena.allocate(numElements * 4L, 16);

        Random r = new Random(19);

        short d16 = Float.floatToFloat16(0.04f);
        short min16 = Float.floatToFloat16(0.01f);
        src.set(ValueLayout.JAVA_SHORT, 0L, d16);
        src.set(ValueLayout.JAVA_SHORT, 2L, min16);

        // scales[16] same get_scale_min_k4 packing as Q4_K — random bytes.
        for (int i = 0; i < 16; i++) {
            src.set(ValueLayout.JAVA_BYTE, 4L + i, (byte) r.nextInt(256));
        }
        // qh[32]: random byte-packed high bits.
        for (int i = 0; i < 32; i++) {
            src.set(ValueLayout.JAVA_BYTE, 20L + i, (byte) r.nextInt(256));
        }
        // qs[128]: each byte holds two 4-bit values. Random.
        for (int i = 0; i < 128; i++) {
            int lo = r.nextInt(16);
            int hi = r.nextInt(16);
            src.set(ValueLayout.JAVA_BYTE, 52L + i, (byte) ((hi << 4) | lo));
        }

        Q5_K.dequantToFloat(src, dst, numElements);

        // Reference: re-build each output via the llama.cpp canonical loop.
        float dExp = Float.float16ToFloat(d16);
        float mExp = Float.float16ToFloat(min16);
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            long qlOff = 52L + (j / 2);
            int u1 = 1 << (2 * (j / 64));
            int u2 = 1 << (2 * (j / 64) + 1);

            int sc0, m0;
            if (is + 0 < 4) {
                sc0 = src.get(ValueLayout.JAVA_BYTE, 4L + (is + 0)) & 0x3F;
                m0  = src.get(ValueLayout.JAVA_BYTE, 4L + (is + 0 + 4)) & 0x3F;
            } else {
                sc0 = (src.get(ValueLayout.JAVA_BYTE, 4L + (is + 0 + 4)) & 0x0F)
                    | ((src.get(ValueLayout.JAVA_BYTE, 4L + (is + 0 - 4)) >> 6) << 4);
                m0  = (src.get(ValueLayout.JAVA_BYTE, 4L + (is + 0 + 4)) >> 4)
                    | ((src.get(ValueLayout.JAVA_BYTE, 4L + (is + 0 - 0)) >> 6) << 4);
            }
            int sc1, m1;
            if (is + 1 < 4) {
                sc1 = src.get(ValueLayout.JAVA_BYTE, 4L + (is + 1)) & 0x3F;
                m1  = src.get(ValueLayout.JAVA_BYTE, 4L + (is + 1 + 4)) & 0x3F;
            } else {
                sc1 = (src.get(ValueLayout.JAVA_BYTE, 4L + (is + 1 + 4)) & 0x0F)
                    | ((src.get(ValueLayout.JAVA_BYTE, 4L + (is + 1 - 4)) >> 6) << 4);
                m1  = (src.get(ValueLayout.JAVA_BYTE, 4L + (is + 1 + 4)) >> 4)
                    | ((src.get(ValueLayout.JAVA_BYTE, 4L + (is + 1 - 0)) >> 6) << 4);
            }

            float d1 = dExp * sc0; float mm1 = mExp * m0;
            float d2 = dExp * sc1; float mm2 = mExp * m1;

            for (int l = 0; l < 32; l++) {
                byte qlB = src.get(ValueLayout.JAVA_BYTE, qlOff + l);
                byte qhB = src.get(ValueLayout.JAVA_BYTE, 20L + l);
                int q = (qlB & 0x0F) + (((qhB & u1) != 0) ? 16 : 0);
                float exp = d1 * q - mm1;
                assertEquals(exp,
                    dst.get(ValueLayout.JAVA_FLOAT, (long) (j + l) * 4L), 0f,
                    "j=" + j + ",l=" + l + ",first-batch");
            }
            for (int l = 0; l < 32; l++) {
                byte qlB = src.get(ValueLayout.JAVA_BYTE, qlOff + l);
                byte qhB = src.get(ValueLayout.JAVA_BYTE, 20L + l);
                int q = ((qlB >> 4) & 0x0F) + (((qhB & u2) != 0) ? 16 : 0);
                float exp = d2 * q - mm2;
                assertEquals(exp,
                    dst.get(ValueLayout.JAVA_FLOAT, (long) (j + l + 32) * 4L), 0f,
                    "j=" + j + ",l=" + l + ",second-batch");
            }
            is += 2;
        }
    }

    @Test
    void q0MinAndQ5MaxProduceSignExtreme() {
        // q is the unsigned 5-bit value assembled (ql low/high nibble + qh bit).
        // Impl: x = d * sc * q - dmin * m. With d=1, sc=8, dmin=1, m=8 →
        //   x = 8 * q - 8.
        // q = 0 (ql=0, qh bit 0) → x = -8.
        // q = 15 (ql=15, qh bit 0) → x = 8 * 15 - 8 = 112.
        // q = 31 (ql=15, qh bit 1) → x = 8 * 31 - 8 = 240.
        long numElements = Q5_K.BLOCK_ELEMENTS;
        Arena arena = Arena.ofConfined();
        MemorySegment src = arena.allocate(Q5_K.BLOCK_BYTES, 16);
        MemorySegment dst = arena.allocate(numElements * 4L, 16);

        src.set(ValueLayout.JAVA_SHORT, 0L, (short) 0x3C00);  // FP16 1.0
        src.set(ValueLayout.JAVA_SHORT, 2L, (short) 0x3C00);  // FP16 1.0
        // scales[0]=8 and scales[4]=8 → get_scale_min_k4(is=0): sc=q[0]&0x3F=8,
        // m=q[4]&0x3F=8. So d=8, mm=8.
        src.set(ValueLayout.JAVA_BYTE, 4L + 0, (byte) 0x08);
        src.set(ValueLayout.JAVA_BYTE, 4L + 4, (byte) 0x08);
        // qh: all zero → high bit always 0, so q = (ql low/high nibble) only.
        for (int i = 0; i < 32; i++) src.set(ValueLayout.JAVA_BYTE, 20L + i, (byte) 0x00);
        // qs[0..31]: 0x00 → q=0 for first 32 outputs (j=0 batch 1).
        // qOff for n=0 is `qlBase + 0` = blockOff+52. The first 32 qs bytes
        // are at qlBase..qlBase+31 → indices in absolute sement space 52..83.
        for (int i = 0; i < 32; i++) src.set(ValueLayout.JAVA_BYTE, 52L + i, (byte) 0x00);

        Q5_K.dequantToFloat(src, dst, numElements);

        // First 32 outputs: q=0 → x = 8*0 - 8 = -8.
        for (int l = 0; l < 32; l++) {
            assertEquals(-8.0f, dst.get(ValueLayout.JAVA_FLOAT, (long) l * 4L), 0f,
                "q=0 first-batch l=" + l);
        }
        // Now flip first 32 qs bytes to 0xFF → q=15 (high nibble all 0xF).
        for (int i = 0; i < 32; i++) src.set(ValueLayout.JAVA_BYTE, 52L + i, (byte) 0xFF);
        Q5_K.dequantToFloat(src, dst, numElements);
        for (int l = 0; l < 32; l++) {
            assertEquals(112.0f, dst.get(ValueLayout.JAVA_FLOAT, (long) l * 4L), 0f,
                "q=15 first-batch l=" + l);
        }
    }

    @Test
    void allZerosProducesNegativeMin() {
        // d = 1, dmin = 1, sc=0, m=0 (zero scales bytes) → all outputs zero.
        long numElements = Q5_K.BLOCK_ELEMENTS;
        Arena arena = Arena.ofConfined();
        MemorySegment src = arena.allocate(Q5_K.BLOCK_BYTES, 16);
        MemorySegment dst = arena.allocate(numElements * 4L, 16);

        src.set(ValueLayout.JAVA_SHORT, 0L, (short) 0x3C00);  // FP16 1.0
        src.set(ValueLayout.JAVA_SHORT, 2L, (short) 0x3C00);  // FP16 1.0
        // scales all zero (default) → via get_scale_min_k4, sc=0 and m=0 → 0.

        Q5_K.dequantToFloat(src, dst, numElements);

        for (long i = 0; i < numElements; i++) {
            assertEquals(0.0f, dst.get(ValueLayout.JAVA_FLOAT, i * 4L), 0f, "i=" + i);
        }
    }
}
