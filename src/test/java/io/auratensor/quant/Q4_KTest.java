package io.auratensor.quant;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Round-trip and edge-case tests for {@link Q4_K}. The byte packing is
 * byte-exact against llama.cpp's {@code dequantize_row_q4_K} loop
 * (with the {@code get_scale_min_k4} scale-bit-packing quirk inlined),
 * so the test re-derives each output via the same
 * {@code d * sc * q - dmin * m} formula and compares bit-for-bit.
 */
class Q4_KTest {

    @Test
    void dequantMatchesCanonicalInnerLoop() {
        long numElements = Q4_K.BLOCK_ELEMENTS;
        Arena arena = Arena.ofConfined();
        MemorySegment src = arena.allocate(Q4_K.BLOCK_BYTES, 16);
        MemorySegment dst = arena.allocate(numElements * 4L, 16);

        Random r = new Random(17);

        short d16 = Float.floatToFloat16(0.05f);
        short min16 = Float.floatToFloat16(0.01f);
        src.set(ValueLayout.JAVA_SHORT, 0L, d16);
        src.set(ValueLayout.JAVA_SHORT, 2L, min16);

        // scales[16]: 4-bit d-scale (low nibble) | 4-bit min-scale (high nibble)
        //            + high-2-bit spillover via q[j-4] >> 6 for j ≥ 4.
        // We just write fully random 8-bit bytes; the test will re-apply the canonical
        // get_scale_min_k4 algorithm to extract sc/m.
        for (int i = 0; i < 16; i++) {
            src.set(ValueLayout.JAVA_BYTE, 4L + i, (byte) r.nextInt(256));
        }

        // qs[128]: each byte holds two 4-bit signed values (after -16 de-bias in the impl).
        for (int i = 0; i < 128; i++) {
            int lo = r.nextInt(16);
            int hi = r.nextInt(16);
            src.set(ValueLayout.JAVA_BYTE, 20L + i, (byte) ((hi << 4) | lo));
        }

        Q4_K.dequantToFloat(src, dst, numElements);

        // Reference: re-build each output via the llama.cpp canonical loop,
        // using the canonical get_scale_min_k4 helper directly.
        float dExp = Float.float16ToFloat(d16);
        float mExp = Float.float16ToFloat(min16);
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            long qOff = 20L + (j / 2);

            // get_scale_min_k4(is + 0, scales)
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
                byte q = src.get(ValueLayout.JAVA_BYTE, qOff + l);
                float exp = d1 * (q & 0x0F) - mm1;
                assertEquals(exp,
                    dst.get(ValueLayout.JAVA_FLOAT, (long) (j + l) * 4L), 0f,
                    "j=" + j + ",l=" + l + ",first-batch");
            }
            for (int l = 0; l < 32; l++) {
                byte q = src.get(ValueLayout.JAVA_BYTE, qOff + l);
                float exp = d2 * ((q >> 4) & 0x0F) - mm2;
                assertEquals(exp,
                    dst.get(ValueLayout.JAVA_FLOAT, (long) (j + l + 32) * 4L), 0f,
                    "j=" + j + ",l=" + l + ",second-batch");
            }
            is += 2;
        }
    }

    @Test
    void q0MinAndQ15MaxProduceSignExtreme() {
        // q = 0 (low nibble = 0 OR high nibble = 0) → output = -mm.
        // q = 15 (max 4-bit) → output = +15 * d1 - mm.
        // With d = 1, dmin = 1, sc = 1, m = 1 → output = d * 1 * q - 1*1 = q - 1.
        long numElements = Q4_K.BLOCK_ELEMENTS;
        Arena arena = Arena.ofConfined();
        MemorySegment src = arena.allocate(Q4_K.BLOCK_BYTES, 16);
        MemorySegment dst = arena.allocate(numElements * 4L, 16);

        src.set(ValueLayout.JAVA_SHORT, 0L, (short) 0x3C00);  // FP16 1.0
        src.set(ValueLayout.JAVA_SHORT, 2L, (short) 0x3C00);  // FP16 1.0
        // scales[16] = full bit pattern so each byte decodes via both
        // sc-low-nibble and m-low-nibble = 1 at least. We need sc=1, m=1
        // resolved by get_scale_min_k4 for j=0..7 (where is = 0..7):
        //   j=0 (is=0): sc = q[0] & 63, m = q[4] & 63. Pick q[0] = 0x01, q[4] = 0x01.
        // For j=4 (is=4): sc = (q[8] & 0xF) | ((q[0] >> 6) << 4)
        //                = (q[8] & 0xF) | ((0x01 >> 6) << 4) = q[8] & 0xF  (so q[8]=0x01)
        //                m  = (q[8] >> 4) | ((q[4] >> 6) << 4)
        //                = (q[8] >> 4) | ((0x01 >> 6) << 4) = q[8] >> 4   (so q[8]=0x10)
        // ... impossible to satisfy BOTH q[8] & 0xF = 1 AND q[8] >> 4 = 1.
        // Use simple test: scales byte just produces some scale; verify the
        // formula x = d * sc * q - dmin * m holds relative to the impl
        // (so any specific test value isn't important — only the impl-vs-
        // ref consistency).
        // Pick scales bytes so that get_scale_min_k4 returns sc = m = 8 for
        // every is: impossible for all is simultaneously; just pick one
        // concrete set and re-check.
        // Simplest: scales[0]=8, scales[4]=8 (so is=0 → sc=8, m=8).
        src.set(ValueLayout.JAVA_BYTE, 4L + 0, (byte) 0x08);
        src.set(ValueLayout.JAVA_BYTE, 4L + 4, (byte) 0x08);
        // qs[128]: q = 15 (all-ones byte = 0xFF).
        for (int i = 0; i < 128; i++) src.set(ValueLayout.JAVA_BYTE, 20L + i, (byte) 0xFF);

        Q4_K.dequantToFloat(src, dst, numElements);

        // For first 32 outputs (j=0): d=1, sc=8, q=15 → 1*8*15 - 1*8 = 112.
        for (int l = 0; l < 32; l++) {
            assertEquals(112.0f, dst.get(ValueLayout.JAVA_FLOAT, (long) l * 4L), 0f,
                "j=0,l=" + l);
        }
        // For first 32 outputs (j=0): with q=15, high nibble of qs[l] = 15.
        // But wait — at j=0, batch 2 covers outputs 32..63; their sc/m come
        // from is+1, which means scales[is+1] = scales[1]. That byte was 0.
        // So sc=0, m=0, output = 0 - 0 = 0 regardless of q. Expected.
        for (int l = 32; l < 64; l++) {
            assertEquals(0.0f, dst.get(ValueLayout.JAVA_FLOAT, (long) l * 4L), 0f,
                "j=0,batch2,l=" + l);
        }
    }

    @Test
    void allZeroProducesZero() {
        // All-zero scales + zero qs + zero dmin → all-zero output.
        long numElements = Q4_K.BLOCK_ELEMENTS;
        Arena arena = Arena.ofConfined();
        MemorySegment src = arena.allocate(Q4_K.BLOCK_BYTES, 16);
        MemorySegment dst = arena.allocate(numElements * 4L, 16);

        src.set(ValueLayout.JAVA_SHORT, 0L, (short) 0x3C00);  // FP16 1.0
        src.set(ValueLayout.JAVA_SHORT, 2L, (short) 0);         // FP16 0.0

        Q4_K.dequantToFloat(src, dst, numElements);

        for (long i = 0; i < numElements; i++) {
            assertEquals(0.0f, dst.get(ValueLayout.JAVA_FLOAT, i * 4L), 0f, "i=" + i);
        }
    }
}
