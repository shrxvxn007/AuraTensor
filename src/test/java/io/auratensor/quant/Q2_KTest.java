package io.auratensor.quant;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Round-trip and edge-case tests for {@link Q2_K}. The byte packing is
 * byte-exact against llama.cpp's 4-way-interleaved
 * {@code dequantize_row_q2_K} loop, so the test re-derives each output
 * with the same {@code dl * q - ml} formula and compares bit-for-bit.
 */
class Q2_KTest {

    @Test
    void dequantMatchesCanonicalInnerLoop() {
        long numElements = Q2_K.BLOCK_ELEMENTS;
        Arena arena = Arena.ofConfined();
        MemorySegment src = arena.allocate(Q2_K.BLOCK_BYTES, 16);
        MemorySegment dst = arena.allocate(numElements * 4L, 16);

        Random r = new Random(13);

        // FP16 d ≈ 0.04.
        short d16 = Float.floatToFloat16(0.04f);
        short min16 = Float.floatToFloat16(0.01f);
        src.set(ValueLayout.JAVA_SHORT, 80L, d16);
        src.set(ValueLayout.JAVA_SHORT, 82L, min16);

        // scales[16]: each byte = 4-bit d-scale (low nibble) | 4-bit min-scale (high nibble).
        for (int i = 0; i < 16; i++) {
            int sd = r.nextInt(16);    // 4-bit
            int sm = r.nextInt(16);    // 4-bit
            src.set(ValueLayout.JAVA_BYTE, i, (byte) ((sm << 4) | sd));
        }

        // qs[64]: each byte holds four 2-bit unsigned values.
        for (int i = 0; i < 64; i++) {
            int b0 = r.nextInt(4), b1 = r.nextInt(4), b2 = r.nextInt(4), b3 = r.nextInt(4);
            int packed = b0 | (b1 << 2) | (b2 << 4) | (b3 << 6);
            src.set(ValueLayout.JAVA_BYTE, 16L + i, (byte) packed);
        }

        Q2_K.dequantToFloat(src, dst, numElements);

        // Reference: re-build each output via the llama.cpp canonical loop.
        // Scales byte at index (n/16) + (j * 2) + batch picks: scales array
        // held constant across the block; offset 0..15 within scales[].
        float dExp = Float.float16ToFloat(d16);
        float mExp = Float.float16ToFloat(min16);
        long yOff = 0L;
        for (int n = 0; n < 256; n += 128) {
            long qOff = 16L + n / 4L;
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                byte sc1 = src.get(ValueLayout.JAVA_BYTE, (n / 16) + (j * 2) + 0);
                float dl1 = dExp * (sc1 & 0x0F);
                float ml1 = mExp * (sc1 >> 4);
                for (int l = 0; l < 16; l++) {
                    byte qb = src.get(ValueLayout.JAVA_BYTE, qOff + l);
                    int q = (qb >> shift) & 0x03;
                    float exp = dl1 * q - ml1;
                    assertEquals(exp, dst.get(ValueLayout.JAVA_FLOAT, yOff), 0f,
                        "n=" + n + ",j=" + j + ",l=" + l + ",first-batch");
                    yOff += 4;
                }
                byte sc2 = src.get(ValueLayout.JAVA_BYTE, (n / 16) + (j * 2) + 1);
                float dl2 = dExp * (sc2 & 0x0F);
                float ml2 = mExp * (sc2 >> 4);
                for (int l = 0; l < 16; l++) {
                    byte qb = src.get(ValueLayout.JAVA_BYTE, qOff + 16L + l);
                    int q = (qb >> shift) & 0x03;
                    float exp = dl2 * q - ml2;
                    assertEquals(exp, dst.get(ValueLayout.JAVA_FLOAT, yOff), 0f,
                        "n=" + n + ",j=" + j + ",l=" + l + ",second-batch");
                    yOff += 4;
                }
                shift += 2;
            }
        }
    }

    @Test
    void allZerosProducesAllNegativeMin() {
        // q = 0 every element → output = -ml where ml = dmin * (sc >> 4).
        // For all sc = 0, output = 0. For nonzero but symmetric scales, output = -dmin.
        long numElements = Q2_K.BLOCK_ELEMENTS;
        Arena arena = Arena.ofConfined();
        MemorySegment src = arena.allocate(Q2_K.BLOCK_BYTES, 16);
        MemorySegment dst = arena.allocate(numElements * 4L, 16);

        // qs[64] all 0 (default).
        // scales[16] all 0 (default) → both dl=0 and ml=0 → output 0.
        src.set(ValueLayout.JAVA_SHORT, 80L, Float.floatToFloat16(0.04f));
        src.set(ValueLayout.JAVA_SHORT, 82L, Float.floatToFloat16(0.01f));

        Q2_K.dequantToFloat(src, dst, numElements);

        for (long i = 0; i < numElements; i++) {
            assertEquals(0.0f, dst.get(ValueLayout.JAVA_FLOAT, i * 4L), 0f, "i=" + i);
        }
    }

    @Test
    void qMaxAndQMinAndDZeroScaleProducesBoundary() {
        // q = 3 (max) with d = 1, scale.d = 1, scale.m = 0 → +3.
        // q = 0 with same → 0. (And scales byte = 0x01 = d-scale 1, m-scale 0.)
        long numElements = Q2_K.BLOCK_ELEMENTS;
        Arena arena = Arena.ofConfined();
        MemorySegment src = arena.allocate(Q2_K.BLOCK_BYTES, 16);
        MemorySegment dst = arena.allocate(numElements * 4L, 16);

        src.set(ValueLayout.JAVA_SHORT, 80L, (short) 0x3C00);  // FP16 1.0
        src.set(ValueLayout.JAVA_SHORT, 82L, (short) 0);         // dmin = 0
        // scales[16]: 0x01 (low nibble = 1, high nibble = 0).
        for (int i = 0; i < 16; i++) src.set(ValueLayout.JAVA_BYTE, i, (byte) 0x01);
        // qs[64]: each byte = 0xFF → four 2-bit values = 3 each.
        for (int i = 0; i < 64; i++) src.set(ValueLayout.JAVA_BYTE, 16L + i, (byte) 0xFF);

        Q2_K.dequantToFloat(src, dst, numElements);

        // (q - 0) * 1 * 1 - 0 = 3 for every output.
        for (long i = 0; i < numElements; i++) {
            assertEquals(3.0f, dst.get(ValueLayout.JAVA_FLOAT, i * 4L), 0f, "i=" + i);
        }
    }
}
