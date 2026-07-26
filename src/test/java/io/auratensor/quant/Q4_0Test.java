package io.auratensor.quant;

import io.auratensor.core.Fp16;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class Q4_0Test {

    @Test
    void dequantReferenceMatches() {
        long numElements = 128;
        long blocks = numElements / 32L;
        Arena arena = Arena.ofConfined();
        MemorySegment src = arena.allocate(blocks * 18L, 16);
        MemorySegment dst = arena.allocate(numElements * 4L, 16);

        Random r = new Random(7);
        for (long b = 0; b < blocks; b++) {
            float scale = (float) (r.nextGaussian() * 0.05);
            int sBits = Float.floatToRawIntBits(scale);
            int half = (sBits >>> 13) & 0x7FF | 0x3C00;
            src.set(ValueLayout.JAVA_SHORT, b * 18L, (short) half);
            for (int i = 0; i < 16; i++) {
                int lo = r.nextInt(16);
                int hi = r.nextInt(16);
                src.set(ValueLayout.JAVA_BYTE, b * 18L + 2L + i, (byte) ((hi << 4) | lo));
            }
        }

        Q4_0.dequantToFloat(src, dst, numElements);

        // Compare against scalar expected
        for (long b = 0; b < blocks; b++) {
            int half = src.get(ValueLayout.JAVA_SHORT, b * 18L) & 0xFFFF;
            float scale = Fp16.toFloat((short) half);
            for (int i = 0; i < 32; i++) {
                int packed = src.get(ValueLayout.JAVA_BYTE, b * 18L + 2L + (i / 2L));
                int q = (i % 2 == 0) ? (packed & 0x0F) : ((packed >>> 4) & 0x0F);
                float expected = (q - 8) * scale;
                float actual   = dst.get(ValueLayout.JAVA_FLOAT, (b * 32L + i) * 4L);
                assertEquals(expected, actual, 1e-3f,
                             "Mismatch at block=" + b + " i=" + i);
            }
        }
    }

    @Test
    void fusedDotMatchesScalar() {
        long numElements = 256;
        long blocks = numElements / 32L;
        Arena arena = Arena.ofConfined();
        MemorySegment q4 = arena.allocate(blocks * 18L, 16);
        MemorySegment weights = arena.allocate(numElements * 4L, 16);

        Random r = new Random(101);
        for (long b = 0; b < blocks; b++) {
            float scale = 0.05f;
            int half = (Float.floatToRawIntBits(scale) >>> 13) | 0x3C00;
            q4.set(ValueLayout.JAVA_SHORT, b * 18L, (short) half);
            for (int i = 0; i < 16; i++) {
                int lo = r.nextInt(16);
                int hi = r.nextInt(16);
                q4.set(ValueLayout.JAVA_BYTE, b * 18L + 2L + i, (byte) ((hi << 4) | lo));
            }
        }
        for (long i = 0; i < numElements; i++) {
            weights.set(ValueLayout.JAVA_FLOAT, i * 4L, r.nextFloat() - 0.5f);
        }

        float fused = Q4_0.dot(q4, weights, numElements);

        // Reference: dequant + dot by hand
        float reference = 0.0f;
        for (long b = 0; b < blocks; b++) {
            int half = q4.get(ValueLayout.JAVA_SHORT, b * 18L) & 0xFFFF;
            float scale = Fp16.toFloat((short) half);
            for (int i = 0; i < 32; i++) {
                int packed = q4.get(ValueLayout.JAVA_BYTE, b * 18L + 2L + (i / 2L));
                int q = (i % 2 == 0) ? (packed & 0x0F) : ((packed >>> 4) & 0x0F);
                float x = (q - 8) * scale;
                reference += x * weights.get(ValueLayout.JAVA_FLOAT, (b * 32L + i) * 4L);
            }
        }
        assertEquals(reference, fused, 5e-2f,
                     "fused=" + fused + " ref=" + reference);
    }
}
