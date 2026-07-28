package io.auratensor.quant;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Round-trip and edge-case tests for {@link Q8_K}. The byte packing is
 * byte-exact against llama.cpp's linear
 * {@code dequantize_row_q8_K} loop (no 4-way interleaving), so the test
 * re-derives each output via the same {@code d * qs[i]} formula and
 * compares bit-for-bit against Q8_K's output.
 */
class Q8_KTest {

    @Test
    void dequantMatchesCanonicalInnerLoop() {
        long numElements = Q8_K.BLOCK_ELEMENTS;
        Arena arena = Arena.ofConfined();
        MemorySegment src = arena.allocate(Q8_K.BLOCK_BYTES, 16);
        MemorySegment dst = arena.allocate(numElements * 4L, 16);

        Random r = new Random(11);

        // FP32 d ≈ 0.04 (typical Q8_K super-block scale for LLMs).
        float dVal = 0.04f;
        src.set(ValueLayout.JAVA_FLOAT, 0L, dVal);

        // qs[256]: each byte is signed int8 in [-128..127].
        for (int i = 0; i < 256; i++) {
            int v = r.nextInt(256) - 128;
            src.set(ValueLayout.JAVA_BYTE, 4L + i, (byte) v);
        }

        Q8_K.dequantToFloat(src, dst, numElements);

        // Reference: re-derive each output via the canonical linear loop.
        long qsBase = 4L;
        for (int i = 0; i < 256; i++) {
            byte q = src.get(ValueLayout.JAVA_BYTE, qsBase + i);
            float expected = dVal * q;
            assertEquals(expected, dst.get(ValueLayout.JAVA_FLOAT, (long) i * 4L), 0f,
                "i=" + i);
        }
    }

    @Test
    void allZeroQsProducesAllZero() {
        long numElements = Q8_K.BLOCK_ELEMENTS;
        Arena arena = Arena.ofConfined();
        MemorySegment src = arena.allocate(Q8_K.BLOCK_BYTES, 16);
        MemorySegment dst = arena.allocate(numElements * 4L, 16);

        src.set(ValueLayout.JAVA_FLOAT, 0L, 1.0f);   // any d
        // qs[256] all zero default

        Q8_K.dequantToFloat(src, dst, numElements);

        for (long i = 0; i < numElements; i++) {
            assertEquals(0.0f, dst.get(ValueLayout.JAVA_FLOAT, i * 4L), 0f, "i=" + i);
        }
    }

    @Test
    void q127MaxAndQM128MinProduceSignExtreme() {
        long numElements = Q8_K.BLOCK_ELEMENTS;
        Arena arena = Arena.ofConfined();
        MemorySegment src = arena.allocate(Q8_K.BLOCK_BYTES, 16);
        MemorySegment dst = arena.allocate(numElements * 4L, 16);

        src.set(ValueLayout.JAVA_FLOAT, 0L, 1.0f);
        // qs[256] = 0x7F (127 = signed byte max). Expected output = +127.
        for (int i = 0; i < 256; i++) {
            src.set(ValueLayout.JAVA_BYTE, 4L + i, (byte) 0x7F);
        }
        Q8_K.dequantToFloat(src, dst, numElements);
        for (long i = 0; i < numElements; i++) {
            assertEquals(127.0f, dst.get(ValueLayout.JAVA_FLOAT, i * 4L), 0f, "max i=" + i);
        }
        // qs[256] = 0x80 (-128 = signed byte min). Expected output = -128.
        for (int i = 0; i < 256; i++) {
            src.set(ValueLayout.JAVA_BYTE, 4L + i, (byte) 0x80);
        }
        Q8_K.dequantToFloat(src, dst, numElements);
        for (long i = 0; i < numElements; i++) {
            assertEquals(-128.0f, dst.get(ValueLayout.JAVA_FLOAT, i * 4L), 0f, "min i=" + i);
        }
    }
}
