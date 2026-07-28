package io.auratensor.quant;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Round-trip and edge-case tests for {@link Q6_K}. The byte packing is
 * byte-exact against the canonical 4-way-interleaved
 * {@code dequantize_row_q6_K} loop in llama.cpp, so a test that
 * (a) re-derives each output via the same bit-extraction logic and
 * (b) compares bit-for-bit against Q6_K is the tightest correctness
 * floor short of decoding a real Llama-3 GGUF dump.
 */
class Q6_KTest {

    @Test
    void dequantMatchesCanonicalInnerLoop() {
        long numElements = Q6_K.BLOCK_ELEMENTS;
        Arena arena = Arena.ofConfined();
        MemorySegment src = arena.allocate(Q6_K.BLOCK_BYTES, 16);
        MemorySegment dst = arena.allocate(numElements * 4L, 16);

        Random r = new Random(7);

        // FP16 d ≈ 0.04 (typical Q6_K super-block scale for LLMs).
        float dVal = 0.04f;
        short d16 = Float.floatToFloat16(dVal);
        src.set(ValueLayout.JAVA_SHORT, 208L, d16);

        // scales[16] = signed int8 in [-128..127], 16 elements per scale.
        for (int i = 0; i < 16; i++) {
            src.set(ValueLayout.JAVA_BYTE, 192L + i, (byte) (r.nextInt(241) - 120));
        }

        // ql[128]: each byte holds two 4-bit unsigned values.
        for (int i = 0; i < 128; i++) {
            int lo = r.nextInt(16);
            int hi = r.nextInt(16);
            src.set(ValueLayout.JAVA_BYTE, i, (byte) ((hi << 4) | lo));
        }

        // qh[64]: each byte holds four 2-bit unsigned values.
        for (int i = 0; i < 64; i++) {
            int b0 = r.nextInt(4);
            int b1 = r.nextInt(4);
            int b2 = r.nextInt(4);
            int b3 = r.nextInt(4);
            int packed = b0 | (b1 << 2) | (b2 << 4) | (b3 << 6);
            src.set(ValueLayout.JAVA_BYTE, 128L + i, (byte) packed);
        }

        Q6_K.dequantToFloat(src, dst, numElements);

        // Reference: re-build each output via the llama.cpp canonical loop
        // (so a pass means both Q6_K and the reference agree on every byte).
        // Byte offsets mirror Q6_K.java: ql at 0..127, qh at 128..191,
        // scales at 192..207, d at 208..209. The impl reads qh from
        // qhBase + n/4 + l (qhBase = 128), so the test must too — reading
        // qh from offset `n / 4 + l` instead picks up ql bytes and the
        // reference diverges from the impl whenever ql and qh carry
        // different bit patterns.
        float dExp = Float.float16ToFloat(d16);
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                byte qlLow = src.get(ValueLayout.JAVA_BYTE, n / 2 + l);
                byte qlHi  = src.get(ValueLayout.JAVA_BYTE, n / 2 + l + 32);
                byte qh    = src.get(ValueLayout.JAVA_BYTE, 128 + n / 4 + l);
                float s0 = src.get(ValueLayout.JAVA_BYTE, 192 + n / 16 + is);
                float s1 = src.get(ValueLayout.JAVA_BYTE, 192 + n / 16 + is + 2);
                float s2 = src.get(ValueLayout.JAVA_BYTE, 192 + n / 16 + is + 4);
                float s3 = src.get(ValueLayout.JAVA_BYTE, 192 + n / 16 + is + 6);

                int q1 = ((qlLow & 0x0F) | (((qh >> 0) & 0x03) << 4)) - 32;
                int q2 = ((qlHi  & 0x0F) | (((qh >> 2) & 0x03) << 4)) - 32;
                // Mirror the Q6_K.java bugfix: `(byte) 0xFF` sign-extends to
                // int 0xFFFFFFFF on `byte → int` promotion, and a bare
                // `>>> 4` then zero-fills 28 high bits of garbage. Use
                // `(x >> 4) & 0x0F` so the reference produces the same
                // unsigned 4-bit nibble the impl does.
                int q3 = (((qlLow >> 4) & 0x0F) | (((qh >> 4) & 0x03) << 4)) - 32;
                int q4 = (((qlHi  >> 4) & 0x0F) | (((qh >> 6) & 0x03) << 4)) - 32;

                float e1 = dExp * s0 * q1;
                float e2 = dExp * s1 * q2;
                float e3 = dExp * s2 * q3;
                float e4 = dExp * s3 * q4;

                long off1 = (long) (n + l) * 4L;
                long off2 = (long) (n + l + 32) * 4L;
                long off3 = (long) (n + l + 64) * 4L;
                long off4 = (long) (n + l + 96) * 4L;

                assertEquals(e1, dst.get(ValueLayout.JAVA_FLOAT, off1), 0f,
                    "n=" + n + ",l=" + l + ",q1");
                assertEquals(e2, dst.get(ValueLayout.JAVA_FLOAT, off2), 0f,
                    "n=" + n + ",l=" + l + ",q2");
                assertEquals(e3, dst.get(ValueLayout.JAVA_FLOAT, off3), 0f,
                    "n=" + n + ",l=" + l + ",q3");
                assertEquals(e4, dst.get(ValueLayout.JAVA_FLOAT, off4), 0f,
                    "n=" + n + ",l=" + l + ",q4");
            }
        }
    }

    @Test
    void zeroBiasProducesAllZero() {
        // q = 32 (bias) for every element, d = 1, scale = 1 → all zeros.
        // q = 32 = 0b100000: ql nibble = 0 (ql byte = 0); qh bits = 0b10
        // packed into every 2-bit slice of every qh byte (byte = 0xAA).
        long numElements = Q6_K.BLOCK_ELEMENTS;
        Arena arena = Arena.ofConfined();
        MemorySegment src = arena.allocate(Q6_K.BLOCK_BYTES, 16);
        MemorySegment dst = arena.allocate(numElements * 4L, 16);

        for (int i = 0; i < 128; i++) {
            src.set(ValueLayout.JAVA_BYTE, i, (byte) 0);
        }
        for (int i = 0; i < 64; i++) {
            src.set(ValueLayout.JAVA_BYTE, 128L + i, (byte) 0xAA);
        }
        for (int i = 0; i < 16; i++) {
            src.set(ValueLayout.JAVA_BYTE, 192L + i, (byte) 1);
        }
        src.set(ValueLayout.JAVA_SHORT, 208L, (short) 0x3C00);  // FP16 1.0

        Q6_K.dequantToFloat(src, dst, numElements);

        for (long i = 0; i < numElements; i++) {
            assertEquals(0.0f, dst.get(ValueLayout.JAVA_FLOAT, i * 4L), 0f, "i=" + i);
        }
    }

    @Test
    void q0MinAndQ63MaxProducedSignExtreme() {
        // q max (=0b111111) → +31; q min (=0) → -32. With d = scale = 1.
        long numElements = Q6_K.BLOCK_ELEMENTS;
        Arena arena = Arena.ofConfined();
        MemorySegment src = arena.allocate(Q6_K.BLOCK_BYTES, 16);
        MemorySegment dst = arena.allocate(numElements * 4L, 16);

        // q = 63 = 0b111111: ql byte = 0xFF (low + high nibbles both 0xF);
        // qh byte = 0xFF (every 2-bit slice = 0b11).
        for (int i = 0; i < 128; i++) {
            src.set(ValueLayout.JAVA_BYTE, i, (byte) 0xFF);
        }
        for (int i = 0; i < 64; i++) {
            src.set(ValueLayout.JAVA_BYTE, 128L + i, (byte) 0xFF);
        }
        for (int i = 0; i < 16; i++) {
            src.set(ValueLayout.JAVA_BYTE, 192L + i, (byte) 1);
        }
        src.set(ValueLayout.JAVA_SHORT, 208L, (short) 0x3C00);  // FP16 1.0

        Q6_K.dequantToFloat(src, dst, numElements);

        // (63 - 32) * 1 * 1 = +31
        for (long i = 0; i < numElements; i++) {
            assertEquals(31.0f, dst.get(ValueLayout.JAVA_FLOAT, i * 4L), 0f,
                "max-side i=" + i);
        }

        // Flip ql + qh to zero: q = 0 → post-bias -32. Scale + d unchanged.
        for (int i = 0; i < 128; i++) {
            src.set(ValueLayout.JAVA_BYTE, i, (byte) 0);
        }
        for (int i = 0; i < 64; i++) {
            src.set(ValueLayout.JAVA_BYTE, 128L + i, (byte) 0);
        }

        Q6_K.dequantToFloat(src, dst, numElements);

        for (long i = 0; i < numElements; i++) {
            assertEquals(-32.0f, dst.get(ValueLayout.JAVA_FLOAT, i * 4L), 0f,
                "min-side i=" + i);
        }
    }
}
