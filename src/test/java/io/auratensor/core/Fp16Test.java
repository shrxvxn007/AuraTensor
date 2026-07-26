package io.auratensor.core;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Fp16Test {

    @Test
    void zeroIsZero() {
        assertEquals(0.0f, Fp16.toFloat((short) 0));
    }

    @Test
    void oneIsApproximatelyOne() {
        // 1.0 fp16 = 0x3C00
        float v = Fp16.toFloat((short) 0x3C00);
        assertEquals(1.0f, v, 1e-6f);
    }

    @Test
    void twoIsApproximatelyTwo() {
        // 2.0 fp16 = 0x4000
        assertEquals(2.0f, Fp16.toFloat((short) 0x4000), 1e-6f);
    }

    @Test
    void negativeZeroIsNegativeZero() {
        short negZero = (short) 0x8000;
        float v = Fp16.toFloat(negZero);
        assertEquals(0.0f, v, 0.0f);
        assertEquals(Float.floatToRawIntBits(-0.0f), Float.floatToRawIntBits(v));
    }

    @Test
    void infinityRoundtrip() {
        // +inf = 0x7C00
        assertEquals(Float.POSITIVE_INFINITY, Fp16.toFloat((short) 0x7C00));
        // -inf = 0xFC00
        assertEquals(Float.NEGATIVE_INFINITY, Fp16.toFloat((short) 0xFC00));
    }

    @Test
    void smallDenormalFlushesToZero() {
        // We use zero-subnormal → 0 (matches CUDA half)
        assertEquals(0.0f, Fp16.toFloat((short) 0x0001));
        assertEquals(0.0f, Fp16.toFloat((short) 0x0200));
    }
}
