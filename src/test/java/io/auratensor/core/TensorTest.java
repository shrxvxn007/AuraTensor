package io.auratensor.core;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TensorTest {

    Tensor allocated;

    @AfterEach
    void cleanup() {
        if (allocated != null) allocated.close();
    }

    @Test
    void rankMustBe1To3() {
        assertThrows(IllegalArgumentException.class,
                     () -> Tensor.allocate(DType.FP32, 4, 4, 4, 4));
        assertThrows(IllegalArgumentException.class,
                     () -> Tensor.allocate(DType.FP32));
    }

    @Test
    void dimsMustBePositive() {
        assertThrows(IllegalArgumentException.class,
                     () -> Tensor.allocate1D(DType.FP32, 0));
        assertThrows(IllegalArgumentException.class,
                     () -> Tensor.allocate2D(DType.FP32, 0, 4));
    }

    @Test
    void roundtripFloatsByIndex() {
        allocated = Tensor.allocate2D(DType.FP32, 2, 3);
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 3; c++) {
                allocated.setFloat(r * 3.f + c, r, c);
            }
        }
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 3; c++) {
                assertEquals(r * 3.f + c, allocated.getFloat(r, c), 1e-6f);
            }
        }
    }

    @Test
    void stridesAreRowMajor() {
        allocated = Tensor.allocate3D(DType.FP32, 2, 3, 4);
        long[] s = allocated.strides();
        assertArrayEquals(new long[]{12, 4, 1}, s);
    }

    @Test
    void elementOffsetHonoursStrides() {
        allocated = Tensor.allocate2D(DType.FP32, 2, 3);
        // (1, 0) -> row 1, col 0 -> offset = 1*3 + 0 = 3
        // (0, 2) -> row 0, col 2 -> offset = 0*3 + 2 = 2
        // (1, 2) -> row 1, col 2 -> offset = 1*3 + 2 = 5
        assertEquals(3, allocated.elementOffset(1, 0));
        assertEquals(2, allocated.elementOffset(0, 2));
        assertEquals(5, allocated.elementOffset(1, 2));
    }

    @Test
    void indexOutOfBounds() {
        allocated = Tensor.allocate2D(DType.FP32, 2, 3);
        assertThrows(IndexOutOfBoundsException.class, () -> allocated.getFloat(2, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> allocated.getFloat(0, 3));
        assertThrows(IllegalArgumentException.class, () -> allocated.getFloat(0, 0, 0));
    }

    @Test
    void writeFromFloatBuffer() {
        allocated = Tensor.allocate1D(DType.FP32, 4);
        java.nio.FloatBuffer fb = java.nio.FloatBuffer.wrap(
            new float[]{ 1.0f, 2.0f, 3.0f, 4.0f });
        allocated.writeFrom(fb);
        assertEquals(1.0f, allocated.getFloat(0));
        assertEquals(4.0f, allocated.getFloat(3));
    }
}
