package io.auratensor.quant;

import java.nio.ByteOrder;
import io.auratensor.core.Simd;
import io.auratensor.format.GgufTensorType;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
/**
 * Q8_0 fast dequantization.
 *
 * <p>Block layout (32 elements/block):
 * <pre>
 *   int16 scale (FP16, little-endian)
 *   int8[32]    quantized values
 * </pre>
 *
 * <p>Dequantized: x_i = scale * q_i.
 */
public final class Q8_0 {

    private Q8_0() {}

    /**
     * Scalar dequantization to a preallocated FloatVector-friendly region. Used
     * for tests and as a reference baseline.
     */
    public static void dequantToFloat(MemorySegment src, MemorySegment dst, long numElements) {
        long blocks = numElements / QuantBlock.BLOCK_SIZE;
        for (long b = 0; b < blocks; b++) {
            float scale = QuantBlock.readScale(src, b, GgufTensorType.Q8_0);
            long srcOff = b * QuantBlock.bytesPerBlock(GgufTensorType.Q8_0)
                        + QuantBlock.q8OffsetInBlock(0);
            long dstOff = b * QuantBlock.BLOCK_SIZE * 4L;
            for (int i = 0; i < QuantBlock.BLOCK_SIZE; i++) {
                byte q = src.get(ValueLayout.JAVA_BYTE, srcOff + i);
                dst.set(ValueLayout.JAVA_FLOAT, dstOff + (long) i * 4L, q * scale);
            }
        }
    }

    /**
     * Fused Q8_0 dequant + dot product. Returns
     * {@code sum_i (dequant(q8_i) * w_i)}.
     *
     * @param q8 raw Q8_0 quantized tensor segment
     * @param weights FP32 vector segment (length must equal {@code numElements})
     * @param numElements = 32 * numBlocks
     */
    public static float dot(MemorySegment q8, MemorySegment weights, long numElements) {
        long blocks = numElements / QuantBlock.BLOCK_SIZE;
        FloatVector acc = FloatVector.zero(Simd.SPECIES);

        for (long b = 0; b < blocks; b++) {
            float scale = QuantBlock.readScale(q8, b, GgufTensorType.Q8_0);
            long srcOff = b * QuantBlock.bytesPerBlock(GgufTensorType.Q8_0)
                        + QuantBlock.q8OffsetInBlock(0);
            long wOff   = b * QuantBlock.BLOCK_SIZE * 4L;

            // Load 32 bytes via two 128-bit ByteVector loads, widen to fp32.
            var bvLo = ByteVector.fromMemorySegment(
                ByteVector.SPECIES_128, q8, srcOff, ByteOrder.nativeOrder());
            var bvHi = ByteVector.fromMemorySegment(
                ByteVector.SPECIES_128, q8, srcOff + 16L, ByteOrder.nativeOrder());

            float[] tmp = new float[16];
            for (int i = 0; i < 16; i++) {
                int bi = bvLo.lane(i);
                tmp[i] = bi * scale;
            }
            FloatVector vLo = FloatVector.fromArray(Simd.SPECIES, tmp, 0);
            for (int i = 0; i < 16; i++) {
                tmp[i] = bvHi.lane(i) * scale;
            }
            FloatVector vHi = FloatVector.fromArray(Simd.SPECIES, tmp, 0);

            FloatVector wLo = FloatVector.fromMemorySegment(
                Simd.SPECIES, weights, wOff, ByteOrder.nativeOrder());
            FloatVector wHi = FloatVector.fromMemorySegment(
                Simd.SPECIES, weights, wOff + (long) Simd.SPECIES.length() * 4L,
                ByteOrder.nativeOrder());

            acc = vLo.fma(wLo, acc);
            acc = vHi.fma(wHi, acc);
        }
        // QuantBlock.BLOCK_SIZE is divided evenly by the lane width on every
        // common SIMD width (256/512/128), so no scalar tail is needed.
        return acc.reduceLanes(VectorOperators.ADD);
    }
}
