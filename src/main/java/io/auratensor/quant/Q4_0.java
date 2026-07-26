package io.auratensor.quant;

import java.nio.ByteOrder;
import io.auratensor.core.Simd;
import io.auratensor.format.GgufTensorType;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
/**
 * Q4_0 fast dequantization.
 *
 * <p>Block layout (32 elements/block):
 * <pre>
 *   int16 scale (FP16, little-endian)
 *   byte[16]   32 nibbles packed low-first
 * </pre>
 *
 * <p>Dequantized: x_i = (q_i - 8) * scale. Nibble ordering: in each byte,
 * low nibble is element (2k), high nibble is element (2k+1).
 *
 * <p>This is the dominant Llama 3 4-bit weight format and the most
 * performance-critical inner loop in AuraTensor.
 */
public final class Q4_0 {

    private Q4_0() {}

    /** Scalar reference dequantization (used by tests). */
    public static void dequantToFloat(MemorySegment src, MemorySegment dst, long numElements) {
        long blocks = numElements / QuantBlock.BLOCK_SIZE;
        for (long b = 0; b < blocks; b++) {
            float scale = QuantBlock.readScale(src, b, GgufTensorType.Q4_0);
            long srcOff = b * QuantBlock.bytesPerBlock(GgufTensorType.Q4_0)
                        + QuantBlock.q4OffsetInBlock(0);
            long dstOff = b * QuantBlock.BLOCK_SIZE * 4L;
            for (int i = 0; i < QuantBlock.BLOCK_SIZE; i++) {
                byte pack = src.get(ValueLayout.JAVA_BYTE, srcOff + (i / 2L));
                int q = (i % 2 == 0) ? (pack & 0x0F) : ((pack >>> 4) & 0x0F);
                dst.set(ValueLayout.JAVA_FLOAT, dstOff + (long) i * 4L, (q - 8) * scale);
            }
        }
    }

    /**
     * Fused Q4_0 dequant + dot product. Returns
     * {@code sum_i (dequant(q4_i) * w_i)}.
     */
    public static float dot(MemorySegment q4, MemorySegment weights, long numElements) {
        long blocks = numElements / QuantBlock.BLOCK_SIZE;
        FloatVector acc = FloatVector.zero(Simd.SPECIES);

        for (long b = 0; b < blocks; b++) {
            float scale = QuantBlock.readScale(q4, b, GgufTensorType.Q4_0);
            long srcOff = b * QuantBlock.bytesPerBlock(GgufTensorType.Q4_0)
                        + QuantBlock.q4OffsetInBlock(0);
            long wOff   = b * QuantBlock.BLOCK_SIZE * 4L;

            // Load 16 packed bytes.
            ByteVector packed = ByteVector.fromMemorySegment(
                ByteVector.SPECIES_128, q4, srcOff, ByteOrder.nativeOrder());

            // Extract low / high nibbles as separate ByteVectors.
            ByteVector lowNib = packed.and((byte) 0x0F);
            ByteVector highNib = packed.lanewise(VectorOperators.LSHR, 4)
                                     .and((byte) 0x0F);

            // Widen each 8-bit nibble to 32-bit floats.
            float[] tmpLo = new float[Simd.SPECIES.length()];
            float[] tmpHi = new float[Simd.SPECIES.length()];
            for (int i = 0; i < Simd.SPECIES.length(); i++) {
                tmpLo[i] = (lowNib.lane(i) - 8) * scale;
                tmpHi[i] = (highNib.lane(i) - 8) * scale;
            }
            FloatVector vLo = FloatVector.fromArray(Simd.SPECIES, tmpLo, 0);
            FloatVector vHi = FloatVector.fromArray(Simd.SPECIES, tmpHi, 0);

            FloatVector wLo = FloatVector.fromMemorySegment(
                Simd.SPECIES, weights, wOff, ByteOrder.nativeOrder());
            FloatVector wHi = FloatVector.fromMemorySegment(
                Simd.SPECIES, weights, wOff + (long) Simd.SPECIES.length() * 4L,
                ByteOrder.nativeOrder());

            acc = vLo.fma(wLo, acc);
            acc = vHi.fma(wHi, acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }
}
