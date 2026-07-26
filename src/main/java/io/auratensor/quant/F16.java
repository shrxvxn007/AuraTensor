package io.auratensor.quant;

import java.nio.ByteOrder;
import io.auratensor.core.Fp16;
import io.auratensor.core.Simd;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
/**
 * F16 (FP16) dequantization and dot product.
 *
 * <p>Reads 2 bytes per element from the source segment, converts to FP32
 * via {@link Fp16}, and accumulates a dot product against an FP32 vector.
 */
public final class F16 {

    private F16() {}

    /** Scalar dequant to FP32 (length-doubling output). */
    public static void dequantToFloat(MemorySegment src, MemorySegment dst, long numElements) {
        for (long i = 0; i < numElements; i++) {
            float v = Fp16.readAtLE(src, i * 2L);
            dst.set(ValueLayout.JAVA_FLOAT, i * 4L, v);
        }
    }

    /** Fused FP16 → FP32 dot product: returns sum_i (fp16_i * fp32_weights_i). */
    public static float dot(MemorySegment f16, MemorySegment weights, long numElements) {
        FloatVector acc = FloatVector.zero(Simd.SPECIES);
        int upper = Simd.loopBound((int) numElements);
        // Inner loop: stream FP16 → FP32 via temp array. (Future optimization:
        // use ByteVector → cast → FloatVector via float16ToFloat intrinsic.)
        for (int i = 0; i < upper; i += Simd.SPECIES.length()) {
            float[] tmp = new float[Simd.SPECIES.length()];
            for (int j = 0; j < Simd.SPECIES.length(); j++) {
                tmp[j] = Fp16.readAtLE(f16, (long)(i + j) * 2L);
            }
            FloatVector v = FloatVector.fromArray(Simd.SPECIES, tmp, 0);
            FloatVector w = FloatVector.fromMemorySegment(
                Simd.SPECIES, weights, (long) i * 4L, ByteOrder.nativeOrder());
            acc = v.fma(w, acc);
        }
        float s = acc.reduceLanes(VectorOperators.ADD);
        for (long i = upper; i < numElements; i++) {
            s += Fp16.readAtLE(f16, i * 2L)
               * weights.get(ValueLayout.JAVA_FLOAT, i * 4L);
        }
        return s;
    }
}
