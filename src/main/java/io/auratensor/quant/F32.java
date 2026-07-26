package io.auratensor.quant;

import java.nio.ByteOrder;
import io.auratensor.core.Simd;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
/**
 * F32-format dot product. Trivially the same as a memory-segment vector
 * dot product, but provided here for symmetry with the quantized types so
 * callers can swap types at one site.
 */
public final class F32 {

    private F32() {}

    public static float dot(MemorySegment f32, MemorySegment weights, long numElements) {
        FloatVector acc = FloatVector.zero(Simd.SPECIES);
        int upper = Simd.loopBound((int) numElements);
        for (int i = 0; i < upper; i += Simd.SPECIES.length()) {
            FloatVector v = FloatVector.fromMemorySegment(
                Simd.SPECIES, f32, (long) i * 4L, ByteOrder.nativeOrder());
            FloatVector w = FloatVector.fromMemorySegment(
                Simd.SPECIES, weights, (long) i * 4L, ByteOrder.nativeOrder());
            acc = v.fma(w, acc);
        }
        float s = acc.reduceLanes(VectorOperators.ADD);
        for (long i = upper; i < numElements; i++) {
            s += f32.get(ValueLayout.JAVA_FLOAT, i * 4L)
               * weights.get(ValueLayout.JAVA_FLOAT, i * 4L);
        }
        return s;
    }
}
