package io.auratensor.core;

import java.nio.ByteOrder;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * IEEE 754 binary16 (= FP16, half precision) conversion helpers.
 *
 * <p>Uses a software bit-manipulation implementation (no subnormals, no NaN
 * payload preservation) so the build is portable across JDK 22+. All
 * routines are allocation-free.
 */
public final class Fp16 {

    private Fp16() {}

    /** Convert a FP16 bit pattern into a FP32 value. Allocation-free. */
    public static float toFloat(short bits) {
        int h = bits & 0xFFFF;
        int s = (h >>> 15) & 0x1;
        int e = (h >>> 10) & 0x1F;
        int m = h & 0x3FF;

        if (e == 0) {
            // Zero / subnormal — flush to zero (matches CUDA half2 semantics).
            return s == 0 ? 0.0f : -0.0f;
        }
        if (e == 0x1F) {
            if (m == 0) {
                return s == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
            }
            return Float.NaN;
        }
        int outBits = (s << 31) | ((e + 112) << 23) | (m << 13);
        return Float.intBitsToFloat(outBits);
    }

    /**
     * Read a FP16 value at byte offset {@code off} in the segment. The
     * segment is assumed to be little-endian (true on every platform
     * AuraTensor supports). JDK 22 dropped the ByteOrder parameter from
     * {@link MemorySegment#get}, so we just pass {@link ValueLayout#JAVA_SHORT}.
     */
    public static float readAt(MemorySegment seg, long off) {
        return toFloat(seg.get(ValueLayout.JAVA_SHORT, off));
    }

    /** Alias of {@link #readAt}; preserved for call-site symmetry. */
    public static float readAtLE(MemorySegment seg, long off) {
        return readAt(seg, off);
    }
}
