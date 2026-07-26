package io.auratensor.quant;

import io.auratensor.core.Fp16;
import io.auratensor.format.GgufTensorType;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
/**
 * Constants and block layout helpers for GGUF quantization formats.
 *
 * <p>AuraTensor implements F32, F16, Q8_0 and Q4_0:
 * <ul>
 *   <li><b>Q8_0</b>: 32 elements per block. 2-byte FP16 scale + 32 int8 values = 34 bytes/block.</li>
 *   <li><b>Q4_0</b>: 32 elements per block. 2-byte FP16 scale + 16 bytes (32 nibbles) = 18 bytes/block.</li>
 * </ul>
 *
 * <p>Both formats symmetric around zero with offset:
 * <ul>
 *   <li>Q8_0: x_i = q_i * scale (q_i int8, scale fp16)</li>
 *   <li>Q4_0: x_i = (q_i - 8) * scale (q_i in [0..15])</li>
 * </ul>
 */
public final class QuantBlock {

    public static final int BLOCK_SIZE = 32;

    public static int bytesPerBlock(GgufTensorType t) {
        return switch (t) {
            case Q8_0 -> 34;
            case Q4_0 -> 18;
            default -> throw new IllegalArgumentException("Not a block-quantized type: " + t);
        };
    }

    /** Read the FP16 scale at the start of the {@code blockIdx}-th block. */
    public static float readScale(MemorySegment seg, long blockIdx, GgufTensorType t) {
        long off = blockIdx * bytesPerBlock(t);
        return Fp16.readAtLE(seg, off);
    }

    /** Write a FP16 scale at the start of a block (for round-trip tests). */
    public static void writeScale(MemorySegment seg, long blockIdx, GgufTensorType t, float v) {
        long off = blockIdx * bytesPerBlock(t);
        // Pack fp32 → fp16 (truncate mantissa) for round-trip tests.
        int fbits = Float.floatToRawIntBits(v);
        int sign = (fbits >>> 16) & 0x8000;
        int val  = (fbits & 0x7FFFFFFF) + 0x1000;
        int val2 = (val >>> 13) + 1;
        int exp  = ((val2 >>> 23) - 0x7F) >> 12;
        int half = sign | (((val2 & 0x007FFFFF) + 0x1000) >> 13);
        half = (half & 0xFFFF);
        half |= (exp + 15) << 10;
        half &= 0xFFFF;
        seg.set(ValueLayout.JAVA_SHORT, off, (short) half);
    }

    public static long q8OffsetInBlock(int idx) {
        return 2L + idx;  // After 2-byte FP16 scale
    }

    public static long q4OffsetInBlock(int idx) {
        return 2L + (idx / 2);  // After 2-byte FP16 scale, then packed nibbles
    }

    private QuantBlock() {}
}
