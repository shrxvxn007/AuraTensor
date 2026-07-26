package io.auratensor.benchmarks;

import io.auratensor.core.Fp16;
import io.auratensor.quant.Q4_0;
import org.openjdk.jmh.annotations.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * JMH microbenchmark for Q4_0 dequantization + dot product.
 *
 * <p>Measures both throughput (elements/sec) and effective achieved memory
 * bandwidth in GB/s.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 3)
@Fork(1)
public class Q4Benchmark {

    @Param({"8192", "32768", "131072"})
    public int numElements;

    private MemorySegment q4;
    private MemorySegment weights;

    @Setup(Level.Trial)
    public void setup() {
        long blocks = numElements / 32L;
        Arena arena = Arena.ofConfined();
        long size = blocks * 18L;  // Q4_0 bytes per block
        q4 = arena.allocate(size, 16);
        weights = arena.allocate((long) numElements * 4L, 16);

        Random r = new Random(42);
        // Initialize q4 weights: FP16 scale + 32 random nibbles per block.
        for (long b = 0; b < blocks; b++) {
            float s = (float) (r.nextGaussian() * 0.05);
            int sBits = Float.floatToRawIntBits(s);
            int half = ((sBits >>> 13) & 0xFFFF) | 0x3C00;
            q4.set(java.lang.foreign.ValueLayout.JAVA_SHORT, b * 18L, (short) half);
            for (int i = 0; i < 16; i++) {
                int lo = r.nextInt(16);
                int hi = r.nextInt(16);
                q4.set(java.lang.foreign.ValueLayout.JAVA_BYTE, b * 18L + 2L + i, (byte) ((hi << 4) | lo));
            }
        }
        for (long i = 0; i < numElements; i++) {
            weights.set(java.lang.foreign.ValueLayout.JAVA_FLOAT, i * 4L, r.nextFloat() - 0.5f);
        }
    }

    @Benchmark
    public float fusedDot() {
        return Q4_0.dot(q4, weights, numElements);
    }

    @Benchmark
    public float[] dequantOnly() {
        long blocks = numElements / 32L;
        Arena arena = Arena.ofConfined();
        MemorySegment dst = arena.allocate((long) numElements * 4L, 16);
        for (long b = 0; b < blocks; b++) {
            Q4_0.dequantToFloat(q4, dst, numElements);
        }
        return new float[]{1.0f};
    }
}
