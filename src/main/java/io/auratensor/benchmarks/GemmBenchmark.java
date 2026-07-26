package io.auratensor.benchmarks;

import java.lang.foreign.MemorySegment;
import io.auratensor.core.Kernels;
import io.auratensor.core.Tensor;
import io.auratensor.core.DType;
import org.openjdk.jmh.annotations.*;

import java.lang.foreign.Arena;
import java.util.concurrent.TimeUnit;

/**
 * JMH microbenchmark for SIMD vs scalar SGEMM.
 *
 * <p>Measures GFLOPS on a fixed-shape matrix product that mimics a small
 * decoder step (M=batch, K=embeddingDim, N=ffnDim).
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 3)
@Fork(1)
public class GemmBenchmark {

    @Param({"512", "1024", "2048"})
    public int K;

    private MemorySegment a;
    private MemorySegment b;
    private MemorySegment c;
    private int M = 1;
    private int N;

    @Setup(Level.Trial)
    public void setup() {
        N = K;
        Arena arena = Arena.ofConfined();
        a = arena.allocate((long) M * K * 4L, 16);
        b = arena.allocate((long) K * N * 4L, 16);
        c = arena.allocate((long) M * N * 4L, 16);
        // Initialize with deterministic values so the JIT cannot specialize
        // too aggressively on zeros.
        for (long i = 0; i < (long) M * K; i++) {
            a.set(java.lang.foreign.ValueLayout.JAVA_FLOAT, i * 4L, (float) ((i * 73) % 1000) / 1000f);
        }
        for (long i = 0; i < (long) K * N; i++) {
            b.set(java.lang.foreign.ValueLayout.JAVA_FLOAT, i * 4L, (float) ((i * 37) % 1000) / 1000f);
        }
    }

    @Benchmark
    public float[] simdSgemm() {
        Kernels.sgemm(a, b, c, M, K, N);
        return new float[]{(float) (M * K * N * 2.0)};
    }

    @Benchmark
    public float[] scalarSgemm() {
        Kernels.sgemmScalar(a, b, c, M, K, N);
        return new float[]{(float) (M * K * N * 2.0)};
    }
}
