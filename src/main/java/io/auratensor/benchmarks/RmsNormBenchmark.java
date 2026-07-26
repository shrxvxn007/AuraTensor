package io.auratensor.benchmarks;

import io.auratensor.core.Kernels;
import io.auratensor.core.Tensor;
import io.auratensor.core.DType;
import org.openjdk.jmh.annotations.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.TimeUnit;

/**
 * JMH microbenchmark for the RMSNorm fused kernel.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 2)
@Fork(1)
public class RmsNormBenchmark {

    @Param({"4096"})
    public int dimension;

    private Tensor x;
    private Tensor w;

    @Setup(Level.Trial)
    public void setup() {
        x = Tensor.allocate1D(DType.FP32, dimension);
        w = Tensor.allocate1D(DType.FP32, dimension);
        for (int i = 0; i < dimension; i++) {
            x.setFloat((float) Math.sin(i), i);
            w.setFloat((float) Math.cos(i * 0.001) + 1.0f, i);
        }
    }

    @Benchmark
    public void rmsNorm() {
        Kernels.rmsNormInPlace(x, w, 1e-5f);
    }
}
