package io.auratensor.benchmarks;

import io.auratensor.inference.Sampler;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

/**
 * JMH microbenchmark for the sampler loop.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 2)
@Fork(1)
public class SamplerBenchmark {

    @Param({"32000", "128256"})
    public int vocab;

    private float[] logits;
    private int[] history;

    @Setup(Level.Trial)
    public void setup() {
        logits = new float[vocab];
        history = new int[64];
        for (int i = 0; i < vocab; i++) {
            logits[i] = (float) (Math.sin(i * 0.001) * 5);
        }
        for (int i = 0; i < history.length; i++) {
            history[i] = i % vocab;
        }
    }

    @Benchmark
    public int sampleDefault() {
        Sampler.Config cfg = new Sampler.Config(0.7f, 40, 0.95f, 1.1f, 0);
        return Sampler.sample(logits, history, cfg);
    }
}
