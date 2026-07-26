package io.auratensor.benchmarks;

import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;

import java.util.concurrent.TimeUnit;

/**
 * Programmatic JMH driver for AuraTensor benchmarks.
 *
 * <p>Bypasses the {@code META-INF/BenchmarkList} classpath resource lookup
 * that drives {@link org.openjdk.jmh.Main} on the command line. Resolving
 * benchmarks by class name via {@link OptionsBuilder#include(String)} makes
 * the runner immune to classpath-resource shenanigans and lets the same
 * short settings be applied uniformly across all benchmarks.
 *
 * <p>Invocation:
 * <pre>
 *   java --enable-native-access=ALL-UNNAMED \
 *        --add-modules jdk.incubator.vector \
 *        -cp "target/classes:&lt;deps&gt;" \
 *        io.auratensor.benchmarks.Bench [io.auratensor.benchmarks.&lt;Name&gt;]
 * </pre>
 *
 * <p>When {@code args} is empty, runs every benchmark under
 * {@code io.auratensor.benchmarks.*} (matches the README's reproducible
 * command intent). When {@code args[0]} is a JMH pattern substring
 * (e.g. {@code TokenThroughputBenchmark}), the runner uses it directly.
 *
 * <p>JMH flags honour via system properties:
 * <ul>
 *   <li>{@code -Daft.bench.threads=N} → forks</li>
 *   <li>{@code -Daft.bench.warmupSec=N} → warmup seconds (each)</li>
 *   <li>{@code -Daft.bench.measureSec=N} → measurement seconds (each)</li>
 * </ul>
 */
public final class Bench {

    public static void main(String[] args) throws Exception {
        int fork           = Integer.parseInt(System.getProperty("at.bench.forks",       "1"));
        int warmupSec      = Integer.parseInt(System.getProperty("at.bench.warmupSec",  "2"));
        int measureSec     = Integer.parseInt(System.getProperty("at.bench.measureSec", "3"));
        int warmupIters    = Integer.parseInt(System.getProperty("at.bench.warmupIters",  "3"));
        int measureIters   = Integer.parseInt(System.getProperty("at.bench.measureIters", "5"));

        String pattern = (args.length > 0 && !args[0].isEmpty())
                ? args[0]
                : "io\\.auratensor\\.benchmarks\\..*";

        Options opt = new OptionsBuilder()
                .include(pattern)
                .mode(org.openjdk.jmh.annotations.Mode.AverageTime)
                .timeUnit(TimeUnit.MILLISECONDS)
                .warmupIterations(warmupIters)
                .warmupTime(TimeValue.seconds(warmupSec))
                .measurementIterations(measureIters)
                .measurementTime(TimeValue.seconds(measureSec))
                .forks(fork)
                .build();

        new Runner(opt).run();
    }
}
