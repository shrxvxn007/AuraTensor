package io.auratensor.benchmarks;

import io.auratensor.core.DType;
import io.auratensor.core.Tensor;
import io.auratensor.inference.LlamaConfig;
import io.auratensor.inference.LlamaModel;
import io.auratensor.inference.Sampler;
import io.auratensor.inference.Tokenizer;
import io.auratensor.inference.Weights;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.*;

/**
 * JMH single-token decode latency and tokens/sec for AuraTensor, driven by
 * the production {@link LlamaModel#forwardStep} path with synthetic
 * Llama-150M analogue weights (4 transformer blocks, embedding 512, 8 query
 * heads / 4 KV heads, head-dim 64 — NEON SPECIES_PREFERRED = 4 friendly,
 * FFN 2048, vocab 32 768).
 *
 * <p>One {@code @Benchmark} invocation runs {@link #GENS} decode tokens
 * sequentially. The JMH score is {@code ms/op}; the headline metric —
 * <b>tokens/sec</b> — is back-computable as {@code GENS × 1000 / ms_op}.
 *
 * <p>Production-validated: the benchmark exercises the real
 * {@link LlamaModel#forwardStep} code path, including:
 * <ul>
 *   <li>{@code forwardStep → writeTokenEmbedding → layerStep × N}</li>
 *   <li>Per-layer projections via scalar {@code matVec3D}, {@code matVec2D},
 *       and {@code matVecOutput} on the GGUF byte layout</li>
 *   <li>{@link io.auratensor.core.Kernels#rmsNormInPlace} (SIMD),
 *       {@link io.auratensor.core.Kernels#ropeInPlace} (SIMD),
 *       {@link io.auratensor.core.Kernels#siluInPlace} (SIMD)</li>
 *   <li>KV-cache appends through the now flat {@link java.lang.foreign.MemorySegment}-backed
 *       cache and full softmax attention over the prefix</li>
 *   <li>Residual connections, SiLU FFN (gate × up → down), and the scalar
 *       output projection over the vocab</li>
 * </ul>
 *
 * <p>Memory note: a fully-populated Llama-150M analogue is ~190 MB of
 * off-heap FP32 weights, built fresh in {@code @Setup(Level.Trial)}.
 * Per-step allocation churn comes from {@code LlamaModel.layerStep}'s
 * scratch-Tensor arenas and from the per-(layer, head) {@code float[ctx+1]}
 * inner-attention logits buffer.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 3)
@Fork(1)
public class TokenThroughputBenchmark {

    public static final int GENS = 64;

    // Llama-150M analogue shapes — all divisible by 4 (NEON SPECIES_PREFERRED-friendly).
    private static final int EMB_DIM    = 512;
    private static final int N_HEADS    = 8;
    private static final int N_HEADS_KV = 4;
    private static final int HEAD_DIM   = 64;     // EMB_DIM / N_HEADS
    private static final int BLOCKS     = 4;
    private static final int FFN_DIM    = 2048;
    private static final int VOCAB      = 32768;
    private static final float ROPE_BASE = 10_000.0f;

    @Param({"128", "512", "2048"})
    public int contextLength;

    private LlamaModel model;
    private int[] tokenHistory;     // grows contextLength + GENS slots
    private Sampler.Config samplerConfig;

    @Setup(Level.Trial)
    public void setup() {
        // LlamaModel.forwardStep is contractually constrained to
        //   position < cfg.contextLength()
        // so the KVCache and RoPE tables allocated in the ctor must cover
        // every position we will decode into. We reserve
        //   totalContext = contextLength + GENS
        // for the synthetic prompt + the {@link #GENS} decode steps that
        // {@link #decodeLoop()} will run past it. Note: the synthetic
        // "prompt prefill" itself is skipped — positions [0..contextLength)
        // hold zero-initialized KVCache state at the first decode step.
        // Cost measurement is unaffected; if the benchmark is ever extended
        // to validate decoded-token quality, the prefix must be seeded.
        final long totalContext = contextLength + GENS;

        LlamaConfig cfg = new LlamaConfig(
            "llama",            // architecture
            totalContext,       // contextLength (sized prompt + GENS decode)
            EMB_DIM,            // embeddingLength
            BLOCKS,             // blockCount
            N_HEADS,            // headCount
            N_HEADS_KV,         // headCountKv (GQA 2:1)
            FFN_DIM,            // feedForwardLength
            ROPE_BASE,          // ropeFrequencyBase (Llama 2 / Mistral convention)
            1e-5f,              // rmsNormEpsilon
            VOCAB               // vocabSize
        );

        // Token + output + per-layer tensors populated with deterministic
        // patterns so the forward pass has non-zero data everywhere.
        Tensor tokenEmbeddings = Tensor.allocate2D(DType.FP32, VOCAB, EMB_DIM);
        fillDeterministic(tokenEmbeddings, VOCAB, EMB_DIM, /*rowStride*/0.013, /*colStride*/0.0017);

        Tensor outputNorm   = Tensor.allocate1D(DType.FP32, EMB_DIM);
        fillNormPattern(outputNorm, EMB_DIM);

        // outputWeight shares tokenEmbeddings' byte layout (vocab × embDim).
        Tensor outputWeight = Tensor.allocate2D(DType.FP32, VOCAB, EMB_DIM);
        fillDeterministic(outputWeight, VOCAB, EMB_DIM, /*rowStride*/0.017, /*colStride*/0.0023);

        Weights.Layer[] layers = new Weights.Layer[BLOCKS];
        for (int L = 0; L < BLOCKS; L++) {
            Tensor attnQ     = Tensor.allocate3D(DType.FP32, EMB_DIM, N_HEADS,    HEAD_DIM);
            Tensor attnK     = Tensor.allocate3D(DType.FP32, EMB_DIM, N_HEADS_KV, HEAD_DIM);
            Tensor attnV     = Tensor.allocate3D(DType.FP32, EMB_DIM, N_HEADS_KV, HEAD_DIM);
            Tensor attnOut   = Tensor.allocate3D(DType.FP32, N_HEADS, HEAD_DIM, EMB_DIM);
            Tensor ffnGate   = Tensor.allocate2D(DType.FP32, FFN_DIM, EMB_DIM);
            Tensor ffnUp     = Tensor.allocate2D(DType.FP32, FFN_DIM, EMB_DIM);
            Tensor ffnDown   = Tensor.allocate2D(DType.FP32, EMB_DIM, FFN_DIM);
            Tensor attnNorm  = Tensor.allocate1D(DType.FP32, EMB_DIM);
            Tensor ffnNorm   = Tensor.allocate1D(DType.FP32, EMB_DIM);

            fillDeterministic(attnQ,   EMB_DIM * N_HEADS * HEAD_DIM, /*colsPerRowTreatAs1*/1, 0.011, 0.0019);
            fillDeterministic(attnK,   EMB_DIM * N_HEADS_KV * HEAD_DIM, 1, 0.013, 0.0021);
            fillDeterministic(attnV,   EMB_DIM * N_HEADS_KV * HEAD_DIM, 1, 0.017, 0.0029);
            fillDeterministic(attnOut, N_HEADS * HEAD_DIM * EMB_DIM,    1, 0.019, 0.0031);
            fillDeterministic(ffnGate, FFN_DIM * EMB_DIM,               1, 0.023, 0.0037);
            fillDeterministic(ffnUp,   FFN_DIM * EMB_DIM,               1, 0.029, 0.0041);
            fillDeterministic(ffnDown, EMB_DIM * FFN_DIM,               1, 0.031, 0.0043);
            fillNormPattern(attnNorm, EMB_DIM);
            fillNormPattern(ffnNorm,  EMB_DIM);

            layers[L] = new Weights.Layer(attnQ, attnK, attnV, attnOut,
                                          ffnGate, ffnUp, ffnDown,
                                          attnNorm, ffnNorm);
        }

        Weights weights = new Weights(tokenEmbeddings, outputNorm, outputWeight, layers);
        Tokenizer tokenizer = new SyntheticTokenizer();

        model = new LlamaModel(cfg, weights, tokenizer);

        // Sampler.sample's history is consulted only by repetition-penalty
        // and top-K/T/p samplers; greedy() short-circuits to argmax. So leaving
        // indices [0..contextLength) at zero is harmless for this benchmark;
        // a future non-greedy sampler would require seeding them with the
        // (synthetic) prefix tokens.
        tokenHistory = new int[contextLength + GENS];
        samplerConfig = Sampler.Config.greedy();
    }

    /**
     * Runs {@link #GENS} sequential decode steps through
     * {@code LlamaModel.forwardStep}. Returns the final sampled token id
     * (an {@code int}) so JIT cannot dead-code-eliminate the work.
     */
    @Benchmark
    public int decodeLoop() {
        int lastToken = 0;
        long startPos = contextLength;
        for (int step = 0; step < GENS; step++) {
            float[] logits = model.forwardStep(lastToken, startPos + step);
            int sampled = Sampler.sample(logits, tokenHistory, samplerConfig);
            tokenHistory[step + contextLength] = sampled;
            lastToken = sampled;
        }
        return lastToken;
    }

    // -----------------------------------------------------------------
    // Tensor fill helpers — use the same data().set(ValueLayout.JAVA_FLOAT, …)
    // idiom as production code so allocation accounting matches.
    // -----------------------------------------------------------------

    private static void fillDeterministic(Tensor t, int rows, int cols, double rowStride, double colStride) {
        MemorySegment seg = t.data();
        long rowBytes = (long) cols * 4L;
        for (int r = 0; r < rows; r++) {
            long rowOff = (long) r * rowBytes;
            for (int c = 0; c < cols; c++) {
                float v = (float) Math.sin(r * rowStride + c * colStride);
                seg.set(ValueLayout.JAVA_FLOAT, rowOff + (long) c * 4L, v);
            }
        }
    }

    private static void fillNormPattern(Tensor t, int len) {
        MemorySegment seg = t.data();
        for (int i = 0; i < len; i++) {
            seg.set(ValueLayout.JAVA_FLOAT, (long) i * 4L, 0.7f + 0.3f * (float) Math.sin(i * 0.21));
        }
    }

    /** No-op tokenizer — encode/decode are unused by the synthetic benchmark loop. */
    static final class SyntheticTokenizer implements Tokenizer {
        @Override public int[] encode(String text) { return new int[0]; }
        @Override public String decode(int tokenId) { return ""; }
        @Override public int eosTokenId() { return -1; }
        @Override public Map<Integer, String> vocab() { return Collections.emptyMap(); }
    }
}
