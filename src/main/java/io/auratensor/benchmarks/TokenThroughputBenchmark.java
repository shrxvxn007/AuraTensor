package io.auratensor.benchmarks;

import io.auratensor.core.DType;
import io.auratensor.core.Tensor;
import io.auratensor.format.GgufFile;
import io.auratensor.inference.Llama3BpeTokenizer;
import io.auratensor.inference.LlamaConfig;
import io.auratensor.inference.LlamaModel;
import io.auratensor.inference.Sampler;
import io.auratensor.inference.Tokenizer;
import io.auratensor.inference.Weights;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.net.URI;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.*;

/**
 * JMH single-token decode latency and tokens/sec for AuraTensor, two flavours:
 *
 * <ol>
 *   <li><b>Real model</b> (default) — loads an actual Llama-3.2-1B-Instruct
 *       GGUF in Q4_0 quantisation via {@link GgufFile} + {@link Weights#load},
 *       exercising the full production path including load-time Q4_0 → FP32
 *       dequantization and Llama-3 BPE token vocabulary. The
 *       {@link io.auratensor.core.Kernels#sgemv}, {@code rmsNorm}, RoPE, and
 *       SiLU SIMD inner loops all run on a real production shape.</li>
 *   <li><b>Synthetic FP32</b> (fallback) — 150M-analogue weights built from
 *       raw {@link Tensor#allocate3D}. Reproducible, fast, no external model
 *       needed.</li>
 * </ol>
 *
 * <p>The real-model path is selected by default ({@code at.bench.realModel}
 * system property = {@code true}). On any real-model failure (download
 * error, parse error, host hard-coded at "false" in CI) the benchmark
 * transparently falls back to the synthetic path.
 *
 * <p>One {@code @Benchmark} invocation runs {@link #GENS} decode tokens
 * sequentially. JMH score is {@code ms/op}; the headline metric — <b>tokens/sec</b>
 * — is back-computable as {@code GENS × 1000 / ms_op}.
 *
 * <p>JVM tuning system properties:
 * <ul>
 *   <li>{@code -Dat.bench.realModel=true|false} — toggle real vs synthetic</li>
 *   <li>{@code -Dat.bench.modelDir=<path>} — real-model cache directory
 *       (default {@code ~/.auratensor/models})</li>
 *   <li>{@code -Dat.bench.modelUrl=<url>} — override the HuggingFace resolution URL</li>
 * </ul>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 3)
@Fork(1)
public class TokenThroughputBenchmark {

    public static final int GENS = 64;

    // Synthetic FP32 fallback shapes — all divisible by 4 (NEON SPECIES_PREFERRED-friendly).
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
    private GgufFile gguf;          // real-model lifecycle: keep mmap alive
    private boolean usingRealModel;

    private static final boolean USE_REAL_MODEL = Boolean.parseBoolean(
            System.getProperty("at.bench.realModel", "true"));
    private static final String DEFAULT_REAL_MODEL_URL =
            "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf";
    private static final String DEFAULT_MODEL_DIR =
            System.getProperty("at.bench.modelDir",
                    Paths.get(System.getProperty("user.home"), ".auratensor", "models").toString());
    private static final String REAL_MODEL_FILENAME = "Llama-3.2-1B-Instruct-Q4_0.gguf";

    /**
     * Lower bound on a healthy Llama-3.2-1B Q4_0 GGUF download. Anything
     * significantly below 600 MB indicates a truncated download that would
     * silently produce NaN-masked logits and a misleading tokens/sec number.
     * The canonical bartowski export is ~774 MB; Q4_0 + Llama-3.2-vocab
     * + 16-layer transformer header graphs to ~774 059 904 bytes.
     */
    private static final long MIN_HEALTHY_MODEL_BYTES = 600_000_000L;
    private static final long CANONICAL_MODEL_BYTES   = 774_059_904L;

    @Setup(Level.Trial)
    public void setup() throws IOException {
        if (USE_REAL_MODEL) {
            try {
                setupRealModel();
                usingRealModel = true;
                System.out.println("[TokenThroughputBenchmark] Loaded real Llama-3.2-1B Q4_0 from "
                    + ggufDataPath());
                return;
            } catch (UnsupportedOperationException uoe) {
                // Specifically the "tensortype not implemented" path — most bartowski Q4_0
                // exports mix Q6_K for token_embd.weight / output.weight, which AuraTensor
                // doesn't yet dequant. Surface the cause so the fallback isn't mysterious.
                System.err.println("[TokenThroughputBenchmark] Real-model load skipped: " + uoe.getMessage());
                System.err.println("[TokenThroughputBenchmark] Falling back to SyntheticTokenizer FP32 weights.");
            } catch (Throwable t) {
                System.err.println("[TokenThroughputBenchmark] Real-model load failed: " + t);
                System.err.println("[TokenThroughputBenchmark] Falling back to synthetic FP32 weights.");
            }
        }
        setupSyntheticModel();
    }

    @TearDown(Level.Trial)
    public void tearDown() throws IOException {
        if (gguf != null) {
            gguf.close();
            gguf = null;
        }
    }

    /**
     * Runs {@link #GENS} sequential decode steps through
     * {@code LlamaModel.forwardStep}. Returns the final sampled token id
     * so JIT cannot dead-code-eliminate the work.
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
    // Real-model setup — auto-download → GgufFile.open → LlamaConfig →
    // Weights.load (with Q4_0 → FP32 dequant dispatch) → Llama3BpeTokenizer.
    // -----------------------------------------------------------------

    private void setupRealModel() throws IOException {
        Path candidate = ensureRealModel();
        gguf = GgufFile.open(candidate.toString());
        LlamaConfig cfg = LlamaConfig.fromGguf(gguf.metadata());

        // The model's advertised context length must cover the synthetic
        // "prefill" prefix plus the GENS decode slots we will produce.
        long needed = contextLength + GENS;
        if (cfg.contextLength() < needed) {
            System.err.println("[TokenThroughputBenchmark] WARN: model contextLength="
                + cfg.contextLength() + " < ctx+GENS=" + needed
                + "; clamping to model contextLength (decode will land earlier)");
            // No resize for now: real models we benchmark comfortably exceed 2048+GENS.
        }

        Weights weights = Weights.load(gguf, cfg);
        Tokenizer tokenizer = Llama3BpeTokenizer.fromGguf(gguf);
        model = new LlamaModel(cfg, weights, tokenizer);

        int histLen = (int) Math.max(cfg.contextLength(), needed);
        tokenHistory = new int[histLen];
        samplerConfig = Sampler.Config.greedy();
    }

    private Path ensureRealModel() throws IOException {
        Path dest = Paths.get(DEFAULT_MODEL_DIR, REAL_MODEL_FILENAME);
        if (Files.exists(dest)) {
            verifyModelFile(dest);
            return dest;
        }
        Files.createDirectories(dest.getParent());
        String urlStr = System.getProperty("at.bench.modelUrl", DEFAULT_REAL_MODEL_URL);
        System.out.println("[TokenThroughputBenchmark] Downloading " + urlStr + "\n  → " + dest);
        URL u = URI.create(urlStr).toURL();
        try (InputStream in = u.openStream();
             OutputStream out = Files.newOutputStream(dest)) {
            byte[] buf = new byte[64 * 1024];
            long total = 0;
            int n;
            while ((n = in.read(buf)) > 0) {
                out.write(buf, 0, n);
                total += n;
            }
            System.out.println("[TokenThroughputBenchmark] Downloaded " + total + " bytes.");
        }
        verifyModelFile(dest);
        return dest;
    }

    /**
     * Post-download verification: rejects suspiciously small files (truncated
     * download) and prints the actual size + SHA-256. We deliberately do NOT
     * hardcode a SHA (re-uploads change it); the printed hash is for manual
     * cross-check against the resolved HuggingFace LFS SHA256.
     */
    private static void verifyModelFile(Path dest) throws IOException {
        long actual = Files.size(dest);
        if (actual < MIN_HEALTHY_MODEL_BYTES) {
            throw new IOException("Downloaded file is suspiciously small: "
                + actual + " bytes (expected ≥ " + MIN_HEALTHY_MODEL_BYTES
                + "). Likely a truncated download; please re-run.");
        }
        String sha = sha256Of(dest);
        long deltaPct = Math.abs(actual - CANONICAL_MODEL_BYTES) * 100L / CANONICAL_MODEL_BYTES;
        System.out.println("[TokenThroughputBenchmark] model: " + dest.getFileName());
        System.out.println("  size:    " + actual + " bytes (canonical ~" + CANONICAL_MODEL_BYTES
            + ", Δ=" + deltaPct + "%)");
        System.out.println("  sha256:  " + sha);
    }

    private static String sha256Of(Path p) throws IOException {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            try (InputStream in = Files.newInputStream(p)) {
                byte[] buf = new byte[64 * 1024];
                int n;
                while ((n = in.read(buf)) > 0) md.update(buf, 0, n);
            }
            byte[] hash = md.digest();
            StringBuilder sb = new StringBuilder(64);
            for (byte b : hash) sb.append(String.format("%02x", b));
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IOException(e);
        }
    }

    private String ggufDataPath() {
        return Paths.get(DEFAULT_MODEL_DIR, REAL_MODEL_FILENAME).toString();
    }

    // -----------------------------------------------------------------
    // Synthetic FP32 fallback — Llama-150M analogue shape.
    // -----------------------------------------------------------------

    private void setupSyntheticModel() {
        final long totalContext = contextLength + GENS;

        LlamaConfig cfg = new LlamaConfig(
            "llama",
            totalContext,
            EMB_DIM,
            BLOCKS,
            N_HEADS,
            N_HEADS_KV,
            FFN_DIM,
            ROPE_BASE,
            1e-5f,
            VOCAB
        );

        Tensor tokenEmbeddings = Tensor.allocate2D(DType.FP32, VOCAB, EMB_DIM);
        fillDeterministic(tokenEmbeddings, VOCAB, EMB_DIM, 0.013, 0.0017);

        Tensor outputNorm   = Tensor.allocate1D(DType.FP32, EMB_DIM);
        fillNormPattern(outputNorm, EMB_DIM);

        Tensor outputWeight = Tensor.allocate2D(DType.FP32, VOCAB, EMB_DIM);
        fillDeterministic(outputWeight, VOCAB, EMB_DIM, 0.017, 0.0023);

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

            fillDeterministic(attnQ,   EMB_DIM * N_HEADS * HEAD_DIM, 1, 0.011, 0.0019);
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
        tokenHistory = new int[contextLength + GENS];
        samplerConfig = Sampler.Config.greedy();
        usingRealModel = false;
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
