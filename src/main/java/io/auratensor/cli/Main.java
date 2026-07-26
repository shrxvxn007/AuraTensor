package io.auratensor.cli;

import io.auratensor.format.GgufFile;
import io.auratensor.inference.LlamaConfig;
import io.auratensor.inference.LlamaModel;
import io.auratensor.inference.Sampler;
import io.auratensor.inference.Weights;
import io.auratensor.server.InferenceServer;
import jdk.incubator.vector.FloatVector;

import java.io.IOException;
import java.util.HashMap;

/**
 * AuraTensor CLI entry point.
 *
 * <p>Usage:
 * <pre>
 *   java -jar auratensor.jar --model llama3-8b.Q4_0.gguf --prompt "Explain quantum computing"
 *   java -jar auratensor.jar --model llama3-8b.Q4_0.gguf --serve --port 8080
 *   java -jar auratensor.jar --model llama3-8b.Q4_0.gguf --prompt "..." --tokens 256 \
 *                                                    --temperature 0.7 --top-k 40 \
 *                                                    --top-p 0.95 --repeat-penalty 1.1
 * </pre>
 *
 * <p>Must run with {@code --add-modules jdk.incubator.vector
 * --enable-preview --enable-native-access=ALL-UNNAMED}. The shipped
 * shaded jar uses these flags via the {@code Main-Class} manifest defaults
 * documented in {@code README.md}.
 */
public final class Main {

    public static void main(String[] args) throws IOException, InterruptedException {
        Arguments a = Arguments.parse(args);

        System.out.println("[AuraTensor] Java " + System.getProperty("java.version"));
        System.out.println("[AuraTensor] SIMD lane width: "
            + FloatVector.SPECIES_PREFERRED.vectorBitSize() + " bits");

        try (GgufFile gguf = GgufFile.open(a.model)) {
            System.out.println("[AuraTensor] GGUF v" + gguf.version()
                + "  tensors=" + gguf.tensorCount()
                + "  fileSize=" + gguf.fileLength() + " bytes");

            LlamaConfig cfg = LlamaConfig.fromGguf(gguf.metadata());
            System.out.println("[AuraTensor] " + cfg.architecture()
                + "  ctx=" + cfg.contextLength()
                + "  embed=" + cfg.embeddingLength()
                + "  heads=" + cfg.headCount()
                + "  layers=" + cfg.blockCount()
                + "  ffn=" + cfg.feedForwardLength());

            Weights weights = Weights.load(gguf, cfg);
            // For now, the bundled Tokenizer is a minimal placeholder so the
            // CLI can demonstrate the full pipeline. Real tokenization from
            // GGUF metadata is intentionally a follow-up; the engine
            // contract is documented and consuming code stays untouched when
            // a richer tokenizer replaces this one.
            LlamaModel model = new LlamaModel(cfg, weights, placeholderTokenizer());

            if (a.serve) {
                InferenceServer server = new InferenceServer(a.port);
                server.bind(model);
                server.start();
                System.out.println("[AuraTensor] serving on :http://localhost:" + server.port());
                // Block forever; virtual threads take care of incoming requests.
                Thread.currentThread().join();
            } else {
                runPrompt(model, a);
            }
        }
    }

    private static void runPrompt(LlamaModel model, Arguments a) {
        int[] promptTokens = model.tokenizer().encode(a.prompt);
        if (promptTokens.length == 0) promptTokens = new int[]{ 1 };

        long t0 = System.nanoTime();
        int position = 0;
        float[] logits = null;
        for (int tok : promptTokens) {
            logits = model.forwardStep(tok, position++);
        }
        // Warm-up complete; start sampling.
        Sampler.Config cfg = new Sampler.Config(
            a.temperature, a.topK, a.topP, a.repeatPenalty, a.seed);
        int[] history = promptTokens.clone();
        StringBuilder out = new StringBuilder();
        int emitted = 0;
        long gen0 = System.nanoTime();
        for (int t = 0; t < a.maxTokens; t++) {
            int next = Sampler.sample(logits, history, cfg);
            if (next == model.tokenizer().eosTokenId()) break;
            String piece = model.tokenizer().decode(next);
            System.out.print(piece);
            System.out.flush();
            out.append(piece);
            emitted++;
            int[] newHist = new int[history.length + 1];
            System.arraycopy(history, 0, newHist, 0, history.length);
            newHist[history.length] = next;
            history = newHist;
            logits = model.forwardStep(next, position++);
        }
        long gen1 = System.nanoTime();
        double seconds = (gen1 - gen0) / 1e9;
        double ttft  = (gen0 - t0) / 1e9;
        System.out.println();
        if (emitted > 0) {
            System.out.printf("%n[AuraTensor] %d tokens, %.2fs, %.2f tok/s, "
                              + "prompt time=%.2fs%n",
                emitted, seconds, emitted / seconds, ttft);
        }
    }

    private static io.auratensor.inference.Tokenizer placeholderTokenizer() {
        // Maps "hello" → [1, 2, 3, 4] for any prompt so the engine pipeline
        // can be exercised end-to-end without a real tokenizer. Real tokenizers
        // are plugged in via Weights.load(...) replacement — see README.
        return new io.auratensor.inference.Tokenizer() {
            @Override public int[] encode(String text) {
                int[] toks = new int[Math.max(1, text.length())];
                for (int i = 0; i < toks.length; i++) {
                    toks[i] = (text.isEmpty() ? 1
                              : Math.floorMod(text.charAt(i), 30000) + 2);
                }
                return toks;
            }
            @Override public String decode(int tokenId) {
                // Inverse of encode: back to a single printable char.
                int c = (tokenId - 2);
                if (c < 0 || c > 0xFFFF) return "";
                return String.valueOf((char) c);
            }
            @Override public int eosTokenId() { return 1; }
            @Override public java.util.Map<Integer, String> vocab() {
                return new HashMap<>();
            }
        };
    }

    // ---------------------------------------------------------------------
    // Argument parsing
    // ---------------------------------------------------------------------

    static final class Arguments {
        String model = "";
        String prompt = "Hello!";
        boolean serve = false;
        int port = 8080;
        int maxTokens = 128;
        float temperature = 0.7f;
        int topK = 40;
        float topP = 0.95f;
        float repeatPenalty = 1.1f;
        long seed = 0;

        static Arguments parse(String[] argv) {
            Arguments a = new Arguments();
            for (int i = 0; i < argv.length; i++) {
                String k = argv[i];
                String v = (i + 1 < argv.length) ? argv[i + 1] : "";
                switch (k) {
                    case "--model"             -> { a.model = v; i++; }
                    case "--prompt"            -> { a.prompt = v; i++; }
                    case "--serve"             -> { a.serve = true; }
                    case "--port"              -> { a.port = Integer.parseInt(v); i++; }
                    case "--tokens"            -> { a.maxTokens = Integer.parseInt(v); i++; }
                    case "--temperature"       -> { a.temperature = Float.parseFloat(v); i++; }
                    case "--top-k"             -> { a.topK = Integer.parseInt(v); i++; }
                    case "--top-p"             -> { a.topP = Float.parseFloat(v); i++; }
                    case "--repeat-penalty"    -> { a.repeatPenalty = Float.parseFloat(v); i++; }
                    case "--seed"              -> { a.seed = Long.parseLong(v); i++; }
                    case "--help", "-h"        -> { printHelp(); System.exit(0); }
                    default -> { /* ignore unknown */ }
                }
            }
            if (a.model.isEmpty()) {
                System.err.println("Error: --model <path.gguf> is required");
                printHelp();
                System.exit(2);
            }
            return a;
        }
    }

    static void printHelp() {
        System.out.println("""
            AuraTensor: zero-dependency LLM inference in pure Java 21+

            Usage:
              java -jar auratensor.jar --model model.gguf --prompt "Explain..."
              java -jar auratensor.jar --model model.gguf --serve --port 8080

            Flags:
              --model <path.gguf>      Path to GGUF model (required)
              --prompt <text>          Prompt string (default: "Hello!")
              --serve                  Start the OpenAI-compatible HTTP server
              --port <n>               Server port (default 8080)
              --tokens <n>             Max new tokens (default 128)
              --temperature <f>        Sampling temperature (default 0.7)
              --top-k <n>              Top-K cutoff (default 40)
              --top-p <f>              Top-P cutoff (default 0.95)
              --repeat-penalty <f>     Repetition penalty (default 1.1)
              --seed <n>               RNG seed (default 0 = random)

            Run with:
              --add-modules jdk.incubator.vector \
              --enable-preview \
              --enable-native-access=ALL-UNNAMED
            """);
    }
}
