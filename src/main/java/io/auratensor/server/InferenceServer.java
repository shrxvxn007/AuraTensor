package io.auratensor.server;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import io.auratensor.inference.LlamaModel;
import io.auratensor.inference.Sampler;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Lightweight HTTP server backed by Java 21 virtual threads.
 *
 * <p>Endpoints:
 * <ul>
 *   <li>{@code GET  /health}                  → JSON status</li>
 *   <li>{@code POST /v1/chat/completions}     → OpenAI-compatible (non-stream + SSE)</li>
 *   <li>{@code POST /v1/completions}          → Legacy prompt completion</li>
 * </ul>
 *
 * <p>Streaming responses use Server-Sent Events ({@code text/event-stream}).
 */
public final class InferenceServer {

    private final HttpServer server;
    private final AtomicReference<LlamaModel> modelRef = new AtomicReference<>();

    public InferenceServer(int port) throws IOException {
        this.server = HttpServer.create(new InetSocketAddress(port), 0);
        // Java 21 virtual threads: each request runs on its own virtual thread.
        this.server.setExecutor(Executors.newVirtualThreadPerTaskExecutor());

        server.createContext("/health", this::handleHealth);
        server.createContext("/v1/chat/completions", this::handleChatCompletions);
        server.createContext("/v1/completions",      this::handleChatCompletions);
    }

    public void bind(LlamaModel model) {
        modelRef.set(model);
    }

    public void start() { server.start(); }

    public int port() { return server.getAddress().getPort(); }

    public void stop() { server.stop(0); }

    // ---------------------------------------------------------------------
    // Handlers
    // ---------------------------------------------------------------------

    private void handleHealth(HttpExchange ex) throws IOException {
        addCors(ex);
        if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
            ex.sendResponseHeaders(204, -1);
            return;
        }
        if (!"GET".equalsIgnoreCase(ex.getRequestMethod())) {
            sendJson(ex, 405, Json.obj("status", Json.quote("error"),
                                       "error",  Json.quote("method not allowed")));
            return;
        }
        LlamaModel m = modelRef.get();
        String status = (m == null) ? "loading" : "ready";
        sendJson(ex, 200, Json.obj("status", Json.quote(status)));
    }

    private void handleChatCompletions(HttpExchange ex) throws IOException {
        addCors(ex);
        if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
            ex.sendResponseHeaders(204, -1);
            return;
        }
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
            sendJson(ex, 405, Json.obj("error", Json.quote("method not allowed")));
            return;
        }
        LlamaModel m = modelRef.get();
        if (m == null) {
            sendJson(ex, 503, Json.obj("error", Json.quote("model not loaded")));
            return;
        }

        String body = RequestParser.readAll(ex.getRequestBody());
        InferenceRequest req = RequestParser.parse(body);

        if (req.stream()) {
            ex.getResponseHeaders().add("Content-Type", "text/event-stream");
            ex.getResponseHeaders().add("Cache-Control", "no-cache");
            ex.sendResponseHeaders(200, 0);
            try (OutputStream os = ex.getResponseBody()) {
                SseFrame sse = new SseFrame(os);
                streamGenerate(m, req, sse);
            }
        } else {
            ex.getResponseHeaders().add("Content-Type", "application/json");
            String resp = blockingGenerate(m, req);
            sendJson(ex, 200, resp);
        }
    }

    /**
     * Synchronous token-by-token generation, returning a complete OpenAI-shaped
     * JSON response body. The caller is responsible for writing it to the
     * response; sending it through a dedicated JSON helper keeps the contract
     * uniform with {@link #handleHealth}.
     */
    private String blockingGenerate(LlamaModel m, InferenceRequest req) {
        int[] promptTokens = m.tokenizer().encode(req.prompt());
        if (promptTokens.length == 0) promptTokens = new int[]{ 1 };  // BOS fallback

        int position = 0;
        float[] logits = null;
        for (int i = 0; i < promptTokens.length; i++) {
            logits = m.forwardStep(promptTokens[i], i);
            position++;
        }
        StringBuilder all = new StringBuilder();
        Sampler.Config cfg = samplerConfig(req);
        int[] history = promptTokens.clone();
        for (int t = 0; t < req.maxTokens(); t++) {
            int next = Sampler.sample(logits, history, cfg);
            if (next == m.tokenizer().eosTokenId()) break;
            String piece = m.tokenizer().decode(next);
            all.append(piece);
            int[] newHist = new int[history.length + 1];
            System.arraycopy(history, 0, newHist, 0, history.length);
            newHist[history.length] = next;
            history = newHist;
            logits = m.forwardStep(next, position++);
        }
        return Json.obj(
            "id", Json.quote("auratensor-" + System.nanoTime()),
            "object", Json.quote("text_completion"),
            "model", Json.quote(req.model()),
            "choices", Json.strArray(all.toString())
        );
    }

    private void streamGenerate(LlamaModel m, InferenceRequest req, SseFrame sse) throws IOException {
        int[] promptTokens = m.tokenizer().encode(req.prompt());
        if (promptTokens.length == 0) promptTokens = new int[]{ 1 };

        int position = 0;
        float[] logits = null;
        for (int i = 0; i < promptTokens.length; i++) {
            logits = m.forwardStep(promptTokens[i], i);
            position++;
        }
        Sampler.Config cfg = samplerConfig(req);
        int[] history = promptTokens.clone();
        for (int t = 0; t < req.maxTokens(); t++) {
            int next = Sampler.sample(logits, history, cfg);
            if (next == m.tokenizer().eosTokenId()) break;
            String piece = m.tokenizer().decode(next);
            String chunk = Json.obj(
                "id", Json.quote("auratensor-" + System.nanoTime()),
                "object", Json.quote("text_completion"),
                "model", Json.quote(req.model()),
                "delta", Json.quote(piece)
            );
            sse.send(chunk);
            int[] newHist = new int[history.length + 1];
            System.arraycopy(history, 0, newHist, 0, history.length);
            newHist[history.length] = next;
            history = newHist;
            logits = m.forwardStep(next, position++);
        }
        sse.done();
    }

    private static Sampler.Config samplerConfig(InferenceRequest req) {
        return new Sampler.Config(
            req.temperature(),
            req.topK(),
            req.topP(),
            req.repetitionPenalty(),
            0
        );
    }

    /** Writes a JSON response with the given status code and body. */
    private static void sendJson(HttpExchange ex, int code, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().add("Content-Type", "application/json");
        ex.sendResponseHeaders(code, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    private static void addCors(HttpExchange ex) {
        var h = ex.getResponseHeaders();
        h.add("Access-Control-Allow-Origin",  "*");
        h.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        h.add("Access-Control-Allow-Headers", "Content-Type, Authorization");
    }
}
