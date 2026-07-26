package io.auratensor.server;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

/**
 * Tiny Server-Sent Events (SSE) frame writer.
 *
 * <p>The OpenAI streaming protocol is a series of {@code data:} lines plus
 * per-frame {@code event:} and {@code id:}; one logical message ends with
 * a literal blank line. SSE is plain text over HTTP/1.1, no chunked-encoding
 * tricks needed.
 */
public final class SseFrame {

    private final OutputStream out;

    public SseFrame(OutputStream out) { this.out = out; }

    public void send(String data) throws IOException {
        writeRaw("data: " + data + "\n\n");
    }

    public void sendEvent(String event, String data) throws IOException {
        writeRaw("event: " + event + "\n");
        writeRaw("data: " + data + "\n\n");
    }

    public void done() throws IOException {
        writeRaw("data: [DONE]\n\n");
    }

    private void writeRaw(String s) throws IOException {
        out.write(s.getBytes(StandardCharsets.UTF_8));
        out.flush();
    }
}
