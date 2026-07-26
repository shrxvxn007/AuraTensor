package io.auratensor.server;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class RequestParserTest {

    @Test
    void parsesMinimalBody() {
        String body = """
            {
              "model": "llama3-8b",
              "prompt": "Hello",
              "temperature": 0.4,
              "top_k": 50,
              "top_p": 0.9,
              "repetition_penalty": 1.05,
              "max_tokens": 256,
              "stream": false
            }
        """;
        InferenceRequest r = RequestParser.parse(body);
        assertEquals("llama3-8b", r.model());
        assertEquals("Hello", r.prompt());
        assertEquals(0.4f, r.temperature());
        assertEquals(50, r.topK());
        assertEquals(0.9f, r.topP());
        assertEquals(1.05f, r.repetitionPenalty());
        assertEquals(256, r.maxTokens());
        assertFalse(r.stream());
    }

    @Test
    void toleratesExtraFields() {
        String body = """
            {
              "messages": [{"role":"user","content":"Hi"}],
              "stream": true,
              "temperature": 1.1,
              "future_field_we_dont_know": 42
            }
        """;
        InferenceRequest r = RequestParser.parse(body);
        assertTrue(r.stream());
        assertEquals(1.1f, r.temperature());
        assertEquals("Hi", r.prompt());
    }

    @Test
    void defaultsToSensibleValuesWhenMissing() {
        InferenceRequest r = RequestParser.parse("{}");
        assertEquals(0.7f, r.temperature(), 1e-6f);
        assertEquals(40, r.topK());
        assertEquals(0.95f, r.topP(), 1e-6f);
    }
}
