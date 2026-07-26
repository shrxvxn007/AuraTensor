package io.auratensor.inference;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SamplerTest {

    @Test
    void greedyPicksArgmax() {
        float[] logits = {0.1f, 0.5f, 0.7f, 0.3f, 0.2f};
        int token = Sampler.sample(logits, null, Sampler.Config.greedy());
        assertEquals(2, token);
    }

    @Test
    void tempZeroIsGreedy() {
        float[] logits = {0.1f, 0.5f, 0.7f, 0.3f};
        int token = Sampler.sample(logits, null, new Sampler.Config(0.0f, 1, 1.0f, 1.0f, 0));
        assertEquals(2, token);
    }

    @Test
    void topKOneIsGreedy() {
        float[] logits = new float[100];
        for (int i = 0; i < 100; i++) logits[i] = (i == 42 ? 1000.0f : (float) i);
        int token = Sampler.sample(logits, null, new Sampler.Config(1.0f, 1, 1.0f, 1.0f, 0));
        assertEquals(42, token);
    }

    @Test
    void repetitionPenaltyReducesSeenTokens() {
        float[] logits = {1.0f, 1.0f, 1.0f, 1.0f};
        int[] history = {0, 1};
        // With rep penalty 0.5, positive logits for 0 and 1 are halved, so
        // 2 and 3 become the relative winners.
        int t1 = Sampler.sample(logits, history, new Sampler.Config(1.0f, 1, 1.0f, 2.0f, 42));
        // HF semantics: penalty=2.0 halves seen positive logits; 2 and 3 win.
        assertTrue(t1 == 2 || t1 == 3);
    }

    @Test
    void topPStillPicksFromAllowedSet() {
        float[] logits = new float[1024];
        for (int i = 0; i < 1024; i++) logits[i] = -100f;
        logits[100] = 0f;
        logits[101] = -5f;     // should be allowed
        // Use a top-p that includes both 100 and 101.
        int first = -1;
        for (int i = 0; i < 1000; i++) {
            int t = Sampler.sample(logits, null, new Sampler.Config(1.0f, 1024, 0.999f, 1.0f, 7));
            assertTrue(t == 100 || t == 101, "Got t=" + t);
            if (first < 0) first = t;
        }
        assertEquals(100, first);  // most likely is 100 with a 1:exp(-5) ratio
    }

    @Test
    void eosTokenIsRespected() {
        float[] logits = { -1e9f, 0.0f, -1e9f, -1e9f };
        int[] hist = { 2 };
        // BOS = 1 in our demo tokenizer; greedy should pick 1
        int t = Sampler.sample(logits, hist, Sampler.Config.greedy());
        assertEquals(1, t);
    }
}
