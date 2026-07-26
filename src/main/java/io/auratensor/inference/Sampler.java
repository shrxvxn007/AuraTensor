package io.auratensor.inference;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Token-sampling strategies for autoregressive generation.
 *
 * <p>Supports any combination of: temperature scaling, top-k truncation,
 * top-p (nucleus) truncation, and repetition penalty.
 */
public final class Sampler {

    private Sampler() {}

    public record Config(
        float temperature,
        int topK,
        float topP,
        float repetitionPenalty,
        long seed
    ) {
        public static Config greedy() {
            return new Config(1.0f, 1, 1.0f, 1.0f, 0);
        }
    }

    /**
     * Pick the next token id from raw logits (not softmaxed).
     *
     * @param logits      pre-softmax logits of shape [vocabSize]
     * @param history     previously emitted token ids for repetition penalty
     * @param config      sampling parameters
     */
    public static int sample(float[] logits, int[] history, Config config) {
        int vocab = logits.length;

        // 1) Repetition penalty — multiply previously-seen token logits by 1/p
        //    (or p if negative).
        if (config.repetitionPenalty() != 1.0f && history != null && history.length > 0) {
            float penalty = config.repetitionPenalty();
            for (int tok : history) {
                if (tok >= 0 && tok < vocab) {
                    if (logits[tok] > 0) logits[tok] /= penalty;
                    else                 if (logits[tok] > 0) logits[tok] /= penalty;
                else                 logits[tok] *= penalty;
                }
            }
        }

        // 2) Greedy fast path
        if (config.temperature() == 0.0f
            || (config.topK() == 1 && config.topP() >= 1.0f)) {
            int best = 0;
            float bestScore = logits[0];
            for (int i = 1; i < vocab; i++) {
                if (logits[i] > bestScore) {
                    bestScore = logits[i];
                    best = i;
                }
            }
            return best;
        }

        // 3) Temperature scaling + numerically-stable softmax over the
        //    top-k / top-p filtered candidates.
        float invT = 1.0f / config.temperature();
        List<int[]> idx = new ArrayList<>(vocab);
        for (int i = 0; i < vocab; i++) idx.add(new int[]{ i });

        float[] adj = new float[vocab];
        for (int i = 0; i < vocab; i++) adj[i] = logits[i] * invT;

        idx.sort(Comparator.comparingDouble((int[] a) -> adj[a[0]]).reversed());

        int topK = Math.min(config.topK(), vocab);
        float maxLogit = adj[idx.get(0)[0]];
        float sumExp = 0.0f;
        float[] probs = new float[topK];

        for (int i = 0; i < topK; i++) {
            int tok = idx.get(i)[0];
            float p = (float) Math.exp(adj[tok] - maxLogit);
            probs[i] = p;
            sumExp += p;
        }

        // Top-p (nucleus) cutoff.
        int cutoff = topK;
        if (config.topP() < 1.0f) {
            float running = 0.0f;
            float total = sumExp;
            for (int i = 0; i < topK; i++) {
                running += probs[i];
                if (running / total >= config.topP()) {
                    cutoff = i + 1;
                    break;
                }
            }
        }
        // Renormalize cutoff region.
        float newSum = 0.0f;
        for (int i = 0; i < cutoff; i++) newSum += probs[i];
        for (int i = 0; i < cutoff; i++) probs[i] /= newSum;

        // Sample from distribution. JDK 22 removed RandomGenerator.factory();
        // we use a seeded java.util.Random instead — it implements
        // RandomGenerator.
        RandomGenerator rng = config.seed() == 0
            ? RandomGenerator.getDefault()
            : new java.util.Random(config.seed());
        float r = rng.nextFloat();
        float cum = 0.0f;
        for (int i = 0; i < cutoff; i++) {
            cum += probs[i];
            if (r <= cum) return idx.get(i)[0];
        }
        return idx.get(cutoff - 1)[0];
    }
}
