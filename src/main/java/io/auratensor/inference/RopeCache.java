package io.auratensor.inference;

import io.auratensor.core.Kernels;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Precomputed Rotary Position Embedding frequency tables.
 *
 * <p>Layout: for each position p and each pair index i in {@code [0, headDim/2)},
 * we duplicate the value across both elements of the pair, so the cache is laid
 * out as {@code [c_0, c_0, c_1, c_1, ...]} for length {@code headDim}. This
 * matches the dense stride-1 access pattern the SIMD {@link Kernels#ropeInPlace}
 * expects (lane i of the QK vector pairs with cos[i]).
 *
 * <p>Llama 3 uses base = 500_000; Llama 2 and Mistral use base = 10_000.
 *
 * <p>The full cos/sin tables live in a single shared {@link Arena} and are
 * exposed as flat {@link MemorySegment}s via {@link #cosSegment()} /
 * {@link #sinSegment()} (full tables) and {@link #cosSegmentFor(long, long)} /
 * {@link #sinSegmentFor(long, long)} (per-position slices via
 * {@link MemorySegment#asSlice}). This lets the inner forward loop
 * ({@link LlamaModel#layerStep}) call {@link Kernels#ropeInPlaceSegment}
 * directly on flat segments without any per-step {@code Arena.ofConfined} or
 * {@code Tensor.wrap} allocation churn.
 *
 * <p>Note: there is no {@code close()} method on this cache. The lifetime of
 * the underlying shared arena is bound to the {@code RopeCache} instance.
 * Callers must hold a reference for as long as the model is alive.
 */
public final class RopeCache {

    /** Single shared Arena owns the lifetime of both cos and sin tables. */
    private final Arena arena;
    /** Cosine table — flat {@code [maxContext * headDim]} FP32. */
    private final MemorySegment cosSeg;
    /** Sine   table — flat {@code [maxContext * headDim]} FP32. */
    private final MemorySegment sinSeg;

    public RopeCache(long headDim, float base, long maxContext) {
        long half = headDim / 2;
        long strideElems = headDim;
        // One row per position, length = headDim (cos duplicated per element).
        long n = maxContext * strideElems;
        arena = Arena.ofShared();
        cosSeg = arena.allocate(n * 4L, 16);
        sinSeg = arena.allocate(n * 4L, 16);
        long rowBytes = strideElems * 4L;
        for (long pos = 0; pos < maxContext; pos++) {
            long rowBase = pos * rowBytes;
            for (long i = 0; i < half; i++) {
                float freq = (float) Math.pow(base, -2.0 * i / (double) headDim);
                float angle = pos * freq;
                float c = (float) Math.cos(angle);
                float s = (float) Math.sin(angle);
                // Duplicate per element of the pair: lanes (2i, 2i+1) both see (c, s).
                cosSeg.set(ValueLayout.JAVA_FLOAT, rowBase + (2L * i) * 4L, c);
                cosSeg.set(ValueLayout.JAVA_FLOAT, rowBase + (2L * i + 1L) * 4L, c);
                sinSeg.set(ValueLayout.JAVA_FLOAT, rowBase + (2L * i) * 4L, s);
                sinSeg.set(ValueLayout.JAVA_FLOAT, rowBase + (2L * i + 1L) * 4L, s);
            }
        }
    }

    /** Full cosine table as a flat {@link MemorySegment} (lifetime owned by this cache). */
    public MemorySegment cosSegment() { return cosSeg; }

    /** Full sine table as a flat {@link MemorySegment} (lifetime owned by this cache). */
    public MemorySegment sinSegment() { return sinSeg; }

    /**
     * Per-position slice of the cosine table. Returns a zero-copy
     * {@link MemorySegment#asSlice} view of length {@code headDim} floats
     * &mdash; no per-call Arena allocation, no Tensor wrap.
     */
    public MemorySegment cosSegmentFor(long pos, long headDim) {
        return cosSeg.asSlice(pos * headDim * 4L, headDim * 4L);
    }

    /**
     * Per-position slice of the sine table. Returns a zero-copy
     * {@link MemorySegment#asSlice} view of length {@code headDim} floats
     * &mdash; no per-call Arena allocation, no Tensor wrap.
     */
    public MemorySegment sinSegmentFor(long pos, long headDim) {
        return sinSeg.asSlice(pos * headDim * 4L, headDim * 4L);
    }
}
