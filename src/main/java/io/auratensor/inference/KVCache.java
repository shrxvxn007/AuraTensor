package io.auratensor.inference;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Off-heap KV-cache for autoregressive transformer inference.
 *
 * <p>For each transformer layer {@code L} and each KV-head {@code h}, the K
 * and V vectors at every position in the context are stored contiguously in
 * off-heap memory. Allocation is done once at model load and reused across
 * all subsequent token generations — no GC pressure during inference.
 *
 * <p>Byte layout (row-major):
 * <pre>
 *   rowBytes       =  maxContext * headDim * 4
 *   kvRowBytes     =  headCountKv * rowBytes
 *   keysOff(l, h, p, d) = (l * headCountKv + h) * rowBytes
 *                       + p * headDim * 4
 *                       + d * 4
 * </pre>
 * Exposed as a flat {@link MemorySegment} per side (& keys, & values); both
 * sides share the same shape so a single {@link #rowBytes()} accessor yields
 * the per-(layer, head) row stride for callers doing manual offset math.
 *
 * <p>The number of KV heads differs from query heads in grouped-query
 * attention (GQA) — see {@code LlamaConfig.headCountKv}.
 */
public final class KVCache {

    private final long blockCount;
    private final long headCountKv;
    private final long headDim;
    private final long maxContext;
    /** Per-(layer, head) row stride in bytes: maxContext * headDim * 4. */
    private final long rowBytes;

    private final Arena arena;
    private final MemorySegment keys;   // flat [blockCount, headCountKv, maxContext, headDim] → FP32
    private final MemorySegment values; // flat, same shape as keys
    private long size;                  // current number of tokens stored

    public KVCache(long blockCount, long headCountKv, long headDim, long maxContext) {
        if (blockCount <= 0 || headCountKv <= 0 || headDim <= 0 || maxContext <= 0) {
            throw new IllegalArgumentException(
                "KVCache dims must be positive: blocks=" + blockCount
                + " kvHeads=" + headCountKv + " headDim=" + headDim
                + " ctx=" + maxContext);
        }
        this.blockCount  = blockCount;
        this.headCountKv = headCountKv;
        this.headDim     = headDim;
        this.maxContext  = maxContext;
        this.rowBytes    = maxContext * headDim * 4L;

        long totalFloats = blockCount * headCountKv * maxContext * headDim;
        long bytes       = totalFloats * 4L;

        this.arena  = Arena.ofConfined();
        this.keys   = arena.allocate(bytes, 16);
        this.values = arena.allocate(bytes, 16);
        this.size   = 0;
    }

    /**
     * Byte offset of element {@code (layer, kvHead, position, dim)} inside
     * the shared layout. Single source of truth for callers doing manual
     * reads/writes against {@link #keys()} or {@link #values()}.
     */
    public long offsetOf(long layer, long kvHead, long position, long dim) {
        if (layer  < 0 || layer  >= blockCount)  throw new IllegalArgumentException("layer oob");
        if (kvHead < 0 || kvHead >= headCountKv) throw new IllegalArgumentException("kvHead oob");
        if (position < 0 || position >= maxContext) throw new IllegalArgumentException("position oob");
        if (dim >= headDim) throw new IllegalArgumentException("dim oob");
        return ((layer * headCountKv) + kvHead) * rowBytes + position * headDim * 4L + dim * 4L;
    }

    /** Appends one K/V slice per KV-head for the given layer at {@code position}. */
    public void append(long layer, long position, MemorySegment kSlice, MemorySegment vSlice) {
        if (layer >= blockCount) {
            throw new IllegalArgumentException("layer " + layer + " out of range " + blockCount);
        }
        if (position >= maxContext) {
            throw new IllegalStateException("position " + position + " exceeds maxContext " + maxContext);
        }
        long layerOff = layer * headCountKv * rowBytes;
        long posByte  = position * headDim * 4L;
        long headBytes = headDim * 4L;
        for (long h = 0; h < headCountKv; h++) {
            long dstBase = layerOff + h * rowBytes + posByte;
            MemorySegment.copy(kSlice, h * headBytes, keys,   dstBase, headBytes);
            MemorySegment.copy(vSlice, h * headBytes, values, dstBase, headBytes);
        }
        size = Math.max(size, position + 1);
    }

    public long size()                  { return size; }
    public long maxContext()            { return maxContext; }
    public MemorySegment keys()         { return keys; }
    public MemorySegment values()       { return values; }
    public long blockCount()            { return blockCount; }
    public long headCountKv()           { return headCountKv; }
    public long headDim()               { return headDim; }
    public long rowBytes()              { return rowBytes; }

    /** Close the off-heap arena holding K and V. Idempotent. */
    public void close() {
        if (arena.scope().isAlive()) {
            arena.close();
        }
    }
}
