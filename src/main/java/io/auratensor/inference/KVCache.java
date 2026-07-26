package io.auratensor.inference;

import io.auratensor.core.Tensor;
import io.auratensor.core.DType;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Off-heap KV-cache for autoregressive transformer inference.
 *
 * <p>For each transformer layer {@code L} and each head {@code h}, K and V at
 * every position are stored contiguously in off-heap memory. Allocation is
 * done once at model load and reused across all subsequent token generations
 * — no GC pressure during inference.
 *
 * <p>The number of KV heads differs from query heads in grouped-query
 * attention (GQA) — see {@code LlamaConfig.headCountKv}.
 */
public final class KVCache {

    private final long maxContext;
    private final long blockCount;
    private final long headCountKv;
    private final long headDim;

    private final Tensor keys;   // [blockCount, headCountKv, maxContext, headDim]
    private final Tensor values; // [blockCount, headCountKv, maxContext, headDim]
    private long size;           // current number of tokens stored

    public KVCache(long blockCount, long headCountKv, long headDim, long maxContext) {
        this.blockCount = blockCount;
        this.headCountKv = headCountKv;
        this.headDim = headDim;
        this.maxContext = maxContext;

        Arena arena = Arena.ofConfined();
        long totalFloats = blockCount * headCountKv * maxContext * headDim;
        MemorySegment kSeg = arena.allocate(totalFloats * 4L, 16);
        MemorySegment vSeg = arena.allocate(totalFloats * 4L, 16);
        int[] shape = new int[]{
            (int) blockCount, (int) headCountKv, (int) maxContext, (int) headDim
        };
        this.keys = Tensor.wrap(kSeg, arena, DType.FP32, shape);
        this.values = Tensor.wrap(vSeg, Arena.ofConfined(), DType.FP32, shape);
        this.size = 0;
    }

    /** Appends one K/V slice per head for the given layer at {@code position}. */
    public void append(long layer, long position, MemorySegment kSlice, MemorySegment vSlice) {
        if (layer >= blockCount) {
            throw new IllegalArgumentException("layer " + layer + " out of range " + blockCount);
        }
        if (position >= maxContext) {
            throw new IllegalStateException("position " + position + " exceeds maxContext " + maxContext);
        }
        long rowStride = maxContext * headDim * 4L;
        long layerOff = layer * headCountKv * maxContext * headDim * 4L;
        long headBytes = headDim * 4L;
        for (long h = 0; h < headCountKv; h++) {
            long headBase = layerOff + h * rowStride + position * headBytes;
            MemorySegment.copy(kSlice, h * headBytes, keys.data(), headBase, headBytes);
            MemorySegment.copy(vSlice, h * headBytes, values.data(), headBase, headBytes);
        }
        size = Math.max(size, position + 1);
    }

    public long size() { return size; }
    public long maxContext() { return maxContext; }
    public Tensor keys() { return keys; }
    public Tensor values() { return values; }
    public long blockCount() { return blockCount; }
    public long headCountKv() { return headCountKv; }
    public long headDim() { return headDim; }
}
