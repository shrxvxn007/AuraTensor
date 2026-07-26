package io.auratensor.core;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.FloatBuffer;
import java.util.Objects;

/**
 * An off-heap tensor backed by a Java 21 {@link MemorySegment}.
 *
 * <p>The Tensor abstraction supports 1D, 2D, and 3D layouts with explicit
 * stride calculations. All element access goes through {@code MemorySegment}
 * load/store intrinsics so the JVM can keep the data in native memory and
 * avoid GC pressure.
 *
 * <p>Tensors are released deterministically by closing the owning
 * {@link Arena}. For weights that should outlive a single request, use
 * {@link Arena#ofShared()} or hold a strong reference to the tensor and let
 * its {@code close()} method handle release.
 */
public final class Tensor implements AutoCloseable {

    /** Underlying off-heap memory. */
    private final MemorySegment data;

    /** Owning arena; closed by {@link #close()}. */
    private final Arena arena;

    /** Element data type. */
    private final DType dtype;

    /** Shape array. Length = rank (1, 2, or 3). */
    private final int[] shape;

    /** Row-major strides (in elements). {@code strides[i] = product(shape[i+1..])}. */
    private final long[] strides;

    /** Total number of elements. */
    private final long numElements;

    /** Cached once: true if the layout is contiguous (row-major) which means
     *  element i is exactly at byte offset i*dtype.bytesPerElement. */
    private final boolean contiguous;

    private Tensor(MemorySegment data,
                   Arena arena,
                   DType dtype,
                   int[] shape) {
        if (shape.length < 1 || shape.length > 3) {
            throw new IllegalArgumentException(
                "AuraTensor supports rank 1, 2, or 3 tensors (got rank " + shape.length + ")");
        }
        for (int s : shape) {
            if (s <= 0) {
                throw new IllegalArgumentException("All dimensions must be positive, got " + s);
            }
        }
        this.data = Objects.requireNonNull(data, "data");
        this.arena = Objects.requireNonNull(arena, "arena");
        this.dtype = Objects.requireNonNull(dtype, "dtype");
        this.shape = shape.clone();
        this.numElements = computeNumElements(shape);
        this.strides = computeStrides(shape);
        this.contiguous = true;  // We always allocate row-major contiguous.
    }

    // ---------------------------------------------------------------------
    // Factory: allocate
    // ---------------------------------------------------------------------

    /** Allocates a 1D contiguous tensor of {@code n} FP32 elements. */
    public static Tensor allocate1D(DType dtype, int n) {
        return allocate(dtype, n);
    }

    /** Allocates a 2D contiguous tensor with shape {@code [rows, cols]} of FP32 elements. */
    public static Tensor allocate2D(DType dtype, int rows, int cols) {
        return allocate(dtype, rows, cols);
    }

    /** Allocates a 3D contiguous tensor with shape {@code [d0, d1, d2]} of FP32 elements. */
    public static Tensor allocate3D(DType dtype, int d0, int d1, int d2) {
        return allocate(dtype, d0, d1, d2);
    }

    /** Allocates a contiguous tensor of arbitrary rank 1..3. */
    public static Tensor allocate(DType dtype, int... shape) {
        long n = computeNumElements(shape);
        long bytes = n * dtype.bytesPerElement();
        Arena arena = Arena.ofConfined();
        MemorySegment seg = arena.allocate(bytes, 16);  // 16-byte aligned
        return new Tensor(seg, arena, dtype, shape);
    }

    // ---------------------------------------------------------------------
    // Factory: from existing segment (zero-copy wrap)
    // ---------------------------------------------------------------------

    /**
     * Wraps an externally-allocated segment as a tensor without copying.
     * The returned Tensor takes ownership of {@code arena} for cleanup.
     */
    public static Tensor wrap(MemorySegment data, Arena arena, DType dtype, int[] shape) {
        long expected = computeNumElements(shape) * dtype.bytesPerElement();
        if (data.byteSize() < expected) {
            throw new IllegalArgumentException(
                "Segment too small: " + data.byteSize() + " < " + expected);
        }
        return new Tensor(data, arena, dtype, shape);
    }

    /**
     * Wraps an externally-allocated segment as a tensor using a confined arena
     * that owns the segment. Useful for {@code FileChannel.map(...)} results.
     */
    public static Tensor wrapMapped(MemorySegment data, DType dtype, int[] shape) {
        Arena closingArena = Arena.ofConfined();
        return new Tensor(data, closingArena, dtype, shape);
    }

    // ---------------------------------------------------------------------
    // Accessors
    // ---------------------------------------------------------------------

    public MemorySegment data() { return data; }
    public DType dtype() { return dtype; }
    public int[] shape() { return shape.clone(); }
    public long[] strides() { return strides.clone(); }
    public long numElements() { return numElements; }
    public boolean contiguous() { return contiguous; }
    public int rank() { return shape.length; }

    public long elementOffset(int... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException(
                "Index rank mismatch: tensor rank=" + shape.length
                + ", got " + indices.length);
        }
        long off = 0;
        for (int i = 0; i < indices.length; i++) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException(
                    "Index " + i + " out of range " + indices[i] + " vs dim " + shape[i]);
            }
            off += (long) indices[i] * strides[i];
        }
        return off;
    }

    // ---------------------------------------------------------------------
    // Typed element access (FP32 only at this layer; quantized types handle
    // their own element semantics in the quant package).
    // ---------------------------------------------------------------------

    public float getFloat(int... indices) {
        if (dtype != DType.FP32) {
            throw new UnsupportedOperationException(
                "Tensor.dtype=" + dtype + " does not support direct float access");
        }
        long off = elementOffset(indices) * 4L;
        return data.get(ValueLayout.JAVA_FLOAT, off);
    }

    public void setFloat(float v, int... indices) {
        if (dtype != DType.FP32) {
            throw new UnsupportedOperationException(
                "Tensor.dtype=" + dtype + " does not support direct float access");
        }
        long off = elementOffset(indices) * 4L;
        data.set(ValueLayout.JAVA_FLOAT, off, v);
    }

    /**
     * Copies the tensor's float32 view into a heap {@link FloatBuffer}.
     *
     * <p>JDK 22+ no longer provides a direct {@code MemorySegment.copy}
     * overload into a {@link FloatBuffer}, so we go through a heap float[]
     * backed by {@link MemorySegment#toArray(ValueLayout)}.
     */
    public FloatBuffer toFloatBuffer() {
        if (dtype != DType.FP32) {
            throw new UnsupportedOperationException("toFloatBuffer only valid for FP32 tensors");
        }
        float[] arr = data.toArray(ValueLayout.JAVA_FLOAT);
        return FloatBuffer.wrap(arr);
    }

    /**
     * Bulk FP32 write into the tensor from a heap-backed {@link FloatBuffer}.
     *
     * <p>JDK 22+ no longer provides a direct {@code MemorySegment.copy}
     * overload from a {@link FloatBuffer}, so we drain into a heap float[]
     * and copy via a heap-allocated {@link MemorySegment}.
     */
    public void writeFrom(FloatBuffer src) {
        if (dtype != DType.FP32) {
            throw new UnsupportedOperationException("writeFrom only valid for FP32 tensors");
        }
        if (src.remaining() != numElements) {
            throw new IllegalArgumentException(
                "Source size " + src.remaining() + " != tensor numElements " + numElements);
        }
        float[] arr = new float[(int) numElements];
        src.get(arr);
        MemorySegment.copy(
            MemorySegment.ofArray(arr), 0L,
            data, 0L, (long) numElements * 4L);
    }

    /**
     * Copy all elements from {@code src} into {@code this} tensor (element-wise).
     * Both tensors must be FP32 and have the same number of elements.
     * Safe when {@code src == this} (no-op self-copy, useful as a test fixture).
     */
    public void copyFromOther(Tensor src) {
        if (src == this) return;
        if (this.dtype != DType.FP32 || src.dtype != DType.FP32) {
            throw new IllegalArgumentException("copyFromOther requires FP32 tensors");
        }
        if (this.numElements != src.numElements) {
            throw new IllegalArgumentException(
                "copyFromOther: size mismatch dst=" + this.numElements
                + " vs src=" + src.numElements);
        }
        long n = this.numElements;
        for (long i = 0; i < n; i++) {
            long off = i * 4L;
            data.set(ValueLayout.JAVA_FLOAT, off,
                     src.data.get(ValueLayout.JAVA_FLOAT, off));
        }
    }

    @Override
    public void close() {
        if (arena.scope().isAlive()) {
            arena.close();
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Tensor{");
        sb.append("dtype=").append(dtype)
          .append(", shape=[");
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(shape[i]);
        }
        sb.append("], strides=[");
        for (int i = 0; i < strides.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(strides[i]);
        }
        sb.append("], bytes=").append(data.byteSize())
          .append('}');
        return sb.toString();
    }

    // ---------------------------------------------------------------------
    // Internals
    // ---------------------------------------------------------------------

    private static long computeNumElements(int[] shape) {
        long n = 1L;
        for (int s : shape) n *= s;
        return n;
    }

    private static long[] computeStrides(int[] shape) {
        long[] s = new long[shape.length];
        long running = 1L;
        for (int i = shape.length - 1; i >= 0; i--) {
            s[i] = running;
            running *= shape[i];
        }
        return s;
    }
}
