package io.auratensor.format;

/**
 * Header entry for a single tensor in a GGUF file.
 *
 * <p>{@code dims} uses GGUF ordering (outer to inner, e.g. for a Llama 3
 * weight {@code [out, in]} the dims are {@code [out, in]}). The byte {@code
 * offset} points into the data segment (region after the alignment padding
 * following the tensor-info section).
 */
public record GgufTensorInfo(
    String name,
    long[] dims,
    GgufTensorType type,
    long offset
) {
    public long numElements() {
        long n = 1L;
        for (long d : dims) n *= d;
        return n;
    }

    public int rank() {
        return dims.length;
    }

    public long byteSize() {
        long elems = numElements();
        // Block-quantized types: ceil(elems / 32) blocks, each of bytesPerBlock bytes.
        // Non-block types (F32, F16): per-element bytes.
        if (type.isBlockQuantized()) {
            return ((elems + 31L) / 32L) * type.bytesPerBlockOrElement;
        }
        return elems * type.bytesPerBlockOrElement;
    }
}
