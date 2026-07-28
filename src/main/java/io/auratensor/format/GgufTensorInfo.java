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
        // Block-quantized types: ceil(elems / blockElems) blocks, each of
        // bytesPerBlockOrElement bytes. The standard 32-element block size
        // applies to all k-quants that store per-32-element scale + nibble
        // packs (Q4_0 = 18 B, Q4_1 = 20 B, Q5_0 = 22 B, Q5_1 = 24 B,
        // Q8_0 = 34 B, Q8_1 = 36 B). Q6_K is the outlier: it stores 256
        // elements per super-block (ql + qh + scales + d = 210 B) and would
        // be over-counted by 8× if the 32-element formula were applied.
        // Non-block types (F32, F16): per-element bytes.
        if (type.isBlockQuantized()) {
            int blockElems = (type == GgufTensorType.Q6_K) ? 256 : 32;
            return ((elems + blockElems - 1L) / blockElems) * type.bytesPerBlockOrElement;
        }
        return elems * type.bytesPerBlockOrElement;
    }
}
