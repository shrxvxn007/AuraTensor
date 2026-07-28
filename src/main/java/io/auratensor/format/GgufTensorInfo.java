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
        // bytesPerBlockOrElement bytes. Two block-size buckets:
        //   * 32-element blocks (Q4_0 = 18 B, Q4_1 = 20 B, Q5_0 = 22 B,
        //     Q5_1 = 24 B, Q8_0 = 34 B, Q8_1 = 36 B) — every legacy Q*_0/Q*_1
        //     variant.
        //   * 256-element super-blocks (Q2_K = 84 B, Q3_K = 110 B, Q4_K = 148 B,
        //     Q5_K = 180 B, Q6_K = 210 B, Q8_K = 260 B) — every k-quant.
        //     Using the 32-element formula on a 256-element super-block
        //     is over-counted by 8× and trips IOOBE on `MappedMemorySegment
        //     .asSlice(start, len)`.
        // Non-block types (F32, F16): per-element bytes.
        if (type.isBlockQuantized()) {
            int blockElems = switch (type) {
                case Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K -> 256;
                default -> 32;
            };
            return ((elems + blockElems - 1L) / blockElems) * type.bytesPerBlockOrElement;
        }
        return elems * type.bytesPerBlockOrElement;
    }
}
