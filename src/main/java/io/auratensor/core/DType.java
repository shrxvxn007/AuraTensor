package io.auratensor.core;

/**
 * Data type of a tensor element or quantized block.
 *
 * <p>Quantized block layouts (Q4_0, Q8_0) report a per-element "nominal" type
 * (INT4 / INT8 respectively) so the rest of the engine can reason uniformly
 * about element counts, while special handling for the embedded FP16 scale
 * lives in {@link io.auratensor.quant.QuantBlock}.
 */
public enum DType {
    FP32(4),
    FP16(2),
    INT8(1),
    INT4(1);  // logical element size (stored as packed nibbles)

    private final int bytesPerElement;

    DType(int bytesPerElement) {
        this.bytesPerElement = bytesPerElement;
    }

    public int bytesPerElement() {
        return bytesPerElement;
    }
}
