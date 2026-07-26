package io.auratensor.core;

/**
 * Defensive helpers for common tensor dimension labels used throughout the
 * core/engine code. We use plain int indexing (rows, cols, channels) so the
 * hot loops do not pay for enum dispatch.
 */
public final class DimOrder {
    public static final int ROWS = 0;
    public static final int COLS = 1;
    public static final int DEPTH = 2;

    private DimOrder() {}
}
