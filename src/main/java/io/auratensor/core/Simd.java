package io.auratensor.core;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Centralized accessor for the runtime-preferred {@link FloatVector} species.
 *
 * <p>Hot kernels query {@link #species()} once for a loop bound and use
 * {@code species.loopBound(n)} to peel tail iterations cleanly. We always use
 * {@link FloatVector} (not {@code Vector}) so kernel dispatch is statically
 * resolved by the JIT.
 *
 * <p>The Vector API was finalized preview/incubator in JDK 21+. Run the JVM
 * with {@code --add-modules jdk.incubator.vector --enable-preview
 * --enable-native-access=ALL-UNNAMED}.
 */
public final class Simd {

    /** Maximum lane width available on this CPU. */
    public static final VectorSpecies<Float> SPECIES =
        FloatVector.SPECIES_PREFERRED;

    private Simd() {}

    /**
     * Returns the loop upper bound (multiple of lane width) for processing
     * {@code length} elements.
     */
    public static int loopBound(int length) {
        return SPECIES.loopBound(length);
    }

    /**
     * True if the host supports at least 256-bit vector lanes.
     */
    public static boolean hasWideVectors() {
        return SPECIES.vectorBitSize() >= 256;
    }
}
