package io.auratensor.format;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies the GGUF v3 binary parser on the spec-conformant
 * tightly-packed layout: every metadata field (key length, key bytes,
 * value-type int, value length, value bytes) is stored back-to-back
 * with no padding between them. GGUF v3 made the previous
 * zero-pad-to-8-byte convention obsolete; only the data section offset
 * honours a per-file alignment declared in metadata (default 32 bytes).
 */
class GgufFileTest {

    @Test
    void parsesMinimalHeader() throws IOException {
        // Build a tiny GGUF v3 file in memory:
        //  - magic "GGUF" (4 bytes)
        //  - version = 3
        //  - tensorCount = 0
        //  - metadataKvCount = 1
        //  - one KV: "general.architecture" = "llama"
        // Tightly-packed layout (per GGUF v3 spec — NO inter-field padding):
        //   pos 24   long: key length = 20
        //   pos 32-51 20 bytes: "general.architecture"
        //   pos 52   int: value-type STRING = 8
        //   pos 56   long: string length = 5
        //   pos 64   5 bytes: "llama"
        // Round file length up to next 32-byte boundary for the data section.
        long fileLen = 24 + 8 + 20 + 4 + 8 + 5;
        fileLen = ((fileLen + 31) / 32) * 32;

        Arena arena = Arena.ofConfined();
        MemorySegment seg = arena.allocate(fileLen, 16);

        seg.set(ValueLayout.JAVA_INT, 0, 0x46554747);          // GGUF magic
        seg.set(ValueLayout.JAVA_INT, 4, 3);                   // v3
        seg.set(ValueLayout.JAVA_LONG, 8, 0L);                 // tensorCount = 0
        seg.set(ValueLayout.JAVA_LONG, 16, 1L);                // metadataKvCount = 1

        long pos = 24;
        seg.set(ValueLayout.JAVA_LONG, pos, 20L);              // key length = 20 (strlen("general.architecture"))
        pos += 8;                                              // pos=32
        seg.setString(pos, "general.architecture");           // writes 20 UTF-8 bytes at 32..51
        pos += 20;                                             // pos=52
        seg.set(ValueLayout.JAVA_INT, pos, 8);                 // value-type STRING (enum code 8)
        pos += 4;                                              // pos=56
        seg.set(ValueLayout.JAVA_LONG, pos, 5L);               // value-length = 5
        pos += 8;                                              // pos=64
        seg.setString(pos, "llama");                           // writes 5 bytes at 64..68

        // Write the built memory into a temp file and round-trip through GgufFile.
        java.nio.file.Path tmp = java.nio.file.Files.createTempFile("auratensor-test", ".gguf");
        try {
            try (java.nio.channels.FileChannel ch = java.nio.channels.FileChannel.open(
                    tmp, java.nio.file.StandardOpenOption.WRITE)) {
                ch.write(java.nio.ByteBuffer.wrap(seg.toArray(ValueLayout.JAVA_BYTE)));
            }
            try (GgufFile gguf = GgufFile.open(tmp.toString())) {
                assertEquals(3, gguf.version());
                assertEquals(0, gguf.tensorCount());
                assertEquals(1, gguf.metadataKvCount());
                assertEquals("llama", gguf.metadata().stringOrDefault("general.architecture", ""));
            }
        } finally {
            java.nio.file.Files.deleteIfExists(tmp);
        }
    }
}
