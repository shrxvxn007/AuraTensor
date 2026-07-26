package io.auratensor.format;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.*;

class GgufFileTest {

    @Test
    void parsesMinimalHeader() throws IOException {
        // Build a tiny GGUF v3 file in memory:
        //  - magic "GGUF" (4 bytes)
        //  - version = 3
        //  - tensorCount = 0
        //  - metadataKvCount = 1
        //  - one KV: "general.architecture" = "llama"
        //  - aligned tensor-info section (empty, since tensorCount=0)
        //  - aligned data section (empty)
        // Layout (with parser 8-byte alignment after key read):
        //   pos 24   long: key length = 20
        //   pos 32-51 20 bytes: "general.architecture"
        //   pos 52-55 4 bytes: zero-pad (parser aligns to 56)
        //   pos 56   int: value-type STRING = 8
        //   pos 60-63 4 bytes: zero-pad (parser aligns to 64)
        //   pos 64   long: string length = 5
        //   pos 72   5 bytes: "llama"
        long fileLen = 24 + 8 + 20 + 4 + 4 + 4 + 8 + 5 + 3;
        fileLen = ((fileLen + 31) / 32) * 32;

        Arena arena = Arena.ofConfined();
        MemorySegment seg = arena.allocate(fileLen, 16);

        seg.set(ValueLayout.JAVA_INT, 0, 0x46554747);          // GGUF magic
        seg.set(ValueLayout.JAVA_INT, 4, 3);                   // v3
        seg.set(ValueLayout.JAVA_LONG, 8, 0L);                 // tensorCount = 0
        seg.set(ValueLayout.JAVA_LONG, 16, 1L);                // metadataKvCount = 1

        long pos = 24;
        seg.set(ValueLayout.JAVA_LONG, pos, 20L);              // key length = 20 bytes (strlen("general.architecture"))
        pos += 8;                                              // pos=32
        seg.setString(pos, "general.architecture");           // writes 20 UTF-8 bytes at 32..51
        pos += 20;                                             // pos=52 (parser will align to 56)
        while (pos < 56) {                                     // zero-pad to int position
            seg.set(ValueLayout.JAVA_BYTE, pos, (byte) 0);
            pos++;
        }
        seg.set(ValueLayout.JAVA_INT, pos, 8);                 // value-type STRING (enum code 8)
        pos += 4;                                              // pos=60
        while (pos < 64) {                                     // zero-pad to long position
            seg.set(ValueLayout.JAVA_BYTE, pos, (byte) 0);
            pos++;
        }
        seg.set(ValueLayout.JAVA_LONG, pos, 5L);               // value-length = 5
        pos += 8;                                              // pos=72
        seg.setString(pos, "llama");                           // pos=72
        pos += 5;                                              // pos=77
        while (pos < fileLen) {                                // zero-pad to fileLen
            seg.set(ValueLayout.JAVA_BYTE, pos, (byte) 0);
            pos++;
        }

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
