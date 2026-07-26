package io.auratensor.format;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Tiny GGUF v3 binary parser.
 *
 * <p>Parses the GGUF magic, version, tensor count, KV metadata, and tensor
 * info records, then mmap()s the file for zero-copy weight access. We do
 * not allocate intermediate Java representations for weights — they live
 * in the mapped MemorySegment and are reached by byte offsets.
 *
 * <p>Supports GGUF version 3 (current Llama.cpp / Llama 3 exports).
 */
public final class GgufFile implements AutoCloseable {

    /** GGUF magic bytes (little-endian): "GGUF" → 0x46554747. */
    public static final int MAGIC = 0x46554747;

    private final RandomAccessFile raf;
    private final FileChannel channel;
    private final MemorySegment mapped;
    private final Arena arena;

    private final int version;
    private final long tensorCount;
    private final long metadataKvCount;
    private final GgufMetadata metadata;
    private final List<GgufTensorInfo> tensorInfos;
    private final long dataOffset;        // byte offset where tensor data starts
    private final long fileLength;

    private GgufFile(RandomAccessFile raf,
                      FileChannel channel,
                      MemorySegment mapped,
                      Arena arena,
                      int version,
                      long tensorCount,
                      long metadataKvCount,
                      GgufMetadata metadata,
                      List<GgufTensorInfo> tensorInfos,
                      long dataOffset,
                      long fileLength) {
        this.raf = raf;
        this.channel = channel;
        this.mapped = mapped;
        this.arena = arena;
        this.version = version;
        this.tensorCount = tensorCount;
        this.metadataKvCount = metadataKvCount;
        this.metadata = metadata;
        this.tensorInfos = tensorInfos;
        this.dataOffset = dataOffset;
        this.fileLength = fileLength;
    }

    /**
     * Opens a GGUF file. The path may be a regular file; we mmap with
     * READ_ONLY for zero-copy weight access.
     */
    public static GgufFile open(String path) throws IOException {
        RandomAccessFile raf = new RandomAccessFile(path, "r");
        FileChannel ch = raf.getChannel();
        long fileLen = ch.size();
        Arena arena = Arena.ofConfined();
        MemorySegment mapped = ch.map(FileChannel.MapMode.READ_ONLY, 0L, fileLen, arena);

        int magic = readInt(mapped, 0);
        if (magic != MAGIC) {
            arena.close();
            raf.close();
            throw new IOException("Not a GGUF file: magic=0x" + Integer.toHexString(magic));
        }

        int version = readInt(mapped, 4);
        if (version != 3) {
            arena.close();
            raf.close();
            throw new IOException("Unsupported GGUF version: " + version
                + " (AuraTensor supports v3)");
        }

        long tensorCount = readLong(mapped, 8);
        long metadataKvCount = readLong(mapped, 16);

        Cursor cur = new Cursor(24);
        GgufMetadata meta = parseMetadata(mapped, cur, metadataKvCount);

        cur.pos = align32(cur.pos);

        int n = (int) tensorCount;
        List<GgufTensorInfo> infos = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            infos.add(parseTensorInfo(mapped, cur));
        }

        long dataStart = align32(cur.pos);

        return new GgufFile(raf, ch, mapped, arena, version, tensorCount, metadataKvCount,
                            meta, infos, dataStart, fileLen);
    }

    public int version() { return version; }
    public long tensorCount() { return tensorCount; }
    public long metadataKvCount() { return metadataKvCount; }
    public GgufMetadata metadata() { return metadata; }
    public List<GgufTensorInfo> tensorInfos() { return tensorInfos; }
    public long dataOffset() { return dataOffset; }
    public long fileLength() { return fileLength; }
    public MemorySegment mapped() { return mapped; }

    /** Returns a slice of the mmap region corresponding to a single tensor's raw bytes. */
    public MemorySegment tensorData(GgufTensorInfo info) {
        long start = dataOffset + info.offset();
        long len = info.byteSize();
        return mapped.asSlice(start, len);
    }

    @Override
    public void close() throws IOException {
        try {
            if (arena.scope().isAlive()) arena.close();
        } finally {
            raf.close();
        }
    }

    // ---------------------------------------------------------------------
    // Parsing helpers
    // ---------------------------------------------------------------------

    private static final class Cursor {
        long pos;
        Cursor(long pos) { this.pos = pos; }
    }

    private static int readInt(MemorySegment s, long off) {
        return s.get(java.lang.foreign.ValueLayout.JAVA_INT, off);
    }

    private static long readLong(MemorySegment s, long off) {
        return s.get(java.lang.foreign.ValueLayout.JAVA_LONG, off);
    }

    private static long align32(long v) {
        return ((v + 31L) / 32L) * 32L;
    }

    private static String readLengthPrefixedString(MemorySegment s, Cursor c, byte[] scratch) {
        long len = readLong(s, c.pos);
        c.pos += 8;
        byte[] buf = (len > scratch.length) ? new byte[(int) len] : scratch;
        // Stable segment-to-segment copy via MemorySegment.ofArray(buf) —
        // works on JDK 22+ FFM without requiring ValueLayout-specific overloads.
        MemorySegment.copy(
            s, c.pos,
            MemorySegment.ofArray(buf), 0L,
            (int) len);
        c.pos += len;
        return new String(buf, 0, (int) len, StandardCharsets.UTF_8);
    }

    private static GgufMetadata parseMetadata(MemorySegment s, Cursor c, long count) {
        GgufMetadata out = new GgufMetadata();
        byte[] scratch = new byte[1024];
        for (long i = 0; i < count; i++) {
            String key = readLengthPrefixedString(s, c, scratch);

            // GGUF v3 spec: align each KV key's read position to 8 bytes so
            // the value-type int read below lands on an int-aligned offset.
            c.pos = (c.pos + 7L) & ~7L;

            int vtCode = readInt(s, c.pos);
            c.pos += 4;
            // GGUF v3 spec: align pos to 8 before reading the value (long/etc.).
            c.pos = (c.pos + 7L) & ~7L;
            GgufMetadataValueType vt = GgufMetadataValueType.fromCode(vtCode);
            Object value = readValue(s, c, vt, scratch);
            out.put(key, value);
        }
        return out;
    }

    private static Object readValue(MemorySegment s, Cursor c,
                                    GgufMetadataValueType vt, byte[] scratch) {
        switch (vt) {
            case UINT8:  { long v = s.get(java.lang.foreign.ValueLayout.JAVA_BYTE, c.pos) & 0xFFL; c.pos += 1; return v; }
            case INT8:   { long v = s.get(java.lang.foreign.ValueLayout.JAVA_BYTE, c.pos);      c.pos += 1; return v; }
            case UINT16: { long v = s.get(java.lang.foreign.ValueLayout.JAVA_SHORT, c.pos) & 0xFFFFL; c.pos += 2; return v; }
            case INT16:  { long v = s.get(java.lang.foreign.ValueLayout.JAVA_SHORT, c.pos);     c.pos += 2; return v; }
            case UINT32: { long v = s.get(java.lang.foreign.ValueLayout.JAVA_INT, c.pos) & 0xFFFFFFFFL; c.pos += 4; return v; }
            case INT32:  { long v = s.get(java.lang.foreign.ValueLayout.JAVA_INT, c.pos);       c.pos += 4; return v; }
            case UINT64: { long v = s.get(java.lang.foreign.ValueLayout.JAVA_LONG, c.pos);      c.pos += 8; return v; }
            case INT64:  { long v = s.get(java.lang.foreign.ValueLayout.JAVA_LONG, c.pos);      c.pos += 8; return v; }
            case FLOAT32:{ float v = s.get(java.lang.foreign.ValueLayout.JAVA_FLOAT, c.pos);    c.pos += 4; return v; }
            case FLOAT64:{ double v = s.get(java.lang.foreign.ValueLayout.JAVA_DOUBLE, c.pos);  c.pos += 8; return v; }
            case BOOL:   { byte b = s.get(java.lang.foreign.ValueLayout.JAVA_BYTE, c.pos);      c.pos += 1; return b != 0; }
            case STRING: { return readLengthPrefixedString(s, c, scratch); }
            case ARRAY: {
                int elemTypeCode = readInt(s, c.pos);
                c.pos += 4;
                GgufMetadataValueType elemType = GgufMetadataValueType.fromCode(elemTypeCode);
                long len = readLong(s, c.pos);
                c.pos += 8;
                List<Object> list = new ArrayList<>((int) len);
                for (long i = 0; i < len; i++) {
                    list.add(readValue(s, c, elemType, scratch));
                }
                return list;
            }
            default:
                throw new IllegalStateException("Unsupported GGUF metadata type: " + vt);
        }
    }

    private static GgufTensorInfo parseTensorInfo(MemorySegment s, Cursor c) {
        byte[] scratch = new byte[256];
        String name = readLengthPrefixedString(s, c, scratch);

        int nDims = readInt(s, c.pos);
        c.pos += 4;
        if (nDims < 1 || nDims > 4) {
            throw new IllegalStateException("Unsupported rank: " + nDims + " (" + name + ")");
        }
        long[] dims = new long[nDims];
        for (int i = 0; i < nDims; i++) {
            dims[i] = readLong(s, c.pos);
            c.pos += 8;
        }
        int typeCode = readInt(s, c.pos);
        c.pos += 4;
        GgufTensorType type = GgufTensorType.fromCode(typeCode);
        long offset = readLong(s, c.pos);
        c.pos += 8;
        return new GgufTensorInfo(name, dims, type, offset);
    }

    /** Looks up a single tensor info by name. Returns null if not found. */
    public GgufTensorInfo findTensor(String name) {
        for (GgufTensorInfo t : tensorInfos) {
            if (t.name().equals(name)) return t;
        }
        return null;
    }
}
