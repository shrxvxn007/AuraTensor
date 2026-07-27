package io.auratensor.format;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
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
 *
 * <p><b>Alignment convention</b> — GGUF v3 spec stores all metadata
 * field bytes (key length, key bytes, value-type int, value bytes) and
 * tensor info field bytes (name, nDims, dims, typeCode, offset) BACK-TO-BACK
 * with <b>no</b> padding between them. The spec REMOVED the v1/v2
 * zero-pad-to-8-byte convention that was applied between fields; only
 * the data-section offset honours a configurable per-file alignment
 * declared in metadata (typically 32 or 256 bytes; default 32).
 *
 * <p><b>Misalignment-safe reads</b> — because the runtime padding-free
 * layout means successive reads land on byte positions that are
 * unpredictable from FFM's {@code ValueLayout} alignment constraints,
 * every multi-byte read in this parser uses {@link #readInt},
 * {@link #readLong}, or {@link #readShort} helpers that copy raw bytes
 * via {@link MemorySegment#copy} and assemble the little-endian value
 * manually. This bypasses {@link MemorySegment#get}'s alignment
 * preconditions outright, so a read at any byte position is correct
 * regardless of preceding variable-length fields.
 *
 * <p><b>Historical bug (now fixed)</b> — earlier revisions of this
 * parser added {@code c.pos = (c.pos + 7L) & ~7L;} round-ups that
 * skipped 1­–7 bytes of real GGUF content. The resulting misaligned
 * {@code readLong} consumed string bytes ("nnnnnnn p" ≈
 * {@code 0x6E6E6E6E6E6E6E70}) and decoded them as the bogus long
 * {@code 7,954,877,566,517,510,144}, igniting
 * {@code IndexOutOfBoundsException} or tripping plausibility checks
 * at huge offsets. With byte-copy readers, ALL round-ups become
 * no-ops and the parser follows GGUF bytes verbatim.
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

        // GGUF v3 spec: KV metadata fields are tightly packed (no
        // inter-field padding). The byte-copy readers handle consequential
        // unaligned positions correctly without any c.pos round-up.
        Cursor cur = new Cursor(24);
        GgufMetadata meta = parseMetadata(mapped, cur, metadataKvCount);

        int n = (int) tensorCount;
        List<GgufTensorInfo> infos = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            infos.add(parseTensorInfo(mapped, cur));
        }

        // GGUF v3 spec: tensor data section begins at the per-file
        // alignment declared in metadata (default 32). GGUF v3 made this
        // per-file configurable; older hardcoded align32 caused IOOBE
        // on files with 64- or 256-byte alignment.
        long alignment = meta.longOrDefault("general.alignment", 32L);
        long dataOffset = alignUp(cur.pos, alignment);

        return new GgufFile(raf, ch, mapped, arena, version, tensorCount, metadataKvCount,
                            meta, infos, dataOffset, fileLen);
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
        long len = info.byteSize();
        if (len <= 0 || info.type().bytesPerBlockOrElement < 0) {
            throw new UnsupportedOperationException(
                "Gguf tensor type " + info.type().label + " for tensor '" + info.name()
                + "' is not yet implemented by AuraTensor (bytesPerBlockOrElement="
                + info.type().bytesPerBlockOrElement + ", byteSize=" + len + "). "
                + "Supported: F32, F16, Q4_0, Q8_0. Use a different export (e.g. all-Q4_0).");
        }
        long start = dataOffset + info.offset();
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
    // Misalignment-safe byte-copy readers — GGUF fields are tightly packed
    // per spec, so reads can land on any byte offset. We bypass the FFM
    // alignment preconditions by copying raw bytes and assembling
    // little-endian values manually.
    // ---------------------------------------------------------------------

    private static final class Cursor {
        long pos;
        Cursor(long pos) { this.pos = pos; }
    }

    /** Read a little-endian int at any offset (no alignment constraint). */
    private static int readInt(MemorySegment s, long off) {
        byte[] b = new byte[4];
        MemorySegment.copy(s, off, MemorySegment.ofArray(b), 0L, 4L);
        return (b[0] & 0xFF)
             | ((b[1] & 0xFF) <<  8)
             | ((b[2] & 0xFF) << 16)
             | ((b[3] & 0xFF) << 24);
    }

    /** Read a little-endian long at any offset (no alignment constraint). */
    private static long readLong(MemorySegment s, long off) {
        byte[] b = new byte[8];
        MemorySegment.copy(s, off, MemorySegment.ofArray(b), 0L, 8L);
        return (b[0] & 0xFFL)
             | ((b[1] & 0xFFL) <<  8)
             | ((b[2] & 0xFFL) << 16)
             | ((b[3] & 0xFFL) << 24)
             | ((b[4] & 0xFFL) << 32)
             | ((b[5] & 0xFFL) << 40)
         | ((b[6] & 0xFFL) << 48)
             | ((b[7] & 0xFFL) << 56);
    }

    /** Read a little-endian short at any offset (no alignment constraint). */
    private static int readShort(MemorySegment s, long off) {
        byte[] b = new byte[2];
        MemorySegment.copy(s, off, MemorySegment.ofArray(b), 0L, 2L);
        return (b[0] & 0xFF) | ((b[1] & 0xFF) << 8);
    }

    /** Round up to the next multiple of `align`. The GGUF v3 spec
     *  defines tensor-data alignment via `general.alignment` metadata
     *  (default 32 on legacy writers; many modern exports use 64 or 256
     *  for mmap page alignment efficiency). */
    private static long alignUp(long v, long align) {
        return ((v + align - 1L) / align) * align;
    }

    private static String readLengthPrefixedString(MemorySegment s, Cursor c, byte[] scratch) {
        long len = readLong(s, c.pos);
        c.pos += 8;
        if (len < 0 || len > Integer.MAX_VALUE) {
            throw new IllegalStateException("Implausible string length: " + len
                + " at offset " + (c.pos - 8));
        }
        byte[] buf = (len > scratch.length) ? new byte[(int) len] : scratch;
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
        // Spec: KV fields are tightly packed — no inter-KV padding.
        for (long i = 0; i < count; i++) {
            String key = readLengthPrefixedString(s, c, scratch);
            int vtCode = readInt(s, c.pos);
            c.pos += 4;
            GgufMetadataValueType vt = GgufMetadataValueType.fromCode(vtCode);
            Object value = readValue(s, c, vt, scratch);
            out.put(key, value);
        }
        return out;
    }

    private static Object readValue(MemorySegment s, Cursor c,
                                    GgufMetadataValueType vt, byte[] scratch) {
        switch (vt) {
            case UINT8:  { long v = s.get(ValueLayout.JAVA_BYTE, c.pos) & 0xFFL; c.pos += 1; return v; }
            case INT8:   { long v = s.get(ValueLayout.JAVA_BYTE, c.pos);      c.pos += 1; return v; }
            case UINT16: { long v = readShort(s, c.pos) & 0xFFFFL;            c.pos += 2; return v; }
            case INT16:  { long v = (short) readShort(s, c.pos);              c.pos += 2; return v; }
            case UINT32: { long v = readInt(s, c.pos) & 0xFFFFFFFFL;          c.pos += 4; return v; }
            case INT32:  { long v = readInt(s, c.pos);                        c.pos += 4; return v; }
            case UINT64: { long v = readLong(s, c.pos);                       c.pos += 8; return v; }
            case INT64:  { long v = readLong(s, c.pos);                       c.pos += 8; return v; }
            case FLOAT32:{ int ibits = readInt(s, c.pos);                     c.pos += 4; return Float.intBitsToFloat(ibits); }
            case FLOAT64:{ long lbits = readLong(s, c.pos);                   c.pos += 8; return Double.longBitsToDouble(lbits); }
            case BOOL:   { byte b = s.get(ValueLayout.JAVA_BYTE, c.pos);      c.pos += 1; return b != 0; }
            case STRING: { return readLengthPrefixedString(s, c, scratch); }
            case ARRAY: {
                // GGUF v3 spec: ARRAY elements are tightly packed.
                int elemTypeCode = readInt(s, c.pos);
                c.pos += 4;
                GgufMetadataValueType elemType = GgufMetadataValueType.fromCode(elemTypeCode);
                long len = readLong(s, c.pos);
                c.pos += 8;
                if (len < 0 || len > 1 << 26) {
                    throw new IllegalStateException("Implausible array length: " + len
                        + " at offset " + (c.pos - 8));
                }
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
        // GGUF v3 spec: tensor info fields are tightly packed (no padding).
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
