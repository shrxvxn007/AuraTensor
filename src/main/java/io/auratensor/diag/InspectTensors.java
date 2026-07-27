package io.auratensor.diag;

import io.auratensor.format.GgufFile;

import java.util.Arrays;

/**
 * Diagnostic helper: open a GGUF, list every tensor's name, type, rank,
 * dims, numElements and byteSize. Used to localise over-large or
 * mis-parsed tensors that produce IOOBE during FP32 dequantization.
 *
 * <p>Usage: {@code java io.auratensor.diag.InspectTensors &lt;path.gguf&gt;}.
 */
public final class InspectTensors {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.err.println("usage: java io.auratensor.diag.InspectTensors <path.gguf>");
            System.exit(2);
        }
        try (GgufFile gguf = GgufFile.open(args[0])) {
            System.out.printf("GGUF v%d, %d tensors, %d KV metadata, fileLen=%d, dataOffset=0x%x%n",
                gguf.version(), gguf.tensorCount(), gguf.metadataKvCount(),
                gguf.fileLength(), gguf.dataOffset());
            long totalBytes = 0;
            int maxDimsByte = 0;
            String worstName = null;
            long worstElems = 0;
            long worstBytes = 0;
            for (var info : gguf.tensorInfos()) {
                long bytes = info.byteSize();
                totalBytes += bytes;
                System.out.printf("  %-50s type=%-7s rank=%d dims=%-32s elems=%-15d bytes=%-15d%n",
                    trunc(info.name(), 50),
                    info.type().label,
                    info.dims().length,
                    Arrays.toString(info.dims()),
                    info.numElements(),
                    bytes);
                if (info.numElements() > 0 && info.numElements() > (Long.MAX_VALUE / 8L)) {
                    System.out.printf("    >>> SUSPICIOUS numElements=%d (very large)%n",
                        info.numElements());
                }
                if (info.numElements() > worstElems) {
                    worstElems = info.numElements();
                    worstBytes = bytes;
                    worstName = info.name();
                }
            }
            System.out.println();
            System.out.printf("Total tensor data bytes: %d (%.2f MB)%n",
                totalBytes, totalBytes / 1e6);
            System.out.printf("Largest tensor by elem: %s elems=%d bytes=%d%n",
                worstName, worstElems, worstBytes);
            // Sanity-check: tensor data section should not exceed file length.
            if (gguf.dataOffset() + totalBytes > gguf.fileLength()) {
                System.out.println("WARN: declared tensor data exceeds file length by "
                    + (gguf.dataOffset() + totalBytes - gguf.fileLength()) + " bytes");
            }
        }
    }

    private static String trunc(String s, int n) {
        return s.length() <= n ? s : s.substring(0, n - 1) + "\u2026";
    }
}
