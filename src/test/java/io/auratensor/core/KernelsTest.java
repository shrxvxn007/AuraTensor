package io.auratensor.core;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.nio.FloatBuffer;

import static org.junit.jupiter.api.Assertions.*;

class KernelsTest {

    Tensor x;
    Tensor w;
    Tensor A;
    Tensor B;
    Tensor C;
    Tensor y;

    @AfterEach
    void cleanup() {
        close(x); close(w); close(A); close(B); close(C); close(y);
    }

    private static void close(Tensor t) {
        if (t != null) t.close();
    }

    @Test
    void rmsNormMatchesScalarBaseline() {
        x = Tensor.allocate1D(DType.FP32, 16);
        w = Tensor.allocate1D(DType.FP32, 16);
        for (int i = 0; i < 16; i++) {
            x.setFloat((float) (Math.sin(i) * 2 + 0.5), i);
            w.setFloat((float) Math.cos(i * 0.01) + 1.0f, i);
        }
        // Reference
        x.copyFromOther(x);  // no-op; we'll just call twice to compare values
        Tensor ref = Tensor.allocate1D(DType.FP32, 16);
        for (int i = 0; i < 16; i++) ref.setFloat(x.getFloat(i), i);
        Kernels.rmsNormInPlace(ref, w, 1e-5f);
        Kernels.rmsNormInPlace(x, w, 1e-5f);
        for (int i = 0; i < 16; i++) {
            assertEquals(ref.getFloat(i), x.getFloat(i), 1e-5f);
        }
        ref.close();
    }

    @Test
    void siluMatchesScalarBaseline() {
        x = Tensor.allocate1D(DType.FP32, 32);
        for (int i = 0; i < 32; i++) x.setFloat((float) (i - 16) * 0.3f, i);
        Tensor ref = Tensor.allocate1D(DType.FP32, 32);
        for (int i = 0; i < 32; i++) {
            float xv = (float) (i - 16) * 0.3f;
            ref.setFloat(xv / (1.0f + (float) Math.exp(-xv)), i);
        }
        Kernels.siluInPlace(x);
        for (int i = 0; i < 32; i++) {
            assertEquals(ref.getFloat(i), x.getFloat(i), 1e-5f);
        }
        ref.close();
    }

    @Test
    void softmaxMatchesScalarBaseline() {
        x = Tensor.allocate1D(DType.FP32, 8);
        for (int i = 0; i < 8; i++) x.setFloat((float) (i * 0.7 - 2), i);
        Tensor ref = Tensor.allocate1D(DType.FP32, 8);
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < 8; i++) max = Math.max(max, x.getFloat(i));
        float sum = 0;
        for (int i = 0; i < 8; i++) {
            float v = (float) Math.exp(x.getFloat(i) - max);
            ref.setFloat(v, i);
            sum += v;
        }
        for (int i = 0; i < 8; i++) ref.setFloat(ref.getFloat(i) / sum, i);
        Kernels.softmaxInPlace(x);
        for (int i = 0; i < 8; i++) {
            assertEquals(ref.getFloat(i), x.getFloat(i), 1e-5f);
        }
        ref.close();
    }

    @Test
    void sumOfSoftmaxProbabilitiesIsOne() {
        x = Tensor.allocate1D(DType.FP32, 64);
        for (int i = 0; i < 64; i++) x.setFloat((float) (i * 0.13 - 4), i);
        Kernels.softmaxInPlace(x);
        float sum = 0;
        for (int i = 0; i < 64; i++) sum += x.getFloat(i);
        assertEquals(1.0f, sum, 1e-4f);
    }

    @Test
    void ropeIsInvertible() {
        x = Tensor.allocate1D(DType.FP32, 8);
        for (int i = 0; i < 8; i++) x.setFloat((float) (i + 1) * 0.3f, i);
        Tensor orig = Tensor.allocate1D(DType.FP32, 8);
        for (int i = 0; i < 8; i++) orig.setFloat(x.getFloat(i), i);

        // Per ropeInPlace docstring: cos/sin must be of length headDim and duplicated
        // per pair element, e.g. [c_0, c_0, c_1, c_1, c_2, c_2, c_3, c_3] for headDim=8.
        Tensor cos = Tensor.allocate1D(DType.FP32, 8);
        Tensor sin = Tensor.allocate1D(DType.FP32, 8);
        for (int i = 0; i < 4; i++) {
            float a = i * 0.7f;
            float cv = (float) Math.cos(a);
            float sv = (float) Math.sin(a);
            cos.setFloat(cv, 2 * i);
            cos.setFloat(cv, 2 * i + 1);
            sin.setFloat(sv, 2 * i);
            sin.setFloat(sv, 2 * i + 1);
        }
        Kernels.ropeInPlace(x, cos, sin, 8);

        // Undo by using -sin
        // ropeInPlace reads species.length() floats per SIMD iteration; sinNeg must match headDim.
        // Iterate over all 8 lanes and negate each sin value directly (sin[i] already duplicated).
        Tensor sinNeg = Tensor.allocate1D(DType.FP32, 8);
        for (int i = 0; i < 8; i++) {
            sinNeg.setFloat(-sin.getFloat(i), i);
        }
        Kernels.ropeInPlace(x, cos, sinNeg, 8);

        for (int i = 0; i < 8; i++) {
            assertEquals(orig.getFloat(i), x.getFloat(i), 1e-5f);
        }
        orig.close(); cos.close(); sin.close(); sinNeg.close();
    }

    @Test
    void sgemmMatchesScalar() {
        int M = 4, K = 16, N = 6;
        A = Tensor.allocate2D(DType.FP32, M, K);
        B = Tensor.allocate2D(DType.FP32, K, N);
        C = Tensor.allocate2D(DType.FP32, M, N);

        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) A.setFloat((float) ((i + 1) * (k + 1)), i, k);
        }
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) B.setFloat((float) ((k + 2) * (j + 3)), k, j);
        }
        Kernels.sgemm(A, B, C);

        // Scalar baseline
        float[][] ref = new float[M][N];
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0;
                for (int k = 0; k < K; k++) {
                    sum += A.getFloat(i, k) * B.getFloat(k, j);
                }
                ref[i][j] = sum;
            }
        }
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                assertEquals(ref[i][j], C.getFloat(i, j), 1e-3f, "(" + i + "," + j + ")");
            }
        }
    }

    @Test
    void sgemvMatchesManual() {
        int M = 8, K = 32;
        A = Tensor.allocate2D(DType.FP32, M, K);
        Tensor x = Tensor.allocate1D(DType.FP32, K);
        y = Tensor.allocate1D(DType.FP32, M);
        for (int i = 0; i < M; i++)
            for (int k = 0; k < K; k++)
                A.setFloat((float) ((i + 1) * (k + 1)), i, k);
        for (int k = 0; k < K; k++) x.setFloat(0.1f * (k + 1), k);
        Kernels.sgemv(A, x, y);

        for (int i = 0; i < M; i++) {
            float sum = 0;
            for (int k = 0; k < K; k++) sum += A.getFloat(i, k) * x.getFloat(k);
            assertEquals(sum, y.getFloat(i), 1e-3f);
        }
        x.close();
    }
}
