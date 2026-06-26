// Exercise 08 harness — DO NOT EDIT.
// Builds data, runs your solve(), checks correctness vs a CPU reference, times
// it, and benchmarks against cuBLAS.
#include <vector>
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include "cuda_utils.cuh"

#ifndef SOLUTION_FILE
#define SOLUTION_FILE "gemm.cu"
#endif
#include SOLUTION_FILE

int main() {
    print_device_banner();

    // Square-ish, divisible by common tile sizes; small enough for a CPU check.
    const int M = 1024, N = 1024, K = 1024;
    const size_t aN = (size_t)M * K, bN = (size_t)K * N, cN = (size_t)M * N;

    std::vector<float> hA(aN), hB(bN), hC(cN);
    for (size_t i = 0; i < aN; ++i) hA[i] = std::sin(i * 0.0007f);
    for (size_t i = 0; i < bN; ++i) hB[i] = std::cos(i * 0.0011f);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, aN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, bN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, cN * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), aN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bN * sizeof(float), cudaMemcpyHostToDevice));

    // ---- Student kernel ----
    solve(dA, dB, dC, M, N, K);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, cN * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- CPU reference (double accumulation), sampled rows for speed ----
    // Full M*N*K in double would be slow; check a deterministic subset of rows
    // plus all columns, which still catches indexing/sync bugs.
    bool ok = true;
    double maxrel = 0.0;
    const int row_step = 37;   // coprime-ish stride over rows
    for (int i = 0; i < M; i += row_step) {
        for (int j = 0; j < N; ++j) {
            double ref = 0.0;
            for (int k = 0; k < K; ++k)
                ref += (double)hA[(size_t)i * K + k] * (double)hB[(size_t)k * N + j];
            double got = hC[(size_t)i * N + j];
            double abserr = std::fabs(ref - got);
            // Combined absolute+relative tolerance (numpy.allclose-style). FP32
            // accumulation over K terms gives an absolute error ~K*eps*|A||B| ~1e-4
            // regardless of the algorithm; near-zero C entries (sin/cos cancellation)
            // make a pure per-element relative error meaningless. atol covers those.
            const double atol = 1e-3, rtol = 1e-3;
            if (abserr > atol + rtol * std::fabs(ref)) ok = false;
            // Report rel error floored at the matrix scale so the metric stays
            // informative instead of being dominated by cancellation entries.
            double denom = std::fabs(ref) > 1e-2 ? std::fabs(ref) : 1e-2;
            double rel = abserr / denom;
            if (rel > maxrel) maxrel = rel;
        }
    }
    report_correct(ok);
    report_metric("max_rel_err", maxrel);

    // ---- Timing: student kernel ----
    float ms = time_kernel([&] { solve(dA, dB, dC, M, N, K); });
    double flops = 2.0 * M * N * K;
    double gflops = flops / (ms * 1e-3) / 1e9;
    report_metric("ms", ms);
    report_metric("gflops", gflops);

    // ---- cuBLAS reference (column-major). ----
    // cuBLAS is column-major. Row-major C(MxN) == column-major C^T(NxM), and
    // C^T = B^T * A^T. Passing our row-major B as the first (col-major) operand
    // gives B^T(NxK), and row-major A gives A^T(KxM); their product is C^T,
    // i.e. exactly our row-major C. So: m=N, n=M, k=K, no transposes,
    // lda=N (for B), ldb=K (for A), ldc=N (for C).
    cublasHandle_t h;
    cublasCreate(&h);
    const float alpha = 1.f, beta = 0.f;
    float *dCb;
    CUDA_CHECK(cudaMalloc(&dCb, cN * sizeof(float)));
    float cublas_ms = time_kernel([&] {
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    dB, N,
                    dA, K,
                    &beta,
                    dCb, N);
    });
    double gflops_cublas = flops / (cublas_ms * 1e-3) / 1e9;
    report_metric("cublas_ms", cublas_ms);
    report_metric("gflops_cublas", gflops_cublas);
    report_metric("frac_cublas", gflops / gflops_cublas);

    cublasDestroy(h);
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dCb));
    return 0;
}
