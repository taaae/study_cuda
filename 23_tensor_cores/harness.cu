// Exercise 23 harness — DO NOT EDIT.
// Builds FP16 A,B; runs your solve() for C(FP32)=A*B; checks vs an FP32 CPU
// reference; times it and an FP32 tiled GEMM baseline; reports gflops + speedup.
#include <vector>
#include <cmath>
#include <cstdio>
#include <cuda_fp16.h>
#include "cuda_utils.cuh"

#ifndef SOLUTION_FILE
#define SOLUTION_FILE "wmma_gemm.cu"
#endif
#include SOLUTION_FILE

// ---- FP32 tiled GEMM baseline (for speedup_vs_fp32) ----
#define BTILE 32
__global__ void gemm_fp32(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    __shared__ float As[BTILE][BTILE];
    __shared__ float Bs[BTILE][BTILE];
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * BTILE + ty;
    int col = blockIdx.x * BTILE + tx;
    float acc = 0.f;
    for (int t = 0; t < K / BTILE; ++t) {
        As[ty][tx] = A[row * K + (t * BTILE + tx)];
        Bs[ty][tx] = B[(t * BTILE + ty) * N + col];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BTILE; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[row * N + col] = acc;
}

int main() {
    print_device_banner();

    // Tensor cores (WMMA FP16) need compute capability >= 7.0.
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 7) {
        std::printf("# SKIP: WMMA tensor cores unsupported (need sm_70+)\n");
        report_correct(true);
        return 0;
    }

    const int M = 1024, N = 1024, K = 1024;   // multiples of 16 (and 32)
    const size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;

    std::vector<float> hAf(nA), hBf(nB), hCf(nC), href(nC);
    std::vector<half>  hAh(nA), hBh(nB);
    for (size_t i = 0; i < nA; ++i) { hAf[i] = std::sin(i * 0.0013f) * 0.5f; hAh[i] = __float2half(hAf[i]); }
    for (size_t i = 0; i < nB; ++i) { hBf[i] = std::cos(i * 0.0007f) * 0.5f; hBh[i] = __float2half(hBf[i]); }

    // CPU reference in FP32 but from the FP16-rounded inputs (fair comparison).
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c) {
            double s = 0.0;
            for (int k = 0; k < K; ++k)
                s += (double)__half2float(hAh[r * K + k]) * (double)__half2float(hBh[k * N + c]);
            href[r * N + c] = (float)s;
        }

    half *dAh, *dBh; float *dC;
    CUDA_CHECK(cudaMalloc(&dAh, nA * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dBh, nB * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dC,  nC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dAh, hAh.data(), nA * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBh, hBh.data(), nB * sizeof(half), cudaMemcpyHostToDevice));

    solve(dAh, dBh, dC, M, N, K);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(hCf.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    double maxrel = 0.0;
    for (size_t i = 0; i < nC; ++i) {
        double ref = href[i];
        double err = std::fabs(ref - hCf[i]);
        double rel = err / (std::fabs(ref) + 1e-3);
        if (rel > maxrel) maxrel = rel;
        if (rel > 1e-2) ok = false;        // FP16-appropriate, generous
    }
    report_correct(ok);
    report_metric("max_rel_err", maxrel);

    double flop = 2.0 * M * N * K;
    float ms_sol = time_kernel([&] { solve(dAh, dBh, dC, M, N, K); });
    report_metric("ms", ms_sol);
    report_metric("gflops", flop / (ms_sol * 1e-3) / 1e9);

    // FP32 baseline for speedup_vs_fp32.
    float *dAf, *dBf;
    CUDA_CHECK(cudaMalloc(&dAf, nA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBf, nB * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dAf, hAf.data(), nA * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBf, hBf.data(), nB * sizeof(float), cudaMemcpyHostToDevice));
    dim3 b(BTILE, BTILE), g(N / BTILE, M / BTILE);
    float ms_fp32 = time_kernel([&] { gemm_fp32<<<g, b>>>(dAf, dBf, dC, M, N, K); });
    report_metric("fp32_gflops", flop / (ms_fp32 * 1e-3) / 1e9);
    report_metric("speedup_vs_fp32", ms_fp32 / ms_sol);

    CUDA_CHECK(cudaFree(dAh));
    CUDA_CHECK(cudaFree(dBh));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dAf));
    CUDA_CHECK(cudaFree(dBf));
    return 0;
}
