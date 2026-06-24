// Exercise 09 harness — DO NOT EDIT.
// Builds data, runs your solve(), checks correctness vs a CPU reference, times
// it vs a basic tiled-GEMM baseline, and benchmarks against cuBLAS.
#include <vector>
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include "cuda_utils.cuh"

#ifndef SOLUTION_FILE
#define SOLUTION_FILE "gemm.cu"
#endif
#include SOLUTION_FILE

// ---- Baseline: exercise-08-style tiled GEMM (for speedup) ------------------
#ifndef BASE_TILE
#define BASE_TILE 16
#endif
__global__ void base_gemm(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    __shared__ float As[BASE_TILE][BASE_TILE];
    __shared__ float Bs[BASE_TILE][BASE_TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * BASE_TILE + ty;
    int col = blockIdx.x * BASE_TILE + tx;
    float sum = 0.f;
    int numTiles = (K + BASE_TILE - 1) / BASE_TILE;
    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * BASE_TILE + tx;
        int bRow = t * BASE_TILE + ty;
        As[ty][tx] = (row < M && aCol < K) ? A[(size_t)row * K + aCol] : 0.f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[(size_t)bRow * N + col] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BASE_TILE; ++k) sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * N + col] = sum;
}
static void base_solve(const float* A, const float* B, float* C,
                       int M, int N, int K) {
    dim3 block(BASE_TILE, BASE_TILE);
    dim3 grid(ceil_div(N, BASE_TILE), ceil_div(M, BASE_TILE));
    base_gemm<<<grid, block>>>(A, B, C, M, N, K);
}

int main() {
    print_device_banner();

    // Multiples of 128 so the register-tiled kernel's float4 loads are aligned
    // and in-bounds with no boundary handling.
    const int M = 2048, N = 2048, K = 2048;
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

    // ---- CPU reference on a sampled set of rows (double accumulation) ----
    bool ok = true;
    double maxrel = 0.0;
    const int row_step = 101;
    for (int i = 0; i < M; i += row_step) {
        for (int j = 0; j < N; ++j) {
            double ref = 0.0;
            for (int k = 0; k < K; ++k)
                ref += (double)hA[(size_t)i * K + k] * (double)hB[(size_t)k * N + j];
            double got = hC[(size_t)i * N + j];
            double denom = std::fabs(ref) > 1e-6 ? std::fabs(ref) : 1e-6;
            double rel = std::fabs(ref - got) / denom;
            if (rel > maxrel) maxrel = rel;
            if (rel > 1e-3) ok = false;
        }
    }
    report_correct(ok);
    report_metric("max_rel_err", maxrel);

    double flops = 2.0 * M * N * K;

    // ---- Timing: student vs baseline tiled GEMM ----
    float ms = time_kernel([&] { solve(dA, dB, dC, M, N, K); });
    float base_ms = time_kernel([&] { base_solve(dA, dB, dC, M, N, K); });
    double gflops = flops / (ms * 1e-3) / 1e9;
    report_metric("ms", ms);
    report_metric("base_ms", base_ms);
    report_metric("gflops", gflops);
    report_metric("speedup", base_ms / ms);

    // ---- cuBLAS reference (column-major; see exercise 08 harness for the
    // row-major <-> column-major mapping: m=N, n=M, k=K, operands swapped). ----
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
