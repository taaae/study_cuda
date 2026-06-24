// Exercise 22 harness — DO NOT EDIT.
// Builds A,B; runs your solve() for C=A*B; checks vs CPU; times it against a
// single-buffer tiled GEMM baseline; reports speedup and gflops.
#include <vector>
#include <cmath>
#include <cstdio>
#include "cuda_utils.cuh"

#ifndef SOLUTION_FILE
#define SOLUTION_FILE "gemm.cu"
#endif
#include SOLUTION_FILE

// ---- Baseline: classic single-buffer tiled GEMM (what we want to beat) ----
#define BTILE 32
__global__ void gemm_single_buffer(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    __shared__ float As[BTILE][BTILE];
    __shared__ float Bs[BTILE][BTILE];
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * BTILE + ty;
    int col = blockIdx.x * BTILE + tx;
    float acc = 0.f;
    int numTiles = K / BTILE;
    for (int t = 0; t < numTiles; ++t) {
        As[ty][tx] = A[row * K + (t * BTILE + tx)];
        Bs[ty][tx] = B[(t * BTILE + ty) * N + col];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BTILE; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[row * N + col] = acc;
}
static void baseline(const float* A, const float* B, float* C,
                     int M, int N, int K) {
    dim3 block(BTILE, BTILE);
    dim3 grid(N / BTILE, M / BTILE);
    gemm_single_buffer<<<grid, block>>>(A, B, C, M, N, K);
}

int main() {
    print_device_banner();

    const int M = 1024, N = 1024, K = 1024;   // multiples of the 32 tile
    const size_t bA = (size_t)M * K * sizeof(float);
    const size_t bB = (size_t)K * N * sizeof(float);
    const size_t bC = (size_t)M * N * sizeof(float);

    std::vector<float> hA(M * K), hB(K * N), hC(M * N), href(M * N);
    for (int i = 0; i < M * K; ++i) hA[i] = std::sin(i * 0.0013f);
    for (int i = 0; i < K * N; ++i) hB[i] = std::cos(i * 0.0007f);

    // CPU reference (row-major).
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) s += (double)hA[r * K + k] * hB[k * N + c];
            href[r * N + c] = (float)s;
        }

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bA));
    CUDA_CHECK(cudaMalloc(&dB, bB));
    CUDA_CHECK(cudaMalloc(&dC, bC));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bB, cudaMemcpyHostToDevice));

    solve(dA, dB, dC, M, N, K);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bC, cudaMemcpyDeviceToHost));

    bool ok = true;
    double maxrel = 0.0;
    for (int i = 0; i < M * N; ++i) {
        double ref = href[i];
        double err = std::fabs(ref - hC[i]);
        double rel = err / (std::fabs(ref) + 1e-6);
        if (rel > maxrel) maxrel = rel;
        if (rel > 1e-3) ok = false;
    }
    report_correct(ok);
    report_metric("max_rel_err", maxrel);

    double flop = 2.0 * M * N * K;
    float ms_sol  = time_kernel([&] { solve(dA, dB, dC, M, N, K); });
    float ms_base = time_kernel([&] { baseline(dA, dB, dC, M, N, K); });
    report_metric("ms", ms_sol);
    report_metric("gflops", flop / (ms_sol * 1e-3) / 1e9);
    report_metric("baseline_ms", ms_base);
    report_metric("speedup", ms_base / ms_sol);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
