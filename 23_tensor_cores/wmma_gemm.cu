// Exercise 23 — Tensor Cores (WMMA half-precision GEMM)
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// C(FP32) = A(FP16) * B(FP16), row-major. One WARP computes one 16x16 tile of C.
// Each lane in the warp must execute every wmma::*_sync call (they are collective).
__global__ void wmma_gemm(const half* A, const half* B, float* C,
                          int M, int N, int K) {
    // Which 16x16 output tile does THIS warp own?
    int warpCol = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;  // tile column
    int warpRow = (blockIdx.y * blockDim.y + threadIdx.y);             // tile row

    int tilesM = M / WMMA_M;
    int tilesN = N / WMMA_N;
    if (warpRow >= tilesM || warpCol >= tilesN) return;

    // TODO: declare the fragments:
    //   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    //   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    //   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    //   wmma::fill_fragment(acc_frag, 0.0f);

    // TODO: loop k0 = 0, 16, ..., K-16
    //   for (int k0 = 0; k0 < K; k0 += WMMA_K) {
    //       wmma::load_matrix_sync(a_frag, A + (warpRow*WMMA_M)*K + k0, K);
    //       wmma::load_matrix_sync(b_frag, B + k0*N + (warpCol*WMMA_N), N);
    //       wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    //   }

    // TODO: store the result tile (ldm = N, row-major):
    //   wmma::store_matrix_sync(C + (warpRow*WMMA_M)*N + (warpCol*WMMA_N),
    //                           acc_frag, N, wmma::mem_row_major);
    (void)tilesM; (void)tilesN;
}

void solve(const half* A, const half* B, float* C, int M, int N, int K) {
    // Each block has several warps along x (32 lanes each) and a few rows along y.
    // TODO: choose a block so warpsPerBlockX = blockDim.x / 32, then size the grid:
    //   dim3 block(128, 4);                       // 4 warps along x, 4 rows along y
    //   int warpsX = block.x / 32;                // = 4
    //   dim3 grid( ceil_div(N/WMMA_N, warpsX), ceil_div(M/WMMA_M, (int)block.y) );
    //   wmma_gemm<<<grid, block>>>(A, B, C, M, N, K);
}
