// Reference solution — Exercise 23 (Tensor Cores / WMMA half-precision GEMM).
#include "cuda_utils.cuh"
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void wmma_gemm(const half* A, const half* B, float* C,
                          int M, int N, int K) {
    int warpCol = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;  // tile column
    int warpRow = (blockIdx.y * blockDim.y + threadIdx.y);             // tile row

    int tilesM = M / WMMA_M;
    int tilesN = N / WMMA_N;
    if (warpRow >= tilesM || warpCol >= tilesN) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        // A 16x16 block at (warpRow tile, k0): top-left = A + warpRow*16*K + k0, ldm = K
        wmma::load_matrix_sync(a_frag, A + (warpRow * WMMA_M) * K + k0, K);
        // B 16x16 block at (k0, warpCol tile): top-left = B + k0*N + warpCol*16, ldm = N
        wmma::load_matrix_sync(b_frag, B + k0 * N + (warpCol * WMMA_N), N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    wmma::store_matrix_sync(C + (warpRow * WMMA_M) * N + (warpCol * WMMA_N),
                            acc_frag, N, wmma::mem_row_major);
}

void solve(const half* A, const half* B, float* C, int M, int N, int K) {
    dim3 block(128, 4);             // 4 warps along x, 4 tile-rows along y
    int warpsX = block.x / 32;      // = 4
    dim3 grid(ceil_div(N / WMMA_N, warpsX),
              ceil_div(M / WMMA_M, (int)block.y));
    wmma_gemm<<<grid, block>>>(A, B, C, M, N, K);
}
