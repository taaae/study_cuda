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

    // TODO: declare 16x16x16 WMMA fragments for A (half), B (half), and the
    //       FP32 accumulator, and zero-fill the accumulator.

    // TODO: loop over K in steps of WMMA_K; each step load_matrix_sync this
    //       warp's A and B 16x16 tiles (mind the leading dimensions) and
    //       mma_sync them into the accumulator.

    // TODO: store_matrix_sync the accumulator into this warp's C tile
    //       (row-major, leading dimension N).
    // (See README's function table and hints.md if stuck.)
    (void)tilesM; (void)tilesN;
}

void solve(const half* A, const half* B, float* C, int M, int N, int K) {
    // Each block has several warps along x (32 lanes each) and a few rows along y.
    // TODO: pick a block, derive warpsPerBlockX = blockDim.x / 32, and size the
    //       grid so the warps cover all 16x16 output tiles. (See README + hints.md.)
}
