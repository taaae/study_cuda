// Exercise 08 — Tiled GEMM (shared memory)
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

// Square tile size. Block is TILE x TILE threads; each block computes one
// TILE x TILE tile of C. Keep this in sync between the kernel and solve().
#ifndef TILE
#define TILE 16
#endif

// C = A * B, all row-major. A is M x K, B is K x N, C is M x N.
// Index as: A[r*K + c], B[r*N + c], C[r*N + c].
//
// Structure (see README):
//   __shared__ float As[TILE][TILE], Bs[TILE][TILE];
//   row = blockIdx.y*TILE + ty;  col = blockIdx.x*TILE + tx;
//   float sum = 0;
//   for each K-tile t:
//       load As[ty][tx] from A (0 if out of range), Bs[ty][tx] from B (0 if out of range)
//       __syncthreads();
//       for k in 0..TILE-1: sum += As[ty][k] * Bs[k][tx];
//       __syncthreads();
//   if (row < M && col < N) C[row*N + col] = sum;
__global__ void gemm(const float* A, const float* B, float* C,
                     int M, int N, int K) {
    // TODO: declare the two __shared__ tiles
    // TODO: compute tx, ty, row, col, and a float accumulator
    // TODO: loop over K-tiles: cooperative load -> __syncthreads -> multiply -> __syncthreads
    // TODO: guarded write of the accumulator to C
}

// Host entry point. All pointers are DEVICE pointers, all matrices row-major.
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: dim3 block(TILE, TILE);
    // TODO: dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
    // TODO: gemm<<<grid, block>>>(A, B, C, M, N, K);
}
