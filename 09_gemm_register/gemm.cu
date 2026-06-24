// Exercise 09 — Register-Blocked GEMM (float4 loads)
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

// Block tile / micro-tile constants (the reference uses these; you may keep them).
// Block is 16 x 16 = 256 threads: BM/TM = 16 and BN/TN = 16.
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

// C = A * B, row-major. A is M x K, B is K x N, C is M x N.
// The harness guarantees M, N, K are multiples of the tile sizes, so every
// float4 load/store below is in-bounds and 16-byte aligned (no boundary check).
//
// Plan (see README):
//   __shared__ float As[BK][BM];   // A slab stored TRANSPOSED: As[k][row]
//   __shared__ float Bs[BK][BN];   // B slab: Bs[k][col]
//   float acc[TM][TN] = {0};       // per-thread micro-tile, in registers
//   for k0 in 0..K step BK:
//       each thread float4-loads one chunk of A (store transposed into As)
//       each thread float4-loads one chunk of B (store into Bs)
//       __syncthreads();
//       for k in 0..BK: stage a_reg[TM] from As[k], b_reg[TN] from Bs[k];
//                       acc[i][j] += a_reg[i]*b_reg[j];
//       __syncthreads();
//   write acc back to C (float4 stores)
__global__ void gemm(const float* __restrict__ A, const float* __restrict__ B,
                     float* __restrict__ C, int M, int N, int K) {
    // TODO: blockRow, blockCol from blockIdx; tx, ty, linear tid
    // TODO: declare As, Bs, acc[TM][TN], a_reg[TM], b_reg[TN]
    // TODO: derive the per-thread float4 load indices for A and B
    // TODO: K-loop: float4 load A (transposed into As) and B (into Bs)
    // TODO: __syncthreads(); inner product over BK into acc; __syncthreads();
    // TODO: float4 write-back of acc to C
}

// Host entry point. All pointers are DEVICE pointers, all matrices row-major.
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: dim3 block(16, 16);
    // TODO: dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    // TODO: gemm<<<grid, block>>>(A, B, C, M, N, K);
}
