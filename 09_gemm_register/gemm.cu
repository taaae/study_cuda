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
__global__ void gemm(const float* __restrict__ A, const float* __restrict__ B,
                     float* __restrict__ C, int M, int N, int K) {
    // TODO: each block computes a BM x BN tile of C; each thread owns a TM x TN
    //       register micro-tile (acc). Loop over K in BK-wide slabs: cooperatively
    //       float4-load A's slab into shared memory TRANSPOSED and B's slab into
    //       shared memory, __syncthreads, then for each k stage a/b vectors from
    //       shared and do the rank-1 update into acc, __syncthreads.
    //       Finally float4-write acc back to C.
    //       (See README's function table + hints.md.)
}

// Host entry point. All pointers are DEVICE pointers, all matrices row-major.
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: configure a 256-thread (16 x 16) block and a grid of BM x BN tiles
    //       covering C, then launch gemm. (See README + hints.md.)
}
