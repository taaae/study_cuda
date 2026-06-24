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
__global__ void gemm(const float* A, const float* B, float* C,
                     int M, int N, int K) {
    // TODO: each block computes one TILE x TILE tile of C. Walk the K dimension
    //       in TILE-wide steps, staging the A and B sub-tiles through shared memory
    //       (zero-padding out-of-range loads), accumulating the dot product in a
    //       register, then do a guarded write of the result to C.
    //       (See README's function table + hints.md.)
}

// Host entry point. All pointers are DEVICE pointers, all matrices row-major.
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: configure a TILE x TILE block and a grid covering all of C, then
    //       launch gemm. (See README + hints.md.)
}
