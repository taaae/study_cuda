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
    int start_m = blockIdx.x * TILE;
    int start_k = blockIdx.y * TILE;
    int start_n = blockIdx.z * TILE;

    __shared__ float A_shared[TILE * TILE];
    __shared__ float B_shared[TILE * TILE];

    int local_a_col = start_k + threadIdx.x; // since row-major, we want to allign x with columns
    int local_a_row = start_m + threadIdx.y; 
    int local_b_col = start_n + threadIdx.x;
    int local_b_row = start_k + threadIdx.y;

    // copy the whole tile to shared memory (2 values each thread) & sync
    if (local_a_row < M && local_a_col < K) {
        A_shared[threadIdx.y * TILE + threadIdx.x] = A[local_a_row * K + local_a_col];
    } else {
        A_shared[threadIdx.y * TILE + threadIdx.x] = 0;
    }
    if (local_b_row < K && local_b_col < N) {
        B_shared[threadIdx.y * TILE + threadIdx.x] = B[local_b_row * N + local_b_col];
    } else {
        B_shared[threadIdx.y * TILE + threadIdx.x] = 0;
    }
    __syncthreads();

    float sum = 0; // accumulate here sum for one tile in C for each thread

    for (int k = 0; k < TILE; ++k) {
        sum += A_shared[threadIdx.y * TILE + k] * B_shared[k * TILE + threadIdx.x];
    }

    __syncthreads();

    int local_c_col = start_n + threadIdx.x;
    int local_c_row = start_m + threadIdx.y;

    if (local_c_col < N && local_c_row < M) {
        atomicAdd(C + local_c_row * N + local_c_col, sum);
    }
}

// Host entry point. All pointers are DEVICE pointers, all matrices row-major.
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: configure a TILE x TILE block and a grid covering all of C, then
    //       launch gemm. (See README + hints.md.)
    dim3 grid(ceil_div(M, TILE), ceil_div(K, TILE), ceil_div(N, TILE));
    dim3 block(TILE, TILE);
    gemm<<<grid, block>>>(A, B, C, M, N, K);
}
