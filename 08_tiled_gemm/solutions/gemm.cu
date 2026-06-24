// Reference solution — Exercise 08.
#include "cuda_utils.cuh"

#ifndef TILE
#define TILE 16
#endif

__global__ void gemm(const float* A, const float* B, float* C,
                     int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.f;
    int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;
        As[ty][tx] = (row < M && aCol < K) ? A[(size_t)row * K + aCol] : 0.f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[(size_t)bRow * N + col] : 0.f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * N + col] = sum;
}

void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
    gemm<<<grid, block>>>(A, B, C, M, N, K);
}
