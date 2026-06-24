// Reference solution — Exercise 22 (Double-Buffered Tiled GEMM).
#include "cuda_utils.cuh"

#define TILE 32

__global__ void gemm_double_buffer(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    __shared__ float As[2][TILE][TILE];
    __shared__ float Bs[2][TILE][TILE];

    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    int numTiles = K / TILE;
    float acc = 0.f;

    // Prologue: load the first k-tile into buffer 0.
    As[0][ty][tx] = A[row * K + (0 * TILE + tx)];
    Bs[0][ty][tx] = B[(0 * TILE + ty) * N + col];
    __syncthreads();

    int cur = 0;
    for (int t = 0; t < numTiles; ++t) {
        int nxt = cur ^ 1;
        // Prefetch the next k-tile into the other buffer while we compute below.
        if (t + 1 < numTiles) {
            As[nxt][ty][tx] = A[row * K + ((t + 1) * TILE + tx)];
            Bs[nxt][ty][tx] = B[((t + 1) * TILE + ty) * N + col];
        }
        // Compute on the current buffer; the prefetch loads are in flight.
        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[cur][ty][k] * Bs[cur][k][tx];
        __syncthreads();   // next tile fully landed; safe to swap roles
        cur = nxt;
    }

    C[row * N + col] = acc;
}

void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, M / TILE);
    gemm_double_buffer<<<grid, block>>>(A, B, C, M, N, K);
}
