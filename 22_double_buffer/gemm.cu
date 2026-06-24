// Exercise 22 — Double-Buffered Tiled GEMM
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

#define TILE 32

// C = A * B, row-major. A is MxK, B is KxN, C is MxN. Dimensions are multiples
// of TILE. Use TWO shared tile buffers in ping-pong fashion across the K-loop.
__global__ void gemm_double_buffer(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    // TODO: two shared buffers (the [2] is the ping-pong dimension):
    //   __shared__ float As[2][TILE][TILE];
    //   __shared__ float Bs[2][TILE][TILE];

    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    int numTiles = K / TILE;
    float acc = 0.f;

    // TODO (prologue): load k-tile 0 into buffer 0, then __syncthreads().
    //   As[0][ty][tx] = A[row*K + (0*TILE + tx)];
    //   Bs[0][ty][tx] = B[(0*TILE + ty)*N + col];

    // TODO (main loop): ping-pong.
    //   int cur = 0;
    //   for (int t = 0; t < numTiles; ++t) {
    //       int nxt = cur ^ 1;
    //       if (t + 1 < numTiles) {
    //           // prefetch next k-tile into buffer `nxt`
    //           As[nxt][ty][tx] = A[row*K + ((t+1)*TILE + tx)];
    //           Bs[nxt][ty][tx] = B[((t+1)*TILE + ty)*N + col];
    //       }
    //       // compute on the CURRENT buffer while the prefetch is in flight
    //       for (int k = 0; k < TILE; ++k) acc += As[cur][ty][k] * Bs[cur][k][tx];
    //       __syncthreads();
    //       cur = nxt;
    //   }

    (void)numTiles; (void)acc; (void)row; (void)col;
    // TODO: C[row*N + col] = acc;
}

void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: launch with TILE x TILE blocks over the C grid.
    //   dim3 block(TILE, TILE);
    //   dim3 grid(N / TILE, M / TILE);
    //   gemm_double_buffer<<<grid, block>>>(A, B, C, M, N, K);
}
