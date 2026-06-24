// Exercise 22 — Double-Buffered Tiled GEMM
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

#define TILE 32

// C = A * B, row-major. A is MxK, B is KxN, C is MxN. Dimensions are multiples
// of TILE. Use TWO shared tile buffers in ping-pong fashion across the K-loop.
__global__ void gemm_double_buffer(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    // TODO: declare two shared TILE x TILE buffers each for A and B (the extra
    //       dimension is the ping-pong index).

    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    int numTiles = K / TILE;
    float acc = 0.f;

    // TODO (prologue): load k-tile 0 into the first buffer, then __syncthreads().

    // TODO (main loop): ping-pong over the K-tiles. On each iteration, prefetch
    //       the NEXT k-tile into the other buffer (if any) while you accumulate
    //       acc from the CURRENT buffer, __syncthreads(), then swap buffers.

    (void)numTiles; (void)acc; (void)row; (void)col;
    // TODO: write acc back to the C element this thread owns.
    // (See README's function table and hints.md if stuck.)
}

void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: launch with TILE x TILE blocks covering the C grid. (See README + hints.md.)
}
