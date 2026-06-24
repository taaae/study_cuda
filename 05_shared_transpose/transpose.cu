// Exercise 05 — Tiled transpose with shared memory + bank-conflict avoidance.
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

#define TILE 32

// out = in^T for an n x n row-major matrix, staged through a shared tile so that
// BOTH the global read and the global write are coalesced. Pad the tile to avoid
// 32-way shared-memory bank conflicts on the transposed read.
__global__ void transpose(const float* in, float* out, int n) {
    // TODO: declare the padded shared tile. You MUST type the "+ 1" yourself:
    //       __shared__ float tile[TILE][TILE + 1];

    // TODO: coalesced LOAD from global into the tile.
    //   x_in = blockIdx.x*TILE + threadIdx.x;  y_in = blockIdx.y*TILE + threadIdx.y;
    //   if (x_in < n && y_in < n) tile[threadIdx.y][threadIdx.x] = in[y_in*n + x_in];

    // TODO: __syncthreads();  // make sure the whole tile is filled

    // TODO: coalesced STORE to the transposed block, reading the tile transposed.
    //   x_out = blockIdx.y*TILE + threadIdx.x;  y_out = blockIdx.x*TILE + threadIdx.y;
    //   if (x_out < n && y_out < n) out[y_out*n + x_out] = tile[threadIdx.x][threadIdx.y];
}

// Host entry point. in, out are DEVICE pointers to n*n floats, row-major.
void solve(const float* in, float* out, int n) {
    // TODO: dim3 block(TILE, TILE);
    // TODO: dim3 grid(ceil_div(n, TILE), ceil_div(n, TILE));
    // TODO: launch transpose<<<grid, block>>>(in, out, n);
}
