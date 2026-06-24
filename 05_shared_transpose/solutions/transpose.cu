// Reference solution — Exercise 05 (shared-memory tiled transpose, padded tile).
#include "cuda_utils.cuh"

#define TILE 32

__global__ void transpose(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE + 1];   // +1 padding avoids bank conflicts

    int x_in = blockIdx.x * TILE + threadIdx.x;
    int y_in = blockIdx.y * TILE + threadIdx.y;
    if (x_in < n && y_in < n)
        tile[threadIdx.y][threadIdx.x] = in[y_in * n + x_in];   // coalesced read

    __syncthreads();

    int x_out = blockIdx.y * TILE + threadIdx.x;
    int y_out = blockIdx.x * TILE + threadIdx.y;
    if (x_out < n && y_out < n)
        out[y_out * n + x_out] = tile[threadIdx.x][threadIdx.y]; // coalesced write
}

void solve(const float* in, float* out, int n) {
    dim3 block(TILE, TILE);
    dim3 grid(ceil_div(n, TILE), ceil_div(n, TILE));
    transpose<<<grid, block>>>(in, out, n);
}
