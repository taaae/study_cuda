// Reference solution — Exercise 04 (coalesced reads, no shared memory).
#include "cuda_utils.cuh"

__global__ void transpose(const float* in, float* out, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // coalesced read of in
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        out[col * n + row] = in[row * n + col];
    }
}

void solve(const float* in, float* out, int n) {
    dim3 block(32, 8);
    dim3 grid(ceil_div(n, block.x), ceil_div(n, block.y));
    transpose<<<grid, block>>>(in, out, n);
}
