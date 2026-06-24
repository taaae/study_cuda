// Exercise 04 — Naive matrix transpose (global memory only), coalesced reads.
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
// Do NOT use __shared__ here — that is exercise 05.
#include "cuda_utils.cuh"

// Arrange indexing so consecutive threads (threadIdx.x) read CONSECUTIVE
// elements of `in` (coalesced reads). The writes to `out` will be strided.
__global__ void transpose(const float* in, float* out, int n) {
    // TODO: col = the contiguous dimension index (uses threadIdx.x) -> coalesced read of in
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: row = the other dimension index (uses threadIdx.y)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        out[row * n + col] = in[col * n + row];
    }
}

// Host entry point. in, out are DEVICE pointers to n*n floats, row-major.
// Launch with a 2-D grid of 2-D blocks covering the whole matrix.
void solve(const float* in, float* out, int n) {
    // TODO: choose a 2-D block (e.g. dim3 block(32, 8)).
    dim3 block(32, 8);
    // TODO: compute a 2-D grid that covers n in both dimensions.
    dim3 grid(ceil_div(n, block.x), ceil_div(n, block.y));
    // TODO: launch transpose<<<grid, block>>>(in, out, n);
    transpose<<<grid, block>>>(in, out, n);
}
