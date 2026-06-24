// Exercise 04 — Naive matrix transpose (global memory only), coalesced reads.
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
// Do NOT use __shared__ here — that is exercise 05.
#include "cuda_utils.cuh"

// Arrange indexing so consecutive threads (threadIdx.x) read CONSECUTIVE
// elements of `in` (coalesced reads). The writes to `out` will be strided.
__global__ void transpose(const float* in, float* out, int n) {
    // TODO: map this thread to a (row, col) so that threadIdx.x drives the contiguous
    //       column (coalesced read of `in`); guard against the matrix bounds, then
    //       write the transposed element to `out`. (See README + hints.md.)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        out[col * n + row] = in[row * n + col];
    }
}

// Host entry point. in, out are DEVICE pointers to n*n floats, row-major.
// Launch with a 2-D grid of 2-D blocks covering the whole matrix.
void solve(const float* in, float* out, int n) {
    // TODO: pick a 2-D block and a 2-D grid that covers n in both dimensions, then
    //       launch transpose. (See README + hints.md.)
    dim3 block(32, 8);
    dim3 grid(ceil_div(n, block.x), ceil_div(n, block.y));
    transpose<<<grid, block>>>(in, out, n);
}
