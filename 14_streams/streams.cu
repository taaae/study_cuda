// Exercise 14 — Streams & Copy/Compute Overlap
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

// Element-wise map: y = sqrt(x) * x + 1. Grid-stride so any chunk size works.
__global__ void map_kernel(const float* x, float* y, int n) {
    // TODO: grid-stride loop computing y[i] = sqrtf(x[i]) * x[i] + 1.0f
}

// Host entry point. h_in / h_out are PINNED host pointers of length n.
// Drive a chunked H2D -> kernel -> D2H pipeline across nStreams streams so
// that copies and compute overlap. You own all device allocation and freeing.
void solve(const float* h_in, float* h_out, int n, int nStreams) {
    // Scaffolding: a reasonable chunking. You may keep or change this.
    const int block = 256;
    int chunk = ceil_div(n, nStreams);   // elements per chunk (last one is smaller)

    // TODO: allocate ONE device buffer for inputs and ONE for outputs, each big
    //       enough to hold the whole array (so every chunk has a home).
    float* d_in  = nullptr;
    float* d_out = nullptr;

    // TODO: create nStreams streams (cudaStream_t / cudaStreamCreate).

    // TODO: for each chunk, on stream (i % nStreams):
    //         1. cudaMemcpyAsync the chunk H2D
    //         2. launch map_kernel<<<grid, block, 0, stream>>> on the chunk
    //         3. cudaMemcpyAsync the chunk D2H
    //       Use pointer offsets (h_in + off, d_in + off, ...) and the chunk's
    //       actual length (clamp the last chunk to n).

    // TODO: synchronize, then destroy the streams and free the device buffers.
}
