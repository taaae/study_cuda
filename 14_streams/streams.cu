// Exercise 14 — Streams & Copy/Compute Overlap
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

// Element-wise map: y = sqrt(x) * x + 1. Grid-stride so any chunk size works.
__global__ void map_kernel(const float* x, float* y, int n) {
    // TODO: with a grid-stride loop, set y = sqrt(x)*x + 1 for each element.
    //       (See README + hints.md.)
}

// Host entry point. h_in / h_out are PINNED host pointers of length n.
// Drive a chunked H2D -> kernel -> D2H pipeline across nStreams streams so
// that copies and compute overlap. You own all device allocation and freeing.
void solve(const float* h_in, float* h_out, int n, int nStreams) {
    // Scaffolding: a reasonable chunking. You may keep or change this.
    const int block = 256;
    int chunk = ceil_div(n, nStreams);   // elements per chunk (last one is smaller)

    // TODO: allocate device input and output buffers large enough for the whole
    //       array, and create nStreams streams.
    float* d_in  = nullptr;
    float* d_out = nullptr;

    // TODO: chunk the array across the streams and overlap copies with compute:
    //       per chunk, async-copy H2D, launch map_kernel, async-copy D2H, each on
    //       that chunk's stream (cudaMemcpyAsync). Mind the smaller last chunk.

    // TODO: synchronize, then destroy the streams and free the device buffers.
    //       (See README's function table and hints.md if stuck.)
}
