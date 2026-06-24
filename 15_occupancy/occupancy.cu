// Exercise 15 — Occupancy & Launch-Config Tuning
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

// Memory-bound element-wise map: out[i] = in[i] * 2 + 1. Grid-stride.
__global__ void map_kernel(const float* in, float* out, int n) {
    // TODO: with a grid-stride loop, set out = in*2 + 1 for each element.
    //       (See README + hints.md.)
}

// Host entry point. in / out are DEVICE pointers of length n.
// Use the occupancy API to choose blockSize AND minGridSize, then launch.
void solve(const float* in, float* out, int n) {
    int minGridSize = 0;
    int blockSize   = 0;

    // TODO: let the occupancy API pick blockSize and minGridSize for map_kernel
    //       (cudaOccupancyMaxPotentialBlockSize), then choose a grid that covers n
    //       without launching more blocks than there is work for, and launch.
    //       (See README's function table and hints.md if stuck.)
}
