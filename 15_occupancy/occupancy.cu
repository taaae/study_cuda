// Exercise 15 — Occupancy & Launch-Config Tuning
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

// Memory-bound element-wise map: out[i] = in[i] * 2 + 1. Grid-stride.
__global__ void map_kernel(const float* in, float* out, int n) {
    // TODO: grid-stride loop computing out[i] = in[i] * 2.0f + 1.0f
}

// Host entry point. in / out are DEVICE pointers of length n.
// Use the occupancy API to choose blockSize AND minGridSize, then launch.
void solve(const float* in, float* out, int n) {
    int minGridSize = 0;
    int blockSize   = 0;

    // TODO: call cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
    //       map_kernel, 0, 0) to pick a launch config for map_kernel.

    // TODO: pick a grid. A grid-stride kernel covers all n with any grid; a
    //       sensible choice is min(minGridSize, ceil_div(n, blockSize)) so you
    //       never launch more blocks than there is work for.

    // TODO: launch map_kernel<<<grid, blockSize>>>(in, out, n);
}
