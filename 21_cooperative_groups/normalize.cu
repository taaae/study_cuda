// Exercise 21 — Cooperative Groups & Grid-Wide Sync
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"
// TODO: include the cooperative groups header and alias its namespace.

// Normalize `data` to unit L2 norm in ONE launch.
//   Phase 1: each thread accumulates data[i]*data[i] over its grid-stride range
//            into a per-block partial, then atomicAdd the partial into *ssq.
//   grid.sync(): wait for ALL blocks, so *ssq is now the complete sum of squares.
//   Phase 2: inv = rsqrtf(*ssq); every thread scales its elements by inv.
__global__ void normalize_kernel(float* data, int n, float* ssq) {
    // TODO: get this grid's cg::grid_group handle for the grid-wide barrier.

    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // --- Phase 1: partial sum of squares ---
    float local = 0.f;
    // TODO: grid-stride accumulate data[i]*data[i] into `local`, then reduce
    //       `local` across the block (shared memory) and have thread 0
    //       atomicAdd the block's sum into *ssq.

    // --- Grid-wide barrier ---
    // TODO: grid.sync() so *ssq holds the complete sum across all blocks.

    // --- Phase 2: scale ---
    // TODO: compute the inverse norm with rsqrtf(*ssq), then grid-stride scale
    //       every element of `data` by it.
    // (See README's function table and hints.md if stuck.)
}

// Host entry point. `data` is a DEVICE pointer of length n; normalize in place.
void solve(float* data, int n) {
    int block = 256;

    // TODO: size the grid so every block is co-resident (required for grid.sync
    //       to be safe): query the occupancy-limited blocks-per-SM and multiply
    //       by the device's SM count.
    int grid = 1;  // <-- replace with the co-resident grid size

    // Global accumulator for the sum of squares.
    float* ssq = nullptr;
    CUDA_CHECK(cudaMalloc(&ssq, sizeof(float)));
    CUDA_CHECK(cudaMemset(ssq, 0, sizeof(float)));

    // TODO: launch the kernel cooperatively with cudaLaunchCooperativeKernel
    //       (build the argument array it expects). (See README + hints.md.)
    (void)grid;

    CUDA_CHECK(cudaFree(ssq));
}
