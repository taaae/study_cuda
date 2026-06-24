// Exercise 21 — Cooperative Groups & Grid-Wide Sync
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"
// TODO: include the cooperative groups header and alias the namespace:
//   #include <cooperative_groups.h>
//   namespace cg = cooperative_groups;

// Normalize `data` to unit L2 norm in ONE launch.
//   Phase 1: each thread accumulates data[i]*data[i] over its grid-stride range
//            into a per-block partial, then atomicAdd the partial into *ssq.
//   grid.sync(): wait for ALL blocks, so *ssq is now the complete sum of squares.
//   Phase 2: inv = rsqrtf(*ssq); every thread scales its elements by inv.
__global__ void normalize_kernel(float* data, int n, float* ssq) {
    // TODO: cg::grid_group grid = cg::this_grid();

    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // --- Phase 1: partial sum of squares ---
    float local = 0.f;
    // TODO: for (int i = tid; i < n; i += stride) local += data[i] * data[i];

    // Reduce `local` across the block into one value, then atomicAdd into *ssq.
    // A shared-memory block reduction is fine here.
    // TODO: __shared__ float sm[...]; reduce; if (threadIdx.x == 0) atomicAdd(ssq, blockSum);

    // --- Grid-wide barrier ---
    // TODO: grid.sync();

    // --- Phase 2: scale ---
    // TODO: float inv = rsqrtf(*ssq);
    // TODO: for (int i = tid; i < n; i += stride) data[i] *= inv;
}

// Host entry point. `data` is a DEVICE pointer of length n; normalize in place.
void solve(float* data, int n) {
    int block = 256;

    // TODO: query the occupancy-limited blocks-per-SM and multiply by SM count so
    //       every block is co-resident (required for grid.sync to be safe):
    //   int blocksPerSM = 0;
    //   cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, normalize_kernel, block, 0);
    //   cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    //   int grid = blocksPerSM * prop.multiProcessorCount;
    int grid = 1;  // <-- replace with the co-resident grid size

    // Global accumulator for the sum of squares.
    float* ssq = nullptr;
    CUDA_CHECK(cudaMalloc(&ssq, sizeof(float)));
    CUDA_CHECK(cudaMemset(ssq, 0, sizeof(float)));

    // TODO: build the argument array and launch cooperatively:
    //   void* args[] = { &data, &n, &ssq };
    //   cudaLaunchCooperativeKernel((void*)normalize_kernel,
    //                               dim3(grid), dim3(block), args, 0, 0);
    (void)grid;

    CUDA_CHECK(cudaFree(ssq));
}
