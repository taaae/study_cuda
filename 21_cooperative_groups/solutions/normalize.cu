// Reference solution — Exercise 21 (Cooperative Groups & Grid-Wide Sync).
#include "cuda_utils.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void normalize_kernel(float* data, int n, float* ssq) {
    cg::grid_group grid = cg::this_grid();

    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // --- Phase 1: partial sum of squares (grid-stride) ---
    float local = 0.f;
    for (int i = tid; i < n; i += stride) local += data[i] * data[i];

    // Block reduction in shared memory.
    __shared__ float sm[256];
    sm[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(ssq, sm[0]);

    // --- Grid-wide barrier: every block has finished its atomicAdd ---
    grid.sync();

    // --- Phase 2: scale by 1/sqrt(total) ---
    float inv = rsqrtf(*ssq);
    for (int i = tid; i < n; i += stride) data[i] *= inv;
}

void solve(float* data, int n) {
    int block = 256;

    int blocksPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM, normalize_kernel, block, 0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int grid = blocksPerSM * prop.multiProcessorCount;
    if (grid < 1) grid = 1;

    float* ssq = nullptr;
    CUDA_CHECK(cudaMalloc(&ssq, sizeof(float)));
    CUDA_CHECK(cudaMemset(ssq, 0, sizeof(float)));

    void* args[] = { (void*)&data, (void*)&n, (void*)&ssq };
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)normalize_kernel,
                                           dim3(grid), dim3(block),
                                           args, 0, 0));
    CUDA_CHECK(cudaFree(ssq));
}
