// Reference solution — Exercise 15.
#include "cuda_utils.cuh"

__global__ void map_kernel(const float* in, float* out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        out[i] = in[i] * 2.0f + 1.0f;
    }
}

void solve(const float* in, float* out, int n) {
    int minGridSize = 0, blockSize = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, map_kernel, 0, 0));
    int grid = ceil_div(n, blockSize);
    if (grid > minGridSize) grid = minGridSize;
    if (grid < 1) grid = 1;
    map_kernel<<<grid, blockSize>>>(in, out, n);
}
