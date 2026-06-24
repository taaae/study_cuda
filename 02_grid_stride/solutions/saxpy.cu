// Reference solution — Exercise 02.
#include "cuda_utils.cuh"

__global__ void saxpy(float a, const float* x, float* y, int n) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

void solve(float a, const float* x, float* y, int n) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int block = 256;
    int grid  = prop.multiProcessorCount * 32;   // a few blocks per SM, independent of n
    saxpy<<<grid, block>>>(a, x, y, n);
}
