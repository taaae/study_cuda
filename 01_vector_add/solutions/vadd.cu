// Reference solution — Exercise 01.
#include "cuda_utils.cuh"

__global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

void solve(const float* a, const float* b, float* c, int n) {
    int block = 256;
    int grid  = ceil_div(n, block);
    vadd<<<grid, block>>>(a, b, c, n);
}
