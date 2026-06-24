// Exercise 02 — Grid-Stride SAXPY (y = a*x + y)
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

// Each thread strides across the whole array, processing many elements.
// y[i] = a*x[i] + y[i] for every i this thread visits.
__global__ void saxpy(float a, const float* x, float* y, int n) {
    // TODO: use a grid-stride loop so each thread processes every element at its
    //       global index plus multiples of the total grid size, computing
    //       y[i] = a*x[i] + y[i]. (See README + hints.md.)
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = gridDim.x * blockDim.x;
    for (int i = start_idx; i < n; i += grid_stride) {
        y[i] = fmaf(a, x[i], y[i]);
    }
}

// Host entry point. x, y are DEVICE pointers of length n; a is a host scalar.
// Launch a FIXED, modest grid sized from the device SM count (NOT ceil_div(n, block)),
// so each thread handles multiple elements via the grid-stride loop.
void solve(float a, const float* x, float* y, int n) {
    // TODO: query the device (cudaGetDeviceProperties) for its SM count, then size a
    //       fixed, modest grid from it (a few blocks per SM, independent of n) and
    //       launch saxpy. (See README + hints.md.)
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    int sms = p.multiProcessorCount;
    dim3 block(256);
    dim3 grid(sms * 1);
    saxpy<<<grid, block>>>(a, x, y, n);
}
