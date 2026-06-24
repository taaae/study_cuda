// Exercise 02 — Grid-Stride SAXPY (y = a*x + y)
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

// Each thread strides across the whole array, processing many elements.
// y[i] = a*x[i] + y[i] for every i this thread visits.
__global__ void saxpy(float a, const float* x, float* y, int n) {
    // TODO: compute this thread's starting global index.
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: compute the grid stride = total number of threads in the grid.
    int grid_stride = gridDim.x * blockDim.x;
    // TODO: loop i from start to n in steps of stride, doing y[i] = a*x[i] + y[i].
    for (int i = start_idx; i < n; i += grid_stride) {
        y[i] = a * x[i] + y[i];
    }
}

// Host entry point. x, y are DEVICE pointers of length n; a is a host scalar.
// Launch a FIXED, modest grid sized from the device SM count (NOT ceil_div(n, block)),
// so each thread handles multiple elements via the grid-stride loop.
void solve(float a, const float* x, float* y, int n) {
    // TODO: query the device with cudaGetDeviceProperties (wrap it in CUDA_CHECK).
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    // TODO: pick a block size and a grid of a few blocks per SM (independent of n).
    int sms = p.multiProcessorCount;
    dim3 block(256);
    dim3 grid(sms * 4);
    // TODO: launch saxpy<<<grid, block>>>(a, x, y, n);
    saxpy<<<grid, block>>>(a, x, y, n);
}
