// Reference solution — Exercise 03.
#include "cuda_utils.cuh"

__global__ void copy_kernel(const float* in, float* out, int n) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) out[i] = in[i];
}

float benchmark_copy(const float* in, float* out, int n, int iters) {
    int block = 256;
    int grid  = ceil_div(n, block);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warmup (discarded)
    copy_kernel<<<grid, block>>>(in, out, n);
    cudaDeviceSynchronize();

    float best = 1e30f;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start);
        copy_kernel<<<grid, block>>>(in, out, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        if (ms < best) best = ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return best;
}
