// Exercise 03 — Benchmarking a copy kernel with CUDA events.
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls benchmark_copy().
#include "cuda_utils.cuh"

// Grid-stride copy: out[i] = in[i] for all i.
__global__ void copy_kernel(const float* in, float* out, int n) {
    // TODO: grid-stride loop copying in[i] -> out[i].
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;
    for (int i = start_idx; i < n; i += grid_stride) {
        out[i] = in[i];
    }
}

// Launch copy_kernel, TIME IT YOURSELF with cudaEvents, and return the best
// elapsed milliseconds over `iters` runs (after one warmup launch).
// On return, `out` must equal `in` (a correct copy happened last).
// Do NOT use the library time_kernel here — write the event code yourself.
float benchmark_copy(const float* in, float* out, int n, int iters) {

    // Get SM num, select dims
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int sms = prop.multiProcessorCount;
    dim3 block(256);
    dim3 grid(sms * 32);
    // int block = 256;
    // int grid  = 0;  // TODO: pick a grid (grid-stride means it need not equal ceil_div(n, block))

    // TODO: create two cudaEvents (start, stop).
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // TODO: one warmup launch (and a sync) — its time is discarded.
    copy_kernel<<<grid, block>>>(in, out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    float best = 1e30f;
    for (int i = 0; i < iters; ++i) {
        // TODO: record 'start', launch copy_kernel, record 'stop'.
        cudaEventRecord(start);
        // CUDA_CHECK(cudaEventSynchronize(start));
        copy_kernel<<<grid, block>>>(in, out, n);
        // CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(stop);
        CUDA_CHECK(cudaEventSynchronize(stop));
        // TODO: synchronize on 'stop', then cudaEventElapsedTime into a float ms.
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        // TODO: keep the minimum ms in `best`.
        if (ms < best) {
            best = ms;
        }
    }

    // TODO: destroy the events.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    (void)block; (void)grid;
    return best;
}
