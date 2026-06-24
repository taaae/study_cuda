// Exercise 03 — Benchmarking a copy kernel with CUDA events.
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls benchmark_copy().
#include "cuda_utils.cuh"

// Grid-stride copy: out[i] = in[i] for all i.
__global__ void copy_kernel(const float* in, float* out, int n) {
    // TODO: grid-stride loop copying in[i] -> out[i].
}

// Launch copy_kernel, TIME IT YOURSELF with cudaEvents, and return the best
// elapsed milliseconds over `iters` runs (after one warmup launch).
// On return, `out` must equal `in` (a correct copy happened last).
// Do NOT use the library time_kernel here — write the event code yourself.
float benchmark_copy(const float* in, float* out, int n, int iters) {
    int block = 256;
    int grid  = 0;  // TODO: pick a grid (grid-stride means it need not equal ceil_div(n, block))

    // TODO: create two cudaEvents (start, stop).

    // TODO: one warmup launch (and a sync) — its time is discarded.

    float best = 1e30f;
    for (int i = 0; i < iters; ++i) {
        // TODO: record 'start', launch copy_kernel, record 'stop'.
        // TODO: synchronize on 'stop', then cudaEventElapsedTime into a float ms.
        // TODO: keep the minimum ms in `best`.
    }

    // TODO: destroy the events.

    (void)block; (void)grid;
    return best;
}
