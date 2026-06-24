// Exercise 14 harness — DO NOT EDIT.
// Builds data in PINNED memory, runs your solve(), checks correctness, and times
// it end-to-end (copies included) against a single-stream synchronous baseline.
#include <vector>
#include <cmath>
#include <cstdio>
#include "cuda_utils.cuh"

#ifndef SOLUTION_FILE
#define SOLUTION_FILE "streams.cu"
#endif
#include SOLUTION_FILE

// Single-stream synchronous baseline: one big H2D, one kernel, one big D2H.
static float baseline_ms(const float* h_in, float* h_out, int n) {
    const size_t bytes = (size_t)n * sizeof(float);
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    const int block = 256;
    const int grid  = ceil_div(n, block);

    auto run = [&] {
        CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
        map_kernel<<<grid, block>>>(d_in, d_out, n);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    };
    float ms = time_kernel(run, 10);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return ms;
}

int main() {
    print_device_banner();

    const int n = 1 << 24;                 // 16M elements (~64 MB per buffer)
    const size_t bytes = (size_t)n * sizeof(float);
    const int nStreams = 4;

    // Pinned host memory is REQUIRED for cudaMemcpyAsync to overlap.
    float *h_in, *h_out;
    CUDA_CHECK(cudaMallocHost(&h_in, bytes));
    CUDA_CHECK(cudaMallocHost(&h_out, bytes));
    for (int i = 0; i < n; ++i) h_in[i] = (float)(i % 1000) * 0.01f;

    // Correctness: run solve once and compare to the CPU map.
    solve(h_in, h_out, n, nStreams);
    CUDA_CHECK_KERNEL();

    bool ok = true;
    double maxerr = 0.0;
    for (int i = 0; i < n; ++i) {
        double ref = std::sqrt((double)h_in[i]) * h_in[i] + 1.0;
        double err = std::fabs(ref - h_out[i]);
        maxerr = err > maxerr ? err : maxerr;
        if (err > 1e-3) ok = false;
    }
    report_correct(ok);
    report_metric("max_abs_err", maxerr);

    // End-to-end timing (copies included) of the streamed solve.
    float ms = time_kernel([&] { solve(h_in, h_out, n, nStreams); }, 10);
    report_metric("ms", ms);

    // Baseline: single-stream synchronous version.
    float base = baseline_ms(h_in, h_out, n);
    report_metric("baseline_ms", base);
    report_metric("speedup", base / ms);

    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    return 0;
}
