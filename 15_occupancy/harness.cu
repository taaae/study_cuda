// Exercise 15 harness — DO NOT EDIT.
// Builds data, runs your solve(), checks correctness, times it, prints metrics.
#include <vector>
#include <cmath>
#include <cstdio>
#include "cuda_utils.cuh"

#ifndef SOLUTION_FILE
#define SOLUTION_FILE "occupancy.cu"
#endif
#include SOLUTION_FILE

int main() {
    print_device_banner();

    const int n = 1 << 24;                 // 16M elements
    const size_t bytes = (size_t)n * sizeof(float);

    std::vector<float> h_in(n), h_out(n);
    for (int i = 0; i < n; ++i) h_in[i] = std::sin(i * 0.001f);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    solve(d_in, d_out, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    double maxerr = 0.0;
    for (int i = 0; i < n; ++i) {
        double ref = (double)h_in[i] * 2.0 + 1.0;
        double err = std::fabs(ref - h_out[i]);
        maxerr = err > maxerr ? err : maxerr;
        if (err > 1e-4) ok = false;
    }
    report_correct(ok);
    report_metric("max_abs_err", maxerr);

    float ms = time_kernel([&] { solve(d_in, d_out, n); });
    double gbps = 2.0 * bytes / (ms * 1e-3) / 1e9;   // read in, write out
    report_metric("ms", ms);
    report_metric("gbps", gbps);
    report_metric("bw_frac", gbps / peak_bandwidth_gbps());

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
