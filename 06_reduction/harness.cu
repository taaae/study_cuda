// Exercise 06 harness — DO NOT EDIT.
// Builds data, runs your solve(), checks correctness, times it, prints metrics.
#include <vector>
#include <cmath>
#include <cstdio>
#include "cuda_utils.cuh"

#ifndef SOLUTION_FILE
#define SOLUTION_FILE "reduce.cu"
#endif
#include SOLUTION_FILE

int main() {
    print_device_banner();

    const int n = 1 << 24;                 // 16M elements
    const size_t bytes = (size_t)n * sizeof(float);

    std::vector<float> hin(n);
    for (int i = 0; i < n; ++i) {
        // Bounded, mixed-sign values keep the float sum well-conditioned.
        hin[i] = std::sin(i * 0.001f) * 0.5f + 0.25f;
    }

    // double-precision CPU reference sum.
    double ref = 0.0;
    for (int i = 0; i < n; ++i) ref += (double)hin[i];

    float *din, *dout;
    CUDA_CHECK(cudaMalloc(&din, bytes));
    CUDA_CHECK(cudaMalloc(&dout, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(din, hin.data(), bytes, cudaMemcpyHostToDevice));

    // Harness zeroes *out before calling solve().
    CUDA_CHECK(cudaMemset(dout, 0, sizeof(float)));
    solve(din, dout, n);
    CUDA_CHECK_KERNEL();

    float hsum = 0.f;
    CUDA_CHECK(cudaMemcpy(&hsum, dout, sizeof(float), cudaMemcpyDeviceToHost));

    double rel = std::fabs((double)hsum - ref) / std::fabs(ref);
    bool ok = rel < 1e-3;   // relative tolerance (float accumulation vs double)
    report_correct(ok);
    report_metric("rel_err", rel);
    report_metric("ref_sum", ref);

    // Time the full solve. *out must be re-zeroed each iteration since the
    // single-pass atomicAdd accumulates into it.
    float ms = time_kernel([&] {
        cudaMemset(dout, 0, sizeof(float));
        solve(din, dout, n);
    });
    double gbps = (double)bytes / (ms * 1e-3) / 1e9;   // read each element once
    report_metric("ms", ms);
    report_metric("gbps", gbps);
    report_metric("bw_frac", gbps / peak_bandwidth_gbps());

    CUDA_CHECK(cudaFree(din));
    CUDA_CHECK(cudaFree(dout));
    return 0;
}
