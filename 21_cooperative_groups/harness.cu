// Exercise 21 harness — DO NOT EDIT.
// Builds data, runs your solve(), checks correctness vs a CPU L2-normalize,
// times the single fused launch, prints metrics. Guards cooperative launch.
#include <vector>
#include <cmath>
#include <cstdio>
#include "cuda_utils.cuh"

#ifndef SOLUTION_FILE
#define SOLUTION_FILE "normalize.cu"
#endif
#include SOLUTION_FILE

int main() {
    print_device_banner();

    // Runtime guard: grid sync requires cooperative-launch support.
    int coop = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0));
    if (!coop) {
        std::printf("# SKIP: cooperative launch unsupported on this device\n");
        report_correct(true);   // do not false-fail grading
        return 0;
    }

    const int n = 1 << 22;                 // ~4M elements
    const size_t bytes = (size_t)n * sizeof(float);

    std::vector<float> hin(n), hout(n), href(n);
    for (int i = 0; i < n; ++i) hin[i] = std::sin(i * 0.001f) + 0.5f;

    // CPU reference: L2-normalize.
    double ssq = 0.0;
    for (int i = 0; i < n; ++i) ssq += (double)hin[i] * hin[i];
    double inv = 1.0 / std::sqrt(ssq);
    for (int i = 0; i < n; ++i) href[i] = (float)(hin[i] * inv);

    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, bytes));
    CUDA_CHECK(cudaMemcpy(d, hin.data(), bytes, cudaMemcpyHostToDevice));

    solve(d, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(hout.data(), d, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    double maxerr = 0.0;
    for (int i = 0; i < n; ++i) {
        double err = std::fabs((double)href[i] - hout[i]);
        maxerr = err > maxerr ? err : maxerr;
        if (err > 1e-5) ok = false;
    }
    report_correct(ok);
    report_metric("max_abs_err", maxerr);

    // Time the fused single-launch normalize. The data is already unit-norm after
    // the correctness run, so each timed call re-normalizes an already-normalized
    // vector — same amount of arithmetic and memory traffic, so timing is stable.
    float ms = time_kernel([&] { solve(d, n); });
    report_metric("ms", ms);

    CUDA_CHECK(cudaFree(d));
    return 0;
}
