// Exercise 12 harness — DO NOT EDIT.
// Builds data, runs your solve(), checks count / compacted set / sum-of-squares
// vs a CPU reference, times it, and prints metrics.
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "cuda_utils.cuh"

#ifndef SOLUTION_FILE
#define SOLUTION_FILE "compact.cu"
#endif
#include SOLUTION_FILE

int main() {
    print_device_banner();

    const int n = 1 << 22;                  // ~4M floats
    const float threshold = 0.0f;
    const size_t bytes = (size_t)n * sizeof(float);

    std::vector<float> hin(n);
    for (int i = 0; i < n; ++i) hin[i] = std::sin(i * 0.0007f) + std::cos(i * 0.013f);

    // CPU reference: compacted set, count, and sum of squares.
    std::vector<float> ref;
    ref.reserve(n);
    double ref_ss = 0.0;
    for (int i = 0; i < n; ++i)
        if (hin[i] > threshold) { ref.push_back(hin[i]); ref_ss += (double)hin[i] * hin[i]; }
    int ref_count = (int)ref.size();

    float *din, *dcomp, *dsumsq;
    int *dcount;
    CUDA_CHECK(cudaMalloc(&din, bytes));
    CUDA_CHECK(cudaMalloc(&dcomp, bytes));
    CUDA_CHECK(cudaMalloc(&dcount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dsumsq, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(din, hin.data(), bytes, cudaMemcpyHostToDevice));

    solve(din, n, threshold, dcomp, dcount, dsumsq);
    CUDA_CHECK_KERNEL();

    int hcount = 0;
    float hss = 0.f;
    CUDA_CHECK(cudaMemcpy(&hcount, dcount, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&hss, dsumsq, sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = (hcount == ref_count);

    // sum of squares within relative tolerance
    double ss_err = std::fabs((double)hss - ref_ss) / (ref_ss + 1e-9);
    if (ss_err > 1e-3) ok = false;

    // compacted set: copy_if preserves order, so compare element-by-element.
    if (ok) {
        std::vector<float> hcomp(hcount);
        if (hcount > 0)
            CUDA_CHECK(cudaMemcpy(hcomp.data(), dcomp, (size_t)hcount * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        for (int i = 0; i < hcount; ++i)
            if (std::fabs(hcomp[i] - ref[i]) > 1e-5f) { ok = false; break; }
    }

    report_correct(ok);
    report_metric("count", hcount);
    report_metric("ss_rel_err", ss_err);

    float ms = time_kernel([&] {
        solve(din, n, threshold, dcomp, dcount, dsumsq);
    });
    report_metric("ms", ms);

    CUDA_CHECK(cudaFree(din));
    CUDA_CHECK(cudaFree(dcomp));
    CUDA_CHECK(cudaFree(dcount));
    CUDA_CHECK(cudaFree(dsumsq));
    return 0;
}
