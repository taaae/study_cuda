// Exercise 01 harness — DO NOT EDIT.
// Builds data, runs your solve(), checks correctness, times it, prints metrics.
#include <vector>
#include <cmath>
#include <cstdio>
#include "cuda_utils.cuh"

#ifndef SOLUTION_FILE
#define SOLUTION_FILE "vadd.cu"
#endif
#include SOLUTION_FILE

int main() {
    print_device_banner();

    const int n = 1 << 24;                 // 16M elements
    const size_t bytes = (size_t)n * sizeof(float);

    std::vector<float> ha(n), hb(n), hc(n);
    for (int i = 0; i < n; ++i) {
        ha[i] = std::sin(i * 0.001f);
        hb[i] = std::cos(i * 0.002f);
    }

    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));
    CUDA_CHECK(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));

    solve(da, db, dc, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    double maxerr = 0.0;
    for (int i = 0; i < n; ++i) {
        double ref = (double)ha[i] + hb[i];
        double err = std::fabs(ref - hc[i]);
        maxerr = err > maxerr ? err : maxerr;
        if (err > 1e-4) ok = false;
    }
    report_correct(ok);
    report_metric("max_abs_err", maxerr);

    float ms = time_kernel([&] { solve(da, db, dc, n); });
    double gbps = 3.0 * bytes / (ms * 1e-3) / 1e9;   // read a, read b, write c
    report_metric("ms", ms);
    report_metric("gbps", gbps);
    report_metric("bw_frac", gbps / peak_bandwidth_gbps());

    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    return 0;
}
