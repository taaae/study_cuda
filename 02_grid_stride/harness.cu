// Exercise 02 harness — DO NOT EDIT.
// Builds data, runs your solve(), checks correctness, times it, prints metrics.
#include <vector>
#include <cmath>
#include <cstdio>
#include "cuda_utils.cuh"

#ifndef SOLUTION_FILE
#define SOLUTION_FILE "saxpy.cu"
#endif
#include SOLUTION_FILE

int main() {
    print_device_banner();

    const int n = 1 << 25;                 // 33M elements
    const size_t bytes = (size_t)n * sizeof(float);
    const float a = 2.5f;

    std::vector<float> hx(n), hy(n), hy0(n);
    for (int i = 0; i < n; ++i) {
        hx[i]  = std::sin(i * 0.001f);
        hy[i]  = std::cos(i * 0.002f);
        hy0[i] = hy[i];                    // remember the original y
    }

    float *dx, *dy;
    CUDA_CHECK(cudaMalloc(&dx, bytes));
    CUDA_CHECK(cudaMalloc(&dy, bytes));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, hy.data(), bytes, cudaMemcpyHostToDevice));

    solve(a, dx, dy, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(hy.data(), dy, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    double maxerr = 0.0;
    for (int i = 0; i < n; ++i) {
        double ref = (double)a * hx[i] + hy0[i];
        double err = std::fabs(ref - hy[i]);
        maxerr = err > maxerr ? err : maxerr;
        if (err > 1e-3) ok = false;
    }
    report_correct(ok);
    report_metric("max_abs_err", maxerr);

    // Time it. Each timed call re-reads the same inputs from device memory.
    // (We reset dy first so repeated timing runs stay numerically sane.)
    CUDA_CHECK(cudaMemcpy(dy, hy0.data(), bytes, cudaMemcpyHostToDevice));
    float ms = time_kernel([&] { solve(a, dx, dy, n); });
    double gbps = 3.0 * bytes / (ms * 1e-3) / 1e9;   // read x, read y, write y
    report_metric("ms", ms);
    report_metric("gbps", gbps);
    report_metric("bw_frac", gbps / peak_bandwidth_gbps());

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    return 0;
}
