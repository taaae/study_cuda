// Exercise 03 harness — DO NOT EDIT.
// Builds data, runs your benchmark_copy(), checks the copy and your timing, prints metrics.
#include <vector>
#include <cmath>
#include <cstdio>
#include "cuda_utils.cuh"

#ifndef SOLUTION_FILE
#define SOLUTION_FILE "bench.cu"
#endif
#include SOLUTION_FILE

int main() {
    print_device_banner();

    const int n = 1 << 25;                 // 33M elements
    const size_t bytes = (size_t)n * sizeof(float);
    const int iters = 50;

    std::vector<float> hin(n), hout(n);
    for (int i = 0; i < n; ++i) hin[i] = std::sin(i * 0.001f);

    float *din, *dout;
    CUDA_CHECK(cudaMalloc(&din, bytes));
    CUDA_CHECK(cudaMalloc(&dout, bytes));
    CUDA_CHECK(cudaMemcpy(din, hin.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dout, 0, bytes));   // ensure out starts wrong

    // The student's own timing run. Must also leave dout == din.
    float student_ms = benchmark_copy(din, dout, n, iters);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(hout.data(), dout, bytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    double maxerr = 0.0;
    for (int i = 0; i < n; ++i) {
        double err = std::fabs((double)hin[i] - hout[i]);
        maxerr = err > maxerr ? err : maxerr;
        if (err != 0.0) ok = false;
    }
    report_correct(ok);
    report_metric("max_abs_err", maxerr);

    // Independent measurement with the library timer for cross-checking.
    int block = 256;
    int grid  = ceil_div(n, block);
    float harness_ms = time_kernel([&] { copy_kernel<<<grid, block>>>(din, dout, n); });

    report_metric("student_ms", student_ms);
    report_metric("harness_ms", harness_ms);
    report_metric("ms_ratio", student_ms / harness_ms);

    double gbps = 2.0 * bytes / (student_ms * 1e-3) / 1e9;   // read in, write out
    report_metric("gbps", gbps);
    report_metric("bw_frac", gbps / peak_bandwidth_gbps());

    CUDA_CHECK(cudaFree(din));
    CUDA_CHECK(cudaFree(dout));
    return 0;
}
