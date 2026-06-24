// Exercise 10 harness — DO NOT EDIT.
// Builds data, runs your solve(), checks correctness vs a CPU exclusive scan,
// times it, and prints metrics.
#include <vector>
#include <cstdio>
#include <cstdlib>
#include "cuda_utils.cuh"

#ifndef SOLUTION_FILE
#define SOLUTION_FILE "scan.cu"
#endif
#include SOLUTION_FILE

int main() {
    print_device_banner();

    const int n = 1 << 20;                  // 1,048,576 ints
    const size_t bytes = (size_t)n * sizeof(int);

    std::vector<int> hin(n), hout(n), ref(n);
    for (int i = 0; i < n; ++i) hin[i] = (i * 1103515245 + 12345) & 0x3f;  // small ints

    // CPU exclusive scan reference.
    long long acc = 0;
    for (int i = 0; i < n; ++i) { ref[i] = (int)acc; acc += hin[i]; }

    int *din, *dout;
    CUDA_CHECK(cudaMalloc(&din, bytes));
    CUDA_CHECK(cudaMalloc(&dout, bytes));
    CUDA_CHECK(cudaMemcpy(din, hin.data(), bytes, cudaMemcpyHostToDevice));

    solve(din, dout, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(hout.data(), dout, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    int firstBad = -1;
    for (int i = 0; i < n; ++i) {
        if (hout[i] != ref[i]) { ok = false; if (firstBad < 0) firstBad = i; }
    }
    report_correct(ok);
    if (!ok)
        std::printf("# first mismatch at i=%d: got=%d want=%d\n",
                    firstBad, hout[firstBad], ref[firstBad]);

    float ms = time_kernel([&] { solve(din, dout, n); });
    // Scan is multi-pass; ~2*bytes is a fair single-pass-equivalent estimate
    // (read input once, write output once).
    double gbps = 2.0 * bytes / (ms * 1e-3) / 1e9;
    report_metric("ms", ms);
    report_metric("gbps", gbps);
    report_metric("bw_frac", gbps / peak_bandwidth_gbps());

    CUDA_CHECK(cudaFree(din));
    CUDA_CHECK(cudaFree(dout));
    return 0;
}
