// Exercise 11 harness — DO NOT EDIT.
// Builds data, runs your solve(), checks correctness vs a CPU histogram, times
// it against a naive global-atomic baseline, and prints metrics.
#include <vector>
#include <cstdio>
#include <cstdlib>
#include "cuda_utils.cuh"

#ifndef SOLUTION_FILE
#define SOLUTION_FILE "histogram.cu"
#endif
#include SOLUTION_FILE

#define HARNESS_NBINS 256

// Naive baseline: every element atomics straight into the global histogram.
__global__ void hist_naive(const unsigned char* data, unsigned int* hist, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        atomicAdd(&hist[data[i]], 1u);
    }
}

int main() {
    print_device_banner();

    const int n = 1 << 24;                  // 16M bytes
    std::vector<unsigned char> hdata(n);
    unsigned int ref[HARNESS_NBINS] = {0};
    unsigned x = 123456789u;
    for (int i = 0; i < n; ++i) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;   // xorshift
        unsigned char v = (unsigned char)(x & 0xff);
        hdata[i] = v;
        ref[v]++;
    }

    unsigned char* ddata;
    unsigned int *dhist, *dhist_naive;
    CUDA_CHECK(cudaMalloc(&ddata, (size_t)n));
    CUDA_CHECK(cudaMalloc(&dhist, HARNESS_NBINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dhist_naive, HARNESS_NBINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(ddata, hdata.data(), (size_t)n, cudaMemcpyHostToDevice));

    // Run student solve (harness zeroes the histogram first, per contract).
    CUDA_CHECK(cudaMemset(dhist, 0, HARNESS_NBINS * sizeof(unsigned int)));
    solve(ddata, dhist, n);
    CUDA_CHECK_KERNEL();

    unsigned int hout[HARNESS_NBINS];
    CUDA_CHECK(cudaMemcpy(hout, dhist, HARNESS_NBINS * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int b = 0; b < HARNESS_NBINS; ++b)
        if (hout[b] != ref[b]) ok = false;
    report_correct(ok);

    // Time student solve (re-zero each iteration so counts don't accumulate).
    float ms = time_kernel([&] {
        CUDA_CHECK(cudaMemset(dhist, 0, HARNESS_NBINS * sizeof(unsigned int)));
        solve(ddata, dhist, n);
    });

    // Time naive baseline the same way.
    float ms_naive = time_kernel([&] {
        CUDA_CHECK(cudaMemset(dhist_naive, 0, HARNESS_NBINS * sizeof(unsigned int)));
        hist_naive<<<1024, 256>>>(ddata, dhist_naive, n);
    });

    report_metric("ms", ms);
    report_metric("ms_naive", ms_naive);
    report_metric("speedup", ms_naive / ms);

    CUDA_CHECK(cudaFree(ddata));
    CUDA_CHECK(cudaFree(dhist));
    CUDA_CHECK(cudaFree(dhist_naive));
    return 0;
}
