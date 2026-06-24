// Exercise 07 harness — DO NOT EDIT.
// Builds data, runs your solve(), checks correctness, times it vs a shared-only
// baseline reduction, prints metrics.
#include <vector>
#include <cmath>
#include <cstdio>
#include "cuda_utils.cuh"

#ifndef SOLUTION_FILE
#define SOLUTION_FILE "reduce.cu"
#endif
#include SOLUTION_FILE

// ---- Baseline: exercise-06-style shared-only reduction (for speedup) -------
#ifndef BASE_BLOCK
#define BASE_BLOCK 256
#endif
__global__ void base_reduce(const float* in, float* out, int n) {
    __shared__ float sdata[BASE_BLOCK];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    float sum = 0.f;
    for (int i = gid; i < n; i += stride) sum += in[i];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}
static void base_solve(const float* in, float* out, int n) {
    int grid = ceil_div(n, BASE_BLOCK);
    if (grid > 4096) grid = 4096;
    base_reduce<<<grid, BASE_BLOCK>>>(in, out, n);
}

int main() {
    print_device_banner();

    const int n = 1 << 24;                 // 16M elements
    const size_t bytes = (size_t)n * sizeof(float);

    std::vector<float> hin(n);
    for (int i = 0; i < n; ++i)
        hin[i] = std::sin(i * 0.001f) * 0.5f + 0.25f;

    double ref = 0.0;
    for (int i = 0; i < n; ++i) ref += (double)hin[i];

    float *din, *dout;
    CUDA_CHECK(cudaMalloc(&din, bytes));
    CUDA_CHECK(cudaMalloc(&dout, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(din, hin.data(), bytes, cudaMemcpyHostToDevice));

    // Correctness of the student's solve.
    CUDA_CHECK(cudaMemset(dout, 0, sizeof(float)));
    solve(din, dout, n);
    CUDA_CHECK_KERNEL();
    float hsum = 0.f;
    CUDA_CHECK(cudaMemcpy(&hsum, dout, sizeof(float), cudaMemcpyDeviceToHost));

    double rel = std::fabs((double)hsum - ref) / std::fabs(ref);
    bool ok = rel < 1e-3;
    report_correct(ok);
    report_metric("rel_err", rel);

    // Timing: student vs shared-only baseline. Re-zero *out each iteration.
    float ms = time_kernel([&] {
        cudaMemset(dout, 0, sizeof(float));
        solve(din, dout, n);
    });
    float base_ms = time_kernel([&] {
        cudaMemset(dout, 0, sizeof(float));
        base_solve(din, dout, n);
    });

    double gbps = (double)bytes / (ms * 1e-3) / 1e9;
    report_metric("ms", ms);
    report_metric("base_ms", base_ms);
    report_metric("gbps", gbps);
    report_metric("bw_frac", gbps / peak_bandwidth_gbps());
    report_metric("speedup", base_ms / ms);

    CUDA_CHECK(cudaFree(din));
    CUDA_CHECK(cudaFree(dout));
    return 0;
}
