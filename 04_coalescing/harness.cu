// Exercise 04 harness — DO NOT EDIT.
// Builds data, runs your solve(), checks correctness, runs a strided baseline,
// reports speedup and bandwidth.
#include <vector>
#include <cmath>
#include <cstdio>
#include "cuda_utils.cuh"

#ifndef SOLUTION_FILE
#define SOLUTION_FILE "transpose.cu"
#endif
#include SOLUTION_FILE

// --- Deliberately-bad baseline: STRIDED reads (threadIdx.x maps to the strided
// row dimension of `in`). This is the "wrong" indexing choice; coalescing the
// reads instead (your job) should beat it clearly. ---
__global__ void transpose_baseline(const float* in, float* out, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // fast index -> strided read of in
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        out[col * n + row] = in[row * n + col];       // strided read, coalesced write
    }
}

static void run_baseline(const float* in, float* out, int n) {
    dim3 block(32, 8);
    dim3 grid(ceil_div(n, block.x), ceil_div(n, block.y));
    transpose_baseline<<<grid, block>>>(in, out, n);
}

int main() {
    print_device_banner();

    const int n = 4096;                         // 4096 x 4096, multiple of 32
    const size_t count = (size_t)n * n;
    const size_t bytes = count * sizeof(float);

    std::vector<float> hin(count), hout(count);
    for (size_t i = 0; i < count; ++i) hin[i] = std::sin(i * 0.0001f);

    float *din, *dout;
    CUDA_CHECK(cudaMalloc(&din, bytes));
    CUDA_CHECK(cudaMalloc(&dout, bytes));
    CUDA_CHECK(cudaMemcpy(din, hin.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dout, 0, bytes));

    solve(din, dout, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(hout.data(), dout, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    double maxerr = 0.0;
    for (int r = 0; r < n && ok; ++r) {
        for (int c = 0; c < n; ++c) {
            double ref = hin[(size_t)c * n + r];          // in^T
            double err = std::fabs(ref - hout[(size_t)r * n + c]);
            maxerr = err > maxerr ? err : maxerr;
            if (err != 0.0) { ok = false; break; }
        }
    }
    report_correct(ok);
    report_metric("max_abs_err", maxerr);

    float ms       = time_kernel([&] { solve(din, dout, n); });
    float ms_naive = time_kernel([&] { run_baseline(din, dout, n); });

    double gbps = 2.0 * bytes / (ms * 1e-3) / 1e9;   // read in, write out
    report_metric("ms", ms);
    report_metric("naive_ms", ms_naive);
    report_metric("speedup", ms_naive / ms);
    report_metric("gbps", gbps);
    report_metric("bw_frac", gbps / peak_bandwidth_gbps());

    CUDA_CHECK(cudaFree(din));
    CUDA_CHECK(cudaFree(dout));
    return 0;
}
