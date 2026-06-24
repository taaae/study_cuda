// Exercise 24 harness — DO NOT EDIT.
// Builds an image; runs your solve(); checks vs the CPU stencil; runs the naive
// baseline kernel; reports speedup and bw_frac.
#include <vector>
#include <cmath>
#include <cstdio>
#include "cuda_utils.cuh"

#ifndef SOLUTION_FILE
#define SOLUTION_FILE "optimize.cu"
#endif
#include SOLUTION_FILE

// ---- Naive baseline: correct but slow. One thread per pixel, all neighbors
// fetched from global memory with a clamp, no shared memory. ----
__device__ __forceinline__ int clampi_base(int v, int n) {
    return v < 0 ? 0 : (v >= n ? n - 1 : v);
}
__global__ void stencil_naive(const float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    float c = in[clampi_base(y, height) * width + clampi_base(x, width)];
    float l = in[clampi_base(y, height) * width + clampi_base(x - 1, width)];
    float r = in[clampi_base(y, height) * width + clampi_base(x + 1, width)];
    float u = in[clampi_base(y - 1, height) * width + clampi_base(x, width)];
    float d = in[clampi_base(y + 1, height) * width + clampi_base(x, width)];
    out[y * width + x] = (c + l + r + u + d) * 0.2f;
}
static void baseline(const float* in, float* out, int width, int height) {
    dim3 block(32, 8);
    dim3 grid(ceil_div(width, 32), ceil_div(height, 8));
    stencil_naive<<<grid, block>>>(in, out, width, height);
}

int main() {
    print_device_banner();

    const int width = 4096, height = 4096;
    const size_t n = (size_t)width * height;
    const size_t bytes = n * sizeof(float);

    std::vector<float> hin(n), hout(n), href(n);
    for (size_t i = 0; i < n; ++i) hin[i] = std::sin(i * 0.0005f);

    // CPU reference with clamped boundaries (exact formula from the README).
    auto cl = [](int v, int m) { return v < 0 ? 0 : (v >= m ? m - 1 : v); };
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            float c = hin[(size_t)cl(y, height) * width + cl(x, width)];
            float l = hin[(size_t)cl(y, height) * width + cl(x - 1, width)];
            float r = hin[(size_t)cl(y, height) * width + cl(x + 1, width)];
            float u = hin[(size_t)cl(y - 1, height) * width + cl(x, width)];
            float d = hin[(size_t)cl(y + 1, height) * width + cl(x, width)];
            href[(size_t)y * width + x] = (c + l + r + u + d) * 0.2f;
        }

    float *din, *dout;
    CUDA_CHECK(cudaMalloc(&din, bytes));
    CUDA_CHECK(cudaMalloc(&dout, bytes));
    CUDA_CHECK(cudaMemcpy(din, hin.data(), bytes, cudaMemcpyHostToDevice));

    solve(din, dout, width, height);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(hout.data(), dout, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    double maxerr = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double err = std::fabs((double)href[i] - hout[i]);
        if (err > maxerr) maxerr = err;
        if (err > 1e-4) ok = false;
    }
    report_correct(ok);
    report_metric("max_abs_err", maxerr);

    float ms_sol  = time_kernel([&] { solve(din, dout, width, height); });
    float ms_base = time_kernel([&] { baseline(din, dout, width, height); });

    // Effective traffic of the stencil: read the image once + write it once.
    double gbps = 2.0 * bytes / (ms_sol * 1e-3) / 1e9;
    report_metric("ms", ms_sol);
    report_metric("gbps", gbps);
    report_metric("bw_frac", gbps / peak_bandwidth_gbps());
    report_metric("baseline_ms", ms_base);
    report_metric("speedup", ms_base / ms_sol);

    CUDA_CHECK(cudaFree(din));
    CUDA_CHECK(cudaFree(dout));
    return 0;
}
