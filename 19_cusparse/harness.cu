// Exercise 19 harness — DO NOT EDIT.
// Builds a CSR matrix, runs your cuSPARSE-based solve(), checks correctness vs a
// CPU SpMV, times it, and compares against a tiny hand-written scalar CSR kernel.
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "cuda_utils.cuh"

#ifndef SOLUTION_FILE
#define SOLUTION_FILE "cusparse_spmv.cu"
#endif
#include SOLUTION_FILE

// ---- Naive baseline: scalar CSR SpMV (one thread per row) -----------------
__global__ void spmv_csr_scalar(const int* rowPtr, const int* colIdx,
                                const float* vals, const float* x, float* y,
                                int nrows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;
    float sum = 0.0f;
    int start = rowPtr[row], end = rowPtr[row + 1];
    for (int j = start; j < end; ++j) sum += vals[j] * x[colIdx[j]];
    y[row] = sum;
}

int main() {
    print_device_banner();

    // ---- Build a CSR matrix ----------------------------------------------
    // Square, ~R nonzeros per row, scattered columns. Large enough that the
    // library has real work to do.
    const int nrows = 1 << 17;   // 131072 rows
    const int ncols = nrows;
    const int R     = 48;        // nonzeros per row (variable around this)

    std::vector<int> rowPtr(nrows + 1, 0);
    std::vector<int> colIdx;
    std::vector<float> vals;
    colIdx.reserve((size_t)nrows * R);
    vals.reserve((size_t)nrows * R);

    for (int r = 0; r < nrows; ++r) {
        // Variable row length in [R/2, R] -> not perfectly uniform.
        int len = R / 2 + (int)(((unsigned)r * 2654435761u >> 26) % (R / 2 + 1));
        for (int k = 0; k < len; ++k) {
            unsigned h = ((unsigned)r * 2246822519u + (unsigned)k * 3266489917u);
            int col = (int)(h % (unsigned)ncols);
            float v = 0.5f + 0.001f * (float)((h >> 8) & 1023);
            colIdx.push_back(col);
            vals.push_back(v);
        }
        rowPtr[r + 1] = (int)colIdx.size();
    }
    const int nnz = (int)colIdx.size();

    std::vector<float> hx(ncols);
    for (int i = 0; i < ncols; ++i) hx[i] = std::sin(i * 0.0007f) + 1.0f;

    // ---- CPU reference y = A*x -------------------------------------------
    std::vector<float> ref(nrows, 0.0f);
    for (int r = 0; r < nrows; ++r) {
        double s = 0.0;
        for (int j = rowPtr[r]; j < rowPtr[r + 1]; ++j)
            s += (double)vals[j] * hx[colIdx[j]];
        ref[r] = (float)s;
    }

    // ---- Device allocations ----------------------------------------------
    int   *d_rowPtr, *d_colIdx;
    float *d_vals, *d_x, *d_y, *d_y_base;
    CUDA_CHECK(cudaMalloc(&d_rowPtr, rowPtr.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIdx, colIdx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals,   vals.size()   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, ncols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, nrows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_base, nrows * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_rowPtr, rowPtr.data(), rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIdx, colIdx.data(), colIdx.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals,   vals.data(),   vals.size()   * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, hx.data(), ncols * sizeof(float), cudaMemcpyHostToDevice));

    // ---- Run student cuSPARSE solve & check correctness ------------------
    // beta=0 inside solve means y is overwritten; still zero it for hygiene.
    CUDA_CHECK(cudaMemset(d_y, 0, nrows * sizeof(float)));
    solve(d_rowPtr, d_colIdx, d_vals, d_x, d_y, nrows, ncols, nnz);
    CUDA_CHECK_KERNEL();

    std::vector<float> hy(nrows);
    CUDA_CHECK(cudaMemcpy(hy.data(), d_y, nrows * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    double maxerr = 0.0;
    for (int r = 0; r < nrows; ++r) {
        double err = std::fabs((double)ref[r] - hy[r]);
        double tol = 1e-3 * (std::fabs((double)ref[r]) + 1.0);
        maxerr = err > maxerr ? err : maxerr;
        if (err > tol) ok = false;
    }
    report_correct(ok);
    report_metric("max_abs_err", maxerr);

    // ---- Time cuSPARSE solve --------------------------------------------
    float ms_lib = time_kernel([&] {
        solve(d_rowPtr, d_colIdx, d_vals, d_x, d_y, nrows, ncols, nnz);
    });
    double gflops = 2.0 * (double)nnz / (ms_lib * 1e-3) / 1e9;
    report_metric("ms", ms_lib);
    report_metric("gflops", gflops);

    // ---- Naive baseline --------------------------------------------------
    int block = 256, grid = ceil_div(nrows, block);
    CUDA_CHECK(cudaMemset(d_y_base, 0, nrows * sizeof(float)));
    float ms_naive = time_kernel([&] {
        spmv_csr_scalar<<<grid, block>>>(d_rowPtr, d_colIdx, d_vals, d_x, d_y_base, nrows);
    });
    report_metric("ms_naive", ms_naive);
    report_metric("speedup_vs_naive", ms_naive / ms_lib);

    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_colIdx));
    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_y_base));
    return 0;
}
