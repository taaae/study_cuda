// Exercise 18 harness — DO NOT EDIT.
// Builds a near-uniform sparse matrix, stores it as COLUMN-MAJOR ELL, runs your
// solve(), checks correctness vs a CPU SpMV, times it, and compares against a
// scalar CSR SpMV baseline to report a speedup.
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "cuda_utils.cuh"

#ifndef SOLUTION_FILE
#define SOLUTION_FILE "spmv_ell.cu"
#endif
#include SOLUTION_FILE

// ---- Baseline: scalar CSR SpMV (one thread per row) -----------------------
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

    // ---- Build a NEAR-UNIFORM matrix --------------------------------------
    // Each row has between (R-1) and R nonzeros, so maxnnz = R and ELL padding
    // is minimal. Square matrix; columns chosen pseudo-randomly but reproducibly.
    const int nrows  = 1 << 16;   // 65536 rows
    const int ncols  = nrows;
    const int R      = 32;        // nominal nonzeros per row -> maxnnz
    const int maxnnz = R;

    // Per-row nonzero count in [R-1, R]: near-uniform, so ELL pays almost no
    // padding tax (this is the regime where ELL is supposed to win).
    std::vector<int> rowlen(nrows);
    long long nnz = 0;
    for (int r = 0; r < nrows; ++r) {
        rowlen[r] = R - (((unsigned)r * 2654435761u >> 28) & 1);  // R or R-1
        nnz += rowlen[r];
    }

    // CSR arrays.
    std::vector<int>   rowPtr(nrows + 1, 0);
    for (int r = 0; r < nrows; ++r) rowPtr[r + 1] = rowPtr[r] + rowlen[r];
    std::vector<int>   colIdx(nnz);
    std::vector<float> vals(nnz);

    // ELL arrays (column-major: (row,k) at k*nrows + row), padded with (0, 0).
    std::vector<int>   ell_cols((size_t)maxnnz * nrows, 0);
    std::vector<float> ell_vals((size_t)maxnnz * nrows, 0.0f);

    // Fill both layouts from the same logical entries. Columns are spread across
    // the row to give realistic, scattered access into x.
    for (int r = 0; r < nrows; ++r) {
        int len = rowlen[r];
        int base = rowPtr[r];
        for (int k = 0; k < len; ++k) {
            unsigned h = ((unsigned)r * 2246822519u + (unsigned)k * 3266489917u);
            int col = (int)(h % (unsigned)ncols);
            float v = 0.5f + 0.001f * (float)((h >> 8) & 1023);
            // CSR
            colIdx[base + k] = col;
            vals[base + k]   = v;
            // ELL (column-major)
            ell_cols[(size_t)k * nrows + r] = col;
            ell_vals[(size_t)k * nrows + r] = v;
        }
        // ELL padding for k in [len, maxnnz) already (0, 0) from initialization.
    }

    // Input vector x.
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
    int   *d_ell_cols, *d_rowPtr, *d_colIdx;
    float *d_ell_vals, *d_vals, *d_x, *d_y, *d_y_base;
    CUDA_CHECK(cudaMalloc(&d_ell_cols, ell_cols.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ell_vals, ell_vals.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rowPtr, rowPtr.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIdx, colIdx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals,   vals.size()   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, ncols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, nrows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_base, nrows * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_ell_cols, ell_cols.data(), ell_cols.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ell_vals, ell_vals.data(), ell_vals.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowPtr, rowPtr.data(), rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIdx, colIdx.data(), colIdx.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals,   vals.data(),   vals.size()   * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, hx.data(), ncols * sizeof(float), cudaMemcpyHostToDevice));

    // ---- Run student ELL solve & check correctness -----------------------
    CUDA_CHECK(cudaMemset(d_y, 0, nrows * sizeof(float)));
    solve(d_ell_cols, d_ell_vals, d_x, d_y, nrows, maxnnz);
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

    // ---- Time ELL --------------------------------------------------------
    float ms_ell = time_kernel([&] {
        solve(d_ell_cols, d_ell_vals, d_x, d_y, nrows, maxnnz);
    });

    // Effective work / traffic.
    // FLOPs counted on the REAL nonzeros (one multiply + one add each).
    double gflops = 2.0 * (double)nnz / (ms_ell * 1e-3) / 1e9;
    // Bytes the ELL kernel actually moves: every padded slot is read too.
    //   per slot: int col + float val + float x[col]  = 12 bytes
    //   plus one float write per row.
    double ell_slots = (double)maxnnz * nrows;
    double bytes_ell = ell_slots * (sizeof(int) + 2 * sizeof(float))
                       + (double)nrows * sizeof(float);
    double gbps = bytes_ell / (ms_ell * 1e-3) / 1e9;
    report_metric("ms", ms_ell);
    report_metric("gflops", gflops);
    report_metric("bw_frac", gbps / peak_bandwidth_gbps());

    // ---- Baseline: scalar CSR SpMV --------------------------------------
    int block = 256, grid = ceil_div(nrows, block);
    CUDA_CHECK(cudaMemset(d_y_base, 0, nrows * sizeof(float)));
    float ms_csr = time_kernel([&] {
        spmv_csr_scalar<<<grid, block>>>(d_rowPtr, d_colIdx, d_vals, d_x, d_y_base, nrows);
    });
    // Sanity: baseline must agree with ELL result too.
    std::vector<float> hy_base(nrows);
    CUDA_CHECK(cudaMemcpy(hy_base.data(), d_y_base, nrows * sizeof(float), cudaMemcpyDeviceToHost));
    bool base_ok = true;
    for (int r = 0; r < nrows; ++r)
        if (std::fabs((double)ref[r] - hy_base[r]) > 1e-3 * (std::fabs((double)ref[r]) + 1.0))
            base_ok = false;

    report_metric("ms_csr_baseline", ms_csr);
    report_metric("speedup", base_ok ? (ms_csr / ms_ell) : 0.0);

    CUDA_CHECK(cudaFree(d_ell_cols));
    CUDA_CHECK(cudaFree(d_ell_vals));
    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_colIdx));
    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_y_base));
    return 0;
}
