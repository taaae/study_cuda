// Exercise 16 harness — DO NOT EDIT.
// Builds a random CSR matrix, runs your solve(), checks vs a CPU SpMV, times it.
#include <vector>
#include <cmath>
#include <cstdio>
#include <random>
#include <algorithm>
#include "cuda_utils.cuh"

#ifndef SOLUTION_FILE
#define SOLUTION_FILE "spmv.cu"
#endif
#include SOLUTION_FILE

// ---- Random CSR construction (host) ---------------------------------------
// We build a square nrows x nrows matrix with, on average, `avg_nnz_per_row`
// nonzeros per row. For each row we pick a random count near the average, then
// pick that many DISTINCT random columns, sort them (so colIdx is ascending
// within a row, like a real CSR), and assign each a random value. rowPtr is the
// prefix sum of the per-row counts. This yields valid CSR: rowPtr is monotone,
// rowPtr[nrows]==nnz, and every colIdx is in [0, nrows).
struct CSR {
    std::vector<int> rowPtr, colIdx;
    std::vector<float> vals;
    int nrows = 0, nnz = 0;
};

static CSR build_csr(int nrows, int avg_nnz_per_row, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> cnt(avg_nnz_per_row / 2,
                                           avg_nnz_per_row + avg_nnz_per_row / 2);
    std::uniform_int_distribution<int> col(0, nrows - 1);
    std::uniform_real_distribution<float> val(-1.0f, 1.0f);

    CSR A;
    A.nrows = nrows;
    A.rowPtr.resize(nrows + 1);
    A.rowPtr[0] = 0;

    std::vector<char> used(nrows, 0);
    for (int r = 0; r < nrows; ++r) {
        int want = cnt(rng);
        if (want < 1) want = 1;
        if (want > nrows) want = nrows;
        std::vector<int> cols;
        cols.reserve(want);
        for (int t = 0; t < want; ++t) {
            int c = col(rng);
            if (!used[c]) { used[c] = 1; cols.push_back(c); }
        }
        std::sort(cols.begin(), cols.end());
        for (int c : cols) used[c] = 0;   // reset for next row
        for (int c : cols) {
            A.colIdx.push_back(c);
            A.vals.push_back(val(rng));
        }
        A.rowPtr[r + 1] = (int)A.colIdx.size();
    }
    A.nnz = (int)A.colIdx.size();
    return A;
}

// CPU reference SpMV.
static void spmv_cpu(const CSR& A, const std::vector<float>& x,
                     std::vector<float>& y) {
    for (int r = 0; r < A.nrows; ++r) {
        double s = 0.0;
        for (int k = A.rowPtr[r]; k < A.rowPtr[r + 1]; ++k)
            s += (double)A.vals[k] * x[A.colIdx[k]];
        y[r] = (float)s;
    }
}

int main() {
    print_device_banner();

    const int nrows = 1 << 16;             // 65536 rows
    const int avg_nnz = 64;
    CSR A = build_csr(nrows, avg_nnz, 1234u);
    std::printf("# nrows=%d nnz=%d avg_nnz/row=%.1f\n", A.nrows, A.nnz,
                (double)A.nnz / A.nrows);

    std::vector<float> hx(nrows), hy(nrows), hy_ref(nrows);
    for (int i = 0; i < nrows; ++i) hx[i] = std::sin(i * 0.001f);
    spmv_cpu(A, hx, hy_ref);

    int *d_rowPtr, *d_colIdx;
    float *d_vals, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_rowPtr, (nrows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIdx, A.nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals,   A.nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x,      nrows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,      nrows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_rowPtr, A.rowPtr.data(), (nrows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIdx, A.colIdx.data(), A.nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, A.vals.data(), A.nnz * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, hx.data(), nrows * sizeof(float),
                          cudaMemcpyHostToDevice));

    solve(d_rowPtr, d_colIdx, d_vals, d_x, d_y, nrows);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(hy.data(), d_y, nrows * sizeof(float),
                          cudaMemcpyDeviceToHost));

    bool ok = true;
    double maxerr = 0.0;
    for (int r = 0; r < nrows; ++r) {
        double err = std::fabs((double)hy_ref[r] - hy[r]);
        double scale = std::fabs((double)hy_ref[r]) + 1e-3;
        maxerr = err > maxerr ? err : maxerr;
        if (err / scale > 1e-3) ok = false;
    }
    report_correct(ok);
    report_metric("max_abs_err", maxerr);

    float ms = time_kernel([&] {
        solve(d_rowPtr, d_colIdx, d_vals, d_x, d_y, nrows);
    });
    report_metric("ms", ms);
    report_metric("gflops", 2.0 * A.nnz / (ms * 1e-3) / 1e9);
    // Bytes moved (rough): vals + colIdx (nnz each) + rowPtr + y. x reads not counted.
    double bytes = (double)A.nnz * (sizeof(float) + sizeof(int))
                 + (double)(nrows + 1) * sizeof(int) + (double)nrows * sizeof(float);
    double gbps = bytes / (ms * 1e-3) / 1e9;
    report_metric("gbps", gbps);
    report_metric("bw_frac", gbps / peak_bandwidth_gbps());

    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_colIdx));
    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return 0;
}
