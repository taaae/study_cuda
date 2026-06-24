// Exercise 17 harness — DO NOT EDIT.
// Builds a CSR matrix with HIGHLY VARIABLE row lengths, runs your solve(),
// checks vs a CPU SpMV, and times it against a scalar (one-thread-per-row) baseline.
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

// ---- Random CSR with highly variable row lengths (host) -------------------
// To make scalar SpMV suffer from divergence/imbalance, most rows are SHORT
// (a few nonzeros) but a minority are VERY LONG (hundreds). For each row we draw
// a length from a heavy-tailed mix, pick that many distinct columns, sort them,
// and assign random values. rowPtr is the prefix sum of per-row counts, so the
// CSR is valid (monotone rowPtr, rowPtr[nrows]==nnz, colIdx in [0, nrows)).
struct CSR {
    std::vector<int> rowPtr, colIdx;
    std::vector<float> vals;
    int nrows = 0, nnz = 0;
};

static CSR build_csr_imbalanced(int nrows, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);
    std::uniform_int_distribution<int> shortLen(1, 8);
    std::uniform_int_distribution<int> longLen(200, 600);
    std::uniform_int_distribution<int> col(0, nrows - 1);
    std::uniform_real_distribution<float> val(-1.0f, 1.0f);

    CSR A;
    A.nrows = nrows;
    A.rowPtr.resize(nrows + 1);
    A.rowPtr[0] = 0;

    std::vector<char> used(nrows, 0);
    for (int r = 0; r < nrows; ++r) {
        int want = (coin(rng) < 0.05f) ? longLen(rng) : shortLen(rng);  // 5% long
        if (want > nrows) want = nrows;
        std::vector<int> cols;
        cols.reserve(want);
        for (int t = 0; t < want; ++t) {
            int c = col(rng);
            if (!used[c]) { used[c] = 1; cols.push_back(c); }
        }
        std::sort(cols.begin(), cols.end());
        for (int c : cols) used[c] = 0;
        for (int c : cols) {
            A.colIdx.push_back(c);
            A.vals.push_back(val(rng));
        }
        A.rowPtr[r + 1] = (int)A.colIdx.size();
    }
    A.nnz = (int)A.colIdx.size();
    return A;
}

static void spmv_cpu(const CSR& A, const std::vector<float>& x,
                     std::vector<float>& y) {
    for (int r = 0; r < A.nrows; ++r) {
        double s = 0.0;
        for (int k = A.rowPtr[r]; k < A.rowPtr[r + 1]; ++k)
            s += (double)A.vals[k] * x[A.colIdx[k]];
        y[r] = (float)s;
    }
}

// Scalar baseline kernel (one thread per row) — the thing the warp kernel beats.
__global__ void spmv_scalar_baseline(const int* rowPtr, const int* colIdx,
                                     const float* vals, const float* x,
                                     float* y, int nrows) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrows) return;
    int start = rowPtr[r], end = rowPtr[r + 1];
    float sum = 0.0f;
    for (int k = start; k < end; ++k) sum += vals[k] * x[colIdx[k]];
    y[r] = sum;
}

int main() {
    print_device_banner();

    const int nrows = 1 << 16;             // 65536 rows
    CSR A = build_csr_imbalanced(nrows, 1234u);
    std::printf("# nrows=%d nnz=%d avg_nnz/row=%.1f (highly variable)\n",
                A.nrows, A.nnz, (double)A.nnz / A.nrows);

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

    // Baseline: scalar one-thread-per-row on the same imbalanced matrix.
    int block = 256, grid = ceil_div(nrows, block);
    float base = time_kernel([&] {
        spmv_scalar_baseline<<<grid, block>>>(d_rowPtr, d_colIdx, d_vals,
                                              d_x, d_y, nrows);
    });
    report_metric("baseline_ms", base);
    report_metric("speedup", base / ms);

    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_colIdx));
    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return 0;
}
