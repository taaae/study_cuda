// Exercise 20 harness — DO NOT EDIT.
// Builds an SPD 2-D 5-point Laplacian in CSR, picks a known x_true, sets
// b = A*x_true, runs your CG solve(), then recomputes the relative residual
// ||b - A x|| / ||b|| and checks it against the tolerance.
#include <vector>
#include <cmath>
#include <cstdio>
#include "cuda_utils.cuh"

#ifndef SOLUTION_FILE
#define SOLUTION_FILE "cg.cu"
#endif
#include SOLUTION_FILE

int main() {
    print_device_banner();

    // ---- Build the 2-D 5-point Laplacian on an m x m grid -----------------
    // Unknown ordering: index = iy*m + ix. Diagonal = 4, each existing neighbor
    // (up/down/left/right) contributes -1. This matrix is symmetric positive
    // definite, so CG is guaranteed to converge.
    const int m = 256;            // grid side
    const int n = m * m;          // 65536 unknowns

    std::vector<int>   rowPtr(n + 1, 0);
    std::vector<int>   colIdx;
    std::vector<float> vals;
    colIdx.reserve((size_t)n * 5);
    vals.reserve((size_t)n * 5);

    for (int iy = 0; iy < m; ++iy) {
        for (int ix = 0; ix < m; ++ix) {
            int row = iy * m + ix;
            // Emit neighbors in column-ascending order, diagonal in the middle.
            if (iy > 0)     { colIdx.push_back(row - m); vals.push_back(-1.0f); } // up
            if (ix > 0)     { colIdx.push_back(row - 1); vals.push_back(-1.0f); } // left
            colIdx.push_back(row);     vals.push_back(4.0f);                      // diag
            if (ix < m - 1) { colIdx.push_back(row + 1); vals.push_back(-1.0f); } // right
            if (iy < m - 1) { colIdx.push_back(row + m); vals.push_back(-1.0f); } // down
            rowPtr[row + 1] = (int)colIdx.size();
        }
    }
    const int nnz = (int)colIdx.size();

    // ---- Known solution and matching right-hand side ----------------------
    std::vector<float> x_true(n);
    for (int i = 0; i < n; ++i) x_true[i] = std::sin(i * 0.001f) + 0.5f;

    // b = A * x_true  (CPU)
    std::vector<float> hb(n, 0.0f);
    for (int row = 0; row < n; ++row) {
        double s = 0.0;
        for (int j = rowPtr[row]; j < rowPtr[row + 1]; ++j)
            s += (double)vals[j] * x_true[colIdx[j]];
        hb[row] = (float)s;
    }

    // ---- Device allocations ----------------------------------------------
    int   *d_rowPtr, *d_colIdx;
    float *d_vals, *d_b, *d_x;
    CUDA_CHECK(cudaMalloc(&d_rowPtr, rowPtr.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIdx, colIdx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals,   vals.size()   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_rowPtr, rowPtr.data(), rowPtr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIdx, colIdx.data(), colIdx.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals,   vals.data(),   vals.size()   * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, hb.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    const int   maxiter = 5000;
    const float tol     = 1e-4f;   // solve harder than the 1e-3 grading bar

    // ---- Run CG (x is pre-zeroed by the harness) -------------------------
    CUDA_CHECK(cudaMemset(d_x, 0, n * sizeof(float)));
    solve(d_rowPtr, d_colIdx, d_vals, d_b, d_x, n, maxiter, tol);
    CUDA_CHECK_KERNEL();

    std::vector<float> hx(n);
    CUDA_CHECK(cudaMemcpy(hx.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- Recompute relative residual ||b - A x|| / ||b|| on the host ------
    double resid2 = 0.0, bnorm2 = 0.0;
    for (int row = 0; row < n; ++row) {
        double Ax = 0.0;
        for (int j = rowPtr[row]; j < rowPtr[row + 1]; ++j)
            Ax += (double)vals[j] * hx[colIdx[j]];
        double d = (double)hb[row] - Ax;
        resid2 += d * d;
        bnorm2 += (double)hb[row] * hb[row];
    }
    double rel_resid = std::sqrt(resid2) / std::sqrt(bnorm2 > 0 ? bnorm2 : 1.0);

    bool ok = rel_resid <= 1e-3;   // matches grading threshold
    report_correct(ok);
    report_metric("rel_resid", rel_resid);

    // ---- Time the whole solve --------------------------------------------
    float ms = time_kernel([&] {
        CUDA_CHECK(cudaMemset(d_x, 0, n * sizeof(float)));
        solve(d_rowPtr, d_colIdx, d_vals, d_b, d_x, n, maxiter, tol);
    }, 5);   // CG is expensive; a few iters is enough for a stable minimum
    report_metric("ms", ms);

    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_colIdx));
    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_x));
    return 0;
}
