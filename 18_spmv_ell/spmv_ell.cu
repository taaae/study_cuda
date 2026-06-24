// Exercise 18 — SpMV in the ELL format.
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

// One thread per row. The ELL arrays are COLUMN-MAJOR: element (row, k) lives at
// index k*nrows + row. Padding slots use column 0 and value 0.0f (a safe no-op).
//
//   ell_cols[k*nrows + row]  column index of the k-th nonzero of `row`
//   ell_vals[k*nrows + row]  its value
//
__global__ void spmv_ell(const int* ell_cols, const float* ell_vals,
                         const float* x, float* y, int nrows, int maxnnz) {
    // TODO: one thread per row (guard against nrows). Walk this row's maxnnz
    //       slots through the column-major ELL layout, accumulating value*x[col],
    //       and store the result into y. (See README + hints.md.)
}

// Host entry point. All pointers are DEVICE pointers.
// ell_cols / ell_vals have length maxnnz*nrows (column-major). y has length nrows.
void solve(const int* ell_cols, const float* ell_vals,
           const float* x, float* y, int nrows, int maxnnz) {
    // TODO: choose a launch config covering all nrows and launch spmv_ell.
    //       (See README + hints.md.)
}
