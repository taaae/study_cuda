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
    // TODO: compute this thread's global row index and guard against nrows.
    // TODO: loop k = 0 .. maxnnz-1, reading ell_cols[k*nrows + row] and
    //       ell_vals[k*nrows + row] (the column-major stride is k*nrows), and
    //       accumulate sum += val * x[col].
    // TODO: write y[row] = sum.
}

// Host entry point. All pointers are DEVICE pointers.
// ell_cols / ell_vals have length maxnnz*nrows (column-major). y has length nrows.
void solve(const int* ell_cols, const float* ell_vals,
           const float* x, float* y, int nrows, int maxnnz) {
    // TODO: pick a block size, compute grid = ceil_div(nrows, block),
    //       and launch spmv_ell<<<grid, block>>>(...).
}
