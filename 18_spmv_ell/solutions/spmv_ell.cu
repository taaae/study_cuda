// Reference solution — Exercise 18 (SpMV in the ELL format).
#include "cuda_utils.cuh"

__global__ void spmv_ell(const int* ell_cols, const float* ell_vals,
                         const float* x, float* y, int nrows, int maxnnz) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;
    float sum = 0.0f;
    for (int k = 0; k < maxnnz; ++k) {
        int   col = ell_cols[k * nrows + row];   // column-major stride: k*nrows
        float val = ell_vals[k * nrows + row];
        sum += val * x[col];                     // padding is (col=0, val=0): no-op
    }
    y[row] = sum;
}

void solve(const int* ell_cols, const float* ell_vals,
           const float* x, float* y, int nrows, int maxnnz) {
    int block = 256;
    int grid  = ceil_div(nrows, block);
    spmv_ell<<<grid, block>>>(ell_cols, ell_vals, x, y, nrows, maxnnz);
}
