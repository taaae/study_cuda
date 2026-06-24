// Exercise 16 — SpMV (CSR, scalar / one thread per row)
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

// One thread per row: y[r] = sum_k vals[k] * x[colIdx[k]] for k in [rowPtr[r], rowPtr[r+1]).
__global__ void spmv_scalar(const int* rowPtr, const int* colIdx,
                            const float* vals, const float* x, float* y, int nrows) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrows) return;
    // TODO: this thread owns row r. Sum vals[k]*x[colIdx[k]] over the row's
    //       nonzeros (the rowPtr[r]..rowPtr[r+1] range) and store it in y[r].
    //       (See README + hints.md.)
}

// Host entry point. All pointers are DEVICE pointers (CSR layout). Launch one
// thread per row.
void solve(const int* rowPtr, const int* colIdx, const float* vals,
           const float* x, float* y, int nrows) {
    // TODO: launch spmv_scalar with one thread per row.
    //       (See README's function table and hints.md if stuck.)
}
