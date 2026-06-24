// Exercise 16 — SpMV (CSR, scalar / one thread per row)
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

// One thread per row: y[r] = sum_k vals[k] * x[colIdx[k]] for k in [rowPtr[r], rowPtr[r+1]).
__global__ void spmv_scalar(const int* rowPtr, const int* colIdx,
                            const float* vals, const float* x, float* y, int nrows) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrows) return;
    // TODO: read start = rowPtr[r] and end = rowPtr[r+1].
    // TODO: loop k from start to end, accumulate sum += vals[k] * x[colIdx[k]].
    // TODO: write y[r] = sum.
}

// Host entry point. All pointers are DEVICE pointers (CSR layout). Launch one
// thread per row.
void solve(const int* rowPtr, const int* colIdx, const float* vals,
           const float* x, float* y, int nrows) {
    // TODO: launch spmv_scalar with one thread per row.
}
