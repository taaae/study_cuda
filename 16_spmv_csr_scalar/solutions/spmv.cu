// Reference solution — Exercise 16.
#include "cuda_utils.cuh"

__global__ void spmv_scalar(const int* rowPtr, const int* colIdx,
                            const float* vals, const float* x, float* y, int nrows) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrows) return;
    int start = rowPtr[r];
    int end   = rowPtr[r + 1];
    float sum = 0.0f;
    for (int k = start; k < end; ++k)
        sum += vals[k] * x[colIdx[k]];
    y[r] = sum;
}

void solve(const int* rowPtr, const int* colIdx, const float* vals,
           const float* x, float* y, int nrows) {
    int block = 256;
    int grid  = ceil_div(nrows, block);
    spmv_scalar<<<grid, block>>>(rowPtr, colIdx, vals, x, y, nrows);
}
