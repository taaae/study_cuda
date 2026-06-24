// Reference solution — Exercise 17.
#include "cuda_utils.cuh"

__global__ void spmv_vector(const int* rowPtr, const int* colIdx,
                            const float* vals, const float* x, float* y, int nrows) {
    int global = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = global >> 5;          // one warp per row
    int lane   = threadIdx.x & 31;     // lane within the warp
    if (warpId >= nrows) return;

    int start = rowPtr[warpId];
    int end   = rowPtr[warpId + 1];

    float sum = 0.0f;
    for (int k = start + lane; k < end; k += 32)
        sum += vals[k] * x[colIdx[k]];

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0) y[warpId] = sum;
}

void solve(const int* rowPtr, const int* colIdx, const float* vals,
           const float* x, float* y, int nrows) {
    int block = 256;                   // 8 warps per block
    int warpsPerBlock = block / 32;
    int grid = ceil_div(nrows, warpsPerBlock);
    spmv_vector<<<grid, block>>>(rowPtr, colIdx, vals, x, y, nrows);
}
