// Exercise 17 — SpMV (CSR, vector / one warp per row)
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

// One WARP per row. The 32 lanes stride across the row's nonzeros, each builds a
// partial sum, then the warp reduces with __shfl_down_sync; lane 0 writes y[r].
__global__ void spmv_vector(const int* rowPtr, const int* colIdx,
                            const float* vals, const float* x, float* y, int nrows) {
    // Scaffolding: identify this thread's warp and lane.
    int global = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = global >> 5;          // one warp per row -> row index
    int lane   = threadIdx.x & 31;     // 0..31 within the warp
    if (warpId >= nrows) return;

    int start = rowPtr[warpId];
    int end   = rowPtr[warpId + 1];

    // TODO: strided loop — each lane handles k = start+lane, start+lane+32, ...
    //       accumulating sum += vals[k] * x[colIdx[k]].
    float sum = 0.0f;

    // TODO: warp-reduce `sum` across all 32 lanes with __shfl_down_sync
    //       (mask 0xffffffff, offsets 16,8,4,2,1).

    // TODO: lane 0 writes y[warpId] = sum.
}

// Host entry point. All pointers are DEVICE pointers (CSR). Launch ONE WARP per
// row (so total threads = nrows * 32, rounded up to whole blocks).
void solve(const int* rowPtr, const int* colIdx, const float* vals,
           const float* x, float* y, int nrows) {
    // TODO: choose a block size (multiple of 32) and a grid that gives nrows
    //       warps total, then launch spmv_vector.
}
