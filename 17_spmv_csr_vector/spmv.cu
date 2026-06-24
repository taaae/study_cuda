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

    // TODO: have the 32 lanes stride across this row's nonzeros (start..end),
    //       each accumulating its own partial of vals[k]*x[colIdx[k]].
    float sum = 0.0f;

    // TODO: reduce the lanes' partials to one value with __shfl_down_sync,
    //       then have lane 0 write it to y[warpId]. (See README + hints.md.)
}

// Host entry point. All pointers are DEVICE pointers (CSR). Launch ONE WARP per
// row (so total threads = nrows * 32, rounded up to whole blocks).
void solve(const int* rowPtr, const int* colIdx, const float* vals,
           const float* x, float* y, int nrows) {
    // TODO: choose a block size (multiple of 32) and a grid that gives nrows
    //       warps total, then launch spmv_vector.
    //       (See README's function table and hints.md if stuck.)
}
