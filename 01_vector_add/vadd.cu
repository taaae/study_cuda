// Exercise 01 — Vector Add
// Fill in the two TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

// One thread computes one output element: c[i] = a[i] + b[i].
__global__ void vadd(const float* a, const float* b, float* c, int n) {
    // TODO: compute this thread's global index, guard against n, write c[i].
    if (i < n) {
        c[i] = a[i] + b[i]
    }
}

// Host entry point. a, b, c are DEVICE pointers of length n.
// Pick a block size, compute the grid size, and launch vadd.
void solve(const float* a, const float* b, float* c, int n) {
    // TODO: launch vadd<<<grid, block>>>(a, b, c, n);
    
}
