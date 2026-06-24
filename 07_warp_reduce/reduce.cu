// Exercise 07 — Warp-Level Reduction
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

#ifndef BLOCK
#define BLOCK 256
#endif

// Reduce one float per lane down to lane 0 of the warp using __shfl_down_sync.
// Lane 0 returns the sum of all 32 lanes' values.
__device__ float warpReduceSum(float v) {
    // TODO: reduce `v` across the warp's 32 lanes with __shfl_down_sync (halving
    //       the offset each step) so lane 0 ends up with the total, then return v.
    //       (See README + hints.md.)
    return v;
}

// Each block: grid-stride load into a register, reduce within each warp via
// warpReduceSum, combine the per-warp partials (store warp results to a small
// __shared__ array, then warp-reduce the first warp), and atomicAdd the block
// total to *out.
__global__ void reduce(const float* in, float* out, int n) {
    // TODO: grid-stride accumulate this thread's slice into a register, then
    //       warpReduceSum within each warp. Stash the per-warp partials in a small
    //       shared array, and have the first warp warpReduceSum those into the block
    //       total. Thread 0 atomicAdds the block total into *out.
    //       (See README + hints.md.)
}

// Host entry point. in and out are DEVICE pointers; *out is already zeroed.
void solve(const float* in, float* out, int n) {
    // TODO: choose block = BLOCK and a capped grid, then launch reduce.
    //       (See README + hints.md.)
}
