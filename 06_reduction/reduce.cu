// Exercise 06 — Parallel Reduction
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

// Block size used by both the kernel's shared array and the launch in solve().
// Keep these in sync.
#ifndef BLOCK
#define BLOCK 256
#endif

// Each block sums a chunk of `in` into a single block-level partial, then
// contributes it to the global total at *out.
//
// Recommended structure (see README's optimization ladder):
//   1. Each thread walks `in` with a grid-stride loop, accumulating into a
//      register `sum` (this is "first add during load", generalized).
//   2. Store `sum` into __shared__ sdata[tid].
//   3. __syncthreads(), then a SEQUENTIAL-ADDRESSING tree:
//        for (s = blockDim.x/2; s > 0; s >>= 1) { if (tid < s) sdata[tid]+=sdata[tid+s]; __syncthreads(); }
//   4. Thread 0 does ONE atomicAdd(out, sdata[0]).
__global__ void reduce(const float* in, float* out, int n) {
    // TODO: declare __shared__ float sdata[BLOCK];
    // TODO: int tid = threadIdx.x;
    // TODO: grid-stride accumulate into a register
    // TODO: write register into sdata[tid], __syncthreads()
    // TODO: sequential-addressing reduction tree with __syncthreads()
    // TODO: if (tid == 0) atomicAdd(out, sdata[0]);
}

// Host entry point. in and out are DEVICE pointers; *out is already zeroed.
// Pick a launch configuration and launch reduce.
void solve(const float* in, float* out, int n) {
    // TODO: choose block = BLOCK and a grid size (cap it so blocks stay full;
    //       the grid-stride loop handles any leftover), then launch
    //       reduce<<<grid, BLOCK>>>(in, out, n);
}
