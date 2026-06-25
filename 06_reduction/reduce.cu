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
__global__ void reduce(const float* in, float* out, int n) {
    // TODO: accumulate this thread's slice of `in` with a grid-stride loop,
    //       then reduce the block's values in shared memory (sequential-addressing
    //       tree) and have thread 0 atomicAdd the block result into *out.
    //       (See README's optimization ladder + hints.md.)
    int local_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (local_i == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += in[i];
        }

        out[0] = sum;
    }
}

// Host entry point. in and out are DEVICE pointers; *out is already zeroed.
// Pick a launch configuration and launch reduce.
void solve(const float* in, float* out, int n) {
    // TODO: choose block = BLOCK and a capped grid size (the grid-stride loop
    //       handles any leftover), then launch reduce. (See README + hints.md.)
    // cudaDeviceProp p;
    // cudeGetDeviceProperties(&p);
    // int sms = p.multiProcessorCount;
    reduce<<<ceil_div(n, BLOCK), BLOCK>>>(in, out, n);
}
