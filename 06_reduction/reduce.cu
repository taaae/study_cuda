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
    __shared__ float partial_sums[BLOCK];

    int stride = blockDim.x * gridDim.x;

    int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_i = threadIdx.x;

    float thread_sum = 0.0f;

    for (int i = local_i; i < n; i += stride) {
        thread_sum += in[i];
    }
    
    // copy required elements into shared memory first
    partial_sum[thread_i] = thread_sum;

    __syncthreads();

    // do recursive partial summation, first each 2, then each 4 etc
     for (int el_to_sum = 2; el_to_sum <= BLOCK; el_to_sum *= 2) {
        if (thread_i % el_to_sum == 0) {
            partial_sums[thread_i] += partial_sums[thread_i + el_to_sum / 2];
        }
        __syncthreads();
     }
    // more optimized version: stride is not 1,2,4, but blockDim/2, /4.. instead
    // for (int s =  blockDim.x / 2; s > 0; s /= 2) {
    //    if (thread_i < s) {
    //        partial_sums[thread_i] += partial_sums[thread_i + s];
    //    }
    //    __syncthreads();
    // }
    // at the end partial, race free add all the partial sums into global out
    if (thread_i == 0) {
        atomicAdd(out, partial_sums[0]);
    }
}

// Host entry point. in and out are DEVICE pointers; *out is already zeroed.
// Pick a launch configuration and launch reduce.
void solve(const float* in, float* out, int n) {
    // TODO: choose block = BLOCK and a capped grid size (the grid-stride loop
    //       handles any leftover), then launch reduce. (See README + hints.md.)
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p);
    int sms = p.multiProcessorCount;
    reduce<<<sms * 32, BLOCK>>>(in, out, n);
}
