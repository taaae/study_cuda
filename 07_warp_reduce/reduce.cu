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
    for (int s = 16; s >= 1; s >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, s);
    }
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
    float v = 0;
    int absolute_i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = absolute_i; i < n; i += stride) {
        v += in[i];
    }

    int num_warps = blockDim.x >> 5;
    __shared__ float warp_sums[BLOCK / 32];

    v = warpReduceSum(v);


    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    if (lane == 0) {
        warp_sums[warp] = v;
    }

    __syncthreads();

    if (warp == 0) {
        float w = 0;
        if (lane < num_warps) {
            w = warp_sums[lane];
        }
        w = warpReduceSum(w);
        if (lane == 0) {
            atomicAdd(out, w);
        }
    }
}

// Host entry point. in and out are DEVICE pointers; *out is already zeroed.
void solve(const float* in, float* out, int n) {
    // TODO: choose block = BLOCK and a capped grid, then launch reduce.
    //       (See README + hints.md.)
    static int grid = 0;
    if (grid == 0) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, 0);
        grid = p.multiProcessorCount * 32;
    }
    reduce<<<grid, BLOCK>>>(in, out, n);
}
