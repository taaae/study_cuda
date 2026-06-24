// Exercise 07 — Warp-Level Reduction
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

#ifndef BLOCK
#define BLOCK 256
#endif

// Reduce one float per lane down to lane 0 of the warp using __shfl_down_sync.
// Lane 0 returns the sum of all 32 lanes' values.
__device__ float warpReduceSum(float v) {
    // TODO: for (int offset = 16; offset > 0; offset >>= 1)
    //           v += __shfl_down_sync(0xffffffff, v, offset);
    // TODO: return v;
    return v;
}

// Each block: grid-stride load into a register, reduce within each warp via
// warpReduceSum, combine the per-warp partials (store warp results to a small
// __shared__ array, then warp-reduce the first warp), and atomicAdd the block
// total to *out.
__global__ void reduce(const float* in, float* out, int n) {
    // TODO: int tid = threadIdx.x; lane = tid & 31; warp = tid >> 5;
    // TODO: grid-stride accumulate into register `sum`
    // TODO: sum = warpReduceSum(sum);
    // TODO: __shared__ float warpSums[32]; lane 0 of each warp writes warpSums[warp] = sum;
    // TODO: __syncthreads();
    // TODO: first warp loads warpSums (0 past the live warp count), warpReduceSum again
    // TODO: if (tid == 0) atomicAdd(out, blockTotal);
}

// Host entry point. in and out are DEVICE pointers; *out is already zeroed.
void solve(const float* in, float* out, int n) {
    // TODO: choose block = BLOCK and a capped grid, then launch
    //       reduce<<<grid, BLOCK>>>(in, out, n);
}
