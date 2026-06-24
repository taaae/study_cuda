// Reference solution — Exercise 07.
#include "cuda_utils.cuh"

#ifndef BLOCK
#define BLOCK 256
#endif

__device__ float warpReduceSum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;   // valid in lane 0
}

__global__ void reduce(const float* in, float* out, int n) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    // First add during load: grid-stride into a register.
    float sum = 0.f;
    for (int i = gid; i < n; i += stride) sum += in[i];

    // Intra-warp reduction in registers (no shared memory, no __syncthreads).
    sum = warpReduceSum(sum);

    // Combine the per-warp partials.
    __shared__ float warpSums[32];
    if (lane == 0) warpSums[warp] = sum;
    __syncthreads();

    int numWarps = blockDim.x >> 5;
    float blockTotal = 0.f;
    if (warp == 0) {
        float v = (lane < numWarps) ? warpSums[lane] : 0.f;
        blockTotal = warpReduceSum(v);
    }
    if (tid == 0) atomicAdd(out, blockTotal);
}

void solve(const float* in, float* out, int n) {
    int block = BLOCK;
    int grid = ceil_div(n, block);
    if (grid > 4096) grid = 4096;
    reduce<<<grid, block>>>(in, out, n);
}
