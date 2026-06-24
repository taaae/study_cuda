// Reference solution — Exercise 06.
#include "cuda_utils.cuh"

#ifndef BLOCK
#define BLOCK 256
#endif

__global__ void reduce(const float* in, float* out, int n) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    // First add during load: grid-stride accumulate into a register.
    float sum = 0.f;
    for (int i = gid; i < n; i += stride) sum += in[i];
    sdata[tid] = sum;
    __syncthreads();

    // Sequential addressing: contiguous active threads, no bank conflicts.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // One atomic per block.
    if (tid == 0) atomicAdd(out, sdata[0]);
}

void solve(const float* in, float* out, int n) {
    int block = BLOCK;
    int grid = ceil_div(n, block);
    if (grid > 4096) grid = 4096;   // cap so blocks stay full; grid-stride covers the rest
    reduce<<<grid, block>>>(in, out, n);
}
