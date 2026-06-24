// Reference solution — Exercise 11. Shared-memory privatized histogram.
#include "cuda_utils.cuh"

#define NBINS 256

__global__ void hist_privatized(const unsigned char* data, unsigned int* hist, int n) {
    __shared__ unsigned int sHist[NBINS];

    for (int b = threadIdx.x; b < NBINS; b += blockDim.x) sHist[b] = 0;
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        atomicAdd(&sHist[data[i]], 1u);
    }
    __syncthreads();

    for (int b = threadIdx.x; b < NBINS; b += blockDim.x)
        atomicAdd(&hist[b], sHist[b]);
}

void solve(const unsigned char* data, unsigned int* hist, int n) {
    int block = 256;
    int grid  = 1024;
    hist_privatized<<<grid, block>>>(data, hist, n);
}
