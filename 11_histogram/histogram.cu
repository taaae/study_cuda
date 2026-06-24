// Exercise 11 — Histogram with shared-memory privatization.
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

#define NBINS 256

// Each block keeps a private 256-bin histogram in shared memory, fills it with
// cheap shared-memory atomics, then merges it into the global histogram once.
__global__ void hist_privatized(const unsigned char* data, unsigned int* hist, int n) {
    // A per-block private histogram in shared memory.
    // TODO: declare __shared__ unsigned int sHist[NBINS];

    // 1) Zero the shared histogram cooperatively (256 bins, blockDim.x threads).
    // TODO: for (int b = threadIdx.x; b < NBINS; b += blockDim.x) sHist[b] = 0;

    // TODO: __syncthreads();

    // 2) Grid-stride over the input; bump shared bins with shared-memory atomics.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        unsigned char v = data[i];
        // TODO: atomicAdd(&sHist[v], 1u);
    }

    // TODO: __syncthreads();

    // 3) Merge the private histogram into the global one (one atomic per bin).
    // TODO: for (int b = threadIdx.x; b < NBINS; b += blockDim.x)
    //           atomicAdd(&hist[b], sHist[b]);
}

// Host entry point. data is a DEVICE pointer of n bytes; hist is 256 DEVICE
// uints already zeroed by the harness.
void solve(const unsigned char* data, unsigned int* hist, int n) {
    int block = 256;
    // Enough blocks to keep the GPU busy; the grid-stride loop covers any n.
    int grid  = 1024;
    // TODO: launch hist_privatized<<<grid, block>>>(data, hist, n);
}
