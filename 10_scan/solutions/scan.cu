// Reference solution — Exercise 10. Work-efficient (Blelloch) exclusive scan.
#include "cuda_utils.cuh"

#define BLOCK 512
#define TILE  (2 * BLOCK)

// Phase 1: exclusive scan of each block's TILE-sized chunk in shared memory,
// recording the chunk total into blockSums[blockIdx.x].
__global__ void scan_block(const int* in, int* out, int* blockSums, int n) {
    extern __shared__ int s[];
    int tid  = threadIdx.x;
    int base = blockIdx.x * TILE;
    int ai   = 2 * tid;
    int bi   = 2 * tid + 1;

    s[ai] = (base + ai < n) ? in[base + ai] : 0;
    s[bi] = (base + bi < n) ? in[base + bi] : 0;

    int offset = 1;

    // up-sweep
    for (int d = TILE >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int x = offset * (2 * tid + 1) - 1;
            int y = offset * (2 * tid + 2) - 1;
            s[y] += s[x];
        }
        offset <<= 1;
    }

    // save total, clear root
    if (tid == 0) {
        blockSums[blockIdx.x] = s[TILE - 1];
        s[TILE - 1] = 0;
    }

    // down-sweep
    for (int d = 1; d < TILE; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int x = offset * (2 * tid + 1) - 1;
            int y = offset * (2 * tid + 2) - 1;
            int t = s[x];
            s[x] = s[y];
            s[y] += t;
        }
    }
    __syncthreads();

    if (base + ai < n) out[base + ai] = s[ai];
    if (base + bi < n) out[base + bi] = s[bi];
}

// Phase 3: add each block's offset to its elements.
__global__ void add_offsets(int* out, const int* blockOffsets, int n) {
    int base = blockIdx.x * TILE;
    int off  = blockOffsets[blockIdx.x];
    int i0   = base + 2 * threadIdx.x;
    int i1   = base + 2 * threadIdx.x + 1;
    if (i0 < n) out[i0] += off;
    if (i1 < n) out[i1] += off;
}

void solve(const int* in, int* out, int n) {
    int numBlocks = ceil_div(n, TILE);
    size_t shBytes = TILE * sizeof(int);

    int *blockSums = nullptr, *blockOffsets = nullptr;
    CUDA_CHECK(cudaMalloc(&blockSums,    numBlocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&blockOffsets, numBlocks * sizeof(int)));

    // Phase 1
    scan_block<<<numBlocks, BLOCK, shBytes>>>(in, out, blockSums, n);

    // Phase 2: scan the block totals in a single block (numBlocks <= TILE).
    int* dummy = nullptr;
    CUDA_CHECK(cudaMalloc(&dummy, sizeof(int)));
    scan_block<<<1, BLOCK, shBytes>>>(blockSums, blockOffsets, dummy, numBlocks);
    CUDA_CHECK(cudaFree(dummy));

    // Phase 3
    add_offsets<<<numBlocks, BLOCK>>>(out, blockOffsets, n);

    CUDA_CHECK(cudaFree(blockSums));
    CUDA_CHECK(cudaFree(blockOffsets));
}
