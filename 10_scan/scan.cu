// Exercise 10 — Work-efficient exclusive scan (prefix sum).
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

// Each block scans TILE = 2*BLOCK elements (2 elements per thread).
// With BLOCK=512, TILE=1024, the single-pass phase-2 design works for any
// n up to TILE*TILE ~= 1M (then numBlocks <= TILE, so one extra scan fits).
#define BLOCK 512
#define TILE  (2 * BLOCK)

// ---------------------------------------------------------------------------
// Phase 1: exclusive Blelloch scan of each block's TILE-sized chunk in shared
// memory. Writes the scanned chunk back to `out` and the chunk's TOTAL into
// blockSums[blockIdx.x]. Out-of-range slots load 0 (identity for addition).
__global__ void scan_block(const int* in, int* out, int* blockSums, int n) {
    extern __shared__ int s[];        // TILE ints of dynamic shared memory
    int tid    = threadIdx.x;
    int base   = blockIdx.x * TILE;   // first global element this block owns
    int ai     = 2 * tid;             // this thread's two local slots
    int bi     = 2 * tid + 1;

    // Load two elements (0 past the end). Use base+ai and base+bi globally.
    // TODO: s[ai] = (base + ai < n) ? in[base + ai] : 0;
    // TODO: s[bi] = (base + bi < n) ? in[base + bi] : 0;

    int offset = 1;

    // --- up-sweep (reduce) over the tile -----------------------------------
    // for d = TILE/2 down to 1: thread tid<d combines s[off*(2tid+1)-1] into
    // s[off*(2tid+2)-1]; double offset each step. Barrier before each level.
    // TODO: write the up-sweep loop here.

    // Save the block total (root of the tree) and clear it for the down-sweep.
    if (tid == 0) {
        // TODO: blockSums[blockIdx.x] = s[TILE - 1];
        // TODO: s[TILE - 1] = 0;
    }

    // --- down-sweep --------------------------------------------------------
    // for d = 1 up to TILE: halve offset; thread tid<d does the swap-and-add:
    //   t = s[ai']; s[ai'] = s[bi']; s[bi'] += t;   (ai',bi' use offset)
    // Barrier before each level.
    // TODO: write the down-sweep loop here.

    __syncthreads();

    // Store the exclusive results back to global memory (guard the range).
    if (base + ai < n) out[base + ai] = s[ai];
    if (base + bi < n) out[base + bi] = s[bi];
}

// ---------------------------------------------------------------------------
// Phase 3: add each block's offset (the scanned block sum) to every element
// that block produced. Block 0 adds 0, so it is a no-op there.
__global__ void add_offsets(int* out, const int* blockOffsets, int n) {
    int base = blockIdx.x * TILE;
    int off  = blockOffsets[blockIdx.x];
    int i0   = base + 2 * threadIdx.x;
    int i1   = base + 2 * threadIdx.x + 1;
    if (i0 < n) out[i0] += off;
    if (i1 < n) out[i1] += off;
}

// ---------------------------------------------------------------------------
// Host entry point. in/out are DEVICE pointers of length n.
void solve(const int* in, int* out, int n) {
    int numBlocks = ceil_div(n, TILE);
    size_t shBytes = TILE * sizeof(int);

    // Scratch: one total per block, plus its exclusive scan.
    int *blockSums = nullptr, *blockOffsets = nullptr;
    CUDA_CHECK(cudaMalloc(&blockSums,    numBlocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&blockOffsets, numBlocks * sizeof(int)));

    // Phase 1: per-block scan + record block totals.
    // TODO: launch scan_block<<<numBlocks, BLOCK, shBytes>>>(in, out, blockSums, n);

    // Phase 2: exclusive-scan the block totals into blockOffsets.
    // For the sizes used here numBlocks <= TILE, so ONE scan_block over the
    // totals suffices (its own block-sum output can be ignored). Reuse the
    // kernel: scan_block(blockSums, blockOffsets, <scratch>, numBlocks).
    // TODO: allocate a 1-int scratch and launch one scan_block to scan the
    //       block totals (grid = 1 block). Then free that scratch.

    // Phase 3: add offsets back.
    // TODO: launch add_offsets<<<numBlocks, BLOCK>>>(out, blockOffsets, n);

    CUDA_CHECK(cudaFree(blockSums));
    CUDA_CHECK(cudaFree(blockOffsets));
}
