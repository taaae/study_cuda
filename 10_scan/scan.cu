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

    // TODO: load this thread's two elements into shared memory, using 0 for
    //       out-of-range slots (the additive identity). (See README + hints.md.)

    int offset = 1;

    // --- up-sweep (reduce) over the tile -----------------------------------
    // TODO: run the Blelloch up-sweep: combine pairs up the tree, doubling the
    //       offset each level, with a barrier before each level.

    // Save the block total (root of the tree) and clear it for the down-sweep.
    if (tid == 0) {
        // TODO: record the tile total into blockSums, then zero the root so the
        //       down-sweep produces an EXCLUSIVE scan.
    }

    // --- down-sweep --------------------------------------------------------
    // TODO: run the Blelloch down-sweep: halve the offset each level and do the
    //       swap-and-add at each node, with a barrier before each level.

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
    // TODO: launch scan_block over the input (one block per tile, shBytes of
    //       dynamic shared memory).

    // Phase 2: exclusive-scan the block totals into blockOffsets.
    // For the sizes used here numBlocks <= TILE, so ONE scan_block over the
    // totals suffices (its own block-sum output can be ignored).
    // TODO: allocate a small scratch and launch one scan_block (grid = 1) to
    //       scan the block totals into blockOffsets, then free the scratch.

    // Phase 3: add offsets back.
    // TODO: launch add_offsets to add each block's offset to its output range.

    CUDA_CHECK(cudaFree(blockSums));
    CUDA_CHECK(cudaFree(blockOffsets));
}
