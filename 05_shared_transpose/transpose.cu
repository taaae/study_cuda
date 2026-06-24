// Exercise 05 — Tiled transpose with shared memory + bank-conflict avoidance.
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
#include "cuda_utils.cuh"

#define TILE 32

// out = in^T for an n x n row-major matrix, staged through a shared tile so that
// BOTH the global read and the global write are coalesced. Pad the tile to avoid
// 32-way shared-memory bank conflicts on the transposed read.
__global__ void transpose(const float* in, float* out, int n) {
    // TODO: declare a TILE x TILE shared tile, padded by one column to dodge bank
    //       conflicts on the transposed access.
    // TODO: do a coalesced load of this block's tile from `in`, then __syncthreads().
    // TODO: do a coalesced store to the transposed block in `out`, reading the tile
    //       with swapped indices. Guard both stages against the matrix bounds.
    // (See README's function table and hints.md if stuck.)
}

// Host entry point. in, out are DEVICE pointers to n*n floats, row-major.
void solve(const float* in, float* out, int n) {
    dim3 block(TILE, TILE);
    dim3 grid(ceil_div(n, TILE), ceil_div(n, TILE));
    transpose<<<grid, block>>>(in, out, n);
}
