// Exercise 24 — Capstone: optimize the stencil.
// Fill in the TODOs. Do NOT add a main(); harness.cu provides it and calls solve().
//
// Formula (5-point average) with CLAMPED boundaries:
//   out(x,y) = ( in(x,y) + in(x-1,y) + in(x+1,y) + in(x,y-1) + in(x,y+1) ) / 5
//   out-of-range neighbor coords clamp to the nearest valid pixel.
#include "cuda_utils.cuh"

#define BX 32
#define BY 8

// Fast stencil. Intended approach: load a (BX+2)x(BY+2) tile (interior + 1-pixel
// halo) into shared memory, __syncthreads(), then read the 5 neighbors from shared.
__global__ void stencil_fast(const float* in, float* out, int width, int height) {
    // TODO: declare a (BX+2)x(BY+2) shared tile (interior + 1-pixel halo).

    int x = blockIdx.x * BX + threadIdx.x;
    int y = blockIdx.y * BY + threadIdx.y;

    // Helper idea: clamp a coordinate into [0, n-1] before reading global memory.

    // TODO: load each thread's interior pixel into the shared tile, clamping
    //       coordinates so threads past the image edge still read a valid pixel.

    // TODO: load the 1-pixel halo border (edge threads fetch the extra row/column,
    //       same clamp), then __syncthreads().

    // TODO: read the 5 neighbors from the shared tile, apply the 5-point average,
    //       and write out the result for in-bounds pixels.
    // (See README's function table and hints.md if stuck.)
    (void)x; (void)y; (void)width; (void)height;
}

void solve(const float* in, float* out, int width, int height) {
    // TODO: launch stencil_fast with BX x BY blocks covering the image. (See README + hints.md.)
}
