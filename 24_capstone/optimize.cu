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
    // TODO: __shared__ float s[BY + 2][BX + 2];

    int x = blockIdx.x * BX + threadIdx.x;
    int y = blockIdx.y * BY + threadIdx.y;

    // Helper idea: clamp a coordinate into [0, n-1] before reading global memory.
    //   auto clampi = [](int v, int n) { return v < 0 ? 0 : (v >= n ? n - 1 : v); };

    // TODO: load the interior pixel for this thread into s[ty+1][tx+1]
    //       (clamp x,y so threads past the image edge still read a valid pixel).

    // TODO: load the halo. The simplest correct scheme: a few threads on each
    //       edge of the block load the extra border row/column, applying the same
    //       clamp. Then __syncthreads().

    // TODO: compute from shared and write out[y*width + x] if (x<width && y<height):
    //   float c = s[ty+1][tx+1];
    //   float l = s[ty+1][tx], r = s[ty+1][tx+2];
    //   float u = s[ty][tx+1], d = s[ty+2][tx+1];
    //   out[y*width + x] = (c + l + r + u + d) * 0.2f;
    (void)x; (void)y; (void)width; (void)height;
}

void solve(const float* in, float* out, int width, int height) {
    // TODO:
    //   dim3 block(BX, BY);
    //   dim3 grid(ceil_div(width, BX), ceil_div(height, BY));
    //   stencil_fast<<<grid, block>>>(in, out, width, height);
}
