// Reference solution — Exercise 24 (Capstone stencil).
// 5-point average with clamped boundaries, via shared-memory halo tiling.
#include "cuda_utils.cuh"

#define BX 32
#define BY 8

__device__ __forceinline__ int clampi(int v, int n) {
    return v < 0 ? 0 : (v >= n ? n - 1 : v);
}

__global__ void stencil_fast(const float* in, float* out, int width, int height) {
    __shared__ float s[BY + 2][BX + 2];

    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x * BX + tx;
    int y = blockIdx.y * BY + ty;

    // Interior: each thread loads its own pixel (clamped so edge blocks are valid).
    int cx = clampi(x, width);
    int cy = clampi(y, height);
    s[ty + 1][tx + 1] = in[cy * width + cx];

    // Halo columns (left/right): the first two columns of the block load them.
    if (tx == 0) {
        int lx = clampi(blockIdx.x * BX - 1, width);
        s[ty + 1][0] = in[cy * width + lx];
    }
    if (tx == BX - 1) {
        int rx = clampi(blockIdx.x * BX + BX, width);
        s[ty + 1][BX + 1] = in[cy * width + rx];
    }
    // Halo rows (top/bottom): the first two rows of the block load them.
    if (ty == 0) {
        int uy = clampi(blockIdx.y * BY - 1, height);
        s[0][tx + 1] = in[uy * width + cx];
    }
    if (ty == BY - 1) {
        int dy = clampi(blockIdx.y * BY + BY, height);
        s[BY + 1][tx + 1] = in[dy * width + cx];
    }
    // (Corners are not read by the 5-point stencil, so we skip them.)

    __syncthreads();

    if (x < width && y < height) {
        float c = s[ty + 1][tx + 1];
        float l = s[ty + 1][tx];
        float r = s[ty + 1][tx + 2];
        float u = s[ty][tx + 1];
        float d = s[ty + 2][tx + 1];
        out[y * width + x] = (c + l + r + u + d) * 0.2f;
    }
}

void solve(const float* in, float* out, int width, int height) {
    dim3 block(BX, BY);
    dim3 grid(ceil_div(width, BX), ceil_div(height, BY));
    stencil_fast<<<grid, block>>>(in, out, width, height);
}
