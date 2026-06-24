# Hints — Exercise 21

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — Why two phases need a grid barrier (concept)</summary>

Phase 2 (scaling) must not start until the **complete** sum of squares is known,
which only happens after **every** block has done its `atomicAdd`. A
`__syncthreads()` only waits for one block, so it can't help. The grid barrier
`this_grid().sync()` waits for all blocks — that's exactly the guarantee you need.
</details>

<details>
<summary>Hint 2 — Co-residency, and why the grid must be "small" (concept)</summary>

Grid sync deadlocks if some blocks haven't started running yet (they can never
reach the barrier). So you must launch only as many blocks as physically fit on
the GPU at once, and use a **grid-stride loop** so that smaller grid still
touches every element. Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` ×
`multiProcessorCount` for the count.
</details>

<details>
<summary>Hint 3 — The group object and the barrier (code)</summary>

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void normalize_kernel(float* data, int n, float* ssq) {
    cg::grid_group grid = cg::this_grid();
    // phase 1 ...
    grid.sync();
    // phase 2 ...
}
```
</details>

<details>
<summary>Hint 4 — Phase 1: partial reduce + atomicAdd (code)</summary>

```cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
float local = 0.f;
for (int i = tid; i < n; i += stride) local += data[i] * data[i];

__shared__ float sm[256];
sm[threadIdx.x] = local;
__syncthreads();
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
    __syncthreads();
}
if (threadIdx.x == 0) atomicAdd(ssq, sm[0]);
```
</details>

<details>
<summary>Hint 5 — Phase 2 after the barrier (code)</summary>

```cpp
grid.sync();
float inv = rsqrtf(*ssq);
for (int i = tid; i < n; i += stride) data[i] *= inv;
```
</details>

<details>
<summary>Hint 6 — The cooperative launch in solve (code)</summary>

```cpp
int block = 256, blocksPerSM = 0;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, normalize_kernel, block, 0);
cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
int grid = blocksPerSM * prop.multiProcessorCount;

float* ssq; cudaMalloc(&ssq, sizeof(float)); cudaMemset(ssq, 0, sizeof(float));
void* args[] = { (void*)&data, (void*)&n, (void*)&ssq };
cudaLaunchCooperativeKernel((void*)normalize_kernel, dim3(grid), dim3(block), args, 0, 0);
cudaFree(ssq);   // implicitly syncs, so the kernel has finished
```

Note `grade.py` compiles with `-rdc=true`; cooperative launch needs relocatable
device code.
</details>
