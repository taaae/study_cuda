# Hints — Exercise 15

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — What occupancy buys you (concept)</summary>

A memory-bound kernel spends most of its time waiting on global-memory loads. The GPU hides that latency by switching to *another* ready warp. So you want enough resident warps per SM to keep the pipes busy — but only *enough*. Once latency is hidden, more warps don't help, and forcing occupancy up by cutting registers can backfire (spills). The occupancy API finds a config that maximizes theoretical occupancy, which is a good default.
</details>

<details>
<summary>Hint 2 — The kernel (code)</summary>

```cpp
__global__ void map_kernel(const float* in, float* out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        out[i] = in[i] * 2.0f + 1.0f;
    }
}
```

Grid-stride means a launch of *any* size still touches every element.
</details>

<details>
<summary>Hint 3 — Asking the API for a config (code)</summary>

```cpp
int minGridSize = 0, blockSize = 0;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                   map_kernel,
                                   0,    // dynamic shared mem per block
                                   0);   // no block-size cap
```

After this, `blockSize` is the occupancy-maximizing block size and `minGridSize` is the smallest grid that saturates the device.
</details>

<details>
<summary>Hint 4 — Choosing the grid (concept)</summary>

`minGridSize` is enough to fill the GPU, but if `n` is small you don't want more blocks than elements. Use `grid = min(minGridSize, ceil_div(n, blockSize))`, and clamp to at least 1. Because the kernel is grid-stride, a grid smaller than `ceil_div(n, blockSize)` is fine too — each thread just loops more times.
</details>

<details>
<summary>Hint 5 — The full solve (code)</summary>

```cpp
void solve(const float* in, float* out, int n) {
    int minGridSize = 0, blockSize = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, map_kernel, 0, 0));
    int grid = ceil_div(n, blockSize);
    if (grid > minGridSize) grid = minGridSize;
    if (grid < 1) grid = 1;
    map_kernel<<<grid, blockSize>>>(in, out, n);
}
```
</details>

<details>
<summary>Hint 6 — Inspecting occupancy for a block size (optional)</summary>

To see how many blocks the chosen size fits per SM (not required to pass):

```cpp
int blocks = 0;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, map_kernel, blockSize, 0);
```

Multiply `blocks * blockSize / warpSize` by the SM count for the resident-warp estimate behind theoretical occupancy.
</details>
