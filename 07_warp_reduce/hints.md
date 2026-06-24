# Hints — Exercise 07

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — Why shuffle exists (no code)</summary>

The 32 threads of a warp already run in lockstep. Shuffle instructions let one lane read another lane's **register** directly — no shared memory, no `__syncthreads()`. So the within-warp part of a reduction needs neither. You only fall back to a tiny shared array to glue the per-warp results together.
</details>

<details>
<summary>Hint 2 — The butterfly (concept)</summary>

To fold 32 lane-values into lane 0, add the value from `delta` lanes higher, halving `delta` each step: `16, 8, 4, 2, 1`. After five steps lane 0 holds the sum. `__shfl_down_sync(mask, v, delta)` is exactly "give me lane (mylane+delta)'s `v`". The mask `0xffffffff` says all 32 lanes participate.
</details>

<details>
<summary>Hint 3 — warpReduceSum (code)</summary>

```cpp
__device__ float warpReduceSum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;   // valid in lane 0
}
```
</details>

<details>
<summary>Hint 4 — Combining warps (concept)</summary>

A block of `BLOCK` threads has `BLOCK/32` warps. After `warpReduceSum`, lane 0 of each warp has that warp's sum. Write those to `__shared__ float warpSums[32]` indexed by warp id. After a `__syncthreads()`, let the **first warp** load `warpSums` (use 0 for slots beyond the number of live warps) and `warpReduceSum` once more. Thread 0 then `atomicAdd`s the block total.
</details>

<details>
<summary>Hint 5 — The kernel body (code)</summary>

```cpp
__global__ void reduce(const float* in, float* out, int n) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    float sum = 0.f;
    for (int i = gid; i < n; i += stride) sum += in[i];

    sum = warpReduceSum(sum);

    __shared__ float warpSums[32];
    if (lane == 0) warpSums[warp] = sum;
    __syncthreads();

    int numWarps = blockDim.x >> 5;
    float blockTotal = 0.f;
    if (warp == 0) {
        float v = (lane < numWarps) ? warpSums[lane] : 0.f;
        blockTotal = warpReduceSum(v);
    }
    if (tid == 0) atomicAdd(out, blockTotal);
}
```
</details>

<details>
<summary>Hint 6 — The full solve (code)</summary>

```cpp
void solve(const float* in, float* out, int n) {
    int block = BLOCK;
    int grid = ceil_div(n, block);
    if (grid > 4096) grid = 4096;   // grid-stride loop covers the rest
    reduce<<<grid, block>>>(in, out, n);
}
```
</details>
