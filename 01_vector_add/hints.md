# Hints — Exercise 01

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — How a thread finds the element it owns (no code)</summary>

Every thread needs a *unique* index into the array. A block knows its own position (`blockIdx.x`) and its size (`blockDim.x`); a thread knows its position inside the block (`threadIdx.x`). Combine them so block 0 covers elements `0..blockDim.x-1`, block 1 the next chunk, and so on.
</details>

<details>
<summary>Hint 2 — The boundary check (concept)</summary>

If `n` isn't a multiple of your block size, the last block has threads whose index is past the end of the array. Those threads must do nothing. One `if` around the write is enough.
</details>

<details>
<summary>Hint 3 — The kernel body (code)</summary>

```cpp
__global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```
</details>

<details>
<summary>Hint 4 — Choosing the launch configuration (concept)</summary>

A block size of 128–256 is a safe default. You need enough blocks to cover all `n` elements, rounding **up** so the last partial chunk still gets a block. `ceil_div(n, block)` from `cuda_utils.cuh` does the rounding.
</details>

<details>
<summary>Hint 5 — The full solve (code)</summary>

```cpp
void solve(const float* a, const float* b, float* c, int n) {
    int block = 256;
    int grid  = ceil_div(n, block);
    vadd<<<grid, block>>>(a, b, c, n);
}
```
</details>
