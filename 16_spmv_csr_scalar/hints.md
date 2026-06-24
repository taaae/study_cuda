# Hints — Exercise 16

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — What CSR actually stores (concept)</summary>

Three arrays. `vals` and `colIdx` list the nonzeros left-to-right, top-to-bottom, with their values and columns. `rowPtr[r]` says where row `r` begins inside those arrays, and `rowPtr[r+1]` where it ends. So row `r`'s nonzeros are at indices `rowPtr[r] .. rowPtr[r+1]-1`. An empty row has `rowPtr[r] == rowPtr[r+1]`.
</details>

<details>
<summary>Hint 2 — The dot product for one row (concept)</summary>

Row `r` of `y` is the dot product of row `r` of `A` with `x`. Only the nonzeros contribute, and for nonzero `k` the matching `x` entry is `x[colIdx[k]]`:

```
y[r] = sum over k in [rowPtr[r], rowPtr[r+1])  of  vals[k] * x[colIdx[k]]
```
</details>

<details>
<summary>Hint 3 — The kernel body (code)</summary>

```cpp
__global__ void spmv_scalar(const int* rowPtr, const int* colIdx,
                            const float* vals, const float* x, float* y, int nrows) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrows) return;
    int start = rowPtr[r];
    int end   = rowPtr[r + 1];
    float sum = 0.0f;
    for (int k = start; k < end; ++k)
        sum += vals[k] * x[colIdx[k]];
    y[r] = sum;
}
```
</details>

<details>
<summary>Hint 4 — Launching one thread per row (code)</summary>

```cpp
void solve(const int* rowPtr, const int* colIdx, const float* vals,
           const float* x, float* y, int nrows) {
    int block = 256;
    int grid  = ceil_div(nrows, block);
    spmv_scalar<<<grid, block>>>(rowPtr, colIdx, vals, x, y, nrows);
}
```
</details>

<details>
<summary>Hint 5 — Common bugs (concept)</summary>

- Off-by-one: the loop bound is `rowPtr[r+1]`, **exclusive**. `rowPtr` is length `nrows+1` exactly so that `rowPtr[r+1]` is always valid.
- Don't forget the `if (r >= nrows) return;` guard — the last block has spare threads.
- Index `x` with the *column*, `x[colIdx[k]]`, not with `k`.
</details>

<details>
<summary>Hint 6 — Why this is slow on uneven matrices (concept, ties to ex17)</summary>

All 32 threads in a warp run in lockstep, so a warp can't retire until its *longest* row is done — a row with 200 nonzeros stalls 31 threads that finished at 5. And neighbouring threads read `vals[rowPtr[r]]` vs `vals[rowPtr[r+1]]`, which are far apart, so the loads aren't coalesced. Exercise 17 puts a whole *warp* on each row to fix both.
</details>
