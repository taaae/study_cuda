# Hints — Exercise 18 (SpMV in ELL)

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — What each thread owns (no code)</summary>

This is still one-thread-per-row, just like CSR-scalar. Thread `row` computes
`y[row]`. The only thing that changed is **where the row's nonzeros live**: not in
a `rowPtr`-delimited slice, but in `maxnnz` fixed slots spread across the two ELL
arrays. So the loop bound is the same for every thread: `k = 0 .. maxnnz-1`.
</details>

<details>
<summary>Hint 2 — Finding slot (row, k) in a column-major array (concept)</summary>

The arrays are column-major **by k**: all the `k=0` entries of every row come
first (one per row), then all the `k=1` entries, and so on. So the entry for
`(row, k)` is at offset `k*nrows + row`. The `k*nrows` part is the stride; adding
`row` picks your row within that block of `nrows` entries.

This is exactly what makes the access coalesced: at a fixed `k`, consecutive
threads (`row`, `row+1`, ...) read consecutive addresses.
</details>

<details>
<summary>Hint 3 — Why you don't need to know each row's real length (concept)</summary>

Padding slots store column index `0` and value `0.0f`. So `val * x[col]` is
`0.0f * x[0] = 0.0f` — it adds nothing. You can blindly loop the full `maxnnz`
range for every row and the padding silently contributes zero. No per-row length,
no `rowPtr`, no branching.
</details>

<details>
<summary>Hint 4 — The kernel body (code)</summary>

```cpp
__global__ void spmv_ell(const int* ell_cols, const float* ell_vals,
                         const float* x, float* y, int nrows, int maxnnz) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;
    float sum = 0.0f;
    for (int k = 0; k < maxnnz; ++k) {
        int   col = ell_cols[k * nrows + row];   // column-major stride
        float val = ell_vals[k * nrows + row];
        sum += val * x[col];
    }
    y[row] = sum;
}
```
</details>

<details>
<summary>Hint 5 — The full solve (code)</summary>

```cpp
void solve(const int* ell_cols, const float* ell_vals,
           const float* x, float* y, int nrows, int maxnnz) {
    int block = 256;
    int grid  = ceil_div(nrows, block);
    spmv_ell<<<grid, block>>>(ell_cols, ell_vals, x, y, nrows, maxnnz);
}
```
</details>

<details>
<summary>Hint 6 — Why ELL beats CSR-scalar here (concept)</summary>

On the near-uniform matrix the grader builds, ELL wins because every warp step is
one coalesced 128-byte load (`k*nrows + row` is contiguous across the warp) and
all threads do the same amount of work. Scalar CSR instead has each thread chase
its own `rowPtr[row]` offset, so the 32 loads scatter and short rows wait on long
ones. If you accidentally index ELL as if it were row-major
(`row*maxnnz + k`), you lose the coalescing and the speedup check will fail.
</details>
