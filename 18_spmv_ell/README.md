# Exercise 18 — SpMV in the ELL format

**New concepts:** the **ELL** sparse storage format, why a **column-major** padded
layout makes the memory accesses **coalesced**, and the storage / load-balancing
**tradeoff** against CSR.

## Recap: what was wrong with CSR-scalar

In `16_spmv_csr_scalar` you ran one thread per row over a CSR matrix
(`rowPtr[nrows+1]`, `colIdx[nnz]`, `vals[nnz]`). Within a warp, thread `r` reads
`vals[rowPtr[r] + k]` while thread `r+1` reads `vals[rowPtr[r+1] + k]`. Because
rows have different lengths and start at different offsets, the 32 threads of a
warp touch 32 *unrelated* addresses on every step — the loads are **uncoalesced**,
and threads on short rows sit idle while one long row finishes (**load imbalance**).

## The ELL format

ELL fixes the *layout* for matrices whose rows have **near-uniform** length.
Pick `maxnnz` = the maximum number of nonzeros in any row. Store **exactly**
`maxnnz` entries per row in two dense `maxnnz × nrows` arrays, padding short rows.

Critically, the arrays are stored **COLUMN-MAJOR by `k`** (the within-row slot):

```
element (row, k)  ->  index  k * nrows + row        // k = 0 .. maxnnz-1
```

So all the *k = 0* entries of every row are contiguous, then all the *k = 1*
entries, and so on. Padding (slots `k >= row_length`) uses **column index 0 and
value 0.0f** — a safe no-op: `0.0f * x[0]` adds nothing.

```
ell_cols[k*nrows + row]   column index of the k-th nonzero of `row` (0 if padding)
ell_vals[k*nrows + row]   its value                                 (0 if padding)
```

### Why column-major → coalesced

A thread handles one row and loops `k = 0 .. maxnnz-1`. On step `k`, thread `row`
reads index `k*nrows + row`. Across a warp the `row` values are consecutive, so
the 32 threads read 32 **consecutive** addresses `k*nrows + (base .. base+31)` —
one coalesced 128-byte transaction. Every step is a clean, regular, coalesced
load. That is the whole point of the column-major layout.

### The tradeoff (read this — it's the lesson)

- **Win:** coalesced, branch-free, perfectly load-balanced when rows are uniform.
- **Loss:** you pad *every* row up to `maxnnz`. One dense row forces a huge
  `maxnnz`, and you then store (and multiply) mountains of explicit zeros for all
  the short rows — wasted memory and wasted FLOPs.
- **Production answer:** the **hybrid ELL+COO** format. Keep a modest `maxnnz` in
  ELL for the bulk of each row, and spill the few overflowing entries of unusually
  long rows into a small COO side-list processed separately. You get ELL's
  coalescing without ELL's padding blowup.

## The task

Compute `y = A * x` with `A` in ELL. Edit `spmv_ell.cu`:

1. `__global__ void spmv_ell(...)` — one thread per row. Loop `k = 0..maxnnz-1`,
   read `ell_cols[k*nrows + row]` and `ell_vals[k*nrows + row]`, accumulate
   `sum += val * x[col]`, then write `y[row] = sum`.
2. `solve(...)` — pick a block size and launch with `ceil_div(nrows, block)` blocks.

You do **not** allocate memory or build the matrix — `harness.cu` does, and calls
your `solve(...)`.

### `solve` signature (the contract)

```cpp
void solve(const int* ell_cols, const float* ell_vals,
           const float* x, float* y, int nrows, int maxnnz);
```

All pointers are **device pointers**. `ell_cols` and `ell_vals` are length
`maxnnz * nrows`, **column-major** (`(row,k)` at `k*nrows + row`). `x` is length
`ncols` (large enough for any column index that appears). `y` is length `nrows`.

## Syntax / reference

- Global row index: `int row = blockIdx.x * blockDim.x + threadIdx.x;`
- Guard: `if (row < nrows) { ... }`.
- The column-major access **must** use the stride `k * nrows`:
  ```cpp
  for (int k = 0; k < maxnnz; ++k) {
      int   col = ell_cols[k * nrows + row];
      float val = ell_vals[k * nrows + row];
      sum += val * x[col];
  }
  ```
- There is **no `rowPtr`** in ELL — every row has exactly `maxnnz` slots.

## Grading (`!python grade.py`)

- **correctness** — `y == A*x` within tolerance (checked vs a CPU SpMV).
- **efficiency** — on the near-uniform matrix the harness also runs a scalar CSR
  SpMV baseline; ELL must be at least **1.2×** faster (`speedup >= 1.2`).
- **source** — you must use the column-major stride `k * nrows`, and you must
  **not** use `rowPtr` (ELL has none).

Run `python grade.py --check-solution` to grade the reference solution instead.
