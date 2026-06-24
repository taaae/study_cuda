# Exercise 18 — SpMV in the ELL format
> Pad every row to the same length and store it column-major — and the loads fall into place, coalesced.

## The idea

CSR is compact, but its variable row lengths are what made the loads scatter and
the lanes idle. **ELL** trades a little memory for a perfectly regular layout, and
it's a great fit when every row has *roughly the same* number of nonzeros.

The recipe: let `maxnnz` be the largest number of nonzeros in any row. Store
**exactly** `maxnnz` entries per row in two dense `maxnnz × nrows` arrays, padding
short rows. Pad with **column index 0 and value 0.0f** — a safe no-op, because
`0.0f * x[0]` adds nothing to the sum.

The twist that makes ELL fast on a GPU: store those arrays **column-major by `k`**
(the within-row slot), not row-major:

```text
element (row, k)  ->  flat index  k*nrows + row        // k = 0 .. maxnnz-1

ell_cols[k*nrows + row]  = column index of the k-th nonzero of `row` (0 if padding)
ell_vals[k*nrows + row]  = its value                                  (0 if padding)
```

So *all* the `k=0` entries of every row sit contiguously, then all the `k=1`
entries, and so on. There is **no `rowPtr`** — every row has exactly `maxnnz`
slots, so the address is pure arithmetic.

## Under the hood

A thread owns one row and loops `k = 0 .. maxnnz-1`. On step `k`, thread `row`
reads index `k*nrows + row`. Across a warp the `row` values are consecutive, so the
32 threads touch addresses `k*nrows + base, …, k*nrows + base+31` — 32 **consecutive**
words, one coalesced 128-byte transaction. *Every* step is a clean coalesced load,
and because every thread does exactly `maxnnz` iterations, there's **no
divergence** and **no load imbalance**. Branch-free, balanced, coalesced — that's
the whole appeal.

If you'd stored row-major (`row*maxnnz + k`), the warp would read addresses
`base*maxnnz + k` strided by `maxnnz` — back to uncoalesced. The column-major
layout is the entire trick.

> **The tradeoff (this is the lesson, not a footnote):**
> - **Win:** coalesced, branch-free, perfectly load-balanced when rows are uniform.
> - **Loss:** you pad *every* row up to `maxnnz`. A single dense row forces a huge
>   `maxnnz`, and then every short row stores and *multiplies* a mountain of
>   explicit zeros — wasted memory and wasted FLOPs.
> - **Production answer:** the **hybrid ELL+COO** format. Keep a modest `maxnnz` in
>   ELL for the bulk of each row and spill the few overflowing entries of unusually
>   long rows into a small COO side-list. You get ELL's coalescing without the
>   padding blowup. This is why ex17's vector kernel, plain ELL, and HYB all coexist
>   in real libraries — the right one depends on the matrix's row-length spread.

The harness builds a **near-uniform** matrix (every row has `R-1` or `R` nonzeros,
`R = 32 = maxnnz`), so ELL pays almost no padding tax — the regime where it's meant
to win. It even reports `bw_frac`: ELL is memory-bound, so you're really racing the
T4's ~320 GB/s of bandwidth, not its FLOPs.

## A picture

```text
Logical matrix rows (varying real length, padded to maxnnz=4):

  row 0:  c0 c1 c2 .            row 1:  c0 c1 .  .            row 2:  c0 c1 c2 c3
          v0 v1 v2 .                    v0 v1 .  .                    v0 v1 v2 v3

Stored COLUMN-MAJOR (k*nrows + row), warp reads down a column each step:

  flat index:   k=0 block       |   k=1 block       |   k=2 block       | ...
  ell_vals:  [ r0 r1 r2 ... rN ][ r0 r1 r2 ... rN ][ r0 r1 r2 ... rN ] ...
               ^^^^^^^^^^                                  step k=0: 32 consecutive
               one warp, step 0                            -> COALESCED 128B load
```

## Your task

Compute `y = A * x` with `A` in ELL. Edit `spmv_ell.cu`:

1. **Kernel `spmv_ell`** — one thread per row. Compute the global `row`, guard
   against `nrows`, loop `k = 0..maxnnz-1` reading `ell_cols[k*nrows+row]` and
   `ell_vals[k*nrows+row]`, accumulate `sum += val * x[col]`, then write `y[row]`.
2. **Host `solve`** — pick a block size and launch `ceil_div(nrows, block)` blocks.

You do **not** allocate memory or build the matrix — `harness.cu` does and calls
your `solve`.

### The `solve` contract

```cpp
void solve(const int* ell_cols, const float* ell_vals,
           const float* x, float* y, int nrows, int maxnnz);
```

All pointers are **device pointers**. `ell_cols`/`ell_vals` have length
`maxnnz * nrows`, **column-major** (`(row,k)` at `k*nrows + row`). `x` has length
`ncols` (big enough for any column index that appears). `y` has length `nrows`.

## Functions & syntax you'll need

| Tool | Form | What it does |
|------|------|--------------|
| global row index | `int row = blockIdx.x*blockDim.x + threadIdx.x;` | one thread per row |
| bounds guard | `if (row < nrows) { ... }` | drop the extra threads in the last block |
| **column-major index** | `k * nrows + row` | the stride that keeps the warp coalesced — **must** use `k*nrows` |
| ELL access | `int col = ell_cols[k*nrows+row]; float val = ell_vals[k*nrows+row];` | the k-th slot of this row (padding is `col=0, val=0`) |
| gather + accumulate | `sum += val * x[col];` | padding contributes `0 * x[0] = 0`, a harmless no-op |
| `ceil_div(a, b)` | from `cuda_utils.cuh` | grid sizing |

There is **no `rowPtr`** in ELL — that's the structural difference from CSR, and
the grader checks you didn't sneak one in.

## How it's graded

Run `python grade.py` (or `--check-solution` for the reference).

- **correctness** — `y == A*x` within tolerance, checked against a CPU SpMV.
- **efficiency** — the harness times a scalar CSR SpMV baseline on the same matrix;
  ELL must be at least **`speedup ≥ 1.2`** faster. It also reports `gflops` and
  `bw_frac` (fraction of peak bandwidth — the metric that actually matters here).
- **source** — you must use the column-major stride `k * nrows`, and you must
  **not** use `rowPtr` (ELL has none).

## Going deeper

- **`maxnnz` matters enormously.** Compute, for *this* harness, how much memory ELL
  uses (`maxnnz*nrows` slots) versus CSR (`nnz` slots). Because rows are uniform
  they're nearly equal — but imagine one row with 10,000 nonzeros and you'll feel
  the padding tax instantly.
- **ELL on the GPU is a memory-bandwidth story.** Two reads (col, val) per slot
  plus a scattered `x` gather; the arithmetic is trivial. Watch `bw_frac` — getting
  close to 1.0 means you're limited by physics, not by your code.
