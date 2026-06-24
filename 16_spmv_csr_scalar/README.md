# Exercise 16 — Sparse Matrix–Vector Multiply (CSR, scalar)

**New concepts:** **sparse linear algebra**, the **CSR (Compressed Sparse Row)** storage format, and **scalar SpMV** — one thread per matrix row.

## The task

Compute `y = A * x` where `A` is a sparse matrix stored in CSR and `x`, `y` are dense vectors. Assign **one thread to each row**; that thread sums the products of its row's nonzeros with the matching entries of `x`.

Edit `spmv.cu` and fill in the `TODO`s:

1. The `__global__` kernel `spmv_scalar` — one thread per row, looping over that row's nonzeros.
2. The host function `solve` — launch one thread per row.

### `solve` signature (the contract)

```cpp
void solve(const int* rowPtr, const int* colIdx, const float* vals,
           const float* x, float* y, int nrows);
```

All pointers are **device pointers**. `rowPtr` has length `nrows + 1`; `colIdx` and `vals` have length `nnz = rowPtr[nrows]`. `x` is length `ncols`, `y` is length `nrows`.

## CSR, with a worked example

CSR stores only the nonzeros, row by row, with three arrays:

- `vals[nnz]`   — the nonzero values, left-to-right, top-to-bottom.
- `colIdx[nnz]` — the column of each value (same order as `vals`).
- `rowPtr[nrows+1]` — where each row *starts* in `vals`/`colIdx`. Row `r` occupies indices `rowPtr[r] .. rowPtr[r+1]-1`. The last entry `rowPtr[nrows] == nnz`.

Take this 4×4 matrix (dots are zeros):

```
        col:  0    1    2    3
row 0 [ 10    .    .   20 ]
row 1 [  .   30    .    . ]
row 2 [  .   40   50    . ]
row 3 [  .    .    .    . ]   <- empty row
```

Reading the nonzeros row by row gives:

```
vals   = [10, 20, 30, 40, 50]
colIdx = [ 0,  3,  1,  1,  2]
rowPtr = [ 0,  2,  3,  5,  5]
          r0  r1  r2  r3  end
```

Row 2 spans `rowPtr[2]=3 .. rowPtr[3]-1=4`, i.e. `vals[3..4]=[40,50]` at columns `[1,2]`. Row 3 is empty: `rowPtr[3]==rowPtr[4]`, so its loop runs zero times.

## The per-row loop

For row `r`, the dot product is:

```
y[r] = sum over k in [rowPtr[r], rowPtr[r+1])  of  vals[k] * x[colIdx[k]]
```

So each thread:

1. reads its `start = rowPtr[r]` and `end = rowPtr[r+1]`,
2. accumulates `sum += vals[k] * x[colIdx[k]]` for `k` in `[start, end)`,
3. writes `y[r] = sum`.

## The weakness (fixed in exercise 17)

Scalar SpMV gives each row to **one** thread. If rows have very different numbers of nonzeros, threads in the same warp finish at wildly different times — the warp stalls on its longest row (**load imbalance** and **divergence**). Worse, consecutive threads read `vals[rowPtr[r]]`, `vals[rowPtr[r+1]]`, … which are *far apart* in memory — the reads of `colIdx`/`vals` are **uncoalesced**. Exercise 17 (one *warp* per row) fixes both.

## Syntax you'll need

Nothing new — a kernel, a grid-stride or one-thread-per-row launch, and a plain inner loop. Helper available: `ceil_div(a, b)`.

## Grading (`!python grade.py`)

- **correctness** — `y` matches a CPU CSR SpMV within tolerance.
- **metric** — the harness reports `ms`, effective `gflops` (`2*nnz/time`), and a `bw_frac` proxy. There's no strict perf threshold beyond correctness; the lesson here is the format, not the speed.
- **source** — you must actually index `rowPtr`, and you must **not** use `__shfl` (that's the warp-per-row technique in exercise 17).

Run `python grade.py --check-solution` to grade the reference solution instead of yours.
