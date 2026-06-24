# Exercise 16 — Sparse matrix–vector multiply (CSR, scalar)
> Most big matrices are 99% zeros. Storing and multiplying them densely is madness — meet CSR.

## The idea
Welcome to sparse linear algebra. A graph's adjacency matrix, a finite-element stiffness matrix,
a recommender's user–item matrix — these are *enormous* but mostly **zero**. A million-row
matrix stored densely is a trillion floats (terabytes); stored as only its nonzeros it might be
a few hundred megabytes. The standard layout for "only the nonzeros, organized by row" is
**CSR — Compressed Sparse Row**, and it's the format you'll meet everywhere.

Our job is **SpMV**: compute `y = A * x`, where `A` is sparse (CSR) and `x`, `y` are dense
vectors. The simplest GPU strategy is **scalar SpMV** — give **one thread to each row**. That
thread walks its row's nonzeros and accumulates the dot product with `x`. Simple, correct, and a
perfect setup for the performance problem exercise 17 will fix.

## Under the hood
CSR stores the matrix as **three arrays**:

- `vals[nnz]`   — the nonzero values, read left-to-right, top-to-bottom (`nnz` = number of nonzeros).
- `colIdx[nnz]` — the column of each value, in the same order as `vals`.
- `rowPtr[nrows+1]` — where each row *starts*. Row `r` occupies indices `rowPtr[r] .. rowPtr[r+1]-1`
  in `vals`/`colIdx`. The final entry `rowPtr[nrows] == nnz`.

`rowPtr` is the clever bit: it's a prefix sum of per-row nonzero counts, so `rowPtr[r+1]-rowPtr[r]`
is exactly how many nonzeros row `r` has — and an empty row just has `rowPtr[r] == rowPtr[r+1]`.

### A worked example
Take this 4×4 matrix (dots are zeros):
```text
        col:  0    1    2    3
row 0 [ 10    .    .   20 ]
row 1 [  .   30    .    . ]
row 2 [  .   40   50    . ]
row 3 [  .    .    .    . ]   <- empty row
```
Reading the nonzeros row by row:
```text
            idx:   0    1    2    3    4
          vals = [10,  20,  30,  40,  50]
        colIdx = [ 0,   3,   1,   1,   2]

        rowPtr = [ 0,   2,   3,   5,   5]
                  r0   r1   r2   r3  end(=nnz)
```
Read it like this:
- **Row 0** spans `rowPtr[0]=0 .. rowPtr[1]-1=1` → `vals[0..1]=[10,20]` at columns `[0,3]`.
- **Row 2** spans `rowPtr[2]=3 .. rowPtr[3]-1=4` → `vals[3..4]=[40,50]` at columns `[1,2]`.
- **Row 3** is empty: `rowPtr[3]==rowPtr[4]==5`, so its loop runs **zero** times.

The per-row dot product is then:
```text
y[r] = Σ  vals[k] * x[colIdx[k]]   for k in [rowPtr[r], rowPtr[r+1])
```
So each thread: reads `start = rowPtr[r]` and `end = rowPtr[r+1]`, accumulates
`sum += vals[k] * x[colIdx[k]]` over `k` in `[start, end)`, and writes `y[r] = sum`.

> **Why SpMV is memory-bound, not compute-bound:** each nonzero does one multiply-add (2 FLOPs)
> but touches a `vals[k]` (4 B), a `colIdx[k]` (4 B), and a *scattered* `x[colIdx[k]]` read. The
> arithmetic intensity is tiny — performance lives or dies on memory traffic, not the ALU.

## The weakness (and a teaser for exercise 17)
Scalar SpMV hands each row to **one** thread, and that causes two problems on a warp (32 threads):

1. **Load imbalance / divergence.** If one row has 200 nonzeros and its 31 neighbors have 10,
   the whole warp stalls on the slow thread — the warp runs as long as its *longest* row.
2. **Uncoalesced reads.** Consecutive threads read `vals[rowPtr[r]]`, `vals[rowPtr[r+1]]`, … which
   are *far apart* in memory. A warp's 32 loads don't form one tidy 128-byte transaction; they
   scatter, wasting bandwidth — and bandwidth is exactly what SpMV is bound by.

```text
scalar SpMV — one thread per row, threads stride across memory:

  vals: [ r0 r0 │ r1 r1 r1 │ r2 │ r3 r3 r3 r3 │ ... ]
          ▲t0        ▲t1      ▲t2   ▲t3
          threads t0..t3 read from scattered offsets → uncoalesced
```
Exercise 17 assigns **one warp per row**: 32 threads sweep a single row's nonzeros *contiguously*
(coalesced!) and combine with a warp shuffle. Keep this version's numbers to compare against.

## Your task
Edit `spmv.cu` and fill the TODOs:
1. The `__global__` kernel `spmv_scalar` — one thread per row; the row index `r` is already
   computed and bounds-checked for you. Read `start`/`end`, loop, accumulate, write `y[r]`.
2. The host function `solve` — launch one thread per row (e.g. `block = 256`,
   `grid = ceil_div(nrows, block)`).

### The `solve` contract
```cpp
void solve(const int* rowPtr, const int* colIdx, const float* vals,
           const float* x, float* y, int nrows);
```
All pointers are **device pointers**. `rowPtr` has length `nrows + 1`; `colIdx` and `vals` have
length `nnz = rowPtr[nrows]`; `x` is length `ncols` and `y` is length `nrows`. (The harness builds
a random 65536×65536 matrix with ~64 nonzeros/row.)

## Functions & syntax you'll need
Nothing new — just a kernel and CSR indexing. The pieces:

| Piece | Meaning |
| --- | --- |
| `rowPtr[r]`, `rowPtr[r+1]` | Start (inclusive) and end (exclusive) of row `r`'s nonzeros. |
| `colIdx[k]` | Column of the `k`-th nonzero — use it to index `x`. |
| `vals[k]` | Value of the `k`-th nonzero. |
| `x[colIdx[k]]` | The scattered ("gather") read of the dense vector. |
| `blockIdx.x*blockDim.x + threadIdx.x` | The usual global thread index → row `r`. |
| `ceil_div(a, b)` | Helper from `cuda_utils.cuh` for grid sizing. |

The inner loop is a plain C loop:
```cpp
float sum = 0.0f;
for (int k = start; k < end; ++k)
    sum += vals[k] * x[colIdx[k]];
```

## How it's graded
Run `python grade.py` (`!python grade.py` on Colab). It checks:
- **correctness** — `y` matches a CPU CSR SpMV within tolerance.
- **metrics** — the harness reports `ms`, effective `gflops` (`2*nnz/time`), and a `bw_frac` proxy.
  There's **no strict perf threshold** here — the lesson is the *format*, not the speed.
- **source** — you must index `rowPtr`, and you must **not** use `__shfl` (warp shuffle is
  exercise 17's technique).

`python grade.py --check-solution` grades the reference solution instead of yours.

## Going deeper
Note how the harness counts bytes: `vals + colIdx` (nnz each) `+ rowPtr + y`, but **not** the
`x` reads — because those are scattered and impossible to count cleanly, which is itself a hint
about why SpMV is hard to optimize. When you reach exercise 17, run both and compare `gflops`:
coalescing the `vals`/`colIdx` reads is where most of the speedup comes from.
