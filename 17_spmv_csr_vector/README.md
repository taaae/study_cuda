# Exercise 17 — SpMV (CSR, vector / one warp per row)

**New concepts:** **warp-per-row SpMV** — a full warp (32 lanes) cooperatively sums one row's nonzeros, reducing the lane partials with `__shfl_down_sync`. This fixes the load imbalance of scalar SpMV (exercise 16) and coalesces the `colIdx`/`vals` reads within a row.

## The task

Same math as exercise 16 — `y = A * x` in CSR — but now assign **one warp to each row** instead of one thread. The 32 lanes split the row's nonzeros between them (lane `l` handles `k = start + l, start + l + 32, …`), each accumulates a partial dot product, and then the warp **reduces** the 32 partials into one value with a shuffle reduction. Lane 0 writes the result.

Edit `spmv.cu` and fill in the `TODO`s:

1. The `__global__` kernel `spmv_vector` — one warp per row, strided lane loop, warp-reduce, lane 0 writes `y[r]`.
2. The host function `solve` — launch `nrows` warps (e.g. a block of 256 threads = 8 warps handles 8 rows).

### `solve` signature (the contract)

Identical to exercise 16:

```cpp
void solve(const int* rowPtr, const int* colIdx, const float* vals,
           const float* x, float* y, int nrows);
```

All pointers are **device pointers** in CSR layout (`rowPtr` length `nrows+1`, `colIdx`/`vals` length `nnz`).

## Why a warp per row helps

**Load balance.** With one thread per row, a warp can't retire until its longest row finishes; a single 1000-nonzero row stalls 31 idle lanes. With one *warp* per row, every lane in the warp works on the *same* row, so they all finish together — no intra-warp imbalance from row-length variation.

**Coalescing.** The lanes of a warp read `vals[start+0], vals[start+1], …, vals[start+31]` on the same iteration — 32 *consecutive* addresses, a single coalesced transaction. (Scalar SpMV had lane `l` reading `vals[rowPtr[r_l]]`, addresses scattered across the matrix.) Same for `colIdx`.

## The shuffle reduction

After the strided loop each lane holds a partial sum. `__shfl_down_sync` lets a lane read another lane's register directly — no shared memory:

```cpp
for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
// lane 0 now holds the full row sum
```

The mask `0xffffffff` means "all 32 lanes participate." `__shfl_down_sync(mask, v, d)` returns lane `(laneId + d)`'s copy of `v`. Halving `offset` from 16 → 1 sums all 32 partials into lane 0 in 5 steps (a tree reduction).

## When scalar beats vector

If the **average nonzeros per row is small** (say < 8), a warp-per-row wastes most lanes — 32 lanes on a 4-nonzero row leaves 28 idle, and you launch 32× more threads than rows. Scalar SpMV (one thread per row) is then better. Vector SpMV wins when rows are **long and/or highly variable**. This exercise uses a matrix with **highly variable row lengths** so the vector kernel clearly wins.

## Syntax you'll need

```cpp
unsigned mask = 0xffffffff;                      // all lanes
int lane = threadIdx.x & 31;                     // lane id within the warp
int warpId = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // global warp id
float v = __shfl_down_sync(mask, partial, offset);
```

Helper available: `ceil_div(a, b)` from `cuda_utils.cuh`.

## Grading (`!python grade.py`)

- **correctness** — `y` matches a CPU CSR SpMV within tolerance.
- **efficiency** — the harness runs the **scalar** (one-thread-per-row) kernel as a baseline on the same (imbalanced) matrix and reports `speedup`. You must reach **speedup ≥ 1.5**.
- **source** — you must use `__shfl_down_sync` (the warp reduction is the point).

Run `python grade.py --check-solution` to grade the reference solution instead of yours.
