# Exercise 17 — SpMV (CSR, vector / one warp per row)
> When one thread per row stalls on the long rows, hand each row to a whole warp.

## The idea

In exercise 16 you wrote *scalar* CSR SpMV: one thread owns one row and walks its
nonzeros sequentially. That's clean, but it has two problems that show up the
moment your rows are uneven in length:

1. **Load imbalance.** A warp runs in lockstep and can't retire until its *slowest*
   lane finishes. If 31 lanes own 4-nonzero rows and one lane owns a 600-nonzero
   row, those 31 lanes sit idle for the whole long row.
2. **Uncoalesced reads.** On a given step, lane `r` reads `vals[rowPtr[r] + k]` and
   lane `r+1` reads `vals[rowPtr[r+1] + k]` — 32 *unrelated* addresses scattered
   across the matrix. The memory system hates that.

The fix is **one warp per row**. All 32 lanes cooperate on the *same* row: lane `l`
handles nonzeros `start+l, start+l+32, …`, each lane builds a partial dot product,
and then the warp adds the 32 partials together with a shuffle reduction. Lane 0
writes the answer. The math is identical to ex16 — only the work assignment changes.

## Under the hood

Why this is faster comes down to two hardware facts.

**Coalescing.** On each loop step the warp reads `vals[start+0 .. start+31]` — 32
*consecutive* 4-byte words, which the memory controller services as a single
128-byte transaction instead of up to 32 separate ones. Same for `colIdx`. (The
gather `x[colIdx[k]]` is still scattered — that's intrinsic to sparse matrices —
but the big `vals`/`colIdx` streams are now perfectly aligned.)

**Intra-warp balance.** Because every lane in a warp is on the same row, they all
finish that row at the same time. Row-length variation no longer makes lanes idle
*within* a warp. (Variation between warps still exists, but the scheduler hides it
by oversubscribing the SM with many warps.)

> **The catch:** if rows are *short* — say average < 8 nonzeros — a warp wastes
> most of its lanes. 32 lanes on a 4-nonzero row leaves 28 idle, and you launch
> 32× more threads than rows. Then scalar SpMV wins. Vector SpMV pays off when
> rows are **long and/or highly variable** — which is exactly the matrix this
> harness builds (5% of rows have 200–600 nonzeros, the rest have 1–8).

This warp-per-row pattern is the workhorse of real sparse libraries; it's roughly
what NVIDIA's classic CUSP/cuSPARSE "CSR-vector" kernel does.

## A picture

```text
ROW r has nonzeros at CSR slots [start .. end). One WARP (32 lanes) strides it:

 slot:  start  +1   +2   ...  +31  +32  +33  ...
 lane:    0     1    2   ...   31    0    1   ...     <- lane l does start+l, +32, ...
          |     |    |         |
          v     v    v         v
  step 0: 32 CONSECUTIVE addresses  -> one coalesced load
  step 1: next 32 consecutive       -> one coalesced load
          ...

Then reduce the 32 lane partials to lane 0 (tree, 5 steps):
 lanes 0..31 :  p0 p1 p2 ... p31
 offset 16   :  p0+=p16  p1+=p17 ...            (16 adds)
 offset  8   :  p0+=p8   ...                    ( 8 adds)
 offset  4   :  ...                             ( 4 adds)
 offset  2   :  ...                             ( 2 adds)
 offset  1   :  p0 += p1                        ( 1 add)  -> lane 0 = full row sum
```

## Your task

Edit `spmv.cu` and fill in the `TODO`s:

1. **Kernel `spmv_vector`** — map the warp to a row, run the strided per-lane loop
   accumulating `sum += vals[k] * x[colIdx[k]]`, warp-reduce `sum` with
   `__shfl_down_sync`, and have lane 0 write `y[warpId]`.
2. **Host `solve`** — launch `nrows` warps total. A block of 256 threads is 8
   warps, so it covers 8 rows; use `ceil_div` to size the grid.

### The `solve` contract

```cpp
void solve(const int* rowPtr, const int* colIdx, const float* vals,
           const float* x, float* y, int nrows);
```

All pointers are **device pointers** in CSR layout: `rowPtr` has length `nrows+1`,
`colIdx`/`vals` have length `nnz`. Identical to exercise 16.

## Functions & syntax you'll need

| Tool | Signature / form | What it does |
|------|------------------|--------------|
| global warp id | `int warpId = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;` | `>> 5` = `/ 32`; one warp ↦ one row |
| lane id | `int lane = threadIdx.x & 31;` | `& 31` = `% 32`; this thread's position in its warp |
| CSR row bounds | `int start = rowPtr[r], end = rowPtr[r+1];` | half-open slot range `[start, end)` for row `r` |
| strided lane loop | `for (int k = start+lane; k < end; k += 32)` | each lane takes every 32nd nonzero → coalesced reads |
| warp shuffle | `float __shfl_down_sync(unsigned mask, float v, int delta)` | returns lane `(laneId+delta)`'s copy of `v`; register-to-register, no shared memory |
| full-warp mask | `0xffffffff` | declares all 32 lanes active — required for correctness on modern CUDA |
| `ceil_div(a, b)` | from `cuda_utils.cuh` | `(a + b - 1) / b`, for sizing the grid |

The reduction itself is the 5-step halving loop:

```cpp
for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
// lane 0 now holds the full row sum
```

After step 1 lane `l` holds `partial[l] + partial[l+16]`; after all five steps
lane 0 holds the sum of all 32. No `__syncthreads`, no shared memory — shuffles
move data straight between registers inside the warp.

## How it's graded

Run `python grade.py` (or `--check-solution` to grade the reference instead).

- **correctness** — your `y` matches a CPU CSR SpMV within tolerance.
- **efficiency** — the harness times the **scalar** one-thread-per-row kernel on
  the same imbalanced matrix and reports `speedup = baseline_ms / ms`. You must
  reach **`speedup ≥ 1.5`**. It also reports `gflops` (counted on real nonzeros).
- **source** — you must use `__shfl_down_sync`: the warp reduction is the point of
  the exercise, so a shared-memory workaround won't pass.

## Going deeper

- **Why `>> 5` and `& 31` instead of `/32` and `%32`?** They're identical for the
  compiler here, but the bit-ops make the warp structure explicit and never tempt
  the compiler into a slow signed division.
- **One-warp-per-row isn't the end.** When *all* rows are long, you can give each
  row a whole *block* and reduce across warps in shared memory; when rows are tiny
  you'd rather pack several rows per warp. Picking the granularity from the matrix
  is exactly the per-matrix tuning a library does for you (next exercise).
