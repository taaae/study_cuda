# Exercise 07 — Warp-Level Reduction

**New concepts:** warp-level primitives — `__shfl_down_sync` — which let the 32 threads (lanes) of a warp exchange registers directly, with **no shared memory and no `__syncthreads()`**. You reduce within each warp by shuffling, then combine the per-warp results.

## The task

Sum a large `float` array again, but the **intra-warp** reduction must use `__shfl_down_sync`. Then combine the (up to 32) per-warp partials within each block and contribute the block total to `*out`.

Edit `reduce.cu` and fill in the `TODO`s:

1. `__device__ float warpReduceSum(float v)` — reduce one value per lane down to lane 0 using `__shfl_down_sync`.
2. The `__global__` kernel `reduce` — grid-stride load into a register, warp-reduce, combine warps, one `atomicAdd` per block.
3. The host function `solve` — choose the launch configuration and launch.

You do **not** write `main()` or manage memory — `harness.cu` does, **zeroes `*out`**, and calls your `solve(...)`.

### `solve` signature (the contract)

```cpp
void solve(const float* in, float* out, int n);
```

`in` is a **device pointer** of length `n`. `out` is a **device pointer to a single `float`**, already zeroed; after `solve` it must hold the total sum.

## What is a warp, and what does shuffle do?

A **warp** is 32 threads that execute in lockstep. Each thread has its own *lane id* `0..31` (`threadIdx.x % 32`). Shuffle instructions let a lane read a register **directly from another lane in the same warp** — no memory round-trip, no barrier.

`__shfl_down_sync(mask, value, delta)` returns the `value` held by the lane `lane_id + delta` (lanes that would read past 31 return their own unchanged value). So a butterfly of halving deltas folds 32 values into lane 0:

```cpp
__device__ float warpReduceSum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;   // lane 0 of the warp now holds the warp's sum
}
```

### The mask

The first argument is a **32-bit lane mask** naming which lanes participate. `0xffffffff` means "all 32 lanes of this warp are active and synchronized here." Every lane named in the mask **must** execute the shuffle (it is a convergence point); if some lanes have exited via a divergent branch, the result is undefined. Here all lanes participate, so the full mask is correct — and because each thread loaded a real value (or 0 via the grid-stride loop), there is no divergence at the shuffle.

### Why shuffle beats shared memory

The exercise-06 tree writes and reads `__shared__` memory at every step and calls `__syncthreads()` between steps. Shuffle keeps the values **in registers** and uses the warp's implicit lockstep instead of a block barrier: no shared-memory traffic and no `__syncthreads()` for the intra-warp part. You still need a tiny bit of shared memory (one slot per warp) to combine the up-to-32 warp results — reduce those with a second shuffle on the first warp.

## Syntax / reference

```cpp
v += __shfl_down_sync(0xffffffff, v, offset);   // full-warp shuffle-down
```

Lane and warp ids inside a block:

```cpp
int lane = threadIdx.x & 31;        // 0..31
int warp = threadIdx.x >> 5;        // which warp in the block
```

A small shared array to hold one partial per warp (≤ 32 warps in a ≤1024-thread block):

```cpp
__shared__ float warpSums[32];
```

## Grading (`!python grade.py`)

- **correctness** — your sum matches a `double` CPU sum within a relative tolerance.
- **speedup** — the harness also runs an exercise-06-style **shared-only** reduction as a baseline and reports `speedup`. On a T4 this is often modest (~1.0–1.3×); shuffle's win is bigger on compute-bound reductions, and this one is memory-bound. There is no hard speedup threshold here.
- **efficiency** — `bw_frac >= 0.50` of peak global-memory bandwidth.
- **source** — you must use `__shfl_down_sync`.

Run `python grade.py --check-solution` to grade the reference solution instead of yours.
