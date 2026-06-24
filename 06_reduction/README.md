# Exercise 06 — Parallel Reduction

**New concepts:** tree-based parallel reduction, `__shared__` memory as a per-block scratchpad, `__syncthreads()` barriers, and the classic reduction optimization ladder — sequential addressing (no bank conflicts), no warp divergence, first-add-during-load, and a grid-stride load for arbitrary `n`.

## The task

Sum a large `float` array of length `n` on the GPU and write the **single** total to `*out`.

Edit `reduce.cu` and fill in the `TODO`s:

1. The `__global__` kernel `reduce` — each block loads a chunk of the input, reduces it in `__shared__` memory, and contributes its block-level partial to the global total.
2. The host function `solve` — choose the launch configuration and launch the kernel.

You do **not** write `main()`, allocate memory, or copy data — `harness.cu` does all of that, **zeroes `*out` for you**, and calls your `solve(...)`.

### `solve` signature (the contract)

```cpp
void solve(const float* in, float* out, int n);
```

`in` is a **device pointer** of length `n`. `out` is a **device pointer to a single `float`** that has already been zeroed by the harness; after `solve` returns it must hold the sum of all `n` elements.

## Combining block results — pick one

A reduction kernel naturally produces **one partial sum per block**. You must combine those partials into one number. Two correct strategies:

- **Single-pass with `atomicAdd`.** Each block computes its partial in shared memory, then **thread 0 does one `atomicAdd(out, block_partial)`**. That is *one* atomic per block (cheap — there are only a few thousand blocks), not one per element. This is the recommended approach here and what the reference solution does.
- **Two-pass.** Write one partial per block to a scratch array, then launch the kernel again on that (much smaller) array. Correct, but needs a second buffer and launch.

> The harness zeroes `*out` before each call, so the single-pass `atomicAdd` accumulation starts from a clean zero every time (including across the timing-loop's repeated calls).

## The optimization ladder

This is *the* canonical CUDA optimization exercise. The naive-to-fast progression:

1. **Interleaved addressing** (`s = 1, 2, 4, …`, active threads `tid % (2s) == 0`): correct but every warp diverges and shared-memory accesses bank-conflict.
2. **Sequential addressing** (`s = blockDim/2, /4, …`, active threads `tid < s`): consecutive threads stay active together (no divergence early) and access `sdata[tid]` / `sdata[tid+s]` — **stride-1, no bank conflicts**. Use this.
3. **First add during load**: instead of loading one element per thread into shared memory, have each thread add **two** (or, via a grid-stride loop, many) global elements *while loading*. This halves (or more) the shared-memory tree work and is essential to hit the bandwidth bar.
4. **Grid-stride load**: to handle arbitrary `n` with a fixed grid, each thread walks the input with stride `gridDim.x * blockDim.x`, accumulating into a register before the shared-memory tree. This also lets you launch fewer, fuller blocks.

### Why a per-element `atomicAdd` kernel is correct but slow

A trivial kernel — `atomicAdd(out, in[i])` for every element — is *correct*. But it serializes **16 million** atomic updates onto one memory location. The hardware turns that into a long line of read-modify-write traffic to a single address; you get a tiny fraction of peak bandwidth and the exercise's bandwidth bar fails. The whole point of the shared-memory tree is to collapse each block's chunk to a single value **on chip** first, so global memory sees only one atomic per block.

## Syntax / reference

Declare a per-block shared array (size known at compile time):

```cpp
__shared__ float sdata[BLOCK];
```

Barrier — **every** thread in the block must reach it before any moves on. Never put it inside a branch that only some threads take:

```cpp
__syncthreads();
```

Sequential-addressing tree (after each thread has put its partial in `sdata[tid]`):

```cpp
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}
// sdata[0] now holds the block's partial sum
```

One atomic per block to accumulate the global total:

```cpp
if (tid == 0) atomicAdd(out, sdata[0]);
```

Helper available: `ceil_div(a, b)` from `cuda_utils.cuh`.

## Grading (`!python grade.py`)

- **correctness** — your sum matches a `double`-precision CPU sum within a relative tolerance.
- **efficiency** — reduction is memory-bound (you read every element once). You must reach **`bw_frac >= 0.50`** of peak global-memory bandwidth. The per-element-atomic and interleaved-addressing kernels cannot.
- **source** — you must use `__shared__` and `__syncthreads`.

Run `python grade.py --check-solution` to grade the reference solution instead of yours.
