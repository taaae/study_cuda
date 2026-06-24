# Exercise 07 — Warp-Level Reduction
> Threads in a warp can hand each other registers directly — no memory, no barrier.

## The idea

In exercise 06 the reduction tree wrote to shared memory and called `__syncthreads()` at *every* level. But here's a secret: the bottom 5 levels of that tree (where 32 or fewer values remain) all happen *within a single warp* — and a warp already runs in lockstep. Paying for shared-memory writes and full block barriers to coordinate 32 threads that are *already synchronized* is wasteful.

Warp **shuffle** instructions let one lane read a register straight out of another lane in the same warp. No shared memory. No `__syncthreads()`. The values never leave the register file. So you reduce each warp to a single number using shuffles, and only then do you need a tiny bit of shared memory — one slot per warp — to combine the (at most 32) warp results.

## Under the hood

A **warp** is 32 threads that issue the same instruction together. Each has a **lane id** `0..31` (`threadIdx.x % 32`). Because all 32 lanes share the execution pipeline, the hardware can expose a register from one lane to another in a single instruction — that's what shuffle is. It reads from the register file, which is the fastest memory on the chip, faster even than shared memory.

`__shfl_down_sync(mask, value, delta)` returns the `value` held by lane `lane_id + delta`. Lanes that would read past lane 31 keep their own value. So a sequence of halving deltas folds 32 lanes down to lane 0:

```cpp
__device__ float warpReduceSum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;   // lane 0 of the warp now holds the warp's sum
}
```

### The mask, and why it matters

The first argument is a **32-bit lane mask** naming which lanes participate. `0xffffffff` means "all 32 lanes, active and converged here." A shuffle is a convergence point: **every lane named in the mask must execute it**, or the result is undefined. If some lanes had bailed out via a divergent branch and you still named them, you'd get garbage. Here every lane loaded a real value (or a clean `0` from the grid-stride loop), so all 32 are live and the full mask is correct.

## A picture

```text
__shfl_down_sync folding 8 lanes (real op uses 32, offsets 16→1):

 lane:    0     1     2     3     4     5     6     7
 v:      [a]   [b]   [c]   [d]   [e]   [f]   [g]   [h]
          ↓←────────────────────┘ (offset 4: lane 0 += lane 4, etc.)
 v:      a+e   b+f   c+g   d+h    .     .     .     .
          ↓←──────────┘             (offset 2)
 v:    a+e+c+g  ...     .     .
          ↓←────┘                   (offset 1)
 v:      SUM   <- lane 0 holds the whole warp's sum; no shared mem, no barrier
```

Two-stage block reduction:

```text
  warp 0 ─shuffle→ partial ─┐
  warp 1 ─shuffle→ partial ─┤  write to warpSums[warp]  ─┐
  warp 2 ─shuffle→ partial ─┤  (one shared slot per warp) │ __syncthreads
  ...                       ─┘                            │
                          warp 0 loads warpSums[], shuffle-reduces ─→ block total
```

## Your task

Sum a `float` array (`n = 2^24`) again — but the **intra-warp** reduction must use `__shfl_down_sync`. Reduce each warp to lane 0, stash the per-warp partials in a small shared array, then warp-reduce *those* with the first warp, and `atomicAdd` the block total once.

### The `solve` contract

```cpp
void solve(const float* in, float* out, int n);
```

`in` is a **device pointer** of length `n`. `out` is a **device pointer to a single `float`**, already zeroed by the harness. After `solve`, `*out` holds the total. In `reduce.cu` you fill in three things: `__device__ float warpReduceSum(float v)`, the `__global__ reduce` kernel, and the host `solve`. `harness.cu` owns `main()` and memory.

## Functions & syntax you'll need

| Thing | Form | What it does |
|---|---|---|
| `__shfl_down_sync` | `T __shfl_down_sync(unsigned mask, T v, int delta);` | lane reads `v` from lane `lane+delta` in the same warp |
| lane id | `int lane = threadIdx.x & 31;` | `0..31` within the warp |
| warp id | `int warp = threadIdx.x >> 5;` | which warp within the block |
| `__device__` | `__device__ float warpReduceSum(...)` | a function callable from the GPU (the warp-reduce helper) |
| `__shared__` | `__shared__ float warpSums[32];` | one slot per warp (≤32 warps in a ≤1024-thread block) |
| `__syncthreads()` | barrier | needed once, between writing and reading `warpSums` |
| `atomicAdd(addr, val)` | `float atomicAdd(float*, float);` | one per block, into the global total |
| launch | `reduce<<<grid, BLOCK>>>(in, out, n);` | start the kernel |
| `ceil_div(a, b)` | from `cuda_utils.cuh` | grid sizing |

When the first warp combines the partials, lanes beyond the live warp count must load `0`: `float v = (lane < numWarps) ? warpSums[lane] : 0.f;` — otherwise you'd sum stale shared memory.

> **Fun fact:** the `_sync` suffix is post-Volta (CUDA 9+). Older code used `__shfl_down` with no mask and relied on implicit warp-lockstep. Volta introduced *independent thread scheduling*, so lanes can diverge more freely — which is exactly why you must now name the participating lanes explicitly. Forgetting the mask is one of the most common warp-primitive bugs.

## How it's graded

`python grade.py` builds, runs, and checks:

- **correctness** — sum matches a `double` CPU sum within `rel_err < 1e-3`.
- **efficiency** — `bw_frac >= 0.50` of peak bandwidth. This is the hard bar.
- **speedup** — the harness also runs the exercise-06 shared-only reduction and reports `speedup = base_ms / your_ms`. On a T4 expect something **modest (~1.0–1.3×)**: this reduction is memory-bound, so the read traffic dominates and shuffle's real win (cutting compute/sync overhead) is small here. **There is no speedup threshold** — don't chase it.
- **source** — must use `__shfl_down_sync`.

Run `python grade.py --check-solution` to grade the reference solution.

## Going deeper

1. Shuffle shines on **compute-bound** reductions and on fusing reductions into bigger kernels (softmax, layernorm, batchnorm) where avoiding shared memory frees it for other data. Memory-bound here, but the technique is everywhere in real kernels.
2. The cooperative-groups API (`cg::reduce`) wraps this pattern in a portable, readable form — once you've hand-written it, that abstraction will make sense.
3. There are also `__shfl_sync`, `__shfl_up_sync`, and `__shfl_xor_sync` (a true butterfly) — the XOR variant gives *every* lane the full sum, handy when all lanes need the result.
