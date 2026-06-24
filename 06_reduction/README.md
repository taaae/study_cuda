# Exercise 06 — Parallel Reduction
> Collapsing 16 million numbers into one — and why the obvious way is the slow way.

## The idea

A reduction takes a whole array and folds it into a single value: a sum, a max, a dot product. On a CPU you'd just loop. On a GPU the challenge is that you have thousands of threads but only *one* answer slot — so the work is fundamentally about **combining results from many threads without them stepping on each other.**

The naive instinct — every thread does `atomicAdd(out, in[i])` — is *correct* and *catastrophically slow*. It funnels 16 million read-modify-write updates onto a single memory address, and the hardware serializes them into a long queue. You get a sliver of peak bandwidth.

The fix is a **tree**. Pair up neighbors, sum each pair, and you've halved the count. Repeat. After `log2(n)` rounds, one value remains. Done right, each block reduces its chunk **entirely on chip** and touches the global answer just once.

## Under the hood

Reduction is **memory-bound**: you must read every input element exactly once, and that read is the bottleneck — the additions are nearly free by comparison. So the game is "read all 16M elements at full bandwidth, and don't waste any other memory traffic." The on-chip tree exists purely so global memory sees as little extra traffic as possible.

Two hardware realities shape the tree:

- **Warps execute in lockstep.** A warp is 32 threads sharing one instruction stream. If half a warp takes a branch and half doesn't, the hardware runs both paths and masks off the idle lanes — that's **warp divergence**, and it wastes cycles. So you want your "active threads" to be a contiguous prefix, not every-other-thread.
- **Shared memory has 32 banks.** The access pattern of your tree decides whether warps hit distinct banks (fast) or collide (serialized).

These two facts are why *sequential addressing* beats *interleaved addressing*, below.

## A picture

```text
Sequential-addressing tree inside one block (8 threads shown):

 tid:    0    1    2    3    4    5    6    7
 sdata: [a]  [b]  [c]  [d]  [e]  [f]  [g]  [h]
          \    \    \    \   /    /    /    /
 s=4:    a+e  b+f  c+g  d+h  .    .    .    .     (threads 0..3 active, 4..7 idle)
          \    \   /    /
 s=2:    .... .... .    .                         (threads 0..1 active)
          \   /
 s=1:    SUM                                      (thread 0 active)

Active threads stay packed at the low end → whole warps go idle together
(no divergence early), and sdata[tid] / sdata[tid+s] are stride-1 (no bank conflicts).
```

## Your task

Sum a `float` array of length `n` (`n = 2^24 = 16M`) and write the single total to `*out`. Each block reduces a chunk in shared memory; combine the per-block partials into one number.

### The `solve` contract

```cpp
void solve(const float* in, float* out, int n);
```

`in` is a **device pointer** of length `n`. `out` is a **device pointer to a single `float`** that the harness has already **zeroed** for you (every call, including inside the timing loop). After `solve` returns, `*out` must hold the sum. You fill in `reduce.cu`: the `__global__` kernel `reduce` and the host `solve`. `harness.cu` owns `main()` and all memory.

## Combining block results

A reduction kernel naturally produces **one partial per block**. To get one final number, the recommended path is **single-pass `atomicAdd`**: each block reduces its chunk in shared memory, then **thread 0 does one `atomicAdd(out, sdata[0])`**. That's one atomic per *block* (a few thousand total) — cheap — not one per element. (A two-pass scheme writing partials to a scratch array and re-launching also works, but needs a second buffer.)

## The optimization ladder

This is *the* canonical CUDA tuning exercise. The path from slow to fast:

1. **Interleaved addressing** (`s = 1,2,4,…`, active when `tid % (2s) == 0`): correct, but every warp diverges and shared accesses bank-conflict. Avoid.
2. **Sequential addressing** (`s = blockDim/2, /4, …`, active when `tid < s`): contiguous active threads, stride-1 access. **Use this.**
3. **First add during load**: don't load just one element per thread — have each thread sum *several* global elements into a register before the tree even starts. This is where the bandwidth comes from.
4. **Grid-stride load**: with a fixed, capped grid, each thread walks the input with stride `gridDim.x * blockDim.x`, accumulating into a register. Handles any `n` and keeps blocks full.

The sequential-addressing tree, once each thread has its partial in `sdata[tid]`:

```cpp
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();        // never inside the if — every thread must reach it
}
// sdata[0] now holds the block's partial
```

## Functions & syntax you'll need

| Thing | Form | What it does |
|---|---|---|
| `threadIdx.x` | built-in | thread index within block (use as `tid`) |
| `blockIdx.x`, `blockDim.x`, `gridDim.x` | built-ins | block index, block size, grid size |
| `__shared__` | `__shared__ float sdata[BLOCK];` | per-block scratchpad (size known at compile time) |
| `__syncthreads()` | barrier | all threads wait; **never** call inside a partial branch |
| `atomicAdd(addr, val)` | `float atomicAdd(float*, float);` | race-free read-modify-write into global memory |
| launch | `reduce<<<grid, BLOCK>>>(in, out, n);` | start the kernel |
| `ceil_div(a, b)` | from `cuda_utils.cuh` | grid sizing helper |

> **Fun fact:** Mark Harris's NVIDIA talk "Optimizing Parallel Reduction in CUDA" walks this exact ladder through 7 versions, going from ~2 GB/s to ~63 GB/s on an old G80. The lessons — divergence, bank conflicts, first-add-during-load — are still the bread and butter of GPU performance today.

## How it's graded

`python grade.py` builds, runs, and checks:

- **correctness** — your sum matches a `double`-precision CPU sum within `rel_err < 1e-3` (float accumulation vs. double, so an exact match isn't expected).
- **efficiency** — `bw_frac >= 0.50` of peak bandwidth. The per-element-atomic and interleaved-addressing kernels can't reach this; the shared-memory tree with first-add-during-load can.
- **source** — must use `__shared__` and `__syncthreads`.

Run `python grade.py --check-solution` to grade the reference solution.

## Going deeper

1. The reference caps the grid at 4096 blocks and leans on the grid-stride loop. Fewer, fuller blocks mean more work per block to amortize launch and reduction overhead — a recurring GPU theme.
2. The intra-warp part of this tree still pays `__syncthreads()` and shared-memory traffic at every step. Exercise 07 shows how warp shuffles eliminate both for the last 5 levels.
