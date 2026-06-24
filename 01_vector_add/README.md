# Exercise 01 — Vector Add
> Write your very first CUDA kernel: add two arrays of 16 million floats in parallel, one thread per element.

## The idea

On a CPU you'd add two arrays with a loop: `for (i...) c[i] = a[i] + b[i];` — one core grinding through elements one after another. A GPU flips this on its head. Instead of *one* worker doing *every* element, you launch *thousands* of tiny workers and tell each one: "you handle exactly one element." Element 0 is added by thread 0, element 1 by thread 1, and so on, all (conceptually) at the same time.

That's the whole mental shift of GPU programming: **you don't write the loop — you write the body of the loop, and the hardware runs it across a huge fleet of threads.** The function you write is called a **kernel**, and a kernel always describes the work of a *single* thread.

The catch is that each thread must figure out *which* element is "mine." There's no loop variable handed to you — instead each thread reads its own coordinates from built-in variables and computes its index. Getting that index right is the entire exercise.

## Under the hood

A GPU doesn't actually run a million threads literally simultaneously. The threads are organized into a hierarchy, and that hierarchy mirrors the physical chip:

- **Thread** — the smallest unit. Runs your kernel body once.
- **Block** — a group of threads (you choose the size, e.g. 256). All threads in a block run on the *same* physical core cluster called an **SM** (Streaming Multiprocessor) and can cooperate. On a T4 there are **40 SMs**.
- **Grid** — all the blocks for one kernel launch. The GPU's scheduler hands blocks out to SMs as they free up.

Within an SM, threads are actually executed 32 at a time in lockstep — a group called a **warp**. You'll meet warps properly in later exercises; for now just know 256 (a multiple of 32) is a friendly block size.

The other thing to internalize: **the GPU has its own memory.** Your `a`, `b`, `c` arrays don't live in CPU RAM — they've been copied into the GPU's global memory (the harness did this with `cudaMalloc` + `cudaMemcpy`). A pointer like `a` here is a *device pointer*: a valid address on the GPU, garbage if you dereference it on the host. Vector add is **memory-bound** — the actual addition is trivial; all the time goes into reading `a` and `b` and writing `c` across that memory bus.

## A picture

```text
   grid (you launch ceil(n/256) blocks)
 ┌─────────────┬─────────────┬─────────────┬─── ...
 │  block 0    │  block 1    │  block 2    │
 │ [t0..t255]  │ [t0..t255]  │ [t0..t255]  │
 └─────────────┴─────────────┴─────────────┴─── ...

 global index  i = blockIdx.x * blockDim.x + threadIdx.x
                        │            │            │
 block 1, thread 5  →   1     *     256    +      5      = 261

 array:  a[0] a[1] ... a[255] | a[256] ... a[261] ...
          └── block 0 ──────┘   └─ block 1 owns these ─┘
```

Each thread computes its own `i` and touches exactly `a[i]`, `b[i]`, `c[i]`.

## Your task

Compute `C = A + B` for float arrays of length `n = 1 << 24` (16M elements), **one thread per element**. Edit `vadd.cu` and fill the two `TODO`s:

1. The `__global__` kernel `vadd` — compute this thread's global index and write one output element (the boundary guard `if (i < n)` is already there; you supply `i`).
2. The host function `solve` — pick a block size, compute how many blocks cover `n`, and launch the kernel.

You do **not** write `main()`, allocate memory, or copy data — `harness.cu` does all that and calls your `solve(...)`.

### The `solve` contract

```cpp
void solve(const float* a, const float* b, float* c, int n);
```

`a`, `b`, `c` are **device pointers** (already on the GPU) of length `n`. Your only job is to launch the kernel. `a` and `b` are inputs; you write results into `c`.

> **Why the boundary check matters.** You'll launch `ceil(n / blockSize)` blocks. If `n` isn't a perfect multiple of the block size, the last block has *extra* threads whose `i >= n`. If those threads write `c[i]`, they scribble past the end of the array — a classic out-of-bounds bug. The `if (i < n)` guard makes them do nothing. Always guard the global write.

## Functions & syntax you'll need

| Thing | Signature / form | What it does |
|-------|------------------|--------------|
| `__global__` | `__global__ void vadd(...)` | Marks a function as a **kernel** — runs on the GPU, callable from the host. |
| `threadIdx.x` | built-in `uint3` | This thread's index *within its block* (`0 .. blockDim.x-1`). |
| `blockIdx.x` | built-in `uint3` | This block's index *within the grid*. |
| `blockDim.x` | built-in `dim3` | Number of threads per block (what you chose). |
| `gridDim.x` | built-in `dim3` | Number of blocks in the grid. |
| launch syntax | `vadd<<<grid, block>>>(a, b, c, n);` | Launches the kernel with `grid` blocks of `block` threads each. |
| `ceil_div` | `int ceil_div(int a, int b)` | From `cuda_utils.cuh`: returns `(a + b - 1) / b` — rounds the division **up** so the last partial chunk still gets a block. |

The canonical 1-D global index, which you'll use in nearly every kernel from here on:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

## How it's graded

Run `python grade.py` (or `!python grade.py` in a Colab cell). It checks three things:

- **correctness** — `C == A + B` within tolerance (`max_abs_err <= 1e-4`).
- **efficiency** — vector add is purely memory-bound, so the harness measures `bw_frac`: achieved bandwidth as a fraction of the device's theoretical peak. It moves `3 * bytes` (read `a`, read `b`, write `c`) and divides by elapsed time. You must reach **`bw_frac >= 0.55`**. A correct one-thread-per-element kernel clears this easily; the threshold mainly catches mistakes like launching a single block (which leaves 39 of 40 SMs idle) or accidentally serializing.
- **source** — you must actually use the `<<< >>>` launch syntax.

Run `python grade.py --check-solution` to grade the reference solution instead of yours.

## Going deeper

- **Why 256?** Block sizes of 128–512 are typical. Too small and you can't keep an SM busy; too large and you may run low on per-thread registers. 256 is a safe default you'll reuse constantly.
- **The bandwidth wall is real.** Even at 100% efficiency, this kernel is limited entirely by how fast the T4 can move bytes (~320 GB/s), not by arithmetic. Adding is free; *moving* isn't. This intuition — "is my kernel memory-bound or compute-bound?" — guides every optimization you'll do later.
- **Try this:** after it passes, change `block` to `32` or `1024` and re-grade. Watch how the launch config nudges `gbps` up and down.
