# Exercise 09 — Register-Blocked GEMM (`float4` loads)

**New concepts:** **register blocking** (thread coarsening) — each thread computes a small **micro-tile** of `C` (here `TM×TN = 8×8`) held entirely in registers — and **vectorized `float4`** global loads. This is how production GEMMs reach a large fraction of peak.

## The task

Compute `C = A * B`, **row-major**, `A` is `M×K`, `B` is `K×N`, `C` is `M×N` — faster than the exercise-08 tiled kernel.

Edit `gemm.cu` and fill in the `TODO`s:

1. The `__global__` kernel `gemm` — a block computes a `BM×BN` output tile; each thread computes a `TM×TN` micro-tile in registers, streaming `A`/`B` tiles through `__shared__` with `float4` loads.
2. The host function `solve` — set up the launch and call it.

You do **not** write `main()` or manage memory — `harness.cu` does, and calls your `solve(...)`.

### `solve` signature (the contract)

```cpp
void solve(const float* A, const float* B, float* C, int M, int N, int K);
```

All pointers are **device pointers**, all matrices **row-major**.

## Why register blocking is faster

In the exercise-08 tiled kernel, each thread computes **one** output element and reads two shared-memory values per multiply-add — so shared-memory bandwidth becomes the new ceiling. If instead each thread computes a `TM×TN` block of outputs, it loads `TM` values from the `A` tile and `TN` from the `B` tile (a total of `TM+TN` shared reads) but performs `TM×TN` multiply-adds. The shared-reads-per-FLOP ratio drops from `2` to `(TM+TN)/(TM*TN)` — for `8×8` that's `16/64 = 0.25`, an **8×** reduction. Those `TM×TN` partial sums live in registers, the fastest storage on the chip. Higher arithmetic intensity at *every* level of the hierarchy is the whole game.

## `float4` loads — alignment and coalescing

A `float4` load moves 16 bytes in one instruction. Two payoffs: fewer load instructions, and one warp's 32 `float4` loads cover 512 contiguous bytes — perfectly coalesced — when the base address is **16-byte aligned** and threads read consecutive `float4`s. The harness allocates with `cudaMalloc` (256-byte aligned) and uses dimensions that are multiples of the tile sizes, so every `float4` access in this exercise is aligned and in-bounds. Cast and dereference like:

```cpp
float4 v = reinterpret_cast<const float4*>(A)[idx4];   // idx4 counts float4s, = byte_off/16
```

## The kernel, step by step

Constants used by the reference (you may keep them): `BM=128, BN=128, BK=8, TM=8, TN=8`, block of `16×16 = 256` threads.

1. **Identify the block's output tile**: rows `blockIdx.y*BM .. +BM`, cols `blockIdx.x*BN .. +BN`.
2. **Per-thread micro-tile**: thread `(tx,ty)` (with linear id `tid = ty*16 + tx`) owns the `TM×TN` outputs at tile-local rows `ty*TM .. +TM`, cols `tx*TN .. +TN`. Keep them in a register array `acc[TM][TN]`.
3. **K-loop over tiles of width `BK`.** Each iteration:
   - Cooperatively load a `BM×BK` slab of `A` and a `BK×BN` slab of `B` into shared memory, using `float4` loads (each of the 256 threads loads exactly one `float4` into each tile).
   - Store the `A` slab **transposed** as `As[BK][BM]` so the inner loop reads it column-major-friendly (stride-1 along `BM`).
   - `__syncthreads()`.
   - **Inner product over `BK`**: for each `k`, load this thread's `TM` values from `As[k][...]` and `TN` values from `Bs[k][...]` into small register arrays, then do the `TM×TN` outer-product updates into `acc`.
   - `__syncthreads()` before the next slab.
4. **Write back** `acc` to `C` (using `float4` stores or scalar stores).

### `__syncthreads()` placement

Same rule as exercise 08: one barrier after filling the shared slabs (before the inner product) and one after the inner product (before overwriting the slabs).

## Syntax / reference

```cpp
__shared__ float As[BK][BM];   // A slab, transposed
__shared__ float Bs[BK][BN];   // B slab

float acc[TM][TN] = {0.f};     // per-thread micro-tile in registers
float a_reg[TM], b_reg[TN];    // staged operands per k

float4 v = reinterpret_cast<const float4*>(A)[i];   // vectorized load
```

## Grading (`!python grade.py`)

- **correctness** — `C` matches a CPU reference (`double` accumulation) within a relative tolerance.
- **speedup** — the harness runs a basic tiled GEMM (like exercise 08) as baseline and reports `speedup`; you need **`speedup >= 1.5`**.
- **performance** — also benchmarked against cuBLAS: **`frac_cublas = gflops/gflops_cublas >= 0.30`**.
- **source** — you must use `float4` and `__shared__`.

Compiled with `-lcublas`. Run `python grade.py --check-solution` to grade the reference solution instead of yours.
