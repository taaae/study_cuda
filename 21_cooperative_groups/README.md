# Exercise 21 — Cooperative Groups & Grid-Wide Sync
> Make *all* blocks in a grid wait for each other — inside a single kernel launch.

## The idea
You already know `__syncthreads()`: it's a barrier for **one block**. But there's
a wall you keep hitting — there is no built-in barrier that synchronizes the
*whole grid*. The classic workaround is to split the algorithm into **two kernel
launches**, because the only guaranteed grid-wide sync point is the kernel
boundary itself. That works, but it's wasteful: between the two launches every
intermediate value goes back out to global memory and gets re-read.

Cooperative groups give you a real grid-wide barrier: `this_grid().sync()`. After
it returns, every thread in *every* block has reached the barrier, so anything
written before it is visible to everyone after it. Now a two-phase algorithm —
reduce, then use the result — fits in **one launch**, with the partial sum living
in fast on-chip / L2 state instead of bouncing through DRAM twice.

Our task is to L2-normalize a vector in place: `out[i] = data[i] / sqrt(Σ data[j]²)`.
Phase 1 computes the sum of squares; phase 2 divides by its square root. The
barrier sits between them.

## Under the hood
There's a catch, and it's the whole reason `grid.sync()` needs special launch
machinery. A grid barrier can only work if **every block is physically resident
on the GPU at the same time** (co-resident). Picture it: if you launched more
blocks than the SMs can hold, some blocks would still be queued, waiting to start
— while the running blocks sit at the barrier waiting for *them*. Deadlock.

So two rules fall out:

1. **You must launch with `cudaLaunchCooperativeKernel`**, not `<<< >>>`. This
   launch path checks co-residency and refuses an over-subscribed grid (rather
   than deadlocking).
2. **You size the grid to what actually fits.** Ask the occupancy API how many
   blocks of your chosen size fit on one SM, multiply by the SM count, and use a
   **grid-stride loop** so a smaller-than-data grid still covers all `n` elements.

> **Fun fact:** the T4 has 40 SMs. With 256-thread blocks you'll typically get a
> grid of a few hundred blocks total — far fewer than the ~16K blocks a 4M-element
> launch would normally spawn. The grid-stride loop is what lets that lean,
> co-resident grid still touch every element.

## A picture
```text
ONE LAUNCH, all blocks co-resident on the SMs:

  block0  block1  block2  ...  blockG-1
    |       |       |            |
  [phase1: local sum-of-squares, then atomicAdd into *ssq]
    |       |       |            |
    v       v       v            v
  =================== grid.sync() ===================   <-- every block waits here
    |       |       |            |                          *ssq is now complete
    v       v       v            v
  [phase2: inv = rsqrtf(*ssq); data[i] *= inv]
```
Without the barrier, a fast block could reach phase 2 and read `*ssq` while a slow
block hasn't added its contribution yet — silent wrong answers.

## Your task
Edit `normalize.cu`:
1. `normalize_kernel` — phase 1 (grid-stride sum of squares → block reduction →
   `atomicAdd` into `*ssq`), then `grid.sync()`, then phase 2 (scale by `rsqrtf`).
2. `solve` — size a co-resident grid, allocate/zero the global accumulator, and
   launch cooperatively.

### The `solve` contract
```cpp
void solve(float* data, int n);   // normalizes data in place
```
`data` is a **device pointer** of length `n`. Allocate a tiny device scratch
`float* ssq` inside `solve`, zero it with `cudaMemset`, and `cudaFree` it after.

## Functions & syntax you'll need
| Function / construct | Signature (essentials) | What it does |
|---|---|---|
| header | `#include <cooperative_groups.h>` | brings in the cooperative-groups API |
| namespace | `namespace cg = cooperative_groups;` | conventional short alias |
| `cg::this_grid()` | `cg::grid_group cg::this_grid()` | handle to the entire grid as one group |
| `grid.sync()` | `void grid_group::sync()` | grid-wide barrier (all blocks wait) |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` | `(int* numBlocks, const void* kernel, int blockSize, size_t dynSmem)` | how many blocks fit per SM |
| `cudaGetDeviceProperties` | `(cudaDeviceProp* p, int dev)` | read `p->multiProcessorCount` (SM count) |
| `cudaLaunchCooperativeKernel` | `((void*)kernel, dim3 grid, dim3 block, void** args, size_t dynSmem, cudaStream_t)` | the co-resident-checked launch path |
| `atomicAdd` | `float atomicAdd(float* addr, float val)` | race-free add into `*ssq` |
| `rsqrtf` | `float rsqrtf(float x)` | fast `1/sqrt(x)` |
| `__syncthreads()` | `void __syncthreads()` | still needed for the per-block reduction |

The cooperative launch takes arguments as an **array of addresses-of-arguments**:
```cpp
cg::grid_group grid = cg::this_grid();   // in the kernel
// ...
grid.sync();
```
```cpp
void* args[] = { &data, &n, &ssq };      // in solve(): each entry is &variable
cudaLaunchCooperativeKernel((void*)normalize_kernel,
                            dim3(grid), dim3(block), args, 0, 0);
```

## How it's graded
`python grade.py` builds with `-rdc=true` (cooperative launch requires
relocatable device code) and checks:
- **correctness** — output equals the CPU L2-normalized vector, `max_abs_err <= 1e-5`.
- **efficiency** — reports `ms` for the single fused launch (the win is *not*
  re-reading 4M floats between two launches).
- **source** — your code must contain `grid.sync` *and* `cudaLaunchCooperativeKernel`.
  A two-launch solution, even if correct, fails the source check — the point is to
  use the real grid barrier.

The harness probes `cudaDevAttrCooperativeLaunch` first; the T4 supports it. On a
GPU that doesn't, it prints `# SKIP` and reports correct so grading won't
false-fail. Use `python grade.py --check-solution` to grade the reference instead.

## Going deeper
Cooperative groups are more than grid sync: `cg::tiled_partition<32>(block)` gives
you a warp-sized group with `.shfl_down()` and `.reduce()` — a cleaner, portable
way to write the block reduction than raw `__shfl_*` intrinsics. And on multi-GPU
boxes, `this_multi_grid()` extends the barrier across devices.
