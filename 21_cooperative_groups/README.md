# Exercise 21 — Cooperative Groups & Grid-Wide Sync

**New concepts:** the *cooperative groups* programming model, the `this_grid()`
group and `grid.sync()` for a **grid-wide barrier inside a single kernel
launch**, and the `cudaLaunchCooperativeKernel` launch path that makes it legal.

## The problem grid sync solves

Normally a `__syncthreads()` only synchronizes **one block**. There is no built-in
way for *all* blocks in a grid to wait for each other — the usual trick is to
**split the work into two kernel launches** (the kernel boundary is the only
guaranteed global sync). That works, but it re-reads everything from global
memory between launches.

Cooperative groups give you a real **grid-wide barrier**: `this_grid().sync()`.
After it returns, every thread in *every* block is guaranteed to have reached the
barrier, so writes made before it are visible to all threads after it. That lets
you do a two-phase algorithm (reduce, then use the result) in **one launch**.

The catch: grid sync only works if **every block is resident on the GPU at the
same time** (co-resident). If you launched more blocks than fit, some would be
waiting to start while others wait at the barrier — deadlock. So you must:

1. Launch with `cudaLaunchCooperativeKernel` (not `<<< >>>`), which *checks*
   co-residency and refuses to launch an over-subscribed grid.
2. Size the grid to the **occupancy-limited** number of blocks, and use a
   **grid-stride loop** so a smaller-than-data grid still covers all `n`
   elements.

## The task

Normalize a float vector to **unit L2 norm, in place, in ONE kernel launch**:

```
out[i] = data[i] / sqrt( sum_j data[j]^2 )
```

The kernel runs in two phases separated by a grid barrier:

- **Phase 1** — every thread accumulates `data[i]*data[i]` (over its grid-stride
  range) into a per-block partial, then one thread `atomicAdd`s the block partial
  into a single global accumulator `*ssq`.
- **`grid.sync()`** — wait until *all* blocks have finished their `atomicAdd`, so
  `*ssq` now holds the complete sum of squares.
- **Phase 2** — compute `inv = rsqrtf(*ssq)` and every thread multiplies its
  elements by `inv`.

Edit `normalize.cu`:

1. The kernel `normalize_kernel` — phases 1 and 2 with `grid.sync()` between them.
2. The host `solve` — size a co-resident grid and launch with
   `cudaLaunchCooperativeKernel`.

### `solve` signature (the contract)

```cpp
void solve(float* data, int n);   // normalizes data in place
```

`data` is a **device pointer** of length `n`. You also need a small device
scratch for the global accumulator (`float* ssq`) — allocate it inside `solve`,
zero it with `cudaMemset`, and free it after.

## Syntax / reference

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void normalize_kernel(float* data, int n, float* ssq) {
    cg::grid_group grid = cg::this_grid();     // the whole grid as a group
    // ... phase 1: atomicAdd into *ssq ...
    grid.sync();                               // grid-wide barrier
    // ... phase 2: read *ssq, scale data ...
}
```

**Sizing a co-resident grid.** Ask the occupancy API how many blocks of your
chosen size fit on **one** SM, multiply by the SM count:

```cpp
int block = 256;
int blocksPerSM = 0;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocksPerSM, normalize_kernel, block, /*dynamicSmem=*/0);
cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
int grid = blocksPerSM * prop.multiProcessorCount;   // all blocks co-resident
```

**Launching cooperatively.** Unlike `<<< >>>`, the cooperative launch takes the
kernel arguments as an **array of pointers-to-arguments**:

```cpp
void* args[] = { &data, &n, &ssq };
cudaLaunchCooperativeKernel((void*)normalize_kernel,
                            dim3(grid), dim3(block),
                            args, /*dynamicSmem=*/0, /*stream=*/0);
```

Each element of `args` is the **address of** the variable holding that argument.

## Grading (`!python grade.py`)

- **correctness** — output equals the CPU L2-normalized vector within tolerance.
- **efficiency** — reports `ms` (one fused launch instead of two).
- **source** — you must use cooperative groups (`grid.sync`) and launch with
  `cudaLaunchCooperativeKernel`.

The harness checks `cudaDevAttrCooperativeLaunch` at runtime; on the T4 it is
supported. If you ever run on a GPU without it, the harness prints
`# SKIP: cooperative launch unsupported` and reports correct so grading does not
false-fail.

Run `python grade.py --check-solution` to grade the reference solution instead.
