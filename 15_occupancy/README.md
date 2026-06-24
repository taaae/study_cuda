# Exercise 15 — Occupancy & launch-config tuning
> Stop guessing `block = 256`. Let the runtime tell you a launch config that keeps the SM busy.

## The idea
Every kernel you've launched picked a block size by feel — usually 256. Sometimes that's great,
sometimes it leaves the GPU half-idle. The number that actually matters is **occupancy**: the
ratio of *active warps on an SM* to the *hardware maximum* (the T4 allows up to 32 resident warps
per SM, 40 SMs total). High occupancy means the scheduler always has another warp ready to run
while one is stalled waiting on memory — that's how a GPU hides latency.

Picking the block size that maximizes occupancy by hand means knowing your kernel's register
count, shared-memory use, and the SM's resource budget. Tedious, and it changes per architecture.
So CUDA gives you an API that does the math: **`cudaOccupancyMaxPotentialBlockSize`** inspects your
kernel and hands back a `(blockSize, minGridSize)` pair that maximizes theoretical occupancy.

Here you run a memory-bound grid-stride map and let the API choose the config. Because the kernel
is grid-stride, *any* positive grid/block is *correct* — the lesson is choosing a *good* one.

## Under the hood
Each SM has a fixed budget shared by all its resident blocks: a register file, shared memory, a
cap on warp slots, and a cap on block slots. Whichever runs out first sets the ceiling.

- Use **64 registers/thread** instead of 32 and half as many warps fit → occupancy halves.
- Pick a **block size that divides the budget cleanly** and you waste fewer slots.

That ceiling is **theoretical occupancy** — what the hardware *allows* given your kernel's
resource use. **Achieved occupancy** is what actually happened at runtime (Nsight measures it;
it's blocked on Colab). The API optimizes the theoretical number.

> **Higher occupancy isn't always faster.** Occupancy only needs to be *high enough* to hide
> memory latency; past that point extra warps don't help. Worse, shrinking registers to chase
> 100% can force **register spills to local memory** (which lives in slow global memory) and make
> the kernel *slower*. The API gives a solid default, not a guaranteed optimum.

## A picture
```text
one SM's warp slots (T4: 32 max).  Each block here = 8 warps (256 threads).

  low-register kernel: 4 blocks fit     high-register kernel: 2 blocks fit
  ┌──────┬──────┬──────┬──────┐         ┌──────┬──────┬░░░░░░┬░░░░░░┐
  │ blk0 │ blk1 │ blk2 │ blk3 │         │ blk0 │ blk1 │ idle │ idle │
  └──────┴──────┴──────┴──────┘         └──────┴──────┴░░░░░░┴░░░░░░┘
  32/32 warps active → 100% occ.        16/32 warps active → 50% occ.
```

## Your task
Edit `occupancy.cu` and fill the TODOs:
1. The `__global__` kernel `map_kernel` — a grid-stride `out[i] = in[i] * 2 + 1`.
2. The host function `solve` — call `cudaOccupancyMaxPotentialBlockSize` to get
   `(minGridSize, blockSize)` for `map_kernel`, choose a grid, and launch.

A sensible grid is `min(minGridSize, ceil_div(n, blockSize))` so you never launch more blocks
than there is work — but with a grid-stride loop, any positive grid covers all of `n`.

### The `solve` contract
```cpp
void solve(const float* in, float* out, int n);
```
`in` / `out` are **device pointers** of length `n`.

## Functions & syntax you'll need

| Function | What it does |
| --- | --- |
| `cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, dynShmem, blockLimit)` | Picks the `blockSize` maximizing theoretical occupancy for `kernel`, plus a `minGridSize` that fully occupies the device. Pass `dynShmem=0`, `blockLimit=0` (no limit) here. |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, blockSize, dynShmem)` | For a *given* block size, how many blocks fit per SM. Handy for inspecting occupancy; not required to solve. |
| `kernel<<<grid, block>>>(args...)` | Launch with the chosen config. |
| `ceil_div(a, b)` | Helper from `cuda_utils.cuh`. |

Sketch:
```cpp
int minGridSize = 0, blockSize = 0;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, map_kernel, 0, 0);
// then choose a grid and launch map_kernel<<<grid, blockSize>>>(in, out, n);
```

## How it's graded
Run `python grade.py` (`!python grade.py` on Colab). It checks:
- **correctness** — `out == in * 2 + 1` within tolerance.
- **efficiency** — the kernel is memory-bound, so you must hit a healthy fraction of peak
  bandwidth: **bw_frac ≥ 0.55** (`gbps = 2*bytes/time`, read in + write out).
- **source** — you must call `cudaOccupancyMaxPotentialBlockSize`.

`python grade.py --check-solution` grades the reference solution instead of yours.

## Going deeper
Try hardcoding `block = 32` and watch `bw_frac` drop — too few warps per block can't hide
latency. Then try `block = 1024`. The occupancy API's pick usually lands at or near the best
`bw_frac`, which is the whole point: a good default without per-architecture guesswork. For a
memory-bound map like this, anything from ~128 to ~512 is typically fine; the API just removes
the guessing.
