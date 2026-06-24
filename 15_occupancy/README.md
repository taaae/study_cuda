# Exercise 15 — Occupancy & Launch-Config Tuning

**New concepts:** **occupancy** — the ratio of active warps on an SM to the hardware maximum — and the CUDA **occupancy API** that picks a launch configuration for you, plus the register/shared-memory vs occupancy tradeoff.

## The task

Run a simple memory-bound element-wise kernel (a grid-stride map), but instead of hardcoding `block = 256`, let the runtime **choose the block size and grid size** for you with `cudaOccupancyMaxPotentialBlockSize`.

Edit `occupancy.cu` and fill in the `TODO`s:

1. The `__global__` kernel `map_kernel` — a grid-stride `out[i] = in[i] * 2 + 1`.
2. The host function `solve` — call `cudaOccupancyMaxPotentialBlockSize` to get a `(minGridSize, blockSize)` pair, then launch the grid-stride kernel with that config.

### `solve` signature (the contract)

```cpp
void solve(const float* in, float* out, int n);
```

`in` and `out` are **device pointers** of length `n`. Because the kernel is grid-stride, *any* positive grid/block is correct — the lesson here is using the API to choose a good one rather than guessing.

## Theoretical vs achieved occupancy

**Theoretical occupancy** is what the hardware *allows* given your kernel's resource use: each SM has a fixed budget of registers, shared memory, warp slots, and block slots. If your kernel uses 64 registers/thread, fewer warps fit, so theoretical occupancy drops. **Achieved occupancy** is what actually happened at runtime (measurable with Nsight, blocked on Colab). You raise theoretical occupancy by using *fewer* registers / less shared memory per thread, or by choosing a block size that divides the SM's resources cleanly.

## The occupancy API

`cudaOccupancyMaxPotentialBlockSize` picks the block size that **maximizes theoretical occupancy** for a given kernel, and also hands back a `minGridSize` — the smallest grid that fully occupies the device:

```cpp
int minGridSize = 0, blockSize = 0;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                   map_kernel,   // the kernel
                                   0,            // dynamic shared mem per block (bytes)
                                   0);           // block-size limit (0 = no limit)
```

A second call inspects occupancy for a *specific* block size — how many blocks fit per SM:

```cpp
int maxBlocks = 0;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks, map_kernel,
                                              blockSize, /*dynShmem=*/0);
```

For a grid-stride kernel a good grid is `minGridSize` (or `minGridSize` scaled so each thread does a few elements). Because every thread loops with stride `blockDim.x * gridDim.x`, the whole array is covered regardless.

> **Higher occupancy isn't always faster.** Occupancy only needs to be *high enough* to hide memory latency; past that point, extra warps don't help, and shrinking registers to chase 100% occupancy can spill to local memory and make things slower. The API gives a solid default, not a guarantee of the optimum.

## Syntax you'll need

```cpp
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, dynShmem, blockLimit);
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, blockSize, dynShmem);
kernel<<<grid, block>>>(args...);
```

Helper available: `ceil_div(a, b)` from `cuda_utils.cuh`.

## Grading (`!python grade.py`)

- **correctness** — `out == in * 2 + 1` within tolerance.
- **efficiency** — memory-bound kernel, so you must reach a healthy fraction of peak bandwidth: **bw_frac ≥ 0.55**.
- **source** — you must call `cudaOccupancyMaxPotentialBlockSize`.

Run `python grade.py --check-solution` to grade the reference solution instead of yours.
