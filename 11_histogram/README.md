# Exercise 11 — Histogram (atomics & privatization)

**New concepts:** atomic read-modify-write (`atomicAdd`), **contention** when many threads hit the same address, and **shared-memory privatization** — the standard cure.

## The task

Build a 256-bin histogram of an `unsigned char` array: bin `b` counts how many bytes equal `b`. Many input elements map to the same bin, so threads *collide* on the counters — this is the whole point.

### `solve` signature (the contract)

```cpp
void solve(const unsigned char* data, unsigned int* hist, int n);
```

`data` is a **device pointer** to `n` bytes. `hist` is a **device pointer** to 256 `unsigned int`s, already **zeroed by the harness**. Add up the counts; launch whatever kernels you need.

## Why the naive version is slow

The obvious kernel is one `atomicAdd(&hist[data[i]], 1)` per element straight to global memory. It is correct, but with 16M elements and only 256 bins, *thousands* of threads pound the same global counter at once. Atomics to a contended address **serialize** — the hardware processes them one after another — so the kernel is bottlenecked on a handful of hot addresses, not on bandwidth.

## The fix: per-block private histograms in shared memory

Give each block its **own** 256-bin histogram in `__shared__` memory:

1. **Zero** the shared histogram cooperatively (256 bins, `blockDim.x` threads → a small strided loop).
2. `__syncthreads()`.
3. Each thread walks its elements (use a grid-stride loop so the kernel works for any `n`) and does `atomicAdd(&sHist[data[i]], 1)` — an atomic on **shared** memory.
4. `__syncthreads()`.
5. **Merge** the shared histogram into global: each thread `atomicAdd`s a few shared bins into the matching global bins.

This shrinks contention two ways: shared-memory atomics are far cheaper than global ones and are contended only *within one block*, and the expensive global atomics now happen just **256 per block** instead of once per element.

> On modern GPUs `atomicAdd` on shared memory is a hardware op (not an emulated lock), which is exactly why this pattern wins.

## Syntax / reference

```cpp
__shared__ unsigned int sHist[256];           // per-block private histogram
atomicAdd(&sHist[b], 1u);                      // shared-memory atomic
atomicAdd(&hist[b], sHist[b]);                 // merge to global
__syncthreads();

// grid-stride loop over n elements:
for (int i = blockIdx.x * blockDim.x + threadIdx.x;
     i < n; i += blockDim.x * gridDim.x) { ... }
```

## Grading (`!python grade.py`)

- **correctness** — your 256 bins match a CPU histogram exactly.
- **efficiency** — the harness also runs a **naive global-atomic** baseline and computes `speedup = baseline_ms / your_ms`. You must reach `speedup >= 2.0`. (`ms` is reported too.)
- **source** — you must use `__shared__` memory and `atomicAdd`.

Run `python grade.py --check-solution` to grade the reference solution instead of yours.
