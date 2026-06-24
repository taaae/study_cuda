# Exercise 13 — CUB (the SOTA performance-primitive library)

**New concepts:** CUB as NVIDIA's tuned, per-architecture primitive library; the **two-call temp-storage idiom**; `cub::DeviceScan::ExclusiveSum`. We re-solve exercise 10's scan to feel the difference between a hand-rolled kernel and a library that has been auto-tuned for every GPU.

## CUB vs Thrust vs your own kernel

- **Your kernel (ex 10):** maximum control, but you hand-tune tile sizes, padding, and the multi-pass plumbing — and re-tune per architecture.
- **Thrust (ex 12):** one-liners, host-side, great productivity. Thrust's device backend *is* CUB under the hood.
- **CUB:** the layer beneath Thrust. It exposes **device-wide** primitives (`DeviceScan`, `DeviceReduce`, `DeviceRadixSort`, `DeviceSelect`) *and* **block/warp** building blocks (`BlockScan`, `BlockReduce`, `WarpReduce`) you can drop inside your own kernels. CUB picks tile sizes and algorithms tuned for the *exact* compute capability at compile time, so it routinely beats a hand-rolled scan.

## The task

Compute the **exclusive** prefix sum of a large `int` array using `cub::DeviceScan::ExclusiveSum` — the same problem as exercise 10, now in a handful of lines.

### `solve` signature (the contract)

```cpp
void solve(const int* in, int* out, int n);
```

Same as exercise 10 (`in`, `out` are device pointers of length `n`) so you can diff the two solutions and the two `bw_frac` numbers.

## The temp-storage idiom (the one thing to memorize)

CUB device functions don't allocate; **you** give them scratch. Every `cub::Device*` call is made **twice**:

```cpp
void*  d_temp = nullptr;
size_t temp_bytes = 0;

// 1) query: d_temp == nullptr → CUB only writes the required size into temp_bytes
cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, in, out, n);

// 2) allocate and call for real
CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, in, out, n);

CUDA_CHECK(cudaFree(d_temp));
```

First call with `d_temp == nullptr` is the *sizing* pass: nothing runs, CUB just fills `temp_bytes`. Then you `cudaMalloc` that many bytes and call again to do the work. This pattern is identical for every `cub::Device*` algorithm.

## Block/warp primitives (brief — for later)

Inside a kernel you can compose CUB's collectives instead of writing your own:

```cpp
using BlockScan = cub::BlockScan<int, 128>;
__shared__ typename BlockScan::TempStorage tmp;
BlockScan(tmp).ExclusiveSum(thread_in, thread_out);   // a whole block scans in 1 call
```

`BlockReduce`, `WarpReduce`, and `WarpScan` work the same way. You won't need them here, but they are how you'd build a custom kernel that still uses CUB's tuned internals.

## Syntax / reference

```cpp
#include <cub/cub.h>
cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_in, d_out, num_items);
```

## Grading (`!python grade.py`)

- **correctness** — output matches a CPU exclusive scan exactly.
- **efficiency** — `bw_frac >= 0.55`. CUB is single-pass-class fast, so the bar is much higher than the hand-rolled scan in exercise 10 (`0.30`). Compare the two numbers — that gap is the lesson.
- **source** — you must call `cub::DeviceScan`.

Run `python grade.py --check-solution` to grade the reference solution instead of yours.
