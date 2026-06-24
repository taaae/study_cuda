# Exercise 13 — CUB, NVIDIA's tuned primitive library
> You wrote a scan by hand in exercise 10. Now watch a library do it in three lines — and faster.

## The idea
Back in exercise 10 you built an exclusive prefix sum from scratch: tiles, padding, a
multi-pass reduce-then-scan. It worked, and you learned a ton. But you also hand-picked a
tile size, and that choice is only *good* for one GPU. Run the same code on a different
architecture and you'd want to re-tune.

CUB is the answer to "I never want to do that again." It's NVIDIA's library of
**device-wide primitives** — `DeviceScan`, `DeviceReduce`, `DeviceRadixSort`,
`DeviceSelect` — that have been auto-tuned for *every* compute capability. When you compile
for `sm_75` (the T4), CUB picks tile sizes and algorithms measured to be fast on exactly
that chip. The result routinely beats a careful hand-rolled kernel.

Here we re-solve exercise 10's scan with `cub::DeviceScan::ExclusiveSum` so you can diff the
two solutions side by side — and compare their `bw_frac` numbers. That gap is the whole lesson.

> **CUB vs Thrust:** Thrust (ex 12) is the friendly host-side layer of one-liners. Its device
> backend *is* CUB. CUB sits one level down: device-wide calls like the one here, plus
> **block/warp** collectives (`BlockScan`, `WarpReduce`) you can drop *inside* your own kernels.

## Under the hood
A device-wide scan is genuinely hard to make fast. The classic textbook approach needs
multiple passes over memory; the bandwidth bound says the *minimum* is to read the input once
and write the output once (`2*bytes`). CUB uses a **single-pass "decoupled look-back" scan**:
blocks compute their local sums, publish them, and look back at predecessors' published
aggregates instead of waiting for a separate global pass. That collapses the algorithm to
roughly memory-bandwidth-bound — which is why the efficiency bar here (`bw_frac >= 0.55`) is
nearly double the hand-rolled bar from exercise 10 (`0.30`).

The one quirk of using CUB device functions: **they never allocate memory for you.** You hand
them a scratch buffer. To find out how big it needs to be, you call the function *twice* — the
**two-call temp-storage idiom** below. Every `cub::Device*` algorithm works this exact way.

## A picture
```text
two-call temp-storage idiom

  call #1  (d_temp == nullptr)        call #2  (d_temp valid)
  ┌──────────────────────────┐        ┌──────────────────────────┐
  │ CUB writes required size  │  ───►  │ CUB actually runs the     │
  │ into temp_bytes.          │        │ scan using your scratch.  │
  │ NOTHING is computed yet.  │        │                           │
  └──────────────────────────┘        └──────────────────────────┘
        │                                      ▲
        └── cudaMalloc(&d_temp, temp_bytes) ───┘
```

## Your task
Compute the **exclusive** prefix sum of a large `int` array (16M elements) using
`cub::DeviceScan::ExclusiveSum`. The `solve` body is the two-call idiom; fill in the TODOs in
`scan_cub.cu`. `#include <cub/cub.h>` is already there.

### The `solve` contract
```cpp
void solve(const int* in, int* out, int n);
```
`in` and `out` are **device pointers** of length `n` — identical to exercise 10, so you can
diff the two.

## Functions & syntax you'll need

| Function | What it does |
| --- | --- |
| `cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_in, d_out, n)` | Exclusive prefix sum. Call once with `d_temp==nullptr` to size, once with real scratch to run. |
| `cudaMalloc(&d_temp, temp_bytes)` | Allocate the scratch CUB asked for, as a `void*`. |
| `cudaFree(d_temp)` | Release the scratch before returning. |
| `#include <cub/cub.h>` | Pulls in all of CUB's device-wide primitives. |

The two-call idiom, in full:
```cpp
void*  d_temp = nullptr;
size_t temp_bytes = 0;

cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, in, out, n);   // 1) size
CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));                     //    allocate
cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, in, out, n);   // 2) run
CUDA_CHECK(cudaFree(d_temp));
```

> **For later:** the same library exposes block/warp collectives. Inside a kernel,
> ```cpp
> using BlockScan = cub::BlockScan<int, 128>;
> __shared__ typename BlockScan::TempStorage tmp;
> BlockScan(tmp).ExclusiveSum(thread_in, thread_out);  // a whole block scans in 1 call
> ```
> `BlockReduce`, `WarpReduce`, and `WarpScan` follow the same shape. Not needed here.

## How it's graded
Run `python grade.py` (or `!python grade.py` in Colab). It checks:
- **correctness** — your output matches a CPU exclusive scan exactly.
- **efficiency** — `bw_frac >= 0.55`. Because CUB's scan is single-pass-class, the bar is much
  higher than exercise 10's `0.30`. Look at both numbers — the difference is the point.
- **source** — you must call `cub::DeviceScan`.

`python grade.py --check-solution` grades the reference solution in `solutions/` instead of yours.

## Going deeper
Try swapping in `cub::DeviceReduce::Sum` or `cub::DeviceRadixSort::SortKeys` — the temp-storage
idiom is identical, so once you know it you know all of CUB's device layer. If you ever profile
a scan and see it land near `bw_frac` 1.0, that's decoupled look-back doing its job: there is
simply no faster way to move the bytes.
