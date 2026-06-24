# Exercise 03 — Benchmarking with CUDA Events

**New concepts:** timing GPU work correctly with **`cudaEvent`**, the **warmup + best-of-N** discipline, computing **achieved bandwidth**, and comparing it to the device's **theoretical peak**.

So far the harness has timed your kernels for you (`time_kernel` in `cuda_utils.cuh`). This time **you** write the timing code, because measuring a kernel honestly is a skill of its own.

## The task

Two pieces, both in `bench.cu`:

1. The `__global__` kernel `copy_kernel` — a **grid-stride copy**: `out[i] = in[i]` for all `i`.
2. The host function `benchmark_copy` — launch that kernel, **time it yourself with CUDA events**, and return the **best elapsed milliseconds** over `iters` runs (after a warmup). On return, `out` must hold a correct copy of `in`.

You do **not** write `main()` — `harness.cu` provides it, calls `benchmark_copy`, and independently verifies both your copy and your timing.

### `benchmark_copy` signature (the contract)

```cpp
float benchmark_copy(const float* in, float* out, int n, int iters);
```

- `in`, `out` are **device pointers** of length `n`.
- Do **one warmup launch**, then time `iters` launches and return the **minimum** elapsed time in **milliseconds**.
- After returning, `out == in` element-wise (a real copy happened last).
- You must use `cudaEvent`s directly here — do **not** call the library `time_kernel`; the whole point is to write the event code yourself.

## How to time GPU work: CUDA events

You cannot time a kernel with a CPU clock (`clock()`, `std::chrono`) naively, because kernel launches are **asynchronous** — the launch returns immediately, before the GPU is done. CUDA **events** are timestamps recorded *in the GPU's stream*, so they measure actual device execution:

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);          // mark "now" in the stream
my_kernel<<<grid, block>>>(...); // enqueue work
cudaEventRecord(stop);           // mark the end in the stream
cudaEventSynchronize(stop);      // wait until 'stop' has actually happened

float ms = 0.f;
cudaEventElapsedTime(&ms, start, stop);   // device time between the two events

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

`cudaEventElapsedTime` returns **milliseconds** (float).

## Two disciplines that make a benchmark trustworthy

- **Warmup.** The first launch pays one-time costs: context/JIT setup, caches cold, clocks not yet boosted. Run the kernel **once and throw the result away** before timing.
- **Best-of-N.** A shared Colab GPU has scheduling jitter and clock fluctuation. Run the kernel `iters` times and keep the **minimum** — the fastest run is the one least disturbed by noise, and best approximates the kernel's true cost.

## Achieved vs theoretical bandwidth

A copy of `n` floats moves `2*bytes`: read `in`, write `out`. Achieved bandwidth is:

```
GB/s = (2 * n * sizeof(float)) / (ms * 1e-3) / 1e9
```

Divide that by `peak_bandwidth_gbps()` (queried from the device) to get the **fraction of peak**. A pure copy is the simplest memory-bound kernel and should reach a high fraction — that's why the threshold here is strict.

## Grading (`!python grade.py`)

- **correctness** — `out == in` after `benchmark_copy` returns.
- **ms_ratio** — the harness *also* times your copy with its own `time_kernel`; your returned `ms` must agree within ~25% (`ms_ratio` in `[0.6, 1.6]`). If you forgot the warmup or didn't take the min, your number drifts and this fails.
- **efficiency** — `bw_frac` computed from *your* `ms` must be `>= 0.70`. A pure copy on a grid-stride kernel clears this.
- **source** — you must use `cudaEventRecord` and `cudaEventElapsedTime` (i.e. you actually wrote the event timing).

Run `python grade.py --check-solution` to grade the reference solution instead of yours.
