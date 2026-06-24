# Exercise 14 — Streams & copy/compute overlap
> Your GPU has separate engines for copying and computing. Stop making them take turns.

## The idea
Every GPU job so far has had the same shape: copy data up (H2D), run a kernel, copy results
back (D2H). The naive way does these strictly in sequence — one big `cudaMemcpy` in, one kernel
over the whole array, one big `cudaMemcpy` out. The problem: while the H2D copy runs, the SMs
sit idle; while the kernel runs, both copy engines sit idle. You're paying for hardware that's
asleep most of the time.

A T4 actually has **three independent units** that can work at once: a copy engine for
host→device, a copy engine for device→host, and the compute SMs. The trick to using all three
is the **stream**. A stream is an ordered queue of GPU operations. Work *within* one stream runs
in order; work in *different* streams can run concurrently when the resources are free.

So you split the array into **chunks** and round-robin them across streams. Once the pipeline
fills up, the hardware copies chunk `i+1` up, computes chunk `i`, and copies chunk `i-1` down —
all at the same time. Wall-clock time drops from `copy + compute + copy` toward
`max(copyTime, computeTime)`.

## Under the hood
Why does this need **pinned (page-locked) host memory**? Normal host allocations are *pageable* —
the OS can move or swap them out at any moment. A DMA engine reading such memory in the
background could find it relocated mid-transfer, so the driver refuses: `cudaMemcpyAsync` on
pageable memory **silently degrades to a synchronous copy**, and you get zero overlap.

`cudaMallocHost` pins the buffer to a fixed physical address. Now the DMA engine can stream it
in the background while the CPU and SMs do other things — true asynchronous transfer. The
harness already allocated `h_in` / `h_out` with `cudaMallocHost`, so you're set; just know that
**async copies only overlap when the host buffer is pinned.**

One more thing the chunking buys you: pinned transfers also hit higher peak PCIe bandwidth than
pageable ones, because the driver doesn't have to stage through an internal pinned bounce buffer.

## A picture
```text
streams round-robined over 3 chunks (steady state = all engines busy):

stream 0:  H2D c0 │ comp c0 │ D2H c0
stream 1:         │ H2D c1  │ comp c1 │ D2H c1
stream 2:                   │ H2D c2  │ comp c2 │ D2H c2
time  ─────────────────────────────────────────────────▶
                  ▲ here the H2D engine, the SMs, and the
                    D2H engine are ALL working at once.
```

## Your task
Edit `streams.cu` and fill the TODOs:
1. The `__global__` kernel `map_kernel` — element-wise `y = sqrt(x)*x + 1`, written grid-stride
   so a chunk of any size is handled correctly.
2. The host function `solve` — allocate device buffers, create the streams, drive the chunked
   `H2D → kernel → D2H` pipeline on stream `i % nStreams`, then synchronize and clean up.

Use pointer offsets (`h_in + off`, `d_in + off`, …) and clamp the last chunk to `n`. Free
everything you allocate before returning.

### The `solve` contract
```cpp
void solve(const float* h_in, float* h_out, int n, int nStreams);
```
`h_in` / `h_out` are **pinned host pointers** of length `n` (allocated by the harness with
`cudaMallocHost`). `nStreams` is how many streams to round-robin over (the harness passes 4).
**You** own the device allocation, the chunked async copies, the kernel launches, and the
synchronization.

## Functions & syntax you'll need

| Function | What it does |
| --- | --- |
| `cudaStream_t s; cudaStreamCreate(&s)` | Create an ordered work queue. |
| `cudaMemcpyAsync(dst, src, bytes, kind, s)` | Copy queued on stream `s`; returns immediately. `kind` is `cudaMemcpyHostToDevice` or `cudaMemcpyDeviceToHost`. |
| `kernel<<<grid, block, 0, s>>>(args...)` | The **4th** launch parameter is the stream. (3rd is dynamic shared memory — 0 here.) |
| `cudaStreamSynchronize(s)` | Block the host until everything on stream `s` is done. |
| `cudaDeviceSynchronize()` | Block until *all* device work (every stream) is done. |
| `cudaStreamDestroy(s)` | Destroy a stream when finished. |
| `cudaMalloc / cudaFree` | Device buffers. Allocate them big enough for the whole array. |
| `ceil_div(a, b)` | Helper from `cuda_utils.cuh` for grid sizing. |

Sketch of one chunk's launch:
```cpp
cudaMemcpyAsync(d_in + off, h_in + off, b, cudaMemcpyHostToDevice, s);
map_kernel<<<ceil_div(len, block), block, 0, s>>>(d_in + off, d_out + off, len);
cudaMemcpyAsync(h_out + off, d_out + off, b, cudaMemcpyDeviceToHost, s);
```

## How it's graded
Run `python grade.py` (`!python grade.py` on Colab). It checks:
- **correctness** — `y == sqrt(x)*x + 1` within tolerance.
- **efficiency** — the harness times your `solve` **end to end** (copies included) and divides by
  a single-stream synchronous baseline it runs itself. You need **speedup ≥ 1.3**.
- **source** — you must use `cudaMemcpyAsync` and create at least one stream (`cudaStreamCreate` /
  `cudaStream_t`).

`python grade.py --check-solution` grades the reference solution instead of yours.

## Going deeper
This kernel is memory-bound, so `copyTime` and `computeTime` are comparable and overlap helps a
lot. For a compute-heavy kernel, copies are tiny and overlap barely matters — profile before you
pipeline. Also try `nStreams = 2` vs `8`: past ~3 in-flight chunks the speedup plateaus, because
you only have two copy engines to saturate.
