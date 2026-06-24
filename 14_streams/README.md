# Exercise 14 — Streams & Copy/Compute Overlap

**New concepts:** CUDA **streams**, **pinned (page-locked) host memory**, `cudaMemcpyAsync`, and overlapping host↔device transfers with kernel execution — the classic *copy / compute / copy* pipeline.

## The task

Apply an element-wise map to a large array that lives in **pinned host memory**:

```
y = sqrt(x) * x + 1
```

The naive way is one big `cudaMemcpy` H2D, one kernel over the whole array, one big `cudaMemcpy` D2H. While the H2D copy runs, the GPU's compute units sit idle; while the kernel runs, both copy engines sit idle. The pipeline fixes that.

Split the array into **chunks**. Process chunk `i` on **stream `i % nStreams`**. Because each stream is an independent queue, the hardware can run the **H2D copy of chunk `i+1`**, the **kernel on chunk `i`**, and the **D2H copy of chunk `i-1`** *at the same time* — the two copy engines (one per direction on a T4) and the SMs all stay busy.

Edit `streams.cu` and fill in the `TODO`s:

1. The `__global__` kernel `map_kernel` — element-wise `y = sqrt(x)*x + 1` (grid-stride, so a chunk of any size works).
2. The host function `solve` — allocate device buffers, create the streams, and drive the chunked async H2D → kernel → async D2H pipeline, then synchronize.

### `solve` signature (the contract)

```cpp
void solve(const float* h_in, float* h_out, int n, int nStreams);
```

`h_in` and `h_out` are **pinned host pointers** (the harness allocated them with `cudaMallocHost`), length `n`. `nStreams` is how many streams to round-robin over. **You** do the device allocation, the chunked async copies, the kernel launches on streams, and the synchronization. Free everything you allocate before returning.

## Why pinned memory is required

Normal (pageable) host memory can be moved or swapped out by the OS, so the GPU's DMA engine can't safely read it asynchronously. `cudaMemcpyAsync` on pageable memory silently degrades to a **synchronous** copy — no overlap. Page-locked memory (`cudaMallocHost` / `cudaFreeHost`) is pinned to a fixed physical address, so the DMA engine can stream it in the background while the CPU and the SMs do other work. **Async copies only overlap when the host buffer is pinned.**

## The overlap model

A T4 has separate **copy engines** for each direction plus the compute SMs. Operations *in the same stream* run in order; operations *in different streams* can run concurrently if the resources are free. With ≥3 chunks in flight you get a steady state:

```
stream 0:  H2D c0 | comp c0 | D2H c0
stream 1:         | H2D c1  | comp c1 | D2H c1
stream 2:                   | H2D c2  | comp c2 | D2H c2
time  ───────────────────────────────────────────────▶
```

The total wall-clock time approaches `max(copyTime, computeTime)` instead of their sum.

## Syntax you'll need

```cpp
cudaStream_t s;
cudaStreamCreate(&s);                                  // make a stream
cudaMemcpyAsync(dst, src, bytes, kind, s);            // queued on stream s
kernel<<<grid, block, 0, s>>>(args...);               // 4th launch param = stream
cudaStreamSynchronize(s);   // wait for one stream
cudaDeviceSynchronize();    // wait for all work on the device
cudaStreamDestroy(s);
```

`kind` is `cudaMemcpyHostToDevice` or `cudaMemcpyDeviceToHost`.
Helper available: `ceil_div(a, b)` from `cuda_utils.cuh`.

## Grading (`!python grade.py`)

- **correctness** — `y == sqrt(x)*x + 1` within tolerance.
- **efficiency** — the harness times your `solve` **end to end** (copies included) and divides by a single-stream synchronous baseline it also runs. You must reach a **speedup ≥ 1.3**.
- **source** — you must use `cudaMemcpyAsync` and create at least one stream (`cudaStreamCreate` / `cudaStream_t`).

Run `python grade.py --check-solution` to grade the reference solution instead of yours.
