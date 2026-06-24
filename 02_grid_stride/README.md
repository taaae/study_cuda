# Exercise 02 — Grid-Stride Loops

**New concepts:** the **grid-stride loop** (decoupling the grid size from the problem size so each thread handles many elements), and the **`CUDA_CHECK`** error-checking macro.

## The task

Compute **SAXPY**: `y = a*x + y` for float arrays of length `n`, where `a` is a scalar. But this time, instead of launching one thread per element, you launch a **fixed, modest grid** (sized from the device's SM count, *not* `ceil_div(n, block)`), and each thread walks the array in a **grid-stride loop**, processing several elements.

Edit `saxpy.cu` and fill in the `TODO`s:

1. The `__global__` kernel `saxpy` — loop over the array with a grid-stride loop, computing `y[i] = a*x[i] + y[i]`.
2. The host function `solve` — query the device, choose a fixed grid based on SM count, and launch the kernel. Wrap your CUDA runtime calls in `CUDA_CHECK(...)`.

You do **not** write `main()`, allocate memory, or copy data — `harness.cu` does all of that and calls your `solve(...)`.

### `solve` signature (the contract)

```cpp
void solve(float a, const float* x, float* y, int n);
```

`x` and `y` are **device pointers** (already on the GPU) of length `n`; `a` is a host scalar. `y` is updated in place. Your job is only to launch the kernel.

## The grid-stride pattern

In exercise 01 you sized the grid to the problem: one thread per element. Here you do the opposite — pick a **fixed** number of threads and let each one stride across the whole array:

```cpp
__global__ void k(/* ... */ int n) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's start
    int stride = gridDim.x * blockDim.x;                 // total threads in grid
    for (int i = idx; i < n; i += stride) {
        // ... work on element i ...
    }
}
```

Thread 0 handles elements `0, stride, 2*stride, …`; thread 1 handles `1, 1+stride, …`; and so on. The whole array is covered no matter how large `n` is, with a grid you chose.

## Why grid-stride beats one-thread-one-element here

- **Flexible sizing.** One launch config handles *any* `n` — even `n` larger than the maximum grid dimension. You never recompute the grid from `n`.
- **Thread/register reuse.** A launched thread amortizes its setup (index math, register allocation, block-scheduling cost) across many elements instead of doing one add and retiring. Fewer blocks means fewer block-scheduling/launch-tail overheads.
- **Tunable occupancy.** You size the grid to *the machine* (a few blocks per SM) rather than to the data, which keeps every SM busy without spawning millions of one-shot blocks.

A good launch is a few blocks per SM. Query the SM count at runtime:

```cpp
cudaDeviceProp prop;
CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
int blocks = prop.multiProcessorCount * SOME_SMALL_FACTOR;
```

## The `CUDA_CHECK` macro

Every CUDA runtime call returns a `cudaError_t`. Silently ignoring it is the #1 source of mysterious CUDA bugs. Wrap calls in `CUDA_CHECK(...)` (from `common/cuda_utils.cuh`): on failure it prints `file:line`, the call, and the error string, then aborts.

```cpp
CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
```

For kernel *launches* (which don't return an error directly) the harness uses `CUDA_CHECK_KERNEL()` after calling your `solve`, so you don't need to.

## Grading (`!python grade.py`)

- **correctness** — `y == a*x + y` within tolerance.
- **efficiency** — SAXPY is memory-bound (read `x`, read `y`, write `y` ⇒ `3*bytes`). You must reach a healthy fraction of peak global-memory bandwidth (`bw_frac >= 0.55`). A correct grid-stride loop over a modest grid clears this with margin.
- **source** — your kernel must contain an actual stride loop: the grade checks for both `gridDim.x` and `blockDim.x` (the stride is `gridDim.x * blockDim.x`).

Run `python grade.py --check-solution` to grade the reference solution instead of yours.
