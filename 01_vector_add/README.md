# Exercise 01 — Vector Add

**New concepts:** the CUDA execution model (grid → blocks → threads), global thread indexing, device memory, and kernel launch syntax.

## The task

Compute the element-wise sum `C = A + B` for float arrays of length `n` on the GPU, with **one thread per element**.

Edit `vadd.cu` and fill in the two `TODO`s:

1. The `__global__` kernel `vadd` — compute this thread's global index and write one output element.
2. The host function `solve` — choose a block size, compute the grid size, and launch the kernel.

You do **not** write `main()`, allocate memory, or copy data — `harness.cu` does all of that and calls your `solve(...)`.

### `solve` signature (the contract)

```cpp
void solve(const float* a, const float* b, float* c, int n);
```

`a`, `b`, `c` are **device pointers** (already on the GPU) of length `n`. Your job is only to launch the kernel.

## Mental model

A kernel runs as a **grid** of **blocks**, each block a group of **threads**. You pick those dimensions at launch:

```cpp
kernel<<<numBlocks, threadsPerBlock>>>(args...);
```

Inside the kernel each thread finds its own unique position from built-in variables:

| variable | meaning |
|----------|---------|
| `threadIdx.x` | this thread's index within its block (`0 .. blockDim.x-1`) |
| `blockIdx.x`  | this block's index within the grid |
| `blockDim.x`  | threads per block |
| `gridDim.x`   | blocks in the grid |

The standard 1-D global index is:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

## Why the boundary check matters

You'll launch `ceil(n / blockSize)` blocks, so the last block usually has **extra threads** whose index `i >= n`. Those must not write to `c[i]`, or you corrupt memory past the array. Guard every global write with `if (i < n)`.

Helper available from `common/cuda_utils.cuh`: `ceil_div(a, b)` returns `(a + b - 1) / b`.

## Grading (`!python grade.py`)

- **correctness** — `C == A + B` within tolerance.
- **efficiency** — vector add is purely memory-bound; you must reach a healthy fraction of peak global-memory bandwidth (a working one-thread-per-element kernel does this easily).
- **source** — you must actually launch with the `<<< >>>` syntax.

Run `python grade.py --check-solution` to grade the reference solution instead of yours.
