# Exercise 04 — Global-Memory Coalescing

**New concepts:** **memory coalescing** — how a 32-thread warp's global loads/stores are serviced in 128-byte segments, and why the *order* in which threads touch memory decides your bandwidth.

We learn it through a **matrix transpose** with **no shared memory**. The key lesson: a naive transpose can coalesce the **reads** *or* the **writes**, but not both. Which one you choose changes performance a lot.

## The task

Transpose an `N × N` row-major float matrix: `out = in^T`, i.e. `out[r][c] = in[c][r]`. One thread per output element, **global memory only** (no `__shared__` — that's the next exercise).

Arrange your indexing so that **consecutive threads in a warp make coalesced reads of `in`**. (The writes to `out` will then be strided — that's expected and is exactly what exercise 05 fixes with shared memory.)

Edit `transpose.cu`:

1. The `__global__` kernel `transpose` — map this thread to a `(row, col)`, read `in` *coalesced*, write the transposed location in `out`.
2. The host function `solve` — launch with a 2-D grid of 2-D blocks covering the matrix.

You do **not** write `main()` — `harness.cu` provides it, checks correctness, runs a deliberately-bad baseline, and reports your speedup.

### `solve` signature (the contract)

```cpp
void solve(const float* in, float* out, int n);
```

`in`, `out` are **device pointers** to `n*n` floats, row-major. `n` is the matrix dimension.

## Coalescing in one picture

Threads execute in **warps** of 32. When a warp issues a global load, the hardware serves it as a small number of aligned **128-byte memory transactions** (a 128-byte segment holds 32 contiguous floats). The best case:

- the 32 threads of a warp touch **32 consecutive floats** → **one** 128-byte transaction → full bandwidth (**coalesced**).

The worst case:

- the 32 threads touch addresses **`n` floats apart** (a column of the matrix) → up to **32 separate transactions**, each delivering one useful float and wasting the rest → ~1/32 of the bandwidth (**uncoalesced / strided**).

The fast-varying thread index is `threadIdx.x`: adjacent `threadIdx.x` should map to adjacent memory addresses for whichever array you want coalesced.

## Reads vs writes — you can only win one

For a transpose, the read pattern and the write pattern are transposes of each other, so making one contiguous forces the other to be strided:

| Choice | `in` access | `out` access |
|--------|-------------|--------------|
| coalesce reads (this exercise) | contiguous ✅ | strided ❌ |
| coalesce writes | strided ❌ | contiguous ✅ |

Either single-sided choice beats the truly-naive version (which manages to make **both** sides strided). The baseline the harness runs is exactly that doubly-strided version, so coalescing *one* side already gives a clear speedup.

To coalesce reads: let `col = blockIdx.x * blockDim.x + threadIdx.x` index the **contiguous** dimension of `in`, and `row = blockIdx.y * blockDim.y + threadIdx.y`. Then `in[row * n + col]` is coalesced across a warp (consecutive `threadIdx.x` → consecutive addresses). The transposed destination is `out[col * n + row]`.

## Launch config

Use square blocks, e.g. `dim3 block(32, 8)` or `dim3 block(16, 16)`, and a 2-D grid that covers `n` in both dimensions:

```cpp
dim3 block(32, 8);
dim3 grid(ceil_div(n, block.x), ceil_div(n, block.y));
```

Guard against `row >= n || col >= n` in the kernel for non-multiple sizes (the harness uses a multiple of 32, but guarding is good habit).

## Grading (`!python grade.py`)

- **correctness** — `out == in^T`.
- **speedup** — `naive_ms / your_ms >= 1.3`. The naive baseline is the doubly-strided transpose; coalescing the reads must beat it clearly.
- **efficiency** — `bw_frac >= 0.35` (2*bytes moved). A one-sided-strided transpose can't hit peak (the strided side caps you), but it should comfortably clear this floor; the truly naive version won't.
- **source** — must use 2-D-style indexing with `blockIdx`/`threadIdx`, and must **NOT** use `__shared__` (that's exercise 05).

Run `python grade.py --check-solution` to grade the reference solution instead of yours.
