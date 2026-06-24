# Exercise 04 — Global-Memory Coalescing
> The single most important performance idea in CUDA: when a warp reads memory, the *order* its threads touch addresses can swing your bandwidth by up to 32×.

## The idea

Two kernels can do the exact same arithmetic and the exact same number of memory accesses, yet one runs many times faster than the other. The difference is **coalescing** — whether the 32 threads of a warp touch memory addresses that are *next to each other* or *scattered apart*.

The GPU never fetches a single float from global memory. It fetches an aligned **128-byte segment** (= 32 consecutive floats) at a time. So when a warp issues a load:

- if its 32 threads want 32 *consecutive* floats → the hardware delivers them in **one** 128-byte transaction → every byte fetched is used → full bandwidth. This is **coalesced**.
- if its 32 threads want floats that are far apart (say, one whole matrix row apart) → it takes up to **32 separate** transactions, each dragging in 128 bytes to use only 4 → ~1/32 of the bandwidth wasted. This is **uncoalesced / strided**.

We learn this through a **matrix transpose**, `out = in^T`, using global memory only. Transpose is the perfect teacher because its read pattern and write pattern are transposes of each other — so you can coalesce the reads *or* the writes, but **never both** with plain global memory. Choosing which to coalesce is the lesson. (Exercise 05 will use shared memory to finally get both.)

## Under the hood

The variable that changes fastest across the threads of a warp is **`threadIdx.x`**. Threads 0–31 of a warp have `threadIdx.x = 0..31` (with `threadIdx.y` fixed). So the rule of thumb is dead simple:

> **Whatever array you want coalesced, index it so that incrementing `threadIdx.x` by 1 moves the address by 1 element.**

For a row-major matrix `in[row * n + col]`, the *contiguous* dimension is `col` — consecutive `col` values are adjacent in memory. So if you let `col` be driven by `threadIdx.x`, then across a warp the reads of `in` land on consecutive addresses: coalesced. The transposed destination `out[col * n + row]` then has `row`... varying with the *slow* index, so consecutive threads write addresses `n` floats apart — strided. That's the trade we're accepting.

The naive baseline the harness pits you against makes the *opposite* choice: it drives `row` with `threadIdx.x`, which strides the read of `in`. That version is doubly bad, and just coalescing your reads will clearly beat it.

> **Fun fact.** Older GPUs (pre-Fermi) had brutal coalescing rules — threads had to access addresses in a strict order or you'd fall off a performance cliff. Modern hardware (including the T4) is more forgiving: it just needs the 32 accesses to fall within as few 128-byte segments as possible. The intuition "adjacent threads → adjacent addresses" still wins every time.

## A picture

```text
 in (row-major, n=8 shown)        a warp reads one ROW of `in`:
   col→ 0  1  2  3  4  5  6  7
 row 0 [ ][ ][ ][ ][ ][ ][ ][ ]   threadIdx.x = 0 1 2 3 4 5 6 7
 row 1 [ ][ ][ ][ ][ ][ ][ ][ ]            ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
 ...                              in[row*n + col]: consecutive → 1 transaction ✅

 COALESCED READ                   STRIDED READ (the naive baseline)
 t0 t1 t2 t3  (one segment)       t0      t1      t2      t3
 [██████████] one fetch           [█···]  [█···]  [█···]  [█···]  many fetches ❌
 addresses: i, i+1, i+2, i+3      addresses: i, i+n, i+2n, i+3n
```

## Your task

Transpose an `N × N` row-major float matrix (`N = 4096`, a multiple of 32): `out[r][c] = in[c][r]`. One thread per output element, **global memory only — no `__shared__`** (that's exercise 05). Arrange your indexing so that **consecutive threads in a warp make coalesced reads of `in`**; the writes to `out` will be strided, and that's fine.

Edit `transpose.cu`:

1. The `__global__` kernel `transpose` — map this thread to `(row, col)` with `col` driven by `threadIdx.x`, read `in` coalesced, write the transposed slot in `out`.
2. The host function `solve` — launch a 2-D grid of 2-D blocks covering the whole matrix.

### The `solve` contract

```cpp
void solve(const float* in, float* out, int n);
```

`in`, `out` are **device pointers** to `n*n` floats, row-major; `n` is the matrix dimension. The harness handles allocation, correctness checking, and running the bad baseline.

The indexing that coalesces reads:

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;  // contiguous dim of in  → coalesced
int row = blockIdx.y * blockDim.y + threadIdx.y;
if (row < n && col < n) out[col*n + row] = in[row*n + col];
```

## Functions & syntax you'll need

| Thing | Signature / form | What it does |
|-------|------------------|--------------|
| `dim3` | `dim3 block(32, 8);` | A 3-D size struct (unused dims default to 1). Used for 2-D blocks/grids. |
| `threadIdx.x` / `.y` | built-in | Thread's index within its block in each dimension. `.x` is the fast-varying one across a warp. |
| `blockIdx.x` / `.y` | built-in | Block's index within the grid in each dimension. |
| `blockDim.x` / `.y` | built-in | Block size in each dimension. |
| 2-D index | `blockIdx.x*blockDim.x + threadIdx.x` | Same formula as 1-D, applied per dimension. |
| launch syntax | `transpose<<<grid, block>>>(in, out, n);` | `grid` and `block` are `dim3`s for a 2-D launch. |
| `ceil_div` | `int ceil_div(int a, int b)` | From `cuda_utils.cuh`; round the grid up to cover `n` in each dim. |

A typical 2-D launch:

```cpp
dim3 block(32, 8);   // block.x = 32 → each warp's threadIdx.x spans one 128-byte segment
dim3 grid(ceil_div(n, block.x), ceil_div(n, block.y));
transpose<<<grid, block>>>(in, out, n);
```

> **Why `block.x = 32`?** A warp is 32 threads taken in `threadIdx.x`-major order. Making `block.x` exactly 32 means each warp's `threadIdx.x` covers a full 32-float (128-byte) segment of a row — precisely one coalesced read transaction per warp. `dim3 block(16, 16)` also works (it just splits a warp across two half-rows).

The boundary guard `if (row < n && col < n)` isn't strictly needed here (`n` is a multiple of 32) but it's good habit for non-multiple sizes.

## How it's graded

The harness times your kernel *and* the doubly-strided naive baseline, both moving `2 * bytes` (read `in`, write `out`). Run `python grade.py`:

- **correctness** — `out == in^T` exactly (`max_abs_err == 0`).
- **speedup** — `naive_ms / your_ms >= 1.3`. Coalescing the read side must clearly beat the baseline that strides *both* sides. If you accidentally drive `row` with `threadIdx.x` (striding your reads), you become the baseline and this fails.
- **efficiency** — `bw_frac >= 0.35`. A one-sided-strided transpose *can't* reach peak (the strided write side caps you at a fraction), but it comfortably clears this floor; the truly naive version won't. Exercise 05's shared-memory version will push `bw_frac` much higher.
- **source** — must use `threadIdx.x` and `blockIdx` (2-D-style indexing), and must **NOT** contain `__shared__` (that's exercise 05).

Run `python grade.py --check-solution` to grade the reference solution instead of yours.

## Going deeper

- **Coalescing is the first thing to check.** Whenever a kernel underperforms, ask "do adjacent threads touch adjacent memory?" before anything else — it's the highest-leverage fix in GPU programming.
- **The strided write still hurts.** Your reads are perfect, but the strided writes to `out` keep you well under peak. There's no indexing trick to fix this in plain global memory — the data genuinely needs to be reordered. Exercise 05 stages a tile in fast on-chip shared memory so *both* the global read and the global write are coalesced.
- **Try this:** swap your `col`/`row` assignment so `row` gets `threadIdx.x` and re-grade. You'll watch `speedup` collapse below 1.0 — a direct measurement of what one indexing choice costs.
