# Exercise 22 — Double-Buffered Tiled GEMM

**New concepts:** software pipelining / **double-buffering** (ping-pong) of
shared-memory tiles to hide global-load latency in a tiled matrix multiply.

## The stall you're removing

A standard tiled GEMM walks the K dimension one tile at a time:

```
for each k-tile:
    load A-tile, B-tile  from global -> shared
    __syncthreads()              // wait so everyone sees the tile
    multiply-accumulate from shared
    __syncthreads()              // wait so nobody overwrites the tile early
```

The threads sit **idle** during that first `__syncthreads()` while the global
loads land — and global memory has hundreds of cycles of latency. The compute
units have nothing to do until the load returns.

**Double-buffering** overlaps that load with useful work. You keep **two** shared
tiles. While you compute on buffer *cur*, you simultaneously **prefetch the next
k-tile into buffer *nxt***. The loads of the next tile are in flight *during* the
math on the current tile, so the latency is hidden behind compute instead of
stalling on it. Each iteration you swap the roles of the two buffers (ping-pong).

This also removes one of the two `__syncthreads()` per iteration: because the
next tile goes into a *different* buffer, you don't need a barrier to protect the
current buffer from being overwritten.

> **On Ampere and newer** you'd express the same overlap with `cp.async` and the
> `cuda::pipeline` primitives, which do the async copy in hardware. The T4
> (sm_75) has no `cp.async`, so we do the classic register-prefetch +
> two-shared-buffer version — the same idea, by hand.

## The task

Compute `C = A * B` (row-major, all dimensions multiples of the tile size) with a
tiled kernel that uses **two shared-memory tile buffers in ping-pong fashion**
across the K-loop.

Edit `gemm.cu`:

1. `gemm_double_buffer` — the double-buffered tiled kernel.
2. `solve` — launch it with a 2-D grid of `TILE × TILE` blocks.

### `solve` signature (the contract)

```cpp
void solve(const float* A, const float* B, float* C, int M, int N, int K);
```

All pointers are **device pointers**. `A` is `M×K`, `B` is `K×N`, `C` is `M×N`,
all **row-major**. The harness uses dimensions that are multiples of `TILE`
(= 32), so you do not need ragged-edge handling.

Row-major indexing: `A[row*K + k]`, `B[k*N + col]`, `C[row*N + col]`.

## Syntax / reference

Declare the two buffers as the leading dimension `[2]`:

```cpp
#define TILE 32
__shared__ float As[2][TILE][TILE];
__shared__ float Bs[2][TILE][TILE];
int cur = 0;                       // which buffer we compute from
```

Sketch of the ping-pong loop:

```cpp
// prologue: load the first k-tile into buffer 0
load_tile(As[0], Bs[0], /*k-tile=*/0);
__syncthreads();

for (int t = 0; t < numTiles; ++t) {
    int nxt = cur ^ 1;
    if (t + 1 < numTiles)
        load_tile(As[nxt], Bs[nxt], /*k-tile=*/t + 1);   // prefetch next
    // compute on the CURRENT buffer while the prefetch is in flight
    for (int k = 0; k < TILE; ++k)
        acc += As[cur][ty][k] * Bs[cur][k][tx];
    __syncthreads();              // next tile is ready; swap
    cur = nxt;
}
```

The single `__syncthreads()` per iteration is what hides the latency: the
`load_tile` issued at the top runs concurrently with the `acc +=` math.

## Grading (`!python grade.py`)

- **correctness** — `C == A*B` within tolerance vs a CPU reference.
- **efficiency** — the harness also runs a plain **single-buffer** tiled GEMM as
  a baseline and requires `speedup >= 1.10`. (Double-buffering gains are real but
  modest on T4.) Also reports `gflops`.
- **source** — you must declare two shared tile buffers in the
  `__shared__ float As[2][...][...]` style.

Run `python grade.py --check-solution` to grade the reference solution instead.
