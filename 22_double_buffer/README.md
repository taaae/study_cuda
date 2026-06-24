# Exercise 22 — Double-Buffered Tiled GEMM
> Hide global-load latency by computing on one tile while the next one is already on its way.

## The idea
A standard tiled GEMM walks the K dimension one tile at a time, and every
iteration looks like this:
```text
for each k-tile:
    load A-tile, B-tile   global -> shared
    __syncthreads()        // wait so the tile is visible to everyone
    multiply-accumulate from shared
    __syncthreads()        // wait so nobody overwrites the tile too early
```
Look at that first `__syncthreads()`. The threads are **stalled** there, doing
nothing, while the global loads crawl back — and global memory has *hundreds* of
cycles of latency. The math units are idle waiting for data.

**Double-buffering** (a.k.a. software pipelining, or ping-pong) fixes this. Keep
**two** shared tiles. While you do the multiply-accumulate on buffer *cur*, you
simultaneously issue the loads for the **next** k-tile into buffer *nxt*. Those
loads are in flight *during* the math, so the latency hides behind useful work
instead of stalling on it. Each iteration you just swap which buffer is "current."

## Under the hood
There are two distinct wins here. First, the obvious one: **latency hiding**. The
load instruction for the next tile returns immediately (it only *issues* the
memory request); the data lands asynchronously into shared memory while the
warp's `acc +=` loop runs. By the time you reach the bottom barrier, the tile is
already there.

Second, a subtle one: **you drop one of the two barriers per iteration**. The
classic loop needs the *second* `__syncthreads()` to stop a fast thread from
overwriting the shared tile before a slow thread finishes reading it. But when the
next tile goes into a *different* buffer, there's nothing to protect — the current
buffer is read-only this iteration. One barrier per iteration instead of two.

> **Forward pointer — the Ampere connection:** On Ampere (sm_80+) and newer you'd
> express this exact overlap with `cp.async` and `cuda::pipeline`, where the
> hardware streams global→shared asynchronously without even occupying registers.
> The **T4 (sm_75) has no `cp.async`** — so we do the portable, by-hand cousin:
> register-prefetch into a second shared buffer. Same idea, older hardware. This
> is why the speedup here is real but *modest* (≈1.1–1.3×); on Ampere the same
> restructuring buys far more.

> **Fun fact:** real libraries (cuBLAS, CUTLASS) pipeline *several* stages deep,
> not just two, and combine it with register-level tiling. That's how they reach
> ~90% of peak. Two-stage is the conceptual first step.

## A picture
```text
Double-buffer ping-pong across the K-loop:

 buffer 0:  [load t0]            [load t2]            ...
 buffer 1:           [load t1]            [load t3]   ...
 compute :        [== t0 ==][== t1 ==][== t2 ==]...
                     ^ prefetch of t1 overlaps with compute on t0

 cur flips:    0 -> 1 -> 0 -> 1 ...   (nxt = cur ^ 1 each iteration)
```

## Your task
Compute `C = A * B`, row-major, all dimensions multiples of `TILE` (= 32), with a
tiled kernel that ping-pongs **two** shared tile buffers across the K-loop.

Edit `gemm.cu`:
1. `gemm_double_buffer` — prologue loads tile 0; the main loop prefetches `t+1`
   into the other buffer, computes on the current buffer, one `__syncthreads()`,
   then swaps.
2. `solve` — launch a 2-D grid of `TILE × TILE` blocks.

### The `solve` contract
```cpp
void solve(const float* A, const float* B, float* C, int M, int N, int K);
```
All device pointers, all **row-major**: `A` is `M×K`, `B` is `K×N`, `C` is `M×N`.
Indexing: `A[row*K + k]`, `B[k*N + col]`, `C[row*N + col]`. Dimensions are
multiples of `TILE`, so no ragged-edge handling needed.

## Functions & syntax you'll need
| Construct | Form | Purpose |
|---|---|---|
| double shared buffers | `__shared__ float As[2][TILE][TILE];` | the `[2]` is the ping-pong dimension |
| | `__shared__ float Bs[2][TILE][TILE];` | one A-buffer and one B-buffer per slot |
| buffer index | `int cur = 0; int nxt = cur ^ 1;` | `^1` toggles between 0 and 1 cheaply |
| `__syncthreads()` | `void __syncthreads()` | one per iteration, *after* the compute, before the swap |
| `#pragma unroll` | before the inner `for (k...)` | lets the compiler unroll the TILE-length MAC loop |
| thread/block coords | `threadIdx.{x,y}`, `blockIdx.{x,y}`, `blockDim` | map `(tx,ty)` to `(col,row)` |

Skeleton of the loop (no full solution):
```cpp
// prologue: load k-tile 0 into buffer 0; __syncthreads();
for (int t = 0; t < numTiles; ++t) {
    int nxt = cur ^ 1;
    if (t + 1 < numTiles)
        /* prefetch k-tile t+1 into As[nxt]/Bs[nxt] */;   // loads in flight
    for (int k = 0; k < TILE; ++k)
        acc += As[cur][ty][k] * Bs[cur][k][tx];           // compute meanwhile
    __syncthreads();                                       // next tile landed
    cur = nxt;                                             // swap roles
}
```
That single barrier — issued *after* the prefetch instruction but used to gate the
swap — is what overlaps the load with the math.

## How it's graded
`python grade.py` (M=N=K=1024) checks:
- **correctness** — `C == A*B` vs CPU, `max_rel_err <= 1e-3`.
- **efficiency** — the harness times your kernel *and* a built-in single-buffer
  tiled GEMM, and requires `speedup >= 1.10`. It also reports `gflops`. A correct
  but single-buffered kernel will pass correctness but **fail the speedup gate**.
- **source** — you must declare two shared buffers in `As[2][...][...]` style
  (the grader greps for `[2][`).

Run `python grade.py --check-solution` to grade the reference solution.

## Going deeper
Try profiling with **Nsight Systems** (`nsys`) to see the two kernels' durations
on a timeline — it usually works on Colab. (Nsight Compute's `ncu`, which would
show you the memory-stall reduction directly, is often blocked on free Colab.) The
natural next step is **register tiling**: have each thread compute a small `n×n`
patch of C, multiplying the per-thread arithmetic intensity — that, stacked on top
of double-buffering, is the real road to high GEMM throughput.
