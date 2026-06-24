# Exercise 08 — Tiled GEMM
> The kernel the whole GPU was built for — and the one place "reuse data on chip" pays off the most.

## The idea

Matrix multiply, `C = A · B`, is the workhorse of graphics, scientific computing, and every neural network you've heard of. The math is trivial: each `C[i][j]` is a dot product of row `i` of `A` and column `j` of `B`. The naive kernel writes exactly that — one thread per output element, looping over `k`.

And it's slow. Not because the GPU can't multiply fast — it can do ~8 TFLOPS — but because the naive kernel **re-reads the same data from global memory over and over**. Every element of `A` and `B` gets fetched from slow global memory hundreds of times.

The cure is **tiling**: load a small square of `A` and a small square of `B` into shared memory *once*, and let the whole block reuse them many times before moving on. You trade a flood of redundant global loads for a handful of shared-memory reads.

## Under the hood: arithmetic intensity

The concept that explains *why* naive GEMM is slow is **arithmetic intensity** — FLOPs performed per byte (or word) moved from global memory.

The naive inner step does **2 FLOPs** (one multiply, one add) for **2 global loads** (`A[i][k]`, `B[k][j]`) → intensity ≈ **1 FLOP/word**. Now the roofline math, for a T4:

```text
   compute peak:  ~8000 GFLOPS
   memory peak:   ~320 GB/s  ≈ 80 Gword/s

   at intensity 1 FLOP/word:  80 Gword/s × 1 = 80 GFLOPS  ← ~1% of compute!
```

You are pinned to the memory wall, idling 99% of the multipliers. To get off the wall you must raise intensity — do more math per byte fetched. Tiling does exactly that: load each global element once per output tile, then reuse it `TILE` times. Intensity jumps by roughly `TILE×`, and now the multipliers have something to chew on.

## Tiling: the K-loop

Each block computes one `TILE × TILE` tile of `C`. It marches across the `K` dimension one tile-width at a time:

```text
          K                          N                         N
      ┌───────┐                  ┌───┬───┬───┐            ┌───┬───┬───┐
   M  │A      │   ×          K   │B          │     =   M  │   │ C │   │
      │  ┌─┐  │                  │  ┌─┐      │            │   │tile│  │
      │  │A│  │ tile             │  │B│ tile │            │   └───┘   │
      └──┴─┴──┘                  └──┴─┴──────┘            └───────────┘
         │                          │
         └──── for each K-tile: load As, Bs into shared ────┘
                __syncthreads()
                for k in 0..TILE-1:  sum += As[ty][k] * Bs[k][tx]
                __syncthreads()       (each shared value reused TILE times)
```

1. Cooperatively load one `TILE×TILE` tile of `A` and one of `B` into `__shared__` arrays (each thread loads one element of each).
2. `__syncthreads()` — the whole tile must be visible before anyone multiplies.
3. Each thread accumulates `sum += As[ty][k] * Bs[k][tx]` over `k = 0..TILE-1`. Every shared value is reused by `TILE` threads.
4. `__syncthreads()` **again**, before the next K-tile overwrites the shared tiles.

### Two barriers per K-tile — both essential

- **After loading, before multiplying:** so no thread reads a tile slot a neighbor hasn't filled yet.
- **After the inner `k` loop, before the next load:** so no thread is still reading the old tile while another overwrites it.

Drop either and you get **wrong, run-to-run-varying** results — the classic tiled-GEMM bug. (Correctness that flickers between runs is almost always a missing `__syncthreads()`.)

### Ragged edges

When `M`, `N`, or `K` aren't multiples of `TILE`, guard each shared load: write `0` when the global row/col is out of range, so out-of-bounds lanes contribute nothing. Guard the final store with `row < M && col < N`. (Here the harness uses `1024×1024×1024`, a clean multiple of `TILE=16`, but write the guards anyway — good habit, and the grader's correctness check is unforgiving.)

## Your task

Compute `C = A · B` for **row-major** matrices: `A` is `M×K`, `B` is `K×N`, `C` is `M×N`. Beat 12% of cuBLAS throughput.

### The `solve` contract

```cpp
void solve(const float* A, const float* B, float* C, int M, int N, int K);
```

All pointers are **device pointers**, all matrices **row-major**: index as `A[r*K + c]`, `B[r*N + c]`, `C[r*N + c]`. You fill in `gemm.cu`: the `__global__ gemm` kernel and the host `solve`. `harness.cu` owns `main()`, the CPU correctness check, the timing, and the cuBLAS comparison.

## Functions & syntax you'll need

| Thing | Form | What it does |
|---|---|---|
| `threadIdx.x/.y` | built-in | thread's position in the 2-D block (`tx`, `ty`) |
| `blockIdx.x/.y` | built-in | block's position; `row = blockIdx.y*TILE + ty`, `col = blockIdx.x*TILE + tx` |
| `__shared__` | `__shared__ float As[TILE][TILE];` | per-block on-chip tile (one each for A and B) |
| `__syncthreads()` | barrier | the two-per-K-tile barriers above |
| `dim3` | `dim3 block(TILE, TILE);` | 2-D block geometry |
| grid | `dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));` | one block per output tile — note grid.x maps to N, grid.y to M |
| launch | `gemm<<<grid, block>>>(A, B, C, M, N, K);` | start the kernel |
| `#pragma unroll` | above the inner `k` loop | unrolls the fixed-length multiply loop (small free win) |
| `ceil_div(a, b)` | from `cuda_utils.cuh` | grid sizing |

> **Fun fact:** real cuBLAS doesn't stop at one level of tiling. It tiles into **shared memory** *and* into **registers** (each thread computes a small micro-tile of C, e.g. 4×4 or 8×8, not a single element), uses double-buffering to overlap loads with compute, and on this GPU class can dispatch the work to Tensor Cores. That register tiling is the single biggest step beyond what you'll write here — and why 12%, not 100%, is the bar.

## How it's graded

`python grade.py` builds (with `-lcublas`), runs, and checks:

- **correctness** — `C` matches a `double`-accumulation CPU reference within `max_rel_err < 1e-3`. The harness checks a strided subset of rows × all columns — enough to catch any indexing or sync bug.
- **performance** — it times you as `gflops = 2*M*N*K / time` and runs `cublasSgemm` for `gflops_cublas`. You must reach **`frac_cublas = gflops / gflops_cublas >= 0.12`**. A naive (non-tiled) kernel — stuck at intensity ~1 — falls well short.
- **source** — must use `__shared__`.

Run `python grade.py --check-solution` to grade the reference solution.

## Going deeper

1. **Register tiling** is your next 3–5×: have each thread compute a `4×4` block of C, keeping 16 accumulators in registers and reusing each loaded shared value across all of them. This is the standard "thread-coarsening" move and the gap between your kernel and cuBLAS.
2. cuBLAS is column-major; the harness cleverly maps your row-major problem onto it via the identity `C = (B^T · A^T)^T` — read the comment in `harness.cu` for a neat lesson in layout algebra.
3. GEMM is so central that NVIDIA ships **CUTLASS**, an open-source library of templated GEMM building blocks, so you can study production-grade tiling without reverse-engineering cuBLAS.
