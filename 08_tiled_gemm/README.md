# Exercise 08 — Tiled GEMM (shared memory)

**New concepts:** shared-memory **tiling** for matrix multiply, the K-loop over tiles, careful `__syncthreads()` placement, and **arithmetic intensity** — the ratio of FLOPs to bytes that determines whether a kernel can be fast at all. You also meet cuBLAS as a performance yardstick.

## The task

Compute `C = A * B` for **row-major** matrices: `A` is `M×K`, `B` is `K×N`, `C` is `M×N`.

Edit `gemm.cu` and fill in the `TODO`s:

1. The `__global__` kernel `gemm` — each block computes one `TILE×TILE` tile of `C` by streaming `A` and `B` tiles through `__shared__` memory.
2. The host function `solve` — set up the 2-D launch configuration and launch.

You do **not** write `main()` or manage memory — `harness.cu` does, and calls your `solve(...)`.

### `solve` signature (the contract)

```cpp
void solve(const float* A, const float* B, float* C, int M, int N, int K);
```

All pointers are **device pointers**, all matrices **row-major**: `A[r*K + c]`, `B[r*N + c]`, `C[r*N + c]`.

## Why naive GEMM is slow: arithmetic intensity

The naive kernel (one thread per `C[i][j]`, looping `k` and reading `A[i][k]`, `B[k][j]` from global memory) does **2 FLOPs per 2 global loads** — arithmetic intensity ≈ 1 FLOP/word. The T4 can do ~8 TFLOPS but only ~320 GB/s (≈80 Gword/s), so at intensity 1 you're capped near 80 GFLOPS — ~1% of peak compute. GEMM is *compute-bound only if you stop re-reading the same data from global memory.*

## Tiling: reuse data on chip

Partition the K dimension into tiles of width `TILE`. Each block computes one `TILE×TILE` output tile by looping over K-tiles:

1. Cooperatively load one `TILE×TILE` tile of `A` and one of `B` into `__shared__` arrays (each thread loads one element of each).
2. `__syncthreads()` so the whole tile is visible.
3. Each thread accumulates `sum += As[ty][k] * Bs[k][tx]` over `k = 0..TILE-1`. Every value loaded from shared memory is reused `TILE` times by the threads of the block.
4. `__syncthreads()` **before** overwriting the shared tiles with the next K-tile.

This raises arithmetic intensity by ~`TILE×` (each global element loaded once per output tile is reused `TILE` times), pushing the kernel off the memory wall.

### `__syncthreads()` placement — two barriers per K-tile

- **After loading** the tile, **before** using it: so no thread multiplies stale/empty shared memory.
- **After** the inner `k` loop, **before** the next iteration overwrites the shared tiles: so no thread is still reading the old tile when another starts writing the new one.

Forgetting either is the classic tiled-GEMM bug and yields wrong, run-to-run-varying results.

### Handling non-multiple sizes

If `M`, `N`, or `K` aren't multiples of `TILE`, guard each shared-memory load: write `0` when the global row/col is out of range, so out-of-tile lanes contribute nothing. The output write is guarded by `row < M && col < N`.

## Syntax / reference

```cpp
__shared__ float As[TILE][TILE];
__shared__ float Bs[TILE][TILE];

int tx = threadIdx.x, ty = threadIdx.y;
int row = blockIdx.y * TILE + ty;   // C row
int col = blockIdx.x * TILE + tx;   // C col

dim3 block(TILE, TILE);
dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
gemm<<<grid, block>>>(A, B, C, M, N, K);
```

## Grading (`!python grade.py`)

- **correctness** — `C` matches a CPU reference (`double` accumulation) within a relative tolerance.
- **performance** — the harness times your kernel as `gflops = 2*M*N*K / time` and also runs **cuBLAS** (`cublasSgemm`) to report `gflops_cublas`. You must reach **`frac_cublas = gflops/gflops_cublas >= 0.12`**. A naive (non-tiled) kernel falls short.
- **source** — you must use `__shared__`.

Compiled with `-lcublas`. Run `python grade.py --check-solution` to grade the reference solution instead of yours.
