# Hints — Exercise 08

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — Why naive is slow (no code)</summary>

The naive kernel reads `A[i][k]` and `B[k][j]` from global memory for every one of the `K` multiply-adds. That's ~1 FLOP per word loaded — far below the ratio the T4 needs to be compute-bound, so it stalls on memory. The fix is to load each value from global memory *once* and reuse it many times from fast on-chip shared memory.
</details>

<details>
<summary>Hint 2 — The tiling idea (concept)</summary>

A block of `TILE×TILE` threads computes a `TILE×TILE` tile of `C`. March across K in steps of `TILE`: each step, the block cooperatively copies one `TILE×TILE` tile of `A` and one of `B` into shared memory, then every thread does `TILE` multiply-adds out of shared memory. Each shared value is reused by `TILE` threads, so global memory is touched ~`TILE×` less.
</details>

<details>
<summary>Hint 3 — Indexing (concept)</summary>

`row = blockIdx.y*TILE + threadIdx.y`, `col = blockIdx.x*TILE + threadIdx.x` is the `C` element this thread owns. For K-tile `t`, the `A` element to load is `A[row][t*TILE + tx]` and the `B` element is `B[t*TILE + ty][col]`, with row-major indexing `A[row*K + (t*TILE+tx)]` and `B[(t*TILE+ty)*N + col]`. Write `0` into shared memory when the source index is out of bounds.
</details>

<details>
<summary>Hint 4 — The two barriers (concept)</summary>

Per K-tile you need `__syncthreads()` **twice**: once after loading the shared tiles (so everyone sees a complete tile before multiplying), and once after the inner `k` loop (so no thread overwrites the tile for the next step while another is still reading it). Both are required.
</details>

<details>
<summary>Hint 5 — The kernel body (code)</summary>

```cpp
__global__ void gemm(const float* A, const float* B, float* C,
                     int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.f;
    int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;
        As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.f;
        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}
```
</details>

<details>
<summary>Hint 6 — The full solve (code)</summary>

```cpp
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
    gemm<<<grid, block>>>(A, B, C, M, N, K);
}
```
</details>
