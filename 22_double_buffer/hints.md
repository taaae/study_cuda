# Hints — Exercise 22

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — What "double buffer" buys you (concept)</summary>

In single-buffer tiling, threads load a tile, `__syncthreads()`, then compute,
then `__syncthreads()` again. During that first barrier they stall waiting for
the global loads. If you instead prefetch the **next** tile into a **second**
buffer while computing on the current one, the load latency hides behind the
math. You also drop one of the two barriers per iteration.
</details>

<details>
<summary>Hint 2 — Declaring the two buffers (code)</summary>

```cpp
#define TILE 32
__shared__ float As[2][TILE][TILE];
__shared__ float Bs[2][TILE][TILE];
int cur = 0;   // buffer we compute from; nxt = cur ^ 1 is where we prefetch
```
</details>

<details>
<summary>Hint 3 — Loading a k-tile (the indexing)</summary>

For k-tile index `t`, thread (ty,tx) loads:

```cpp
As[buf][ty][tx] = A[row * K + (t * TILE + tx)];   // row = blockIdx.y*TILE + ty
Bs[buf][ty][tx] = B[(t * TILE + ty) * N + col];   // col = blockIdx.x*TILE + tx
```
</details>

<details>
<summary>Hint 4 — The prologue + ping-pong loop (concept)</summary>

Load tile 0 into buffer 0 before the loop and `__syncthreads()`. Then on each
iteration: compute `nxt = cur ^ 1`, prefetch tile `t+1` into `nxt` (guard
`t+1 < numTiles`), do the multiply-accumulate from `cur`, `__syncthreads()`, then
`cur = nxt`.
</details>

<details>
<summary>Hint 5 — The full kernel loop (code)</summary>

```cpp
// prologue
As[0][ty][tx] = A[row*K + tx];
Bs[0][ty][tx] = B[ty*N + col];
__syncthreads();

int cur = 0, numTiles = K / TILE;
for (int t = 0; t < numTiles; ++t) {
    int nxt = cur ^ 1;
    if (t + 1 < numTiles) {
        As[nxt][ty][tx] = A[row*K + ((t+1)*TILE + tx)];
        Bs[nxt][ty][tx] = B[((t+1)*TILE + ty)*N + col];
    }
    #pragma unroll
    for (int k = 0; k < TILE; ++k) acc += As[cur][ty][k] * Bs[cur][k][tx];
    __syncthreads();
    cur = nxt;
}
C[row*N + col] = acc;
```
</details>

<details>
<summary>Hint 6 — The launch (code)</summary>

```cpp
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, M / TILE);
    gemm_double_buffer<<<grid, block>>>(A, B, C, M, N, K);
}
```
</details>
