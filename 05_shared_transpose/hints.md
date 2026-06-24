# Hints — Exercise 05

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — Why shared memory breaks the tradeoff (no code)</summary>

In exercise 04 you could coalesce reads *or* writes because global memory forces one side to be strided. Shared memory is on-chip and lets any thread read any element cheaply. So: bring a tile into shared memory with a **coalesced** global read, then write it back with a **coalesced** global write — and do the actual transpose by *reordering inside the tile*, where strided access is fast (almost — see the bank-conflict hint).
</details>

<details>
<summary>Hint 2 — The three phases (concept)</summary>

1. **Load**: each thread reads one global element (coalesced) into `tile[threadIdx.y][threadIdx.x]`.
2. **Barrier**: `__syncthreads()` so the entire tile is filled before anyone reads it.
3. **Store**: each thread writes one global element (coalesced) of the *transposed* block, pulling its value from the tile with the indices **swapped**: `tile[threadIdx.x][threadIdx.y]`.

The block you write to is the transpose of the block you read from, so the block indices swap between load and store.
</details>

<details>
<summary>Hint 3 — The index math (concept)</summary>

Load coordinates (normal block):
- `x_in = blockIdx.x * TILE + threadIdx.x`  (contiguous → coalesced read)
- `y_in = blockIdx.y * TILE + threadIdx.y`

Store coordinates (transposed block):
- `x_out = blockIdx.y * TILE + threadIdx.x`  (contiguous → coalesced write)
- `y_out = blockIdx.x * TILE + threadIdx.y`

Note `threadIdx.x` stays the contiguous one in *both* the read and the write — that's what keeps both coalesced.
</details>

<details>
<summary>Hint 4 — The bank-conflict fix (concept + the line)</summary>

The store reads `tile[threadIdx.x][threadIdx.y]` — a *column* of the tile. With `tile[32][32]`, a column's 32 elements are 32 words apart, so they all hit the same memory bank and serialize (32× slow). Pad the row:

```cpp
__shared__ float tile[TILE][TILE + 1];
```

Now a column steps by 33 words, which spreads the 32 elements across all 32 banks — conflict-free. Type the `+ 1` yourself; the grader requires it.
</details>

<details>
<summary>Hint 5 — The kernel body (code)</summary>

```cpp
__global__ void transpose(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE + 1];

    int x_in = blockIdx.x * TILE + threadIdx.x;
    int y_in = blockIdx.y * TILE + threadIdx.y;
    if (x_in < n && y_in < n)
        tile[threadIdx.y][threadIdx.x] = in[y_in * n + x_in];

    __syncthreads();

    int x_out = blockIdx.y * TILE + threadIdx.x;
    int y_out = blockIdx.x * TILE + threadIdx.y;
    if (x_out < n && y_out < n)
        out[y_out * n + x_out] = tile[threadIdx.x][threadIdx.y];
}
```
</details>

<details>
<summary>Hint 6 — The full solve (code)</summary>

```cpp
void solve(const float* in, float* out, int n) {
    dim3 block(TILE, TILE);
    dim3 grid(ceil_div(n, TILE), ceil_div(n, TILE));
    transpose<<<grid, block>>>(in, out, n);
}
```
</details>
