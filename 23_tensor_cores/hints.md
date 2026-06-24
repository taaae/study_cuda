# Hints — Exercise 23

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — A warp, not a thread, owns a tile (concept)</summary>

Unlike every previous GEMM, here the unit of work is a **warp** (32 lanes), and
the unit of data is a **fragment** spread across those lanes. Every lane calls
`load_matrix_sync` / `mma_sync` / `store_matrix_sync` together — never guard them
with `if (lane == 0)`. Your only job per warp is to pick which 16×16 output tile
it computes.
</details>

<details>
<summary>Hint 2 — Declaring fragments (code)</summary>

```cpp
using namespace nvcuda;
wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator, 16,16,16, float> acc_frag;
wmma::fill_fragment(acc_frag, 0.0f);
```

Both A and B are stored row-major, so both input fragments use `row_major`.
</details>

<details>
<summary>Hint 3 — The leading dimension is the FULL matrix stride (concept)</summary>

`load_matrix_sync(frag, ptr, ldm)` reads a 16×16 block starting at `ptr`, where
consecutive rows are `ldm` elements apart. For a row-major `M×K` A that's `ldm =
K`; for a row-major `K×N` B that's `ldm = N`; for storing into `M×N` C that's
`ldm = N`. The pointer is the **top-left element of the tile**.
</details>

<details>
<summary>Hint 4 — Pointers into A, B, C for tile (warpRow, warpCol) (code)</summary>

```cpp
// A tile at rows [warpRow*16 .. +16), cols [k0 .. +16):
A + (warpRow*16)*K + k0          // ldm = K
// B tile at rows [k0 .. +16), cols [warpCol*16 .. +16):
B + k0*N + (warpCol*16)          // ldm = N
// C tile:
C + (warpRow*16)*N + (warpCol*16)  // ldm = N
```
</details>

<details>
<summary>Hint 5 — The K-loop and store (code)</summary>

```cpp
for (int k0 = 0; k0 < K; k0 += 16) {
    wmma::load_matrix_sync(a_frag, A + (warpRow*16)*K + k0, K);
    wmma::load_matrix_sync(b_frag, B + k0*N + (warpCol*16), N);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);   // acc = a*b + acc
}
wmma::store_matrix_sync(C + (warpRow*16)*N + (warpCol*16),
                        acc_frag, N, wmma::mem_row_major);
```
</details>

<details>
<summary>Hint 6 — Mapping warps to tiles in solve (code)</summary>

```cpp
dim3 block(128, 4);            // 128/32 = 4 warps along x, 4 tile-rows along y
int warpsX = block.x / 32;     // = 4
dim3 grid(ceil_div(N/16, warpsX), ceil_div(M/16, (int)block.y));
wmma_gemm<<<grid, block>>>(A, B, C, M, N, K);
```

In the kernel: `warpCol = (blockIdx.x*blockDim.x + threadIdx.x)/warpSize;`
`warpRow = blockIdx.y*blockDim.y + threadIdx.y;`
</details>
