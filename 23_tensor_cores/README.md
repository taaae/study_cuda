# Exercise 23 — Tensor Cores (WMMA half-precision GEMM)

**New concepts:** the **WMMA API** (`nvcuda::wmma`) for programming the T4's
**Tensor Cores** — fragment-based, warp-level, mixed-precision matrix multiply
(FP16 inputs, FP32 accumulate) at the fixed `16×16×16` shape.

## What a Tensor Core does

A Tensor Core performs a small matrix multiply-accumulate `D = A*B + C` in
hardware, far faster than the FP32 ALUs. On T4 the natural unit is a `16×16×16`
MMA: a `16×16` chunk of A times a `16×16` chunk of B, accumulated into a `16×16`
chunk of C. Inputs are **FP16**, accumulation is in **FP32** (so you keep
precision while halving load bandwidth).

You don't address individual elements. Instead a **whole warp** (32 threads)
cooperatively owns a **fragment** — an opaque, register-resident piece of a tile,
distributed across the warp's lanes in a layout the hardware defines. You only:

- declare fragments,
- `load_matrix_sync` to fill input fragments from memory,
- `mma_sync` to multiply-accumulate,
- `store_matrix_sync` to write the result fragment back to memory.

All four are **warp-collective**: every lane in the warp must call them together.

## The task

Compute `C(FP32) = A(FP16) * B(FP16)`, row-major, with `M`, `N`, `K` all
multiples of 16. **One warp computes one 16×16 output tile of C.**

For its output tile, a warp loops over the K dimension in steps of 16:

```
acc = 0
for k0 in 0, 16, 32, ... K-16:
    load a_frag from A's [16 x 16] block at (tileRow, k0)
    load b_frag from B's [16 x 16] block at (k0, tileCol)
    acc = a_frag * b_frag + acc          // mma_sync
store acc to C's [16 x 16] block at (tileRow, tileCol)
```

Edit `wmma_gemm.cu`:

1. `wmma_gemm` — the warp-per-tile kernel using fragments.
2. `solve` — launch enough warps to cover all `(M/16)×(N/16)` tiles.

### `solve` signature (the contract)

```cpp
void solve(const half* A, const half* B, float* C, int M, int N, int K);
```

`A` is `M×K` FP16, `B` is `K×N` FP16, `C` is `M×N` FP32, all **row-major**,
all device pointers. Include `<cuda_fp16.h>` and `<mma.h>`.

## Syntax / reference

```cpp
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

const int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

// Fragment types. matrix_a uses row_major, matrix_b uses row_major here,
// because both A and B are stored row-major. The accumulator is FP32.
wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

wmma::fill_fragment(acc_frag, 0.0f);

// Load a 16x16 block. The pointer is the top-left element; the last argument is
// the LEADING DIMENSION (row stride) of the FULL matrix:
//   A row-major MxK  -> ldm = K, top-left = A + tileRow*16*K + k0
//   B row-major KxN  -> ldm = N, top-left = B + k0*N + tileCol*16
wmma::load_matrix_sync(a_frag, A + (tileRow*WMMA_M)*K + k0,      K);
wmma::load_matrix_sync(b_frag, B + k0*N + (tileCol*WMMA_N),      N);

wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);   // acc = a*b + acc

// Store the 16x16 result. ldm = N (C is MxN row-major); use mem_row_major.
wmma::store_matrix_sync(C + (tileRow*WMMA_M)*N + (tileCol*WMMA_N), acc_frag,
                        N, wmma::mem_row_major);
```

**Mapping warps to tiles.** A warp is 32 lanes. If you launch blocks of, say,
`(128, 4)` threads, then `warpId = (blockIdx.x*blockDim.x + threadIdx.x) / 32`
along x picks the tile **column**, and `blockIdx.y*blockDim.y + threadIdx.y`
picks the tile **row**. Every lane in a warp must execute the fragment calls — so
do **not** mask them behind `if (laneId == 0)`.

**Alignment.** WMMA loads want the base pointer aligned (the harness allocates
with `cudaMalloc`, which is suitably aligned, and uses 16-multiple dimensions so
every tile's base stays aligned).

> **Stretch (optional):** have each warp compute several 16×16 tiles (multiple
> accumulator fragments) to amortize the A/B loads — the standard next step
> toward a fast WMMA GEMM. Not required to pass.

## Grading (`!python grade.py`)

- **correctness** — vs an FP32 CPU reference, **relative** tolerance ~1e-2
  (FP16 inputs lose precision, so the bar is generous).
- **efficiency** — reports `gflops = 2*M*N*K / time`. Threshold `gflops >= 12000`
  (>12 TFLOPS — comfortably above the ~8.1 TFLOPS FP32 peak, well under the ~65
  TFLOPS tensor peak). Also reports `speedup_vs_fp32` against an FP32 tiled GEMM.
- **source** — you must use `wmma::fragment` and `mma_sync`.

Run `python grade.py --check-solution` to grade the reference solution instead.
