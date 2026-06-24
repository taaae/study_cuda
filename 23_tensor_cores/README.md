# Exercise 23 — Tensor Cores (WMMA half-precision GEMM)
> Reach for the T4's specialized matrix-multiply hardware — a whole warp cooperatively multiplies 16×16 tiles.

## The idea
Up to now every multiply-add went through the ordinary FP32 ALUs. But the T4 has a
second kind of math unit: **Tensor Cores**, hardware that does a small matrix
multiply-accumulate `D = A*B + C` in one shot — far faster than looping the ALUs.
The natural unit on T4 is a `16×16×16` MMA: a `16×16` block of A times a `16×16`
block of B, accumulated into a `16×16` block of C.

The trade is precision for speed: **inputs are FP16, accumulation is FP32**. You
keep good accuracy (the running sum is full-width float) while halving the memory
bandwidth and feeding the cores their native format. This FP16-storage /
FP32-accumulate pattern is exactly how modern deep-learning GEMMs run.

The programming model is unusual. You **don't** index individual elements. A whole
**warp** (32 lanes) cooperatively owns a **fragment** — an opaque, register-
resident slice of a tile, spread across the warp's lanes in a layout the hardware
defines. You only declare fragments and call four warp-collective ops:
`load_matrix_sync`, `fill_fragment`, `mma_sync`, `store_matrix_sync`.

## Under the hood
"Warp-collective" is the key phrase. Every lane in the warp must execute each
`*_sync` call **together** — they cooperate to feed and drain the Tensor Core.
Never hide them behind `if (laneId == 0)`; that's a deadlock / garbage recipe. The
fragment is distributed register storage, so you can't peek at element `[i][j]` —
you just hand fragments to `mma_sync` and trust the hardware's layout.

A fragment's type encodes everything: its **role** (`matrix_a`, `matrix_b`,
`accumulator`), the `16,16,16` shape, the **element type** (`half` for inputs,
`float` for the accumulator), and the memory **layout** (`row_major` here, since
both A and B are stored row-major). `load_matrix_sync` needs the pointer to the
tile's top-left element plus the **leading dimension** — the row stride of the
*full* matrix, not the tile.

> **Fun fact:** the T4's tensor cores peak around **65 FP16 TFLOPS**, versus
> ~8.1 TFLOPS for its FP32 ALUs — roughly 8× on paper. This exercise asks for
> >12 TFLOPS, which a naive one-tile-per-warp kernel clears comfortably while
> staying far below peak (memory-bound, not compute-bound, at this size).

## A picture
```text
One WARP owns one 16x16 output tile of C. It marches along K in steps of 16:

   A (MxK)                 B (KxN)
   +----+----+----+        +----+
   |a_fr| .. | .. |  row   |b_fr|  k0=0
   +----+----+----+        +----+
                           |b_fr|  k0=16
   k0= 0   16   32         +----+
                           | .. |
   acc_frag (FP32, 16x16) += a_frag(k0) * b_frag(k0)   for each k0
                           +----+

   after the K-loop:  store_matrix_sync(acc_frag) -> C tile (FP32, row-major)
```

## Your task
Compute `C(FP32) = A(FP16) * B(FP16)`, row-major, with `M`, `N`, `K` all multiples
of 16. **One warp computes one 16×16 output tile.**

Edit `wmma_gemm.cu`:
1. `wmma_gemm` — declare the fragments, zero the accumulator, loop `k0` in steps
   of 16 loading/MMA-ing, then store the result tile.
2. `solve` — launch enough warps to cover all `(M/16)×(N/16)` tiles.

### The `solve` contract
```cpp
void solve(const half* A, const half* B, float* C, int M, int N, int K);
```
`A` is `M×K` FP16, `B` is `K×N` FP16, `C` is `M×N` FP32, all row-major device
pointers. Include `<cuda_fp16.h>` and `<mma.h>`; `using namespace nvcuda;`.

## Functions & syntax you'll need
Required headers / namespace:
```cpp
#include <cuda_fp16.h>     // the half type
#include <mma.h>           // the WMMA API
using namespace nvcuda;    // wmma:: lives in nvcuda
const int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
```
| WMMA call | Signature (essentials) | What it does |
|---|---|---|
| `wmma::fragment<...>` | `fragment<Use, M, N, K, T, Layout>` | declares a fragment; `Use` ∈ {`matrix_a`,`matrix_b`,`accumulator`}, `T`=`half` (inputs) or `float` (acc) |
| `wmma::fill_fragment` | `(frag&, T value)` | sets every element (use `0.0f` to zero the accumulator) |
| `wmma::load_matrix_sync` | `(frag&, const T* ptr, unsigned ldm)` | loads a 16×16 block; `ptr`=top-left, `ldm`=full-matrix row stride |
| `wmma::mma_sync` | `(acc&, a&, b&, acc&)` | the multiply-accumulate: `acc = a*b + acc` |
| `wmma::store_matrix_sync` | `(T* ptr, const frag&, unsigned ldm, layout_t)` | writes the result tile; use `wmma::mem_row_major` |

Leading-dimension / pointer math (the part that's easy to get wrong):
```cpp
// A row-major MxK: ldm = K, tile top-left = A + (warpRow*16)*K + k0
wmma::load_matrix_sync(a_frag, A + (warpRow*WMMA_M)*K + k0, K);
// B row-major KxN: ldm = N, tile top-left = B + k0*N + warpCol*16
wmma::load_matrix_sync(b_frag, B + k0*N + (warpCol*WMMA_N), N);
// store: C row-major MxN, ldm = N
wmma::store_matrix_sync(C + (warpRow*WMMA_M)*N + (warpCol*WMMA_N),
                        acc_frag, N, wmma::mem_row_major);
```
**Mapping warps to tiles.** A warp is 32 lanes. With a block like `(128, 4)`:
`warpCol = (blockIdx.x*blockDim.x + threadIdx.x) / warpSize` (=32) picks the tile
column; `warpRow = blockIdx.y*blockDim.y + threadIdx.y` picks the tile row. Size
the grid with `ceil_div` so all `(M/16)×(N/16)` tiles are covered.

## How it's graded
`python grade.py` (M=N=K=1024) checks:
- **correctness** — vs an FP32 CPU reference computed from the *FP16-rounded*
  inputs (a fair comparison), with a generous **relative** tolerance of `1e-2`
  (FP16 inputs genuinely lose precision).
- **efficiency** — reports `gflops = 2*M*N*K / time` and requires `gflops >= 12000`
  (>12 TFLOPS — above the ~8.1 TFLOPS FP32 peak, so you *cannot* hit it on the
  ALUs; only the tensor cores get you there). Also reports `speedup_vs_fp32`.
- **source** — must contain `wmma::fragment` and `mma_sync`.

The harness skips (and reports correct) on pre-sm_70 GPUs; the T4 is sm_75, so it
runs. Run `python grade.py --check-solution` to grade the reference solution.

## Going deeper
The optional stretch: have each warp compute **several** 16×16 tiles with multiple
accumulator fragments, reusing each loaded A/B fragment across them — this raises
arithmetic intensity and is the standard next step toward a fast WMMA GEMM. Beyond
WMMA, the lower-level `mma.sync` PTX and CUTLASS expose more shapes and deeper
pipelines. To inspect tensor-core utilization you'd normally use Nsight Compute
(`ncu`) — note its hardware counters are often blocked on free Colab, though
Nsight Systems (`nsys`) timelines usually work.
