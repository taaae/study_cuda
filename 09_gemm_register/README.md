# Exercise 09 — Register-Blocked GEMM (`float4` loads)
> One thread, one output element is a trap. Give each thread an 8×8 tile of answers and watch the FLOPs pour in.

## The idea
In exercise 08 you built a shared-memory tiled GEMM: each thread computed **one**
element of `C`, reading two values from shared memory per multiply-add. That kills
the global-memory bottleneck, but it just moves the wall — now you're limited by
**shared-memory bandwidth**, because every single FMA needs two fresh shared reads.

The fix is **register blocking** (a.k.a. thread coarsening): make each thread
responsible for a small `TM×TN` *micro-tile* of `C` — here `8×8 = 64` outputs —
held entirely in registers. The payoff is arithmetic, and it's beautiful: to
update all 64 accumulators for one step of the inner product, a thread loads `TM`
values from the `A` tile and `TN` from the `B` tile (`TM+TN = 16` shared reads),
then does `TM*TN = 64` FMAs. Shared-reads-per-FLOP drops from `2` down to
`16/64 = 0.25` — an **8× cut**. This is the single trick that takes you from a
toy GEMM to within striking distance of cuBLAS.

## Under the hood
Three levels of the memory hierarchy, each with rising arithmetic intensity:

- **Global → shared:** you stream `BM×BK` and `BK×BN` slabs in, using **`float4`
  vectorized loads**. One `float4` moves 16 bytes in a single instruction; a warp
  of 32 threads reading consecutive `float4`s covers 512 contiguous bytes —
  perfectly coalesced — *provided* the base address is 16-byte aligned.
  `cudaMalloc` returns 256-byte-aligned memory and the harness uses dimensions
  that are multiples of the tile, so every `float4` access here is aligned and
  in-bounds with **no boundary checks**.
- **Shared → registers:** the inner loop stages `a_reg[TM]` and `b_reg[TN]`.
- **Registers:** the `acc[TM][TN]` array. Registers are the fastest storage on the
  chip — zero-latency, no bank conflicts. On a T4 (sm_75) each SM has a 64K-register
  file; an 8×8 tile costs ~64 accumulator registers per thread, so occupancy stays
  reasonable. That register pressure is the real budget you're spending.

One more trick: you store the `A` slab **transposed** as `As[BK][BM]`, so the inner
loop reads it stride-1 along the `BM` dimension instead of stride-`BK`.

## A picture
```text
  Block output tile C: BM x BN = 128 x 128, owned by 16 x 16 = 256 threads.
  Thread (tx,ty) owns one TM x TN = 8 x 8 micro-tile, all in registers:

        b_reg[0..7]  (8 values from one row of Bs)
        ┌───────────────────────┐
   a    │ acc acc acc ... acc    │  acc[i][j] += a_reg[i]*b_reg[j]
   _    │ acc acc acc ... acc    │
   r 8  │  .                  .  │   16 shared reads (8 a + 8 b)
   e    │  .                  .  │   -> 64 FMAs.  intensity = 64/16 = 4x
   g    │ acc acc acc ... acc    │
        └───────────────────────┘
              8 columns
```

## Your task
Edit `gemm.cu` and fill in the `TODO`s:

1. The `__global__` kernel `gemm` — a block computes a `BM×BN` tile of `C`; each
   thread computes its `TM×TN` micro-tile, streaming `A`/`B` slabs through
   `__shared__` with `float4` loads (storing `A` transposed), and accumulating in
   a register array `acc[TM][TN]`.
2. The host `solve` — set up the launch (`dim3 block(16,16)`, grid sized by
   `BM`/`BN`) and call the kernel.

You do **not** write `main()` or manage memory — `harness.cu` does. Reference tile
constants (keep them): `BM=128, BN=128, BK=8, TM=8, TN=8`.

### The `solve` contract
```cpp
void solve(const float* A, const float* B, float* C, int M, int N, int K);
```
All pointers are **device pointers**; all matrices are **row-major**. `A` is `M×K`,
`B` is `K×N`, `C` is `M×N`. The harness guarantees `M,N,K` are multiples of the
tile sizes.

## Functions & syntax you'll need
| Construct | Signature / form | What it does |
|-----------|------------------|--------------|
| `float4` | `struct { float x,y,z,w; }` | 16-byte vector type; one load moves 4 floats |
| vectorized load | `reinterpret_cast<const float4*>(p)[i]` | reads `float4` at float-offset `4*i`; must be 16B aligned |
| vectorized store | `reinterpret_cast<float4*>(p)[i] = v` | writes 4 contiguous floats at once |
| `__shared__` | `__shared__ float As[BK][BM];` | per-block scratch; declare slabs here |
| `__syncthreads()` | `void __syncthreads()` | block-wide barrier; needed after filling/before reusing shared |
| register tile | `float acc[TM][TN] = {0.f};` | per-thread accumulators, live in registers |
| `#pragma unroll` | before a fixed-bound `for` | unrolls the inner loops so `acc` stays in registers |
| `__restrict__` | `const float* __restrict__ A` | promises no aliasing; lets the compiler reorder loads |
| `ceil_div` | `ceil_div(a,b)` (from `cuda_utils.cuh`) | `(a+b-1)/b`, for grid sizing |

```cpp
__shared__ float As[BK][BM];        // A slab, stored TRANSPOSED: As[k][row]
__shared__ float Bs[BK][BN];        // B slab: Bs[k][col]
float acc[TM][TN] = {0.f};          // micro-tile in registers
float a_reg[TM], b_reg[TN];         // operands staged per inner-k
float4 v = reinterpret_cast<const float4*>(A)[idx4];  // idx4 = byte_off / 16
```

Barrier rule (same as ex. 08): one `__syncthreads()` after filling the shared
slabs (before the inner product), one after the inner product (before overwriting
the slabs next iteration).

## How it's graded
`python grade.py` builds with `-lcublas` and checks:

- **correctness** — `C` matches a CPU reference (`double` accumulation, sampled
  rows) within relative tolerance `1e-3`.
- **speedup** — the harness runs an exercise-08-style tiled GEMM as baseline; you
  need `speedup >= 1.5`. A correct-but-naive (one-element-per-thread) kernel will
  *fail* this even though it passes correctness.
- **performance** — `frac_cublas = gflops / gflops_cublas >= 0.30`. Reaching 30%
  of vendor-tuned cuBLAS by hand is a genuine milestone.
- **source** — you must use `float4` and `__shared__`.

Run `python grade.py --check-solution` to grade the reference solution instead of
yours.

## Going deeper
- **Why not bigger tiles?** `TM=TN=16` (256 outputs/thread) gives even higher
  intensity but blows the register file, slashing occupancy until the SM can't
  hide latency. The sweet spot is hardware-specific; 8×8 is a solid T4 choice.
- **Double buffering:** real GEMMs prefetch the *next* slab into a second shared
  buffer while computing on the current one, hiding load latency behind math. The
  next step after this exercise.
- cuBLAS itself dispatches to hand-tuned SASS (and on newer GPUs, Tensor Cores via
  `cublasGemmEx`/WMMA). Your scalar-FMA kernel can't touch those, which is why 30%,
  not 100%, is the bar.
