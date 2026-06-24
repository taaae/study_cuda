# Exercise 11 — Histogram (atomics & privatization)
> 16 million threads, 256 counters, everyone fighting over the same handful of addresses. Here's how you stop the brawl.

## The idea
A histogram is dead simple to state: for a 256-bin histogram of bytes, bin `b`
counts how many input elements equal `b`. The trouble is *concurrency*. Many
elements map to the same bin, so threads must increment the **same counter at the
same time** — and a plain `hist[b]++` from two threads at once corrupts the count.
You need an **atomic** read-modify-write: `atomicAdd(&hist[b], 1)`.

But correctness isn't the whole story. With 16M elements and only 256 bins,
*thousands* of threads hammer the same global counter simultaneously. Atomics to a
contended address **serialize** — the hardware processes them one at a time — so a
naive kernel is bottlenecked on a few hot addresses, nowhere near memory bandwidth.
That contention, and how to defeat it, is the real lesson.

## Under the hood
The cure is **privatization**: give each block its *own* 256-bin histogram in
`__shared__` memory.

1. **Zero** the shared histogram cooperatively (256 bins, `blockDim.x` threads → a
   short strided loop).
2. `__syncthreads()`.
3. Each thread walks its elements (grid-stride loop, so any `n` works) and does
   `atomicAdd(&sHist[v], 1u)` — an atomic on **shared** memory.
4. `__syncthreads()`.
5. **Merge:** each thread `atomicAdd`s some shared bins into the matching global
   bins — just **256 global atomics per block**, not one per element.

Two wins stack here. First, shared-memory atomics are far cheaper than global ones:
on Maxwell and later (the T4 is sm_75) `atomicAdd` on shared memory is a native
hardware instruction, not an emulated lock. Second, contention is now scoped to a
*single block's* warps instead of the whole grid — and the expensive global
atomics collapse from ~16M down to `256 × numBlocks`.

> Fun fact: if the input were *uniformly* random across 256 bins, contention is
> mild. The privatized pattern really shines on **skewed** data, where one bin is
> red-hot — exactly the case that flattens a naive global-atomic kernel.

## A picture
```text
  NAIVE: every thread -> global hist (one shared set of 256 counters)
     t0 t1 t2 t3 ... t_millions
       \  \  |  /  ... /
        ▼  ▼ ▼ ▼      ▼
        [ global hist[256] ]   <- thousands serialize on hot bins

  PRIVATIZED: each block owns a shared copy, merged once at the end
   block0          block1          block2
   sHist[256]      sHist[256]      sHist[256]   <- cheap shared atomics,
      │               │               │            contention only within block
      └────────┬──────┴───────┬───────┘
               ▼ atomicAdd     ▼   (256 per block)
            [   global hist[256]   ]
```

## Your task
Edit `histogram.cu` and fill in the `TODO`s:

1. **`hist_privatized`** — declare `__shared__ unsigned int sHist[NBINS]`, zero it
   cooperatively, barrier, grid-stride over `data` bumping shared bins with
   `atomicAdd`, barrier, then merge shared bins into `hist` with `atomicAdd`.
2. **`solve`** — launch the kernel (`block = 256`, `grid = 1024` are provided; the
   grid-stride loop covers any `n`).

### The `solve` contract
```cpp
void solve(const unsigned char* data, unsigned int* hist, int n);
```
`data` is a **device pointer** to `n` bytes. `hist` is a **device pointer** to 256
`unsigned int`s, **already zeroed by the harness** — don't zero it yourself. Add up
the counts; launch whatever kernels you need.

## Functions & syntax you'll need
| Construct | Signature / form | What it does |
|-----------|------------------|--------------|
| `__shared__` | `__shared__ unsigned int sHist[256];` | per-block private histogram |
| `atomicAdd` (shared) | `atomicAdd(&sHist[b], 1u)` | conflict-safe increment of a shared bin |
| `atomicAdd` (global) | `atomicAdd(&hist[b], sHist[b])` | merge a private bin into the global one |
| `__syncthreads()` | `void __syncthreads()` | barrier after zeroing and after counting |
| grid-stride loop | see below | each thread covers many elements, any `n` |

```cpp
__shared__ unsigned int sHist[256];
for (int b = threadIdx.x; b < 256; b += blockDim.x) sHist[b] = 0;   // cooperative zero
__syncthreads();
for (int i = blockIdx.x*blockDim.x + threadIdx.x;                   // grid-stride
     i < n; i += blockDim.x*gridDim.x) {
    atomicAdd(&sHist[data[i]], 1u);
}
__syncthreads();
for (int b = threadIdx.x; b < 256; b += blockDim.x)                 // merge
    atomicAdd(&hist[b], sHist[b]);
```
Note `atomicAdd` on `unsigned int` takes a `1u` literal — match the type.

## How it's graded
`python grade.py` checks:

- **correctness** — your 256 bins match a CPU histogram **exactly**.
- **efficiency** — the harness runs a **naive global-atomic** baseline and reports
  `speedup = ms_naive / ms`; you need `speedup >= 2.0`. A correct kernel that
  atomics straight to global memory passes correctness but *fails* this. (`ms` is
  also reported.)
- **source** — you must use `__shared__` and `atomicAdd`.

Run `python grade.py --check-solution` to grade the reference solution instead of
yours.

## Going deeper
- **Sub-privatization / warp-level histograms:** if even per-block contention
  hurts, give each *warp* its own copy, or replicate bins to spread hot addresses.
- **Coarsening:** having each thread process several elements before merging
  amortizes the merge cost — the grid-stride loop already does a version of this.
- The merge step is itself a reduction across blocks; for very many bins you might
  reduce private histograms with a separate kernel instead of global atomics. For
  256 bins, the one-atomic-per-bin merge is plenty.
