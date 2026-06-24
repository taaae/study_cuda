# Exercise 05 — Shared-Memory Transpose
> The first kernel where you, not the hardware, decide what lives in fast memory.

## The idea

Back in exercise 04 you hit a wall: a matrix transpose in global memory can coalesce the **reads** or the **writes**, but never both. One side always walks down a column with a stride of `n` floats — and a strided access throws away most of the memory bus.

Shared memory dissolves that tradeoff. The trick is to stop thinking of the transpose as "move element from here to there" and instead stage a whole `TILE × TILE` square on chip:

1. **Read** a tile of `in` the natural way — consecutive threads touch consecutive addresses (coalesced).
2. Park it in a fast on-chip buffer all threads in the block can see.
3. **Write** that buffer out to the transposed location, but read it back from the buffer *with the indices swapped* — so the global write is *also* consecutive.

The transpose still happens — but it happens inside the chip, in the cheap memory, where strided access barely costs anything.

## Under the hood

Shared memory is a slab of SRAM that physically lives on each Streaming Multiprocessor (SM). On a T4 there are 40 SMs and up to 64 KB of shared memory per SM. It is roughly **100× lower latency** than global memory (a few cycles vs. hundreds) and — crucially — it is *programmer-managed*. The hardware won't stage data here for you; you declare the buffer with `__shared__` and choreograph the loads yourself. Think of it as a manually-controlled cache for one block's working set.

Because every thread in the block can read what every other thread wrote, you need a referee: `__syncthreads()`. It's a barrier — no thread crosses it until all threads in the block have arrived. Without it, a thread might read a tile slot before the thread responsible for filling it has done so.

## A picture

```text
GLOBAL (in)                SHARED tile               GLOBAL (out)
row-major                  TILE x (TILE+1)           row-major

 ┌───────────┐  coalesced   ┌───────────┐  read       ┌───────────┐
 │ a b c d → │  load (rows)  │ a b c d   │  COLUMN     │ a e i m → │
 │ e f g h → │  ===========> │ e f g h   │  =========> │ b f j n → │
 │ i j k l → │               │ i j k l   │  then       │ c g k o → │
 │ m n o p → │               │ m n o p   │  coalesced  │ d h l p → │
 └───────────┘               └───────────┘  store      └───────────┘
   thread x → contiguous       │ │ │ │                   thread x → contiguous
                               └─┴─┴─┴── one warp reads a column
                                         (this is where banks bite)
```

The transpose is the "read a column of the tile, write it as a row of out" step. Both global trips stay contiguous; the awkward stride is confined to fast on-chip memory.

## Your task

Transpose an `N × N` row-major float matrix, `out = in^T`, beating the exercise-04 single-sided transpose. `N = 4096`. Stage each `TILE × TILE` block through a padded shared tile.

### The `solve` contract

```cpp
void solve(const float* in, float* out, int n);
```

`in` and `out` are **device pointers** to `n*n` floats, row-major. You implement two things in `transpose.cu`: the `__global__` kernel `transpose` (declare the shared tile, coalesced load, sync, transposed coalesced store) and the host `solve` (launch with `dim3 block(TILE, TILE)` and a 2-D grid). `harness.cu` owns `main()`, builds the data, checks correctness, runs the baseline, and reports your speedup — don't touch it.

## The bank-conflict trap (and the `+ 1` fix)

This is the part that separates "uses shared memory" from "uses shared memory *well*."

Shared memory is split into **32 banks**. Consecutive 4-byte words live in consecutive banks (word *i* → bank `i % 32`). A warp of 32 threads can service all 32 of its accesses in a single cycle **only if** they land in 32 distinct banks. If `k` threads hit the same bank, those accesses **serialize** — a `k`-way bank conflict is `k×` slower.

Now look at the transposed read `tile[threadIdx.x][threadIdx.y]`: a warp walks straight *down a column*. With an unpadded `tile[32][32]`, column elements are exactly 32 words apart — so all 32 land in the **same bank**. That's a 32-way conflict, and it erases the whole point of using shared memory.

The fix is one extra, unused column:

```cpp
__shared__ float tile[TILE][TILE + 1];   // type the "+ 1" yourself
```

Now stepping down a column advances the address by 33 words. Because 33 is coprime with 32, the 32 column elements scatter across all 32 banks — conflict-free. The padding column never stores data; it just nudges the alignment. The grader greps for `TILE + 1`, so write it literally.

## Functions & syntax you'll need

| Thing | Signature / form | What it does |
|---|---|---|
| `threadIdx.x/.y` | built-in `uint3` | this thread's position within its block |
| `blockIdx.x/.y` | built-in `uint3` | this block's position within the grid |
| `__shared__` | `__shared__ float tile[TILE][TILE+1];` | per-block on-chip buffer, visible to all threads in the block |
| `__syncthreads()` | `void __syncthreads();` | block-wide barrier; **all** threads must reach it |
| `dim3` | `dim3 block(TILE, TILE);` | 2-D/3-D launch geometry |
| launch | `transpose<<<grid, block>>>(in, out, n);` | start the kernel |
| `ceil_div(a, b)` | from `cuda_utils.cuh` | `(a + b - 1) / b`, for grid sizing |

> **Fun fact:** NVIDIA's own optimized transpose sample uses exactly this padded-tile pattern. The `+1` padding trick is so idiomatic it shows up in their CUDA Best Practices Guide as the canonical example of avoiding bank conflicts.

## How it's graded

`python grade.py` builds and runs the harness, then checks:

- **correctness** — `out == in^T`, exactly (`max_abs_err == 0`).
- **speedup** — `naive_ms / your_ms >= 1.5` versus the exercise-04 single-sided (strided-write) transpose. Coalescing *both* sides is what buys this.
- **efficiency** — `bw_frac >= 0.60` of peak bandwidth (the harness counts `2 × bytes`, one read + one write of the whole matrix). Drop the `+ 1` and the 32-way bank conflicts will likely sink this number even though correctness still passes.
- **source** — must contain `__shared__`, `__syncthreads`, and a `TILE + 1` padded tile.

Run `python grade.py --check-solution` to grade the reference solution instead of yours.

## Going deeper

1. A T4's theoretical peak is ~320 GB/s. A pure-copy kernel (read, write, no transpose) is the real ceiling for a memory-bound op like this — see how close your transpose gets to it.
2. Want more? Have each thread transpose several rows of the tile (a thread-coarsening loop) to amortize index arithmetic — that's how the fastest hand-written transposes squeeze out the last few percent.
