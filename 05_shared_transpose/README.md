# Exercise 05 — Shared-Memory Transpose (+ bank-conflict avoidance)

**New concepts:** **shared memory** as a fast, programmer-managed staging buffer inside a block, `__syncthreads()` to coordinate threads, and **bank-conflict avoidance** by **padding** the shared tile.

In exercise 04 you saw that a global-only transpose can coalesce reads *or* writes, never both. Shared memory breaks that tradeoff: stage a tile in fast on-chip memory so **both** the global read and the global write are coalesced.

## The task

Tiled transpose of an `N × N` row-major float matrix, `out = in^T`, using a `__shared__` tile:

1. **Read** a `TILE × TILE` block of `in` into the shared tile — *coalesced* (consecutive `threadIdx.x` → consecutive global addresses).
2. `__syncthreads()` so the whole tile is populated.
3. **Write** the tile out to `out`, reading it **transposed from shared memory**, so the global writes are *also* coalesced.
4. **Pad** the tile's second dimension by 1 (`tile[TILE][TILE+1]`) to eliminate shared-memory bank conflicts.

Edit `transpose.cu`:

1. The `__global__` kernel `transpose` — declare the padded `__shared__` tile, do the coalesced load, sync, and the coalesced transposed store.
2. The host function `solve` — launch with `dim3 block(TILE, TILE)` and a 2-D grid.

You do **not** write `main()` — `harness.cu` provides it, checks correctness, runs the exercise-04-style single-sided transpose as the baseline, and reports your speedup.

### `solve` signature (the contract)

```cpp
void solve(const float* in, float* out, int n);
```

`in`, `out` are **device pointers** to `n*n` floats, row-major.

## Shared memory in one paragraph

`__shared__` memory is on-chip, per-block, ~100× lower latency than global memory, and visible to all threads in the block. You declare it inside the kernel:

```cpp
#define TILE 32
__shared__ float tile[TILE][TILE + 1];   // note the + 1 (explained below)
```

Threads cooperatively fill it, call `__syncthreads()` (a barrier — every thread waits until all threads in the block reach it), then read it back in a different order. That reordering is how we transpose while keeping *both* global accesses coalesced.

## The transpose dance

```
load:   tile[threadIdx.y][threadIdx.x] = in[ y_in  * n + x_in ];   // coalesced read
        __syncthreads();
store:   out[ y_out * n + x_out ] = tile[threadIdx.x][threadIdx.y]; // coalesced write
```

- For the **load**, `x_in = blockIdx.x*TILE + threadIdx.x` is contiguous → coalesced.
- For the **store**, you target the *transposed* block, so `x_out = blockIdx.y*TILE + threadIdx.x` is contiguous → coalesced. The transpose happens by **swapping the indices when you read the shared tile** (`tile[threadIdx.x][threadIdx.y]`), not in global memory.

## Bank conflicts and the `+ 1` trick

Shared memory is split into **32 banks**; consecutive 4-byte words live in consecutive banks. A warp can read 32 words in one shot *only if* they fall in 32 different banks. Otherwise the accesses to the same bank **serialize** (an *N*-way bank conflict is *N*× slower).

The transposed read `tile[threadIdx.x][threadIdx.y]` walks down a **column** of the tile. With an unpadded `tile[32][32]`, a column's elements are exactly 32 words apart — so all 32 land in the **same bank** → a 32-way conflict, killing the benefit of shared memory.

Fix: pad the row to 33 floats — declare `tile[TILE][TILE + 1]`. Now stepping down a column advances the address by 33, which is coprime with 32, so the 32 column elements spread across **all 32 banks** → conflict-free. The extra column is never used for data; it just shifts the alignment. **Type the `+ 1` yourself** — the grader checks for it.

## Grading (`!python grade.py`)

- **correctness** — `out == in^T`.
- **speedup** — `naive_ms / your_ms >= 1.5` over the exercise-04 single-sided (strided-write) transpose.
- **efficiency** — `bw_frac >= 0.60` (2*bytes). Coalescing both sides + no bank conflicts gets you here; drop the `+ 1` and the conflicts will likely sink it.
- **source** — must use `__shared__` and `__syncthreads`, and the tile must be padded (`TILE + 1`).

Run `python grade.py --check-solution` to grade the reference solution instead of yours.
