# Exercise 10 — Parallel Prefix Sum (Scan)

**New concepts:** the *scan* primitive (prefix sum), the **work-efficient Blelloch** algorithm (up-sweep / down-sweep) in shared memory, composing a large scan out of per-block scans, and **bank conflicts** in shared-memory scans.

## Inclusive vs exclusive scan

Given `in = [3, 1, 7, 0, 4]`:

| scan kind | result | rule |
|-----------|--------|------|
| **inclusive** | `[3, 4, 11, 11, 15]` | `out[i] = in[0] + ... + in[i]` |
| **exclusive** | `[0, 3, 4, 11, 11]` | `out[i] = in[0] + ... + in[i-1]`, `out[0] = 0` |

This exercise computes the **exclusive** scan. (Inclusive is just exclusive shifted, or `exclusive[i] + in[i]`.)

## Why "work-efficient"?

The simplest parallel scan (Hillis–Steele) does `O(n log n)` adds — every element is touched at each of the `log n` passes. The **Blelloch** algorithm does only `O(n)` adds, the same as the sequential loop, in two phases over a balanced binary tree:

- **Up-sweep (reduce):** walk the tree from leaves to root, summing pairs. After this the last slot holds the total.
- **Down-sweep:** set the root to the identity (0), then walk back down, at each node passing the left child's old value to the right and the sum down the left.

That `O(n)` work is why it scales: it does not waste memory bandwidth re-reading data.

## The task

Compute the exclusive prefix sum of a large `int` array. A single block cannot scan millions of elements, so use the classic **three-phase** design:

1. **Per-block scan.** Each block loads a chunk into `__shared__` memory and Blelloch-scans it. Process **2 elements per thread** (a block of `B` threads scans `2*B` elements). Each block also records the *total* of its chunk into an auxiliary array `blockSums[blockId]`.
2. **Scan the block sums.** Exclusive-scan `blockSums` so that `blockSums[b]` becomes the offset that block `b`'s results must be shifted by. (For the array sizes here a single extra scan launch over the block totals is enough — but think about how you would recurse if there were too many blocks.)
3. **Add offsets.** Each block adds its `blockSums[b]` to every element it scanned.

### `solve` signature (the contract)

```cpp
void solve(const int* in, int* out, int n);
```

`in` and `out` are **device pointers** of length `n`. You own all kernel launches and any scratch (`blockSums`) allocation; free what you allocate. `n` is not necessarily a power of two or a multiple of the block tile.

## Bank conflicts (optional, but it's why scan code looks weird)

Shared memory has 32 banks. The Blelloch index pattern `2*offset*(tid+1)-1` makes many threads hit the *same* bank, serializing accesses. The standard fix is **conflict-free padding**: spread indices out by adding `index >> LOG_NUM_BANKS` extra slots. A macro makes this readable:

```cpp
#define LOG_NUM_BANKS 5                       // 32 banks on all CUDA GPUs
#define CONFLICT_FREE_OFFSET(i) ((i) >> LOG_NUM_BANKS)
// then index shared memory as  s[i + CONFLICT_FREE_OFFSET(i)]
```

You can pass the exercise without padding (correctness does not depend on it), but it is the difference between a slow and a fast scan, and it is *the* thing that makes real scan code look the way it does.

## Syntax / reference

```cpp
extern __shared__ int s[];                    // dynamic shared memory
kernel<<<grid, block, sharedBytes>>>(...);    // 3rd launch arg = bytes of shared mem
__syncthreads();                              // barrier across the block
CUDA_CHECK(cudaMalloc(&blockSums, ...));      // scratch for phase 1/2
```

Up-sweep / down-sweep skeleton for a tile of size `m == 2*blockDim.x` in `s[]`:

```cpp
int tid = threadIdx.x, offset = 1;
// up-sweep
for (int d = m >> 1; d > 0; d >>= 1) {
    __syncthreads();
    if (tid < d) {
        int ai = offset*(2*tid+1) - 1;
        int bi = offset*(2*tid+2) - 1;
        s[bi] += s[ai];
    }
    offset <<= 1;
}
// clear last element, then down-sweep
if (tid == 0) s[m-1] = 0;
for (int d = 1; d < m; d <<= 1) {
    offset >>= 1;
    __syncthreads();
    if (tid < d) {
        int ai = offset*(2*tid+1) - 1;
        int bi = offset*(2*tid+2) - 1;
        int t = s[ai]; s[ai] = s[bi]; s[bi] += t;
    }
}
__syncthreads();
```

## Grading (`!python grade.py`)

- **correctness** — output matches a CPU exclusive scan exactly (integers).
- **efficiency** — `bw_frac >= 0.30`. Scan is multi-pass (it reads and writes the data more than once and touches the block sums), so the bar is deliberately lenient compared to a single-pass kernel.
- **source** — you must use `__shared__` memory and `__syncthreads()`.

Run `python grade.py --check-solution` to grade the reference solution instead of yours.
