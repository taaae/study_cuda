# Exercise 10 — Parallel Prefix Sum (Scan)
> Turning a stubbornly sequential running total into an O(n)-work parallel tree climb.

## The idea
A **scan** (prefix sum) turns `[3,1,7,0,4]` into its running totals. It looks
hopelessly sequential — each output seems to need the one before it — yet it's one
of the most important parallel primitives there is: it's the engine behind stream
compaction, radix sort, sparse-matrix ops, and allocation of variable-length
output. Learn to parallelize scan and a whole class of "I thought this had to be a
loop" problems opens up.

Two flavors, given `in = [3, 1, 7, 0, 4]`:

| kind | result | rule |
|------|--------|------|
| **inclusive** | `[3, 4, 11, 11, 15]` | `out[i] = in[0] + … + in[i]` |
| **exclusive** | `[0, 3, 4, 11, 11]` | `out[i] = in[0] + … + in[i-1]`, `out[0]=0` |

This exercise computes the **exclusive** scan. (Inclusive is just
`exclusive[i] + in[i]`.)

## Under the hood
The naive parallel scan (Hillis–Steele) does `O(n log n)` adds — every element is
touched in each of the `log n` passes. The **work-efficient Blelloch** algorithm
does only `O(n)` adds, the same as a sequential loop, by sweeping a balanced binary
tree **twice**:

- **Up-sweep (reduce):** leaves → root, summing pairs in place. When it finishes,
  the last slot holds the grand total.
- **Down-sweep:** set the root to the identity `0`, then root → leaves: at each
  node, hand the left child's *old* value to the right child and push the running
  sum down the left.

That `O(n)` work is why it scales — it doesn't waste bandwidth re-reading data.

**Bank conflicts** are the catch. Shared memory has 32 banks; the Blelloch index
pattern `offset*(2*tid+1)-1` makes many threads land in the *same* bank, serializing
the access. The classic fix is **conflict-free padding** — spread indices apart by
`index >> 5` extra slots. You can pass correctness without it, but padding is *the*
reason real scan code looks the way it does.

A single block can't scan a million elements, so you compose: scan each block's
chunk, scan the per-block totals, then add each block's offset back.

## A picture
```text
  Blelloch on [3 1 7 0]  (TILE = 4)

  UP-SWEEP (sum pairs, climb)        DOWN-SWEEP (clear root, descend)
   3  1  7  0                          3  4  7  11   <- total at root
   3 [4] 7 [7]    bi += ai             3 [0] 7 [4]   root := 0
   3  4  7 [11]   total at last        3  0  7 [0]   swap+add left/right
                                       ----------------------------------
   exclusive result:  [0, 3, 4, 11]
```

Three-phase composition for the whole array:
```text
  in ─► [scan_block]×G ─► out (each chunk scanned) + blockSums[g] (chunk totals)
                                         │
                          [scan_block]×1 ▼  exclusive-scan the totals
                                  blockOffsets[g]
                                         │
  out ◄────────── [add_offsets]: out[i] += blockOffsets[g]  ◄──────────┘
```

## Your task
Edit `scan.cu` and fill in the `TODO`s. `BLOCK=512`, `TILE=2*BLOCK=1024` (each
thread handles **2 elements**).

1. **`scan_block`** — load `TILE` elements (0 past the end) into dynamic shared
   memory, run the up-sweep, save the root into `blockSums[blockIdx.x]` and clear
   it, run the down-sweep, write results back to `out` (range-guarded).
2. **`solve`** — launch phase 1; phase 2 re-uses `scan_block` on the block totals
   in a single block (here `numBlocks <= TILE`, so one launch suffices); phase 3
   launches `add_offsets`. Allocate and **free** any scratch.

`add_offsets` is already written for you.

### The `solve` contract
```cpp
void solve(const int* in, int* out, int n);
```
`in` and `out` are **device pointers** of length `n`. You own all kernel launches
and any scratch (`blockSums`, `blockOffsets`) allocation — free what you allocate.
`n` need not be a power of two or a multiple of `TILE`.

## Functions & syntax you'll need
| Construct | Form | What it does |
|-----------|------|--------------|
| dynamic shared mem | `extern __shared__ int s[];` | size set at launch (3rd `<<<>>>` arg) |
| launch w/ shared | `kernel<<<grid, block, shBytes>>>(…)` | `shBytes = TILE*sizeof(int)` |
| `__syncthreads()` | `void __syncthreads()` | block barrier; needed before each sweep level |
| `cudaMalloc` | `cudaMalloc(&p, bytes)` | scratch for `blockSums` / `blockOffsets` |
| `cudaFree` | `cudaFree(p)` | release scratch |
| `ceil_div` | `ceil_div(n, TILE)` (`cuda_utils.cuh`) | number of blocks |
| `CONFLICT_FREE_OFFSET` | `((i) >> 5)` (optional) | conflict-free padding macro |

Up/down-sweep skeleton for a tile of `m == 2*blockDim.x` in `s[]`:
```cpp
int tid = threadIdx.x, offset = 1;
for (int d = m >> 1; d > 0; d >>= 1) {            // up-sweep
    __syncthreads();
    if (tid < d) {
        int ai = offset*(2*tid+1) - 1, bi = offset*(2*tid+2) - 1;
        s[bi] += s[ai];
    }
    offset <<= 1;
}
if (tid == 0) s[m-1] = 0;                          // clear root
for (int d = 1; d < m; d <<= 1) {                 // down-sweep
    offset >>= 1; __syncthreads();
    if (tid < d) {
        int ai = offset*(2*tid+1) - 1, bi = offset*(2*tid+2) - 1;
        int t = s[ai]; s[ai] = s[bi]; s[bi] += t;
    }
}
__syncthreads();
```

## How it's graded
`python grade.py` checks:

- **correctness** — output matches a CPU exclusive scan **exactly** (integers, no
  tolerance). The harness prints the first mismatch index if you're off.
- **efficiency** — `bw_frac >= 0.30` (achieved GB/s vs T4 peak). Scan is
  multi-pass — it reads/writes the data more than once and touches the block sums —
  so the bar is deliberately lenient versus a single-pass kernel. A Hillis–Steele
  (`O(n log n)`) kernel does far more memory traffic and tends to miss this bar.
- **source** — you must use `__shared__` and `__syncthreads`.

Run `python grade.py --check-solution` to grade the reference solution instead of
yours.

## Going deeper
- **Recursion:** if `numBlocks > TILE`, phase 2 itself needs more than one block —
  you'd recurse the same three-phase scheme on the block totals.
- **Real libraries:** in production you'd just call `cub::DeviceScan::ExclusiveSum`
  or `thrust::exclusive_scan`. CUB uses the **temp-storage two-call idiom**: call
  once with `d_temp_storage = nullptr` to learn the byte count, `cudaMalloc` it,
  then call again to run. CUB's chained-scan / decoupled-look-back does the whole
  array in a **single pass** — far faster than this three-pass version, but this
  exercise is where you learn *why* it works.
