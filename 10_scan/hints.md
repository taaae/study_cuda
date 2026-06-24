# Hints — Exercise 10

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — The shape of the whole thing (concept)</summary>

Three launches, in order:
1. `scan_block` — every block scans its own `TILE` elements and drops its total into `blockSums[blockId]`.
2. one more scan over `blockSums` → `blockOffsets` (block `b`'s offset is the sum of all earlier blocks' totals).
3. `add_offsets` — block `b` adds `blockOffsets[b]` to each of its elements.

The trick that makes step 2 a single launch: with `BLOCK=512`/`TILE=1024`, the number of blocks is at most `TILE`, so one `scan_block` covers all the block sums.
</details>

<details>
<summary>Hint 2 — Loading the tile and handling ragged ends (concept)</summary>

Each thread owns two slots, `ai = 2*tid` and `bi = 2*tid+1`, mapping to globals `base+ai`, `base+bi` where `base = blockIdx.x * TILE`. If a global index is `>= n`, load `0` — zero is the additive identity, so it doesn't change any prefix sum, and it lets the algorithm pretend the tile is full.
</details>

<details>
<summary>Hint 3 — Up-sweep (code)</summary>

```cpp
int offset = 1;
for (int d = TILE >> 1; d > 0; d >>= 1) {
    __syncthreads();
    if (tid < d) {
        int ai = offset * (2*tid + 1) - 1;
        int bi = offset * (2*tid + 2) - 1;
        s[bi] += s[ai];
    }
    offset <<= 1;
}
```
After this, `s[TILE-1]` holds the total of the tile.
</details>

<details>
<summary>Hint 4 — Clear the root, then down-sweep (code)</summary>

```cpp
if (tid == 0) {
    blockSums[blockIdx.x] = s[TILE - 1];   // save total
    s[TILE - 1] = 0;                        // seed exclusive scan
}
for (int d = 1; d < TILE; d <<= 1) {
    offset >>= 1;
    __syncthreads();
    if (tid < d) {
        int ai = offset * (2*tid + 1) - 1;
        int bi = offset * (2*tid + 2) - 1;
        int t = s[ai];
        s[ai] = s[bi];
        s[bi] += t;
    }
}
__syncthreads();
```
</details>

<details>
<summary>Hint 5 — The three launches in solve (code)</summary>

```cpp
int numBlocks = ceil_div(n, TILE);
size_t shBytes = TILE * sizeof(int);

scan_block<<<numBlocks, BLOCK, shBytes>>>(in, out, blockSums, n);

// Scan the block totals (one block; its own total output is unused scratch).
int* dummy = nullptr;
CUDA_CHECK(cudaMalloc(&dummy, sizeof(int)));
scan_block<<<1, BLOCK, shBytes>>>(blockSums, blockOffsets, dummy, numBlocks);
CUDA_CHECK(cudaFree(dummy));

add_offsets<<<numBlocks, BLOCK>>>(out, blockOffsets, n);
```
Note `scan_block` already guards with `< n`, so scanning `numBlocks` totals (`numBlocks <= TILE`) in a single block is fine.
</details>

<details>
<summary>Hint 6 — Going faster: bank-conflict padding (concept + code)</summary>

The index `offset*(2*tid+1)-1` makes strided shared-memory accesses that collide on the 32 banks. Pad every shared index by `i >> 5`:

```cpp
#define CONFLICT_FREE_OFFSET(i) ((i) >> 5)
// declare shared as TILE + CONFLICT_FREE_OFFSET(TILE) ints, and index with
// s[i + CONFLICT_FREE_OFFSET(i)] everywhere (load, sweeps, store, clear).
```
Remember to grow the dynamic shared-memory size you pass at launch by the same padding. This is optional for correctness but is what pushes the bandwidth up.
</details>
