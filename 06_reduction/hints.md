# Hints — Exercise 06

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — The shape of a reduction (no code)</summary>

You can't sum `n` numbers with one global accumulator without serializing everything. Instead, do it as a **tree**: pairs add into halves, halves into quarters, until one value remains. Each **block** does this tree over its own chunk in fast on-chip `__shared__` memory, producing one partial sum. Then you only have a few thousand block-partials left to combine — cheap.
</details>

<details>
<summary>Hint 2 — Sequential vs interleaved addressing (concept)</summary>

In the tree loop you halve a stride `s` each step and let threads with `tid < s` add `sdata[tid] + sdata[tid + s]`. Start `s` at `blockDim.x/2` and shift right. This keeps the *active* threads contiguous (`0..s-1`) — so early steps have no warp divergence — and the two addresses `tid` and `tid+s` never collide in a shared-memory bank. The alternative (active threads = `tid % (2s) == 0`) is correct but diverges and bank-conflicts; avoid it.
</details>

<details>
<summary>Hint 3 — First add during load + grid-stride (concept)</summary>

Don't load just one element per thread. Have each thread sum *many* input elements into a register first, using a grid-stride loop:
`for (int i = global_id; i < n; i += gridDim.x * blockDim.x) sum += in[i];`
Then store that single `sum` into `sdata[tid]` and run the tree. This does most of the addition before the shared-memory tree even starts, handles any `n`, and is what gets you over the bandwidth bar. Launch a modest, fixed grid (e.g. a few thousand blocks) so each block stays full.
</details>

<details>
<summary>Hint 4 — Combining block partials (code)</summary>

After the tree, `sdata[0]` holds this block's partial. Add it to the global total with **one** atomic per block:

```cpp
if (tid == 0) atomicAdd(out, sdata[0]);
```

The harness zeroed `*out`, so this accumulates correctly. (One atomic per *block* is fine; one per *element* is not — that's the slow trap.)
</details>

<details>
<summary>Hint 5 — The kernel body (code)</summary>

```cpp
__global__ void reduce(const float* in, float* out, int n) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    float sum = 0.f;
    for (int i = gid; i < n; i += stride) sum += in[i];
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}
```
</details>

<details>
<summary>Hint 6 — The full solve (code)</summary>

```cpp
void solve(const float* in, float* out, int n) {
    int block = BLOCK;
    int grid = ceil_div(n, block);
    if (grid > 4096) grid = 4096;   // cap so blocks stay full; grid-stride covers the rest
    reduce<<<grid, block>>>(in, out, n);
}
```
</details>
