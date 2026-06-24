# Hints — Exercise 17

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — From one thread per row to one warp per row (concept)</summary>

Keep the same math, but spread each row across 32 lanes. Lane `l` takes nonzeros `start+l, start+l+32, start+l+64, …`, accumulating a partial sum. When the loop ends, the warp adds the 32 partials together and lane 0 writes the answer. Because all 32 lanes are on the *same* row, none idle waiting for a longer row — that's the load-balance win.
</details>

<details>
<summary>Hint 2 — Mapping threads to (row, lane) (code)</summary>

```cpp
int global = blockIdx.x * blockDim.x + threadIdx.x;
int warpId = global >> 5;        // = global / 32  -> which row
int lane   = threadIdx.x & 31;   // = threadIdx.x % 32 -> lane in warp
if (warpId >= nrows) return;
```
</details>

<details>
<summary>Hint 3 — The strided per-lane loop (code)</summary>

```cpp
int start = rowPtr[warpId], end = rowPtr[warpId + 1];
float sum = 0.0f;
for (int k = start + lane; k < end; k += 32)
    sum += vals[k] * x[colIdx[k]];
```

On each iteration the 32 lanes read 32 consecutive `vals`/`colIdx` entries — a coalesced load.
</details>

<details>
<summary>Hint 4 — The warp shuffle reduction (concept)</summary>

Each lane holds a partial; you need their total. `__shfl_down_sync(mask, v, offset)` returns the value `v` from the lane `offset` positions higher. Add in offsets 16, 8, 4, 2, 1: after step one, lane `l` holds `partial[l] + partial[l+16]`; after all five steps lane 0 holds the sum of all 32. No shared memory, no `__syncthreads`.
</details>

<details>
<summary>Hint 5 — The reduction and write (code)</summary>

```cpp
for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
if (lane == 0) y[warpId] = sum;
```

The mask `0xffffffff` declares all 32 lanes active — required for correctness on modern CUDA.
</details>

<details>
<summary>Hint 6 — Launching nrows warps (code)</summary>

You need `nrows * 32` threads total, in blocks that are a multiple of 32:

```cpp
void solve(const int* rowPtr, const int* colIdx, const float* vals,
           const float* x, float* y, int nrows) {
    int block = 256;                       // 8 warps per block
    int warpsPerBlock = block / 32;        // = 8
    int grid = ceil_div(nrows, warpsPerBlock);
    spmv_vector<<<grid, block>>>(rowPtr, colIdx, vals, x, y, nrows);
}
```
</details>
