# Hints — Exercise 24

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — Why the naive kernel is slow (concept)</summary>

Each output pixel reads 5 input pixels straight from global memory, and adjacent
output pixels overlap heavily — so every input pixel is fetched ~5 times from
DRAM. The fix is to load each input pixel **once** into fast shared memory and
let the whole block reuse it. That's the entire game here.
</details>

<details>
<summary>Hint 2 — The halo (concept)</summary>

A block computes a `BX × BY` output tile, but to compute the edge pixels it needs
their neighbors, which lie one pixel **outside** the tile. So the shared tile must
be `(BX+2) × (BY+2)`: the interior plus a 1-pixel border ("halo") on all sides.
Apply the clamp **while loading the halo**, so the compute step is branch-free.
</details>

<details>
<summary>Hint 3 — Coalescing + the shared declaration (code)</summary>

```cpp
#define BX 32
#define BY 8
__shared__ float s[BY + 2][BX + 2];
int x = blockIdx.x * BX + threadIdx.x;   // consecutive threads -> consecutive x
int y = blockIdx.y * BY + threadIdx.y;
```
A clamp helper:
```cpp
__device__ int clampi(int v, int n){ return v<0?0:(v>=n?n-1:v); }
```
</details>

<details>
<summary>Hint 4 — Loading interior + halo (code)</summary>

```cpp
int tx = threadIdx.x, ty = threadIdx.y;
int cx = clampi(x, width), cy = clampi(y, height);
s[ty+1][tx+1] = in[cy*width + cx];                       // interior

if (tx == 0)      s[ty+1][0]      = in[cy*width + clampi(blockIdx.x*BX - 1, width)];
if (tx == BX-1)   s[ty+1][BX+1]   = in[cy*width + clampi(blockIdx.x*BX + BX, width)];
if (ty == 0)      s[0][tx+1]      = in[clampi(blockIdx.y*BY - 1, height)*width + cx];
if (ty == BY-1)   s[BY+1][tx+1]   = in[clampi(blockIdx.y*BY + BY, height)*width + cx];
__syncthreads();
```
The 5-point stencil never reads diagonal corners, so you can skip them.
</details>

<details>
<summary>Hint 5 — Compute from shared (code)</summary>

```cpp
if (x < width && y < height) {
    float c = s[ty+1][tx+1];
    float l = s[ty+1][tx],   r = s[ty+1][tx+2];
    float u = s[ty][tx+1],   d = s[ty+2][tx+1];
    out[y*width + x] = (c + l + r + u + d) * 0.2f;
}
```
</details>

<details>
<summary>Hint 6 — The launch + going further (code + concept)</summary>

```cpp
dim3 block(BX, BY);
dim3 grid(ceil_div(width, BX), ceil_div(height, BY));
stencil_fast<<<grid, block>>>(in, out, width, height);
```

If you want more: vectorize the interior load with `float4` when `BX` is a
multiple of 4, and try a couple of block shapes (32×8 vs 16×16) — profile each
with `nsys` timeline durations to compare.
</details>
