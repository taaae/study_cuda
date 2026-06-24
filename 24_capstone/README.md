# Exercise 24 — Capstone: Optimize an Unknown Kernel
> One correct-but-slow kernel, everything you've learned, and a 3× target. Go.

## The idea
This is the capstone — no new API, just the craft. You're handed a **5-point
stencil** (a simple blur) that's correct but slow, and your job is to make it fast
by stacking the techniques from the whole course: coalescing, shared-memory tiling
with a halo, occupancy, and optionally vectorized loads.

Each output pixel is the average of itself and its 4 orthogonal neighbors:
```text
out(x,y) = ( in(x,y) + in(x-1,y) + in(x+1,y) + in(x,y-1) + in(x,y+1) ) / 5
```
**Boundary rule (clamp / replicate):** an out-of-range neighbor reads the nearest
valid pixel — `x=-1` reads column `0`, `x=width` reads `width-1`, same for `y`.
This matches the CPU reference *exactly*, so reproduce it precisely. Images are
row-major: `in[y*width + x]`.

## Under the hood
Why is the naive version slow? It's **memory-bound and wasteful**. One thread per
pixel, every neighbor fetched straight from global memory — so each input pixel
gets re-read up to **5 times** (once as a center, four times as someone's
neighbor). That's 5× the necessary DRAM traffic on a kernel whose arithmetic is
trivial (four adds and a multiply). The bottleneck is bytes moved, not flops.

The fix is **shared-memory halo tiling**. A block of `BX×BY` output pixels reads,
all together, a `(BX+2)×(BY+2)` region — its tile *plus a 1-pixel border* (the
halo) — into shared memory **once**. Then every thread reads its 5 neighbors from
shared memory, which is on-chip and ~100× lower latency than global. Each input
pixel now crosses the DRAM bus essentially once instead of five times.

Two details make it fast *and* clean:
- **Coalescing:** map `threadIdx.x → x` so a warp's 32 threads read 32 consecutive
  addresses — one wide memory transaction instead of 32 scattered ones.
- **Clamp while loading the halo**, so shared memory already holds the correct
  border values. The compute step then has zero boundary branches — every thread
  runs the same straight-line code.

> **Fun fact:** the T4's peak global bandwidth is ~320 GB/s. The grading harness
> measures *effective* traffic as "read the image once + write it once" and reports
> `bw_frac` against that peak. Hitting 40%+ means you're genuinely streaming the
> image near memory speed — the naive kernel can't, because it moves ~5× the bytes.

## A picture
```text
A BX x BY output tile needs a (BX+2) x (BY+2) shared tile (1-pixel halo):

      +---------------------------+
      | H  H  H  H  H  H  H  H  H |   <- top halo row    (s[0][..])
      | H  .  .  .  .  .  .  .  H |
      | H  .  . interior  .  .  H |   <- s[ty+1][tx+1]
      | H  .  .  .  .  .  .  .  H |
      | H  H  H  H  H  H  H  H  H |   <- bottom halo row  (s[BY+1][..])
      +---------------------------+
        ^left halo col       ^right halo col
        (s[..][0])           (s[..][BX+1])

  load interior + halo -> __syncthreads() -> each thread reads its 5
  neighbors from shared. (Corners aren't used by a 5-point stencil.)
```

## Your task
Edit `optimize.cu`:
1. A fast stencil kernel of your design — shared-memory halo tiling is the
   intended approach. Load the interior, load the halo rows/columns (with the
   clamp), `__syncthreads()`, then compute from shared.
2. `solve` — launch it over the image.

### The `solve` contract
```cpp
void solve(const float* in, float* out, int width, int height);
```
`in`, `out` are device pointers to `width*height` floats, row-major. Same formula
and clamp rule as above. (`BX`/`BY` are predefined as 32 and 8 in the file.)

## Functions & syntax you'll need
| Construct | Form | Purpose |
|---|---|---|
| shared halo tile | `__shared__ float s[BY+2][BX+2];` | holds the tile + 1-pixel border |
| `__syncthreads()` | `void __syncthreads()` | barrier *after* all loads, *before* compute |
| clamp helper | `int clampi(int v,int n){ return v<0?0:(v>=n?n-1:v); }` | replicate-boundary index clamp |
| thread/block coords | `threadIdx.{x,y}`, `blockIdx.{x,y}`, `blockDim` | map `tx→x`, `ty→y` for coalescing |
| `ceil_div` | `ceil_div(a,b)` (provided in `cuda_utils.cuh`) | grid sizing that covers the whole image |
| launch | `stencil_fast<<<grid, block>>>(in,out,width,height);` | `block(BX,BY)`, `grid(ceil_div(width,BX), ceil_div(height,BY))` |
| *(optional)* `float4` | `reinterpret_cast<const float4*>(...)` | vectorized 16-byte loads if tile width % 4 == 0 (extra credit) |

Edge threads load the halo (clamping so edge blocks stay valid):
```cpp
s[ty+1][tx+1] = in[clampi(y,height)*width + clampi(x,width)];  // interior
if (tx == 0)      /* load left  halo column s[ty+1][0]      */ ;
if (tx == BX-1)   /* load right halo column s[ty+1][BX+1]   */ ;
if (ty == 0)      /* load top    halo row   s[0][tx+1]      */ ;
if (ty == BY-1)   /* load bottom halo row   s[BY+1][tx+1]   */ ;
__syncthreads();
// then: c/l/r/u/d from s[...], write out[y*width+x] if (x<width && y<height)
```

## Optimization checklist
- **Coalesce:** `threadIdx.x → x` so consecutive threads hit consecutive addresses.
- **Tile with a halo:** load interior + border into shared once, then read neighbors from shared.
- **Clamp on the halo load** so the compute step is branch-free.
- **Sane block** (32×8 or 16×16) for occupancy and coalescing.
- **(Optional) `float4`** interior loads if the tile width is a multiple of 4.

## How it's graded
`python grade.py` (4096×4096 image) checks:
- **correctness** — matches the CPU stencil (formula + clamp) within `max_abs_err <= 1e-4`.
- **efficiency** — **both** gates must pass: `speedup >= 3.0` over the naive
  baseline **and** `bw_frac >= 0.40` of peak bandwidth. A correct kernel that's
  still global-memory-bound (no shared tiling) will clear correctness but miss both.
- **source** — must contain `__shared__` (halo tiling is the expected approach).

Run `python grade.py --check-solution` to grade the reference solution.

## Real-world tooling note
On a real machine you'd profile this with **Nsight Systems** (`nsys`) for the
timeline and kernel durations — that *usually* works on Colab — and **Nsight
Compute** (`ncu`) for per-kernel memory/occupancy counters, which would show you
the re-read traffic directly. Heads up: `ncu`'s hardware counters are **often
blocked on free Colab**. Here the harness stands in for `ncu` by measuring
achieved bandwidth (`gbps`, `bw_frac`) and `speedup` for you.

## Going deeper
Once you pass, try widening the tile, comparing 32×8 vs 16×16 blocks, or the
`float4` interior load — and watch `bw_frac` climb toward the memory ceiling. A
memory-bound stencil can't beat "read once, write once," so `bw_frac` near 1.0 is
the real finish line. That's the whole game for bandwidth-bound kernels, and it's a
fitting place to end the course.
