# Exercise 24 — Capstone: Optimize an Unknown Kernel

**New concepts:** none new — this is the **capstone**. You take a correct-but-slow
kernel and make it fast, combining everything: coalescing, shared-memory tiling
with a halo, occupancy, and (optionally) vectorized loads.

## The problem

A **5-point stencil** (a simple blur) over a large `width × height` float image.
Each output pixel is the average of itself and its 4 orthogonal neighbors:

```
out(x,y) = ( in(x,y)
           + in(x-1,y) + in(x+1,y)
           + in(x,y-1) + in(x,y+1) ) / 5
```

**Boundary rule (clamped / replicate):** out-of-range neighbor coordinates are
**clamped** to the nearest valid pixel. I.e. a neighbor at `x = -1` reads column
`0`; at `x = width` reads column `width-1`; likewise for `y`. This is the exact
formula the CPU reference uses, so match it precisely.

Images are row-major: `in[y * width + x]`.

## The starting point

`harness.cu` contains a **naive baseline** kernel that is correct but slow: one
thread per output pixel, every neighbor fetched straight from global memory (so
each input pixel is re-read up to 5 times), with no coalescing discipline and no
shared memory. The harness runs it to establish the baseline you must beat.

Your job: rewrite `solve` (and the kernel it launches) to be fast. The intended
technique is **shared-memory tiling with a halo** — each block loads its tile
*plus a 1-pixel border* into shared memory once, then every thread reads its 5
neighbors from shared memory instead of global.

Edit `optimize.cu`:

1. A fast stencil kernel of your design (shared-memory halo tiling expected).
2. `solve` — launch it.

### `solve` signature (the contract)

```cpp
void solve(const float* in, float* out, int width, int height);
```

`in`, `out` are device pointers to `width*height` floats, row-major. Same formula
and boundary rule as above.

## Optimization checklist

- **Coalesce.** Threads in a warp (consecutive `x`) should read consecutive
  addresses. Map `threadIdx.x → x`, `threadIdx.y → y`.
- **Tile in shared memory with a halo.** A `BX × BY` output tile needs a
  `(BX+2) × (BY+2)` shared tile (1-pixel border on each side). Load the interior,
  then the halo rows/columns, then `__syncthreads()`, then compute from shared.
- **Apply the clamp when loading the halo**, so shared memory already holds the
  clamped border values and the compute step has no branches.
- **Pick a sane block** (e.g. 32×8 or 16×16) for occupancy and coalescing.
- **(Optional) float4** loads for the interior if your tile width is a multiple
  of 4 — extra credit, not required to pass.

## Real-world tooling note

On a real machine you'd profile this with **Nsight Systems** (`nsys`, timeline /
kernel durations — usually works on Colab) and **Nsight Compute** (`ncu`,
per-kernel memory/occupancy counters — often **blocked** on free Colab). Here the
harness stands in for `ncu` by measuring achieved bandwidth and speedup.

## Grading (`!python grade.py`)

- **correctness** — matches the CPU stencil (formula + clamp above) within
  tolerance.
- **efficiency** — `speedup >= 3.0` over the naive baseline, **and**
  `bw_frac >= 0.40` of the device's peak bandwidth.
- **source** — you must use `__shared__` (halo tiling is the expected approach).

Run `python grade.py --check-solution` to grade the reference solution instead.
