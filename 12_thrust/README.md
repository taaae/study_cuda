# Exercise 12 — Thrust (the high-level productivity library)
> After nine exercises of hand-rolled kernels, here's the one-line-each way the pros actually ship.

## The idea
Thrust is the **"STL for the GPU"** — battle-tested parallel algorithms (`sort`,
`reduce`, `scan`, `copy_if`, `transform_reduce`) you call in a single line. The
skills you've built — tiling, scan, atomics — are exactly what's running *inside*
these algorithms, but you no longer have to write them. The judgment to learn here
is **when** to reach for Thrust: when you want a correct, decently fast result now
and the operation is a standard primitive. Drop to a custom kernel only when a
profiler proves a specific step is the bottleneck and you can beat the library
(rare for these primitives).

Your task, given a `float` array `in` of length `n` and a `threshold`:

1. **Stream-compact** every element `> threshold` into `out_compacted`, preserving
   order — `thrust::copy_if`.
2. Compute the **sum of squares** of the *kept* elements — `thrust::transform_reduce`
   (a fused map-then-reduce, one pass, no intermediate array).

## Under the hood
- **`copy_if` is a scan in disguise.** To compact in parallel you must know *where*
  each kept element lands before you write it. Thrust runs the predicate to make a
  `0/1` flag per element, **exclusive-scans** the flags (exactly exercise 10!) to
  get each survivor's output index, then scatters. That's why compaction is
  "embarrassingly parallel" only after you've internalized scan.
- **`transform_reduce` is fused.** Naively you'd `transform` into a temp array then
  `reduce` it — two passes, extra memory. The fused version applies `x*x` to each
  element *as it's reduced*, so it reads the data once and allocates nothing extra.
  Same arithmetic-intensity instinct you used in the GEMM, handed to you for free.
- **Functors run on the device.** A predicate/op is a callable struct marked
  `__host__ __device__`. Thrust instantiates your algorithm as a templated kernel
  with the functor inlined — there's no function-pointer indirection, so it's as
  fast as if you'd hand-written the comparison.

## A picture
```text
  copy_if(in, pred = (_ > threshold))

  in:    [ -0.4   1.2  -0.1   3.5   0.0   2.1 ]
  flags: [   0     1     0     1     0     1   ]   pred(x)
  scan:  [   0     0     1     1     2     2   ]   exclusive prefix sum  -> dest index
  out:   [ 1.2   3.5   2.1 ]                       count = end - out_ptr = 3
                │     │     │
                ▼     ▼     ▼   transform_reduce(square, +)
              1.44 +12.25+ 4.41  = sum of squares
```

## Your task
Edit `compact.cu` and fill in the `TODO`s:

1. Finish the `keep_above` predicate and `square` functor (`__host__ __device__`).
2. In `solve`: wrap the raw device pointers in `thrust::device_ptr`, call
   `copy_if` to compact, derive `count` from the returned iterator, call
   `transform_reduce` over the kept range, and copy `count` and the sum-of-squares
   back through the single-value device outputs.

### The `solve` contract
```cpp
void solve(const float* in, int n, float threshold,
           float* out_compacted, int* out_count, float* out_sumsq);
```
All pointers are **device pointers**. `out_compacted` has room for up to `n` floats.
`out_count` and `out_sumsq` are **single device values** — write the result through
them (e.g. via `cudaMemcpy` from a host scalar).

## Functions & syntax you'll need
| Function | Signature (sketch) | What it does |
|----------|--------------------|--------------|
| `thrust::device_ptr<T>` | `device_ptr<T> p(raw);` | wraps a `cudaMalloc`'d pointer as a Thrust iterator |
| `thrust::device` | (policy object) | execution policy → runs the algorithm on the GPU |
| `thrust::copy_if` | `copy_if(policy, first, last, result, pred)` → end iterator | copies elements where `pred` is true, order-preserving |
| `thrust::transform_reduce` | `transform_reduce(policy, first, last, unary_op, init, binary_op)` → value | fused map-then-reduce, single pass |
| `thrust::plus<float>` | functor | the `+` reduction operator |
| custom functor | `struct{ __host__ __device__ R operator()(T) const; }` | predicate / unary op |
| placeholders | `using namespace thrust::placeholders; _1 > t; _1*_1` | inline lambdas, no struct needed |

Putting it together:
```cpp
thrust::device_ptr<const float> in_ptr(in);
thrust::device_ptr<float>       out_ptr(out_compacted);

auto end  = thrust::copy_if(thrust::device, in_ptr, in_ptr + n,
                            out_ptr, keep_above{threshold});
int count = (int)(end - out_ptr);                 // iterator arithmetic = count

float ss  = thrust::transform_reduce(thrust::device, out_ptr, out_ptr + count,
                                     square{}, 0.0f, thrust::plus<float>());
CUDA_CHECK(cudaMemcpy(out_count, &count, sizeof(int),   cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(out_sumsq, &ss,    sizeof(float), cudaMemcpyHostToDevice));
```
Headers: `<thrust/copy.h>`, `<thrust/transform_reduce.h>`, `<thrust/device_ptr.h>`,
`<thrust/execution_policy.h>`, `<thrust/functional.h>`.

## How it's graded
`python grade.py` (which adds `--extended-lambda` to nvcc so placeholders/device
lambdas compile) checks:

- **correctness** — `count`, the compacted set (order-preserving, so an
  element-by-element compare works), and the sum of squares (within `1e-3` relative)
  all match a CPU reference.
- **efficiency** — `ms` is reported but there is **no strict threshold**:
  productivity is the point, not squeezing the library.
- **source** — you must use `thrust::` and at least one of `copy_if` /
  `transform_reduce`.

Run `python grade.py --check-solution` to grade the reference solution instead of
yours.

## Going deeper
- **`thrust::device_vector<T>`** owns memory and frees it for you (RAII). Here we
  wrap raw pointers because the harness owns the buffers.
- **One-pass alternative:** `transform_reduce` with a predicate-aware op (return
  `x*x` if kept else `0`) gets the sum of squares *without* materializing the
  compacted array — but you still need `copy_if` for the array itself.
- **CUB** sits one level below Thrust (`cub::DeviceSelect::If`,
  `cub::DeviceScan`) and exposes the **temp-storage two-call idiom**: call once with
  `d_temp_storage = nullptr` to size the scratch, `cudaMalloc` it, then call again.
  Thrust hides that allocation for you. Reach for CUB when you need that control.
