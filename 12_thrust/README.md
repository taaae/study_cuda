# Exercise 12 — Thrust (the high-level productivity library)

**New concepts:** Thrust as the "STL for the GPU" — **functors / predicates**, fused **transform-reduce**, **stream compaction** (`copy_if`), `thrust::device_ptr` to wrap raw device memory, and the `thrust::device` execution policy.

## When to reach for Thrust

Thrust gives you battle-tested parallel algorithms — `sort`, `reduce`, `scan`, `copy_if`, `transform_reduce` — in one line each. Reach for it when you want a correct, decently fast result *now* and the operation is a standard primitive. Drop to hand-written kernels only when profiling says a specific step is the bottleneck and you can beat the library (rare for these primitives).

## The task

Given a `float` array `in` of length `n` and a `threshold`:

1. **Stream-compact** every element with value `> threshold` into `out_compacted`, preserving order. (`thrust::copy_if`)
2. Compute the **sum of squares** of the *kept* elements. (`thrust::transform_reduce`)

Return the compacted array, the count of kept elements, and the sum of squares.

### `solve` signature (the contract)

```cpp
void solve(const float* in, int n, float threshold,
           float* out_compacted, int* out_count, float* out_sumsq);
```

All pointers are **device pointers**. `out_compacted` has room for up to `n` floats. `out_count` and `out_sumsq` are **single device values** (write the result through them). Wrap the raw pointers with `thrust::device_ptr<...>` so Thrust algorithms can operate on them.

## Functors & predicates

Thrust algorithms take *callable objects*. Two ways to make one:

**A custom struct** marked `__host__ __device__`:

```cpp
struct greater_than {
    float t;
    __host__ __device__ bool operator()(float x) const { return x > t; }
};
// use: greater_than{threshold}
```

**Placeholders** for quick arithmetic/comparisons:

```cpp
#include <thrust/functional.h>
using namespace thrust::placeholders;
// _1 > threshold     is a predicate
// _1 * _1            squares its argument
```

Either is fine. The square-it functor for the sum of squares is just `_1 * _1`, or a struct returning `x * x`.

## device_ptr and execution policy

Thrust needs *iterators*. To use Thrust on memory you allocated with `cudaMalloc`, wrap the raw pointer:

```cpp
#include <thrust/device_ptr.h>
thrust::device_ptr<const float> in_ptr(in);
thrust::device_ptr<float>       out_ptr(out_compacted);
```

Pass `thrust::device` (from `<thrust/execution_policy.h>`) as the first argument so the algorithm runs on the GPU:

```cpp
auto end = thrust::copy_if(thrust::device, in_ptr, in_ptr + n, out_ptr, pred);
int count = (int)(end - out_ptr);                 // iterator arithmetic gives the count
```

For the sum of squares, `transform_reduce` applies a unary op then reduces with `+`:

```cpp
#include <thrust/transform_reduce.h>
float ss = thrust::transform_reduce(thrust::device,
                                    out_ptr, out_ptr + count,   // the KEPT elements
                                    square_op, 0.0f, thrust::plus<float>());
```

Write `count` and `ss` back to the single-value device outputs (e.g. with `cudaMemcpy`).

## Syntax / reference

```cpp
#include <thrust/copy.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

thrust::device_ptr<T> p(raw);
thrust::copy_if(thrust::device, first, last, result, predicate);
thrust::transform_reduce(thrust::device, first, last, unary_op, init, binary_op);
```

## Grading (`!python grade.py`)

- **correctness** — `count`, the compacted set (order-preserving, so a direct compare works), and the sum of squares (within tolerance) all match a CPU reference.
- **efficiency** — `ms` is reported but there is **no strict threshold**: the point here is productivity, not squeezing the library.
- **source** — you must use `thrust::` and at least one of `copy_if` / `transform_reduce`.

`grade.py` adds `--extended-lambda` to nvcc so device lambdas/placeholders compile. Run `python grade.py --check-solution` to grade the reference solution instead of yours.
