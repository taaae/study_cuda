# Hints — Exercise 12

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — The four pieces (concept)</summary>

You need: (1) wrap the raw device pointers in `thrust::device_ptr`, (2) `copy_if` with a "greater than threshold" predicate to compact, (3) `transform_reduce` with a "square" op over the kept range to get the sum of squares, (4) copy the two scalars back to `out_count` / `out_sumsq`. Pass `thrust::device` as the first argument to each algorithm.
</details>

<details>
<summary>Hint 2 — Writing the functors (code)</summary>

```cpp
struct keep_above {
    float t;
    __host__ __device__ bool operator()(float x) const { return x > t; }
};
struct square {
    __host__ __device__ float operator()(float x) const { return x * x; }
};
```
(Equivalently, `thrust::placeholders::_1 > t` and `_1 * _1` — needs `--extended-lambda`, which `grade.py` already passes.)
</details>

<details>
<summary>Hint 3 — Wrapping pointers and getting the count (code)</summary>

```cpp
thrust::device_ptr<const float> in_ptr(in);
thrust::device_ptr<float>       out_ptr(out_compacted);

auto end = thrust::copy_if(thrust::device, in_ptr, in_ptr + n,
                           out_ptr, keep_above{threshold});
int count = (int)(end - out_ptr);   // iterator difference = #kept
```
</details>

<details>
<summary>Hint 4 — The fused sum of squares (code)</summary>

```cpp
float ss = thrust::transform_reduce(thrust::device,
                                    out_ptr, out_ptr + count,   // kept elements only
                                    square{}, 0.0f,
                                    thrust::plus<float>());
```
`transform_reduce` applies `square` to each element and folds with `+`, all in one pass — no temporary array.
</details>

<details>
<summary>Hint 5 — Writing scalars back (code)</summary>

```cpp
CUDA_CHECK(cudaMemcpy(out_count, &count, sizeof(int),   cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(out_sumsq, &ss,    sizeof(float), cudaMemcpyHostToDevice));
```
`count` and `ss` are host variables here (Thrust returned them to the host), so the copy direction is Host→Device.
</details>
