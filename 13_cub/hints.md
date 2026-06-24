# Hints — Exercise 13

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — The whole shape (concept)</summary>

CUB device algorithms don't allocate their own scratch — you do. So every call happens twice: once with a null scratch pointer just to *learn the size*, then once for real after you `cudaMalloc` that many bytes. The algorithm itself is a single function: `cub::DeviceScan::ExclusiveSum`.
</details>

<details>
<summary>Hint 2 — The sizing call (code)</summary>

```cpp
void*  d_temp = nullptr;
size_t temp_bytes = 0;
cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, in, out, n);
// nothing ran; temp_bytes now holds how much scratch CUB needs
```
</details>

<details>
<summary>Hint 3 — Allocate then call for real (code)</summary>

```cpp
CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, in, out, n);
CUDA_CHECK(cudaFree(d_temp));
```
The second call is byte-for-byte the same as the first; the only difference is `d_temp` is now a real buffer.
</details>

<details>
<summary>Hint 4 — The complete solve (code)</summary>

```cpp
void solve(const int* in, int* out, int n) {
    void*  d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, in, out, n);
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, in, out, n);
    CUDA_CHECK(cudaFree(d_temp));
}
```
Compare `bw_frac` here against your exercise-10 number — that gap is the point of the exercise.
</details>
