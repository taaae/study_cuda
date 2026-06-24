# Hints — Exercise 02

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — What "grid-stride" means (no code)</summary>

You are *not* making one thread per element this time. You launch a fixed pool of threads (say a few thousand) and each one walks the array in jumps. The size of each jump is the total number of threads in the grid, so the threads together tile the array with no gaps and no overlaps, regardless of how big `n` is.
</details>

<details>
<summary>Hint 2 — Sizing the grid to the machine, not the data (concept)</summary>

The whole point is that the grid size does **not** depend on `n`. Ask the device how many SMs it has (`cudaDeviceProp::multiProcessorCount`) and launch a small multiple of that many blocks — enough blocks per SM to hide memory latency. A factor like 32 blocks per SM with 256 threads/block is plenty on a T4. Use `CUDA_CHECK` around `cudaGetDeviceProperties`.
</details>

<details>
<summary>Hint 3 — The two numbers the kernel needs (concept)</summary>

Inside the kernel each thread needs:
- its **starting index**: the same `blockIdx.x * blockDim.x + threadIdx.x` as before.
- its **stride**: the total thread count, which is `gridDim.x * blockDim.x`.

Then a simple `for` loop from start, incrementing by stride, until you pass `n`. No `if (i < n)` guard needed separately — the loop condition is the guard.
</details>

<details>
<summary>Hint 4 — The kernel body (code)</summary>

```cpp
__global__ void saxpy(float a, const float* x, float* y, int n) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}
```
</details>

<details>
<summary>Hint 5 — Querying SM count safely (code)</summary>

```cpp
cudaDeviceProp prop;
CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
int block = 256;
int grid  = prop.multiProcessorCount * 32;   // a few blocks per SM, independent of n
```
</details>

<details>
<summary>Hint 6 — The full solve (code)</summary>

```cpp
void solve(float a, const float* x, float* y, int n) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int block = 256;
    int grid  = prop.multiProcessorCount * 32;
    saxpy<<<grid, block>>>(a, x, y, n);
}
```
</details>
