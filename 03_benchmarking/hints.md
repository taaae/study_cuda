# Hints — Exercise 03

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — Why you can't just use a CPU timer (no code)</summary>

A kernel launch is **asynchronous**: `kernel<<<...>>>(...)` returns to the CPU almost instantly, long before the GPU finishes. If you read a CPU clock right after the launch you measure the *launch overhead*, not the kernel. CUDA events are timestamps recorded inside the GPU stream, so they capture real device execution time. That's why the task forces you to use them.
</details>

<details>
<summary>Hint 2 — Warmup and best-of-N (concept)</summary>

The first run of any kernel is unusually slow (context setup, cold caches, clocks not boosted). So launch once and ignore that time — that's the **warmup**. Then run `iters` times and keep the **minimum**, not the average: the fastest run is the one least disturbed by scheduling noise on a shared GPU, and best reflects the kernel's true cost.
</details>

<details>
<summary>Hint 3 — The copy kernel (code)</summary>

It's the same grid-stride shape as exercise 02, just a copy:

```cpp
__global__ void copy_kernel(const float* in, float* out, int n) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) out[i] = in[i];
}
```
</details>

<details>
<summary>Hint 4 — The event timing dance (code, one iteration)</summary>

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
copy_kernel<<<grid, block>>>(in, out, n);
cudaEventRecord(stop);
cudaEventSynchronize(stop);          // wait for the GPU to reach 'stop'

float ms = 0.f;
cudaEventElapsedTime(&ms, start, stop);   // milliseconds between the events
```
Remember to `cudaEventDestroy` both events when you're done.
</details>

<details>
<summary>Hint 5 — Picking the grid (concept)</summary>

Because it's a grid-stride loop, the grid doesn't have to equal `ceil_div(n, block)`. `ceil_div(n, block)` works fine, or size it to the SM count like exercise 02. Either clears the bandwidth threshold; just make sure every element gets copied.
</details>

<details>
<summary>Hint 6 — The full benchmark_copy (code)</summary>

```cpp
float benchmark_copy(const float* in, float* out, int n, int iters) {
    int block = 256;
    int grid  = ceil_div(n, block);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warmup (discarded)
    copy_kernel<<<grid, block>>>(in, out, n);
    cudaDeviceSynchronize();

    float best = 1e30f;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start);
        copy_kernel<<<grid, block>>>(in, out, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        if (ms < best) best = ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return best;
}
```
</details>
