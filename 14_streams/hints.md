# Hints — Exercise 14

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — Why a single big copy wastes time (concept)</summary>

The naive flow is `copy-in → compute → copy-out`, three phases done strictly one after another. During the copy-in the SMs are idle; during the compute both copy engines are idle. If you cut the array into chunks and put different chunks on different streams, the hardware can run a copy-in, a compute, and a copy-out *at the same time* (on different chunks). Wall-clock time drops toward `max(copy, compute)` instead of `copy + compute`.
</details>

<details>
<summary>Hint 2 — The kernel (code)</summary>

A grid-stride loop so the same kernel works for a full array or a small last chunk:

```cpp
__global__ void map_kernel(const float* x, float* y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        float v = x[i];
        y[i] = sqrtf(v) * v + 1.0f;
    }
}
```
</details>

<details>
<summary>Hint 3 — Creating the streams and device buffers (code)</summary>

Allocate device buffers big enough for the whole array (every chunk has a fixed home), then make the streams:

```cpp
float *d_in, *d_out;
CUDA_CHECK(cudaMalloc(&d_in,  (size_t)n * sizeof(float)));
CUDA_CHECK(cudaMalloc(&d_out, (size_t)n * sizeof(float)));

std::vector<cudaStream_t> streams(nStreams);
for (int s = 0; s < nStreams; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));
```
</details>

<details>
<summary>Hint 4 — Computing each chunk's offset and length (concept)</summary>

Chunk `i` starts at `off = i * chunk` and runs for `len = min(chunk, n - off)` elements (the last chunk is short). The byte count is `len * sizeof(float)`. Pass `h_in + off`, `d_in + off`, etc., and use stream `i % nStreams`. Loop while `off < n`.
</details>

<details>
<summary>Hint 5 — The pipeline loop (code)</summary>

```cpp
for (int off = 0, i = 0; off < n; off += chunk, ++i) {
    int len = min(chunk, n - off);
    size_t b = (size_t)len * sizeof(float);
    cudaStream_t s = streams[i % nStreams];
    CUDA_CHECK(cudaMemcpyAsync(d_in + off, h_in + off, b,
                               cudaMemcpyHostToDevice, s));
    int grid = ceil_div(len, block);
    map_kernel<<<grid, block, 0, s>>>(d_in + off, d_out + off, len);
    CUDA_CHECK(cudaMemcpyAsync(h_out + off, d_out + off, b,
                               cudaMemcpyDeviceToHost, s));
}
```

Issuing all three ops per chunk back-to-back lets the scheduler overlap *across* streams; the in-order rule only constrains ops within one stream.
</details>

<details>
<summary>Hint 6 — Finishing cleanly (code)</summary>

```cpp
CUDA_CHECK(cudaDeviceSynchronize());            // all streams done
for (auto s : streams) CUDA_CHECK(cudaStreamDestroy(s));
CUDA_CHECK(cudaFree(d_in));
CUDA_CHECK(cudaFree(d_out));
```

If you forget the sync, `h_out` may be read by the harness before the D2H copies land — correctness fails intermittently.
</details>
