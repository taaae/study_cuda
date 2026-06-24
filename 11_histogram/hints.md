# Hints — Exercise 11

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — Why naive is slow, and the cure (concept)</summary>

With 256 bins and millions of elements, a single global counter is hit by thousands of threads at once; the hardware serializes those atomics. Fix: each block keeps its own 256-bin histogram in shared memory, counts into *that*, and only at the end folds 256 values into the global histogram. Far fewer global atomics, and shared-memory atomics are cheap.
</details>

<details>
<summary>Hint 2 — The five steps inside the kernel (concept)</summary>

1. declare a `__shared__ unsigned int sHist[256]`,
2. cooperatively zero it,
3. `__syncthreads()`,
4. grid-stride over the input doing `atomicAdd(&sHist[v], 1u)`,
5. `__syncthreads()`, then each thread merges some bins with `atomicAdd(&hist[b], sHist[b])`.

Steps 2 and 5 are both "256 bins shared among `blockDim.x` threads" loops.
</details>

<details>
<summary>Hint 3 — Zeroing and merging the shared histogram (code)</summary>

```cpp
__shared__ unsigned int sHist[256];
for (int b = threadIdx.x; b < 256; b += blockDim.x) sHist[b] = 0;
__syncthreads();
// ... fill sHist ...
__syncthreads();
for (int b = threadIdx.x; b < 256; b += blockDim.x)
    atomicAdd(&hist[b], sHist[b]);
```
</details>

<details>
<summary>Hint 4 — The counting loop (code)</summary>

```cpp
for (int i = blockIdx.x * blockDim.x + threadIdx.x;
     i < n; i += blockDim.x * gridDim.x) {
    atomicAdd(&sHist[data[i]], 1u);
}
```
The grid-stride form means a fixed grid size (e.g. 1024 blocks) handles any `n`.
</details>

<details>
<summary>Hint 5 — The launch (code)</summary>

```cpp
void solve(const unsigned char* data, unsigned int* hist, int n) {
    hist_privatized<<<1024, 256>>>(data, hist, n);
}
```
The harness already zeroed `hist`, so you only add into it.
</details>
