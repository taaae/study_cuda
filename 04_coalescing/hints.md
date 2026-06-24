# Hints — Exercise 04

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — What "coalesced" actually means (no code)</summary>

A warp is 32 threads that issue memory together. The hardware fetches memory in aligned 128-byte chunks (= 32 consecutive floats). If the 32 threads of a warp ask for 32 *consecutive* floats, that's **one** fetch — full speed. If they ask for floats `n` apart (a matrix column), it can take up to 32 separate fetches, each wasting 31/32 of what it brought back. The variable that changes fastest across a warp is `threadIdx.x`, so whatever array you want fast must be indexed so adjacent `threadIdx.x` lands on adjacent addresses.
</details>

<details>
<summary>Hint 2 — Why you can only coalesce one side (concept)</summary>

The read pattern of a transpose is the transpose of its write pattern. If reads are contiguous, writes are strided, and vice-versa — you can't have both with plain global memory. This exercise picks the **reads**. (Exercise 05 uses shared memory to get both.) The baseline you must beat picks the worse option, so just coalescing the reads wins.
</details>

<details>
<summary>Hint 3 — Which index gets threadIdx.x (concept)</summary>

You want `in` read coalesced. In `in[row*n + col]`, the contiguous dimension is `col`. So `col` must be the one driven by `threadIdx.x`:
- `col = blockIdx.x * blockDim.x + threadIdx.x`
- `row = blockIdx.y * blockDim.y + threadIdx.y`

Then `in[row*n + col]` is coalesced, and you write the transposed slot `out[col*n + row]`.
</details>

<details>
<summary>Hint 4 — The kernel body (code)</summary>

```cpp
__global__ void transpose(const float* in, float* out, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // coalesced read of in
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        out[col * n + row] = in[row * n + col];
    }
}
```
</details>

<details>
<summary>Hint 5 — The 2-D launch config (code)</summary>

```cpp
dim3 block(32, 8);
dim3 grid(ceil_div(n, block.x), ceil_div(n, block.y));
```
A `block.x` of 32 means each warp's `threadIdx.x` spans a full 32-float (128-byte) segment — exactly one coalesced transaction per warp on the read side.
</details>

<details>
<summary>Hint 6 — The full solve (code)</summary>

```cpp
void solve(const float* in, float* out, int n) {
    dim3 block(32, 8);
    dim3 grid(ceil_div(n, block.x), ceil_div(n, block.y));
    transpose<<<grid, block>>>(in, out, n);
}
```
</details>
