# Hints — Exercise 09

Reveal one at a time. Each summary says what the hint gives away.

<details>
<summary>Hint 1 — Why one-output-per-thread stalls (no code)</summary>

Exercise 08 reads two shared-memory values for every single multiply-add, so it just trades the global-memory wall for a shared-memory wall. Give each thread a small block of outputs (a `TM×TN` micro-tile) and the same shared reads feed many more FLOPs: `TM` values of `A` and `TN` of `B` drive `TM×TN` multiply-adds. Those `TM×TN` running sums live in registers.
</details>

<details>
<summary>Hint 2 — The register accumulator (concept)</summary>

Declare `float acc[TM][TN] = {0}` per thread — it lives in registers. The K-loop accumulates an **outer product** each step: stage `a_reg[TM]` and `b_reg[TN]` from shared memory for the current `k`, then `acc[i][j] += a_reg[i] * b_reg[j]`. With `TM=TN=8` that's 64 FMAs per `k` from only 16 shared reads.
</details>

<details>
<summary>Hint 3 — float4 loads and the transposed A slab (concept)</summary>

256 threads load a `128×8` slab of `A` and a `8×128` slab of `B` each K-step. As a `float4`, each thread moves exactly one 16-byte chunk: 1024 floats / 4 = 256. Store the `A` slab **transposed** (`As[k][row]`) so the inner loop reads `As[k][...]` with stride 1 across the micro-tile rows. `B` is stored straight (`Bs[k][col]`). Use `reinterpret_cast<const float4*>(ptr)[idx]`.
</details>

<details>
<summary>Hint 4 — Load index decomposition (code)</summary>

```cpp
int tid = ty * 16 + tx;             // 0..255
int innerRowA = tid / 2;            // 0..127
int innerColA = (tid % 2) * 4;      // 0 or 4
int innerRowB = tid / 32;           // 0..7  (== k within the slab)
int innerColB = (tid % 32) * 4;     // 0,4,..,124
// A chunk: &A[(blockRow+innerRowA)*K + (k0+innerColA)]  -> store transposed
// B chunk: &B[(k0+innerRowB)*N + (blockCol+innerColB)]  -> store straight
```
</details>

<details>
<summary>Hint 5 — The K-loop and inner product (code)</summary>

```cpp
__shared__ float As[BK][BM], Bs[BK][BN];
float acc[TM][TN] = {0.f}, a_reg[TM], b_reg[TN];

for (int k0 = 0; k0 < K; k0 += BK) {
    float4 av = reinterpret_cast<const float4*>(
        &A[(size_t)(blockRow+innerRowA)*K + (k0+innerColA)])[0];
    As[innerColA+0][innerRowA]=av.x; As[innerColA+1][innerRowA]=av.y;
    As[innerColA+2][innerRowA]=av.z; As[innerColA+3][innerRowA]=av.w;

    float4 bv = reinterpret_cast<const float4*>(
        &B[(size_t)(k0+innerRowB)*N + (blockCol+innerColB)])[0];
    Bs[innerRowB][innerColB+0]=bv.x; Bs[innerRowB][innerColB+1]=bv.y;
    Bs[innerRowB][innerColB+2]=bv.z; Bs[innerRowB][innerColB+3]=bv.w;
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < BK; ++k) {
        #pragma unroll
        for (int i = 0; i < TM; ++i) a_reg[i] = As[k][ty*TM + i];
        #pragma unroll
        for (int j = 0; j < TN; ++j) b_reg[j] = Bs[k][tx*TN + j];
        #pragma unroll
        for (int i = 0; i < TM; ++i)
            #pragma unroll
            for (int j = 0; j < TN; ++j) acc[i][j] += a_reg[i]*b_reg[j];
    }
    __syncthreads();
}
```
</details>

<details>
<summary>Hint 6 — Write-back and solve (code)</summary>

```cpp
#pragma unroll
for (int i = 0; i < TM; ++i) {
    int row = blockRow + ty*TM + i;
    int col = blockCol + tx*TN;
    #pragma unroll
    for (int j = 0; j < TN; j += 4) {
        float4 out = {acc[i][j], acc[i][j+1], acc[i][j+2], acc[i][j+3]};
        reinterpret_cast<float4*>(&C[(size_t)row*N + col + j])[0] = out;
    }
}
```
```cpp
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    gemm<<<grid, block>>>(A, B, C, M, N, K);
}
```
</details>
