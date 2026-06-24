// Reference solution — Exercise 09.
// Register-blocked GEMM: 128x128 block tile, 8x8 per-thread micro-tile, BK=8,
// 256 threads, float4 vectorized global loads, A staged transposed in shared.
// Assumes M, N, K are multiples of the tile sizes (the harness guarantees this);
// a general kernel would add boundary handling on the float4 loads.
#include "cuda_utils.cuh"

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
// Block: 16 x 16 = 256 threads. (BM/TM = 16, BN/TN = 16.)

__global__ void gemm(const float* __restrict__ A, const float* __restrict__ B,
                     float* __restrict__ C, int M, int N, int K) {
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int tx = threadIdx.x;          // 0..15
    const int ty = threadIdx.y;          // 0..15
    const int tid = ty * 16 + tx;        // 0..255

    __shared__ float As[BK][BM];         // A slab, transposed: As[k][row]
    __shared__ float Bs[BK][BN];         // B slab: Bs[k][col]

    // Load-index decomposition (each thread loads exactly one float4 per slab).
    const int innerRowA = tid / 2;       // 0..127  (which slab row of A)
    const int innerColA = (tid % 2) * 4; // 0 or 4  (start col within BK)
    const int innerRowB = tid / 32;      // 0..7    (which slab row of B == k)
    const int innerColB = (tid % 32) * 4;// 0,4,..124 (start col within BN)

    float acc[TM][TN] = {0.f};
    float a_reg[TM];
    float b_reg[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        // --- Load A slab (BM x BK) into As transposed ---
        const float4 av = reinterpret_cast<const float4*>(
            &A[(size_t)(blockRow + innerRowA) * K + (k0 + innerColA)])[0];
        As[innerColA + 0][innerRowA] = av.x;
        As[innerColA + 1][innerRowA] = av.y;
        As[innerColA + 2][innerRowA] = av.z;
        As[innerColA + 3][innerRowA] = av.w;

        // --- Load B slab (BK x BN) into Bs ---
        const float4 bv = reinterpret_cast<const float4*>(
            &B[(size_t)(k0 + innerRowB) * N + (blockCol + innerColB)])[0];
        Bs[innerRowB][innerColB + 0] = bv.x;
        Bs[innerRowB][innerColB + 1] = bv.y;
        Bs[innerRowB][innerColB + 2] = bv.z;
        Bs[innerRowB][innerColB + 3] = bv.w;

        __syncthreads();

        // --- Inner product over this BK slab ---
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) a_reg[i] = As[k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b_reg[j] = Bs[k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }
        __syncthreads();
    }

    // --- Write back the micro-tile (float4 stores, 2 per row) ---
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = blockRow + ty * TM + i;
        const int col = blockCol + tx * TN;
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            float4 out;
            out.x = acc[i][j + 0];
            out.y = acc[i][j + 1];
            out.z = acc[i][j + 2];
            out.w = acc[i][j + 3];
            reinterpret_cast<float4*>(&C[(size_t)row * N + col + j])[0] = out;
        }
    }
}

void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    gemm<<<grid, block>>>(A, B, C, M, N, K);
}
