#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <torch/extension.h>

#include "sparse_kernels.h"
#include "sparse_layout.cuh"

#define USE_VERY_OPTIMIZED_KERNEL

/**
 * Compute sparse matrix multiplication with SDD mode.
 * 
 * It multiplies a dense matrix with other dense matrix and create a new sparse
 * matrix through corresponding sparse layout.
 * 
 * Blocks               : (Sparse Blocks, Total Batches)
 * Threads per Block    : 256
 */
#ifdef USE_32x8_TILING
template <bool trans_a, bool trans_b>
__global__ void __launch_bounds__(256, 8) sparse_matmul_sdd_32x32x8_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
          float* __restrict__ matrix_c,
    sparse_layout layout, uint num_blocks,
    uint size_m, uint size_n, uint size_k
) {
    /******** Define shared memory ********/
    constexpr int TILE_SIZE = 32 * 8;
    constexpr int PADDING = 8;

    __shared__ float tile_a[2][(TILE_SIZE + PADDING + 32 - 1) / 32 * 32];
    __shared__ float tile_b[2][(TILE_SIZE + PADDING + 32 - 1) / 32 * 32];

    /******** Fetch sparse block descriptor ********/
    auto block = layout.get(blockIdx.x);
    int m = block.row() * 32;
    int n = block.col() * 32;

    /******** Define accumulator and warp informations ********/
    float accum[2][2] = { { 0.0f, 0.0f }, { 0.0f, 0.0f } };

    int tid = threadIdx.x;

    /******** Prefetch first tiles ********/
    int load_a = blockIdx.y * size_m * size_k;
    int load_b = blockIdx.y * size_k * size_n;
    
    float buffer_a = matrix_a[
        load_a
        + ((trans_a ? 0 : m) + (trans_a ? tid / 32 : tid / 8)) * (trans_a ? size_m : size_k)
        + ((trans_a ? m : 0) + (trans_a ? tid % 32 : tid % 8))
    ];
    float buffer_b = matrix_b[
        load_b
        + ((trans_b ? n : 0) + (trans_b ? tid / 8 : tid / 32)) * (trans_b ? size_k : size_m)
        + ((trans_b ? 0 : n) + (trans_b ? tid % 8 : tid % 32))
    ];

    /******** Iterate over k-dim ********/
    #pragma unroll 1
    for (int k = 0; k < size_k; k += 8) {
        int page = k / 8 % 2;
        int next_k = k + 8;

        /******** Commit the prefetched buffers to the shared memory ********/
        tile_a[page][(trans_a ? tid % 32 : tid / 8) * 8 + (trans_a ? tid / 32 : tid % 8) + (trans_a ? tid % 32 / 4 : tid / 32)] = buffer_a;
        tile_b[page][(trans_b ? tid / 8 : tid % 32) * 8 + (trans_b ? tid % 8 : tid / 32) + (trans_b ? tid / 32 : tid % 32 / 4)] = buffer_b;
        __syncthreads();

        /******** Prefetch next tiles if available ********/
        if (next_k < size_k) {
            buffer_a = matrix_a[
                load_a
                + ((trans_a ? next_k : m) + (trans_a ? tid / 32 : tid / 8)) * (trans_a ? size_m : size_k)
                + ((trans_a ? m : next_k) + (trans_a ? tid % 32 : tid % 8))
            ];
            buffer_b = matrix_b[
                load_b
                + ((trans_b ? n : next_k) + (trans_b ? tid / 8 : tid / 32)) * (trans_b ? size_k : size_m)
                + ((trans_b ? next_k : n) + (trans_b ? tid % 8 : tid % 32))
            ];
        }

        /******** Accumulate tile matmul by using register file ********/
        #pragma unroll
        for (int i = 0; i < 8; ++ i) {
            float local_a[2], local_b[2];

            local_a[0] = tile_a[page][(tid / 16 * 2 + 0) * 8 + i + (tid / 32)];
            local_a[1] = tile_a[page][(tid / 16 * 2 + 1) * 8 + i + (tid / 32)];
            local_b[0] = tile_b[page][(tid % 16 * 2 + 0) * 8 + i + (tid % 16 / 2)];
            local_b[1] = tile_b[page][(tid % 16 * 2 + 1) * 8 + i + (tid % 16 / 2)];

            accum[0][0] += local_a[0] * local_b[0];
            accum[0][1] += local_a[0] * local_b[1];
            accum[1][0] += local_a[1] * local_b[0];
            accum[1][1] += local_a[1] * local_b[1];
        }
    }

    /******** Apply accumulation to output matrix ********/
    int load_c = (blockIdx.y * num_blocks + block.idx()) * 32 * 32;

    matrix_c[load_c + (tid / 16 * 2 + 0) * 32 + (tid % 16 * 2 + 0)] = accum[0][0];
    matrix_c[load_c + (tid / 16 * 2 + 0) * 32 + (tid % 16 * 2 + 1)] = accum[0][1];
    matrix_c[load_c + (tid / 16 * 2 + 1) * 32 + (tid % 16 * 2 + 0)] = accum[1][0];
    matrix_c[load_c + (tid / 16 * 2 + 1) * 32 + (tid % 16 * 2 + 1)] = accum[1][1];
}
#endif

#ifdef USE_8x32_TILING
template <bool trans_a, bool trans_b>
__global__ void __launch_bounds__(256, 8) sparse_matmul_sdd_32x32x8_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
          float* __restrict__ matrix_c,
    sparse_layout layout, uint num_blocks,
    uint size_m, uint size_n, uint size_k
) {
    /******** Define shared memory ********/
    constexpr int TILE_SIZE = 8 * 32;
    constexpr int PADDING = 8;

    __shared__ float tile_a[2][(TILE_SIZE + PADDING + 32 - 1) / 32 * 32];
    __shared__ float tile_b[2][(TILE_SIZE + PADDING + 32 - 1) / 32 * 32];

    /******** Fetch sparse block descriptor ********/
    auto block = layout.get(blockIdx.x);
    int m = block.row() * 32;
    int n = block.col() * 32;

    /******** Define accumulator and warp informations ********/
    float accum[2][2] = { { 0.0f, 0.0f }, { 0.0f, 0.0f } };

    int tid = threadIdx.x;

    /******** Prefetch first tiles ********/
    int load_a = blockIdx.y * size_m * size_k;
    int load_b = blockIdx.y * size_k * size_n;
    
    float buffer_a = matrix_a[
        load_a
        + ((trans_a ? 0 : m) + (trans_a ? tid / 32 : tid / 8 % 4 * 8 + tid / 32)) * (trans_a ? size_m : size_k)
        + ((trans_a ? m : 0) + (trans_a ? tid % 32 : tid % 8))
    ];
    float buffer_b = matrix_b[
        load_b
        + ((trans_b ? n : 0) + (trans_b ? tid / 8 % 4 * 8 + tid / 32 : tid / 32)) * (trans_b ? size_k : size_m)
        + ((trans_b ? 0 : n) + (trans_b ? tid % 8 : tid % 32))
    ];

    /******** Iterate over k-dim ********/
    #pragma unroll 1
    for (int k = 0; k < size_k; k += 8) {
        int page = k / 8 % 2;
        int next_k = k + 8;

        /******** Commit the prefetched buffers to the shared memory ********/
        tile_a[page][(trans_a ? tid / 32 : tid % 8) * (32 + 1) + (trans_a ? tid % 32 : tid / 8 % 4 * 8 + tid / 32)] = buffer_a;
        tile_b[page][(trans_b ? tid % 8 : tid / 32) * (32 + 1) + (trans_b ? tid / 8 % 4 * 8 + tid / 32 : tid % 32)] = buffer_b;
        __syncthreads();

        /******** Prefetch next tiles if available ********/
        if (next_k < size_k) {
            buffer_a = matrix_a[
                load_a
                + ((trans_a ? next_k : m) + (trans_a ? tid / 32 : tid / 8 % 4 * 8 + tid / 32)) * (trans_a ? size_m : size_k)
                + ((trans_a ? m : next_k) + (trans_a ? tid % 32 : tid % 8))
            ];
            buffer_b = matrix_b[
                load_b
                + ((trans_b ? n : next_k) + (trans_b ? tid / 8 % 4 * 8 + tid / 32 : tid / 32)) * (trans_b ? size_k : size_m)
                + ((trans_b ? next_k : n) + (trans_b ? tid % 8 : tid % 32))
            ];
        }

        /******** Accumulate tile matmul by using register file ********/
        #pragma unroll
        for (int i = 0; i < 8; ++ i) {
            float local_a[2], local_b[2];

            local_a[0] = tile_a[page][i * (32 + 1) + (tid / 16 * 2 + 0)];
            local_a[1] = tile_a[page][i * (32 + 1) + (tid / 16 * 2 + 1)];
            local_b[0] = tile_b[page][i * (32 + 1) + (tid % 16 * 2 + 0)];
            local_b[1] = tile_b[page][i * (32 + 1) + (tid % 16 * 2 + 1)];

            accum[0][0] += local_a[0] * local_b[0];
            accum[0][1] += local_a[0] * local_b[1];
            accum[1][0] += local_a[1] * local_b[0];
            accum[1][1] += local_a[1] * local_b[1];
        }
    }

    /******** Apply accumulation to output matrix ********/
    int load_c = (blockIdx.y * num_blocks + block.idx()) * 32 * 32;

    matrix_c[load_c + (tid / 16 * 2 + 0) * 32 + (tid % 16 * 2 + 0)] = accum[0][0];
    matrix_c[load_c + (tid / 16 * 2 + 0) * 32 + (tid % 16 * 2 + 1)] = accum[0][1];
    matrix_c[load_c + (tid / 16 * 2 + 1) * 32 + (tid % 16 * 2 + 0)] = accum[1][0];
    matrix_c[load_c + (tid / 16 * 2 + 1) * 32 + (tid % 16 * 2 + 1)] = accum[1][1];
}
#endif

#ifdef USE_32x32_TILING
template <bool trans_a, bool trans_b>
__global__ void __launch_bounds__(256, 3) sparse_matmul_sdd_32x32x8_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
          float* __restrict__ matrix_c,
    sparse_layout layout, uint num_blocks,
    uint size_m, uint size_n, uint size_k
    //bool trans_a, bool trans_b
) {
    /******** Define shared memory ********/
    __shared__ float tile_a[2 * 32 * (32 + 1)];
    __shared__ float tile_b[2 * 32 * (32 + 1)];

    /******** Fetch sparse block descriptor ********/
    auto block = layout.get(blockIdx.x);
    int m = block.row() * 32;
    int n = block.col() * 32;

    /******** Define accumulator and warp informations ********/
    float accum[2][2] = { { 0.0f, 0.0f }, { 0.0f, 0.0f } };

    int tid = threadIdx.x;

    /******** Prefetch first tiles ********/
    int load_a = blockIdx.y * size_m * size_k;
    int load_b = blockIdx.y * size_k * size_n;
    
    float buffer_a[4], buffer_b[4];

    #pragma unroll
    for (int i = 0; i < 4; ++ i) {
        buffer_a[i] = matrix_a[
            load_a
            + ((trans_a ? 0 : m) + (tid / 32 + i * 8)) * (trans_a ? size_m : size_k)
            + ((trans_a ? m : 0) + (tid % 32))
        ];
        buffer_b[i] = matrix_b[
            load_b
            + ((trans_b ? n : 0) + (tid / 32 + i * 8)) * (trans_b ? size_k : size_m)
            + ((trans_b ? 0 : n) + (tid % 32))
        ];
    }

    /******** Iterate over k-dim ********/
    #pragma unroll 1
    for (int k = 0; k < size_k; k += 32) {
        int page = k / 32 % 2;
        int next_k = k + 32;

        /******** Commit the prefetched buffers to the shared memory ********/
        #pragma unroll
        for (int i = 0; i < 4; ++ i) {
            tile_a[page * 32 * (32 + 1) + (trans_a ? tid % 32 : tid / 32 + i * 8) * (32 + 1) + (trans_a ? tid / 32 + i * 8 : tid % 32)] = buffer_a[i];
            tile_b[page * 32 * (32 + 1) + (trans_b ? tid / 32 + i * 8 : tid % 32) * (32 + 1) + (trans_b ? tid % 32 : tid / 32 + i * 8)] = buffer_b[i];
        }
        __syncthreads();

        /******** Prefetch next tiles if available ********/
        if (next_k < size_k) {
            #pragma unroll
            for (int i = 0; i < 4; ++ i) {
                buffer_a[i] = matrix_a[
                    load_a
                    + ((trans_a ? next_k : m) + (tid / 32 + i * 8)) * (trans_a ? size_m : size_k)
                    + ((trans_a ? m : next_k) + (tid % 32))
                ];
                buffer_b[i] = matrix_b[
                    load_b
                    + ((trans_b ? n : next_k) + (tid / 32 + i * 8)) * (trans_b ? size_k : size_m)
                    + ((trans_b ? next_k : n) + (tid % 32))
                ];
            }
        }

        /******** Accumulate tile matmul by using register file ********/
        #pragma unroll
        for (int i = 0; i < 32; ++ i) {
            float local_a[2], local_b[2];

            local_a[0] = tile_a[page * 32 * (32 + 1) + (tid / 16 * 2 + 0) * (32 + 1) + i];
            local_a[1] = tile_a[page * 32 * (32 + 1) + (tid / 16 * 2 + 1) * (32 + 1) + i];
            local_b[0] = tile_b[page * 32 * (32 + 1) + (tid % 16 * 2 + 0) * (32 + 1) + i];
            local_b[1] = tile_b[page * 32 * (32 + 1) + (tid % 16 * 2 + 1) * (32 + 1) + i];

            accum[0][0] += local_a[0] * local_b[0];
            accum[0][1] += local_a[0] * local_b[1];
            accum[1][0] += local_a[1] * local_b[0];
            accum[1][1] += local_a[1] * local_b[1];
        }
    }

    /******** Apply accumulation to output matrix ********/
    int load_c = (blockIdx.y * num_blocks + block.idx()) * 32 * 32;

    matrix_c[load_c + (tid / 16 * 2 + 0) * 32 + (tid % 16 * 2 + 0)] = accum[0][0];
    matrix_c[load_c + (tid / 16 * 2 + 0) * 32 + (tid % 16 * 2 + 1)] = accum[0][1];
    matrix_c[load_c + (tid / 16 * 2 + 1) * 32 + (tid % 16 * 2 + 0)] = accum[1][0];
    matrix_c[load_c + (tid / 16 * 2 + 1) * 32 + (tid % 16 * 2 + 1)] = accum[1][1];
}
#endif

#ifdef USE_32x32_TILING_NO_BUFFERING
template <bool trans_a, bool trans_b>
__global__ void sparse_matmul_sdd_32x32x8_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
          float* __restrict__ matrix_c,
    sparse_layout layout, uint num_blocks,
    uint size_m, uint size_n, uint size_k
    //bool trans_a, bool trans_b
) {
    /******** Define shared memory ********/
    __shared__ float tile_a[32 * (32 + 1)];
    __shared__ float tile_b[32 * (32 + 1)];

    /******** Fetch sparse block descriptor ********/
    auto block = layout.get(blockIdx.x);
    int m = block.row() * 32;
    int n = block.col() * 32;

    /******** Define accumulator and warp informations ********/
    float accum[2][2] = { { 0.0f, 0.0f }, { 0.0f, 0.0f } };

    int tid = threadIdx.x;

    /******** Prefetch first tiles ********/
    int load_a = blockIdx.y * size_m * size_k;
    int load_b = blockIdx.y * size_k * size_n;
    
    float buffer_a[4], buffer_b[4];

    #pragma unroll
    for (int i = 0; i < 4; ++ i) {
        buffer_a[i] = matrix_a[
            load_a
            + ((trans_a ? 0 : m) + (tid / 32 + i * 8)) * (trans_a ? size_m : size_k)
            + ((trans_a ? m : 0) + (tid % 32))
        ];
        buffer_b[i] = matrix_b[
            load_b
            + ((trans_b ? n : 0) + (tid / 32 + i * 8)) * (trans_b ? size_k : size_m)
            + ((trans_b ? 0 : n) + (tid % 32))
        ];
    }

    /******** Iterate over k-dim ********/
    #pragma unroll 1
    for (int k = 0; k < size_k; k += 32) {
        int next_k = k + 32;

        /******** Commit the prefetched buffers to the shared memory ********/
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 4; ++ i) {
            tile_a[(trans_a ? tid % 32 : tid / 32 + i * 8) * (32 + 1) + (trans_a ? tid / 32 + i * 8 : tid % 32)] = buffer_a[i];
            tile_b[(trans_b ? tid / 32 + i * 8 : tid % 32) * (32 + 1) + (trans_b ? tid % 32 : tid / 32 + i * 8)] = buffer_b[i];
        }
        __syncthreads();

        /******** Prefetch next tiles if available ********/
        if (next_k < size_k) {
            #pragma unroll
            for (int i = 0; i < 4; ++ i) {
                buffer_a[i] = matrix_a[
                    load_a
                    + ((trans_a ? next_k : m) + (tid / 32 + i * 8)) * (trans_a ? size_m : size_k)
                    + ((trans_a ? m : next_k) + (tid % 32))
                ];
                buffer_b[i] = matrix_b[
                    load_b
                    + ((trans_b ? n : next_k) + (tid / 32 + i * 8)) * (trans_b ? size_k : size_m)
                    + ((trans_b ? next_k : n) + (tid % 32))
                ];
            }
        }

        /******** Accumulate tile matmul by using register file ********/
        #pragma unroll
        for (int i = 0; i < 32; ++ i) {
            float local_a[2], local_b[2];

            local_a[0] = tile_a[(tid / 16 * 2 + 0) * (32 + 1) + i];
            local_a[1] = tile_a[(tid / 16 * 2 + 1) * (32 + 1) + i];
            local_b[0] = tile_b[(tid % 16 * 2 + 0) * (32 + 1) + i];
            local_b[1] = tile_b[(tid % 16 * 2 + 1) * (32 + 1) + i];

            accum[0][0] += local_a[0] * local_b[0];
            accum[0][1] += local_a[0] * local_b[1];
            accum[1][0] += local_a[1] * local_b[0];
            accum[1][1] += local_a[1] * local_b[1];
        }
    }

    /******** Apply accumulation to output matrix ********/
    int load_c = (blockIdx.y * num_blocks + block.idx()) * 32 * 32;

    matrix_c[load_c + (tid / 16 * 2 + 0) * 32 + (tid % 16 * 2 + 0)] = accum[0][0];
    matrix_c[load_c + (tid / 16 * 2 + 0) * 32 + (tid % 16 * 2 + 1)] = accum[0][1];
    matrix_c[load_c + (tid / 16 * 2 + 1) * 32 + (tid % 16 * 2 + 0)] = accum[1][0];
    matrix_c[load_c + (tid / 16 * 2 + 1) * 32 + (tid % 16 * 2 + 1)] = accum[1][1];
}
#endif

#ifdef USE_32x32_TILING_FUSED_COPY
template <bool trans_a, bool trans_b>
__global__ void sparse_matmul_sdd_32x32x8_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
          float* __restrict__ matrix_c,
    sparse_layout layout, uint num_blocks,
    uint size_m, uint size_n, uint size_k
    //bool trans_a, bool trans_b
) {
    /******** Define shared memory ********/
    __shared__ float tile_a[32 * (32 + 1)];
    __shared__ float tile_b[32 * (32 + 1)];

    /******** Fetch sparse block descriptor ********/
    auto block = layout.get(blockIdx.x);
    int m = block.row() * 32;
    int n = block.col() * 32;

    /******** Define accumulator and warp informations ********/
    float accum[2][2] = { { 0.0f, 0.0f }, { 0.0f, 0.0f } };

    int tid = threadIdx.x;
    int i = tid / 8;
    int j = tid % 8 * 4;

    /******** Prefetch first tiles ********/
    int load_a = blockIdx.y * size_m * size_k;
    int load_b = blockIdx.y * size_k * size_n;
    
    float4 buffer_a, buffer_b;

    buffer_a = *(float4 *) &matrix_a[
        load_a
        + ((trans_a ? 0 : m) + i) * (trans_a ? size_m : size_k)
        + ((trans_a ? m : 0) + j)
    ];
    buffer_b = *(float4 *) &matrix_b[
        load_b
        + ((trans_a ? n : 0) + i) * (trans_a ? size_k : size_n)
        + ((trans_a ? 0 : n) + j)
    ];

    /******** Iterate over k-dim ********/
    #pragma unroll 1
    for (int k = 0; k < size_k; k += 32) {
        int next_k = k + 32;

        /******** Commit the prefetched buffers to the shared memory ********/
        __syncthreads();
        tile_a[(trans_a ? j + 0 : i) * (32 + 1) + (trans_a ? i : j + 0)] = buffer_a.x;
        tile_a[(trans_a ? j + 1 : i) * (32 + 1) + (trans_a ? i : j + 1)] = buffer_a.y;
        tile_a[(trans_a ? j + 2 : i) * (32 + 1) + (trans_a ? i : j + 2)] = buffer_a.z;
        tile_a[(trans_a ? j + 3 : i) * (32 + 1) + (trans_a ? i : j + 3)] = buffer_a.w;
        tile_b[(trans_a ? i : j + 0) * (32 + 1) + (trans_a ? j + 0 : i)] = buffer_b.x;
        tile_b[(trans_a ? i : j + 1) * (32 + 1) + (trans_a ? j + 1 : i)] = buffer_b.y;
        tile_b[(trans_a ? i : j + 2) * (32 + 1) + (trans_a ? j + 2 : i)] = buffer_b.z;
        tile_b[(trans_a ? i : j + 3) * (32 + 1) + (trans_a ? j + 3 : i)] = buffer_b.w;
        __syncthreads();

        /******** Prefetch next tiles if available ********/
        if (next_k < size_k) {
            buffer_a = *(float4 *) &matrix_a[
                load_a
                + ((trans_a ? next_k : m) + tid / 8) * (trans_a ? size_m : size_k)
                + ((trans_a ? m : next_k) + tid % 8 * 4)
            ];
            buffer_b = *(float4 *) &matrix_b[
                load_b
                + ((trans_a ? n : next_k) + tid / 8) * (trans_a ? size_k : size_n)
                + ((trans_a ? next_k : n) + tid % 8 * 4)
            ];
        }

        /******** Accumulate tile matmul by using register file ********/
        #pragma unroll
        for (int i = 0; i < 32; ++ i) {
            float local_a[2], local_b[2];

            local_a[0] = tile_a[(tid / 16 * 2 + 0) * (32 + 1) + i];
            local_a[1] = tile_a[(tid / 16 * 2 + 1) * (32 + 1) + i];
            local_b[0] = tile_b[(tid % 16 * 2 + 0) * (32 + 1) + i];
            local_b[1] = tile_b[(tid % 16 * 2 + 1) * (32 + 1) + i];

            accum[0][0] += local_a[0] * local_b[0];
            accum[0][1] += local_a[0] * local_b[1];
            accum[1][0] += local_a[1] * local_b[0];
            accum[1][1] += local_a[1] * local_b[1];
        }
    }

    /******** Apply accumulation to output matrix ********/
    int load_c = (blockIdx.y * num_blocks + block.idx()) * 32 * 32;

    matrix_c[load_c + (tid / 16 * 2 + 0) * 32 + (tid % 16 * 2 + 0)] = accum[0][0];
    matrix_c[load_c + (tid / 16 * 2 + 0) * 32 + (tid % 16 * 2 + 1)] = accum[0][1];
    matrix_c[load_c + (tid / 16 * 2 + 1) * 32 + (tid % 16 * 2 + 0)] = accum[1][0];
    matrix_c[load_c + (tid / 16 * 2 + 1) * 32 + (tid % 16 * 2 + 1)] = accum[1][1];
}
#endif

#ifdef USE_VERY_OPTIMIZED_KERNEL

template <bool tr_a, bool tr_b>
__global__ void sparse_matmul_sdd_32x32x32_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
          float* __restrict__ matrix_c,
    sparse_layout layout, int num_blocks,
    int size_m, int size_n, int size_k
) {
    float accum[2][2] = { 0 };

    float4 buffer_a, buffer_b;
    __shared__ float shared_a[32 * 33], shared_b[32 * 33];

    // Fetch current block and get corresponding row and column positions.
    auto block = layout.get(blockIdx.x);
    int m = block.row() * 32;
    int n = block.col() * 32;

    // Get an offset of each matrix and calculate mapping indices.
    int offset_a = blockIdx.y * size_m * size_k + (tr_a ? m : m * size_k);
    int offset_b = blockIdx.y * size_k * size_n + (tr_b ? n * size_k : n);
    int offset_c = (blockIdx.y * num_blocks + block.idx()) * 32 * 32;

    int stride_a = tr_a ? size_m : size_k;
    int stride_b = tr_b ? size_k : size_n;

    int p = threadIdx.x / 8;
    int q = threadIdx.x % 8 * 4;
    int r = threadIdx.x / 16 * 2;
    int s = threadIdx.x % 16 * 2;

    // Prefetch first tiles from matrices in global memory.
    buffer_a = *(float4 *) &matrix_a[offset_a + p * stride_a + q]; //(tr_a ? ((0 + p) * size_m + (m + q)) : ((m + p) * size_k + (0 + q)))];
    buffer_b = *(float4 *) &matrix_b[offset_b + p * stride_b + q]; //(tr_b ? ((n + p) * size_k + (0 + q)) : ((0 + p) * size_n + (n + q)))];

    #pragma unroll 1
    for (int k = 0; k < size_k; k += 32) {
        // Commit the prefetched tiles to the shared memory storage.
        __syncthreads();
        shared_a[tr_a ? ((q + 0) * 33 + p) : (p * 33 + (q + 0))] = buffer_a.x;
        shared_a[tr_a ? ((q + 1) * 33 + p) : (p * 33 + (q + 1))] = buffer_a.y;
        shared_a[tr_a ? ((q + 2) * 33 + p) : (p * 33 + (q + 2))] = buffer_a.z;
        shared_a[tr_a ? ((q + 3) * 33 + p) : (p * 33 + (q + 3))] = buffer_a.w;

        shared_b[tr_b ? (p * 33 + (q + 0)) : ((q + 0) * 33 + p)] = buffer_b.x;
        shared_b[tr_b ? (p * 33 + (q + 1)) : ((q + 1) * 33 + p)] = buffer_b.y;
        shared_b[tr_b ? (p * 33 + (q + 2)) : ((q + 2) * 33 + p)] = buffer_b.z;
        shared_b[tr_b ? (p * 33 + (q + 3)) : ((q + 3) * 33 + p)] = buffer_b.w;
        __syncthreads();

        // Prefetch next tiles from matrices in global memory.
        if (k + 32 < size_k) {
            offset_a += 32 * (tr_a ? size_m : 1);
            offset_b += 32 * (tr_b ? 1 : size_n);

            buffer_a = *(float4 *) &matrix_a[offset_a + p * stride_a + q]; //(tr_a ? ((k + p) * size_m + (m + q)) : ((m + p) * size_k + (k + q)))];
            buffer_b = *(float4 *) &matrix_b[offset_b + p * stride_b + q]; //(tr_b ? ((n + p) * size_k + (k + q)) : ((k + p) * size_n + (n + q)))];
        }

        // Accumulate the tiled matrix multiplications by loading sliced vectors
        // from the shared memory to local register file.
        #pragma unroll
        for (int i = 0; i < 32; ++ i) {
            float reg_a[2], reg_b[2];

            reg_a[0] = shared_a[(r + 0) * 33 + i];
            reg_a[1] = shared_a[(r + 1) * 33 + i];
            reg_b[0] = shared_b[(s + 0) * 33 + i];
            reg_b[1] = shared_b[(s + 1) * 33 + i];

            accum[0][0] += reg_a[0] * reg_b[0];
            accum[0][1] += reg_a[0] * reg_b[1];
            accum[1][0] += reg_a[1] * reg_b[0];
            accum[1][1] += reg_a[1] * reg_b[1];
        }
    }

    // Write the accumulated results to the output matrix.
    matrix_c[offset_c + (r + 0) * 32 + (s + 0)] = accum[0][0];
    matrix_c[offset_c + (r + 0) * 32 + (s + 1)] = accum[0][1];
    matrix_c[offset_c + (r + 1) * 32 + (s + 0)] = accum[1][0];
    matrix_c[offset_c + (r + 1) * 32 + (s + 1)] = accum[1][1];
}
#endif

torch::Tensor sparse_matmul(
    torch::Tensor a, torch::Tensor b, const std::string& mode,
    const layout_tensors& row_layout, const layout_tensors& col_layout,
    bool trans_a, bool trans_b
) {
    // Select current sparse layout by the given sparse mode.
    auto layout = (mode == "sdd"
                   || mode == "dsd" && !trans_a
                   || mode == "dds" && trans_b) ? row_layout : col_layout;
    uint num_blocks = std::get<0>(layout).size(0) / 2;
    uint sparse_width = (std::get<1>(layout).size(0) - 1) * 32;

    // Get the dimension sizes from the tensors.
    uint size_m = mode.at(1) == 'd' ? a.size(trans_a ? -1 : -2) : sparse_width;
    uint size_n = mode.at(2) == 'd' ? b.size(trans_b ? -2 : -1) : sparse_width;
    uint size_k = mode.at(2) == 'd' ? b.size(trans_b ? -1 : -2)
                                    : a.size(trans_a ? -2 : -1);

    // Construct output tensor shape with preserving multiple batch dimensions.
    auto dense = mode.at(1) == 'd' ? a : b;
    auto shape = dense.sizes().slice(0, dense.dim() - 2).vec();

    if (mode.at(0) == 'd') shape.insert(shape.end(), { size_m, size_n });
    else shape.insert(shape.end(), { num_blocks, 32, 32 });

    // Merge the batch dimensions to one.
    a = a.flatten(0, mode.at(1) == 'd' ? -3 : -4);
    b = b.flatten(0, mode.at(2) == 'd' ? -3 : -4);
    uint num_batches = a.size(0);

    // Create an empty output tensor to store the multiplication result.
    torch::Tensor c;
    if (mode.at(0) == 'd') c = a.new_empty({ num_batches, size_m, size_n });
    else c = a.new_empty({ num_batches, num_blocks, 32, 32 });

    // Launch CUDA kernel with corresponding sparse mode and dimension sizes.
    dim3 blocks;
    if (mode == "sdd") blocks = dim3(num_blocks, num_batches);
    else blocks = dim3(num_batches,
                       (size_m + 32 - 1) / 32, (size_n + 32 - 1) / 32);

    auto kernel = mode == "sdd" ? sparse_matmul_sdd_32x32x32_kernel<false, false> :
                  mode == "dsd" ? sparse_matmul_sdd_32x32x32_kernel<false, false> :
                                  sparse_matmul_sdd_32x32x32_kernel<false, false>;
    kernel<<<blocks, 256>>>( //tile<float, 32, 8>::THREADS>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
        layout, num_blocks, size_m, size_n, size_k //,
        //trans_a, trans_b
    );

    // Return the output tensor with multiple batch dimensions.
    return c.reshape(shape);
}
