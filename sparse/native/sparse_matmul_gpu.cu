#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <torch/extension.h>

#include "sparse_kernels.h"
#include "sparse_layout.cuh"

#define USE_32x32_TILING

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

    auto kernel = mode == "sdd" ? sparse_matmul_sdd_32x32x8_kernel<false, false> :
                  mode == "dsd" ? sparse_matmul_sdd_32x32x8_kernel<false, false> :
                                  sparse_matmul_sdd_32x32x8_kernel<false, false>;
    kernel<<<blocks, 256>>>( //tile<float, 32, 8>::THREADS>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
        layout, num_blocks, size_m, size_n, size_k //,
        //trans_a, trans_b
    );

    // Return the output tensor with multiple batch dimensions.
    return c.reshape(shape);
}
