#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <sys/types.h>
#include <torch/extension.h>

#include "sparse_ops.h"
#include "sparse_layout.cuh"


/**
 * Load matrix tile from global memory to shared memory in sync.
 * 
 * Threads per Block    : (32, 32)
 */
__device__ __forceinline__ void load_matrix_sync(
    const float* __restrict__ src,
          float* __restrict__ dst,
    uint stride_src, uint stride_dst, bool transpose
) {
    uint offset_dst = !transpose ? threadIdx.y * stride_dst + threadIdx.x
                                 : threadIdx.x * stride_dst + threadIdx.y;
    dst[offset_dst] = src[threadIdx.y * stride_src + threadIdx.x];
}

/**
 * Compute sparse matrix multiplication with SDD mode.
 * 
 * It multiplies a dense matrix with other dense matrix and create a new sparse
 * matrix through corresponding sparse layout.
 * 
 * Blocks               : (Sparse Blocks, Total Batches)
 * Threads per Block    : (32, 32)
 */
__global__ void sparse_matmul_sdd_32x32_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
          float* __restrict__ matrix_c,
    sparse_layout layout, uint num_blocks,
    uint size_m, uint size_n, uint size_k,
    bool trans_a, bool trans_b
) {
    __shared__ float tile_a[TILE_32x32_WIDTH][TILE_32x32_WIDTH + 1];
    __shared__ float tile_b[TILE_32x32_WIDTH][TILE_32x32_WIDTH + 1];

    auto block = layout.blocks[blockIdx.x];
    uint m = block.row() * TILE_32x32_WIDTH;
    uint n = block.col() * TILE_32x32_WIDTH;

    // Move to the current batch and sparse block.
    matrix_a += blockIdx.y * size_m * size_k;
    matrix_b += blockIdx.y * size_k * size_n;
    matrix_c += (blockIdx.y * num_blocks + block.idx) * TILE_32x32_SIZE;

    float accumulator = 0.0f;
    for (uint k = 0; k < size_k; k += TILE_32x32_WIDTH) {
        uint offset_a = trans_a ? k * size_m + m : m * size_k + k;
        uint offset_b = trans_b ? n * size_k + k : k * size_n + n;

        load_matrix_sync(matrix_a + offset_a, (float *) tile_a,
                         trans_a ? size_m : size_k, TILE_32x32_WIDTH + 1,
                         trans_a);        
        load_matrix_sync(matrix_b + offset_b, (float *) tile_b,
                         trans_b ? size_k : size_n, TILE_32x32_WIDTH + 1,
                         trans_b);

        // Accumulate the tiled matrix multiplications.
        for (uint i = 0; i < TILE_32x32_SIZE; i ++)
            accumulator += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
        __syncthreads();
    }

    matrix_c[m * TILE_32x32_WIDTH + n] = accumulator;
}

/**
 * Compute sparse matrix multiplication with DSD mode.
 * 
 * It multiplies a sparse matrix with a dense matrix and create new dense matrix
 * through corresponding sparse layout.
 * 
 * Blocks               : (Total Batches, Sparse Block Rows, Dense Block Cols)
 * Threads per Block    : (32, 32)
 */
__global__ void sparse_matmul_dsd_32x32_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
          float* __restrict__ matrix_c,
    sparse_layout layout, uint num_blocks,
    uint size_m, uint size_n, uint size_k,
    bool trans_a, bool trans_b
) {
    __shared__ float tile_a[TILE_32x32_WIDTH][TILE_32x32_WIDTH + 1];
    __shared__ float tile_b[TILE_32x32_WIDTH][TILE_32x32_WIDTH + 1];

    uint m = blockIdx.y * TILE_32x32_WIDTH;
    uint n = blockIdx.z * TILE_32x32_WIDTH;

    // Move to the current batch.
    matrix_a += blockIdx.x * num_blocks * TILE_32x32_SIZE;
    matrix_b += blockIdx.x * size_k * size_n;
    matrix_c += blockIdx.x * size_m * size_n;

    float accumulator = 0.0f;
    for (auto desc = layout.begin(blockIdx.y); desc.valid(); desc.next()) {
        auto block = *desc;
        uint k = (trans_a ? block.row() : block.col()) * TILE_32x32_WIDTH;

        uint offset_a = block.idx * TILE_32x32_SIZE;
        uint offset_b = trans_b ? n * size_k + k : k * size_n + n;

        load_matrix_sync(matrix_a + offset_a, (float *) tile_a,
                         TILE_32x32_WIDTH, TILE_32x32_WIDTH + 1, trans_a);
        load_matrix_sync(matrix_b + offset_b, (float *) tile_b,
                         trans_b ? size_k : size_n, TILE_32x32_WIDTH + 1,
                         trans_b);

        // Accumulate the tiled matrix multiplications.
        for (uint i = 0; i < TILE_32x32_WIDTH; i ++)
            accumulator += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
        __syncthreads();
    }

    matrix_c[(m + threadIdx.y) * size_n + (n + threadIdx.x)] = accumulator;
}

/**
 * Compute sparse matrix multiplication with DDS mode.
 * 
 * It multiplies a dense matrix with a sparse matrix and create new dense matrix
 * through corresponding sparse layout.
 * 
 * Blocks               : (Total Batches, Dense Block Rows, Sparse Block Cols)
 * Threads per Block    : (32, 32)
 */
__global__ void sparse_matmul_dds_32x32_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
          float* __restrict__ matrix_c,
    sparse_layout layout, uint num_blocks,
    uint size_m, uint size_n, uint size_k,
    bool trans_a, bool trans_b
) {
    __shared__ float tile_a[TILE_32x32_WIDTH][TILE_32x32_WIDTH + 1];
    __shared__ float tile_b[TILE_32x32_WIDTH][TILE_32x32_WIDTH + 1];

    uint m = blockIdx.y * TILE_32x32_WIDTH;
    uint n = blockIdx.z * TILE_32x32_WIDTH;

    // Move to the current batch.
    matrix_a += blockIdx.x * num_blocks * TILE_32x32_SIZE;
    matrix_b += blockIdx.x * size_k * size_n;
    matrix_c += blockIdx.x * size_m * size_n;

    float accumulator = 0.0f;
    for (auto desc = layout.begin(blockIdx.z); desc.valid(); desc.next()) {
        auto block = *desc;
        uint k = (trans_b ? block.col() : block.row()) * TILE_32x32_WIDTH;

        uint offset_a = trans_a ? k * size_m + m : m * size_k + k;
        uint offset_b = block.idx * TILE_32x32_SIZE;

        load_matrix_sync(matrix_a + offset_a, (float *) tile_a,
                         trans_a ? size_m : size_k, TILE_32x32_WIDTH + 1,
                         trans_a);
        load_matrix_sync(matrix_b + offset_b, (float *) tile_b,
                         TILE_32x32_WIDTH, TILE_32x32_WIDTH + 1, trans_b);

        // Accumulate the tiled matrix multiplications.
        for (uint i = 0; i < TILE_32x32_WIDTH; i ++)
            accumulator += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
        __syncthreads();
    }

    matrix_c[(m + threadIdx.y) * size_n + (n + threadIdx.x)] = accumulator;
}


torch::Tensor sparse_matmul(
    torch::Tensor a, torch::Tensor b, const std::string& mode,
    const layout_tensors& row_layout, const layout_tensors& col_layout,
    bool trans_a, bool trans_b
) {
    // Select current sparse layout.
    auto layout = (mode == "ssd"
                   || mode == "dsd" && !trans_a
                   || mode == "dds" && trans_b) ? row_layout : col_layout;
    int64_t num_blocks = std::get<0>(layout).size(0) / 2;
    int64_t sparse_width = (std::get<1>(layout).size(0) - 1) * TILE_32x32_WIDTH;

    // Get the dimension sizes from the tensors and corresponding sparse mode.
    int64_t size_m = mode.at(1) == 'd' ? a.size(trans_a ? -1 : -2)
                                       : sparse_width;
    int64_t size_n = mode.at(2) == 'd' ? b.size(trans_b ? -2 : -1)
                                       : sparse_width;
    int64_t size_k = mode.at(2) == 'd' ? b.size(trans_b ? -1 : -2)
                                       : a.size(trans_a ? -2 : -1);

    // Construct output tensor shape with preserving multiple batch dimensions.
    auto dense = mode.at(1) == 'd' ? a : b;
    auto shape = dense.sizes().slice(0, dense.dim() - 2).vec();

    if (mode.at(0) == 'd') shape.insert(shape.end(), { size_m, size_n });
    else shape.insert(shape.end(), { num_blocks,
                                     TILE_32x32_WIDTH, TILE_32x32_WIDTH });

    // Merge the batch dimensions to one.
    a = a.flatten(0, mode.at(1) == 'd' ? -3 : -4);
    b = b.flatten(0, mode.at(2) == 'd' ? -3 : -4);

    int64_t num_batches = a.size(0);

    // Create an empty output tensor to store the multiplication result.
    torch::Tensor c;
    if (mode.at(0) == 'd') c = a.new_empty({ num_batches, size_m, size_n });
    else c = a.new_empty({ num_batches, num_blocks,
                           TILE_32x32_WIDTH, TILE_32x32_WIDTH });

    // Launch CUDA kernel with corresponding sparse mode and dimension sizes.
    dim3 blocks, threadsPerBlock(TILE_32x32_WIDTH, TILE_32x32_WIDTH);
    if (mode == "sdd") blocks = dim3(num_blocks, num_batches);
    else blocks = dim3(num_batches,
                       (size_m + TILE_32x32_WIDTH - 1) / TILE_32x32_WIDTH,
                       (size_n + TILE_32x32_WIDTH - 1) / TILE_32x32_WIDTH);

    auto kernel = mode == "sdd" ? sparse_matmul_sdd_32x32_kernel :
                  mode == "dsd" ? sparse_matmul_dsd_32x32_kernel :
                                  sparse_matmul_dds_32x32_kernel;
    kernel<<<blocks, threadsPerBlock>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
        layout, num_blocks, size_m, size_n, size_k, trans_a, trans_b
    );

    // Return the output tensor with the multiple batch dimensions.
    return c.reshape(shape);
}
