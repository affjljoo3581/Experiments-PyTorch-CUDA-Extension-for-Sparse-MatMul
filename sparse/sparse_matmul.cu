#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

#include "sparse_ops.h"


/**
 * Load 32x32 2d tile of matrix to the shared memory efficiently.
 * 
 * Shared Memory        : (32, 32 + ?)
 * Threads per Block    : (32, 32)
 */
__device__ void load_matrix_32x32_tile_sync(
    const   float*  __restrict__    src,
            float*  __restrict__    dst,
            uint                    offset_row,
            uint                    offset_col,
            uint                    total_rows,
            uint                    total_cols,
            uint                    stride_src,
            uint                    stride_dst,
            bool                    transpose
) {
    uint row = offset_row + threadIdx.y;
    uint col = offset_col + threadIdx.x;
    uint offset_dst = transpose ? threadIdx.x * stride_dst + threadIdx.y
                                : threadIdx.y * stride_dst + threadIdx.x;

    *(dst + offset_dst) = (row < total_rows && col < total_cols)
                          ? *(src + row * stride_src + col)
                          : 0.0f;
    __syncthreads();
}

/**
 * Batched sparse matrix multiplication for SDD mode in single precision.
 * 
 * Multiply a dense matrix with another dense matrix and create a sparse matrix
 * according to the given sparse layout.
 * 
 * Blocks               : (Blocks, Batches)
 * Threads per Block    : (32, 32)
 */
__global__ void batched_sparse_matmul_32x32_sdd_kernel(
    const   float*  __restrict__    matrix_a,
    const   float*  __restrict__    matrix_b,
            float*  __restrict__    matrix_c,
    const   short*  __restrict__    sparse_blocks,
            uint                    total_blocks,
            uint                    total_m,
            uint                    total_n,
            uint                    total_k,
            bool                    trans_a,
            bool                    trans_b
) {
    __shared__ float tile_a[TILE_32x32_WIDTH][TILE_32x32_WIDTH + 1];
    __shared__ float tile_b[TILE_32x32_WIDTH][TILE_32x32_WIDTH + 1];

    // Load current sparse-block information.
    short2 sparse_block = *((short2 *) sparse_blocks + blockIdx.x);
    uint block_stride = (total_n + TILE_32x32_WIDTH - 1) / TILE_32x32_WIDTH;
    uint offset_m = sparse_block.y / block_stride * TILE_32x32_WIDTH;
    uint offset_n = sparse_block.y % block_stride * TILE_32x32_WIDTH;

    // Move to the current batch and sparse block.
    matrix_a += blockIdx.y * total_m * total_k;
    matrix_b += blockIdx.y * total_k * total_n;
    matrix_c += (blockIdx.y * total_blocks + sparse_block.x) * TILE_32x32_SIZE;

    // Copy sub-matrices to the shared memory and accumulate tiled matrix
    // multiplications.
    float accumulator = 0.0f;
    for (uint offset_k = 0; offset_k < total_k; offset_k += TILE_32x32_WIDTH) {
        load_matrix_32x32_tile_sync(
            matrix_a, (float *) tile_a,
            trans_a ? offset_k : offset_m, trans_a ? offset_m : offset_k,
            trans_a ?  total_k :  total_m, trans_a ?  total_m :  total_k,
            trans_a ?  total_m :  total_k, TILE_32x32_WIDTH + 1, trans_a
        );

        load_matrix_32x32_tile_sync(
            matrix_b, (float *) tile_b,
            trans_b ? offset_n : offset_k, trans_b ? offset_k : offset_n,
            trans_b ?  total_n :  total_k, trans_b ?  total_k :  total_n,
            trans_b ?  total_k :  total_n, TILE_32x32_WIDTH + 1, !trans_b
        );

        // Compute tiled matrix multiplication.
        for (uint k = 0; k < TILE_32x32_WIDTH; k ++)
            if (offset_k + k < total_k)
                accumulator += tile_a[threadIdx.y][k] * tile_b[threadIdx.x][k];
        __syncthreads();
    }

    // Assign the accumulation of tiled matrix multiplication.
    if (offset_m + threadIdx.y < total_m && offset_n + threadIdx.x < total_n)
        matrix_c[threadIdx.y * TILE_32x32_WIDTH + threadIdx.x] = accumulator;
}

/**
 * Batched sparse matrix multiplication for DSD mode in single precision.
 * 
 * Multiply a sparse matrix with a dense matrix and create new dense matrix
 * according to the given sparse layout.
 * 
 * Blocks               : (Batches, Block Rows, Block Columns)
 * Threads per Block    : (32, 32)
 */
__global__ void batched_sparse_matmul_32x32_dsd_kernel(
    const   float*  __restrict__    matrix_a,
    const   float*  __restrict__    matrix_b,
            float*  __restrict__    matrix_c,
    const   short*  __restrict__    sparse_blocks,
    const   int*    __restrict__    sparse_table,
            uint                    total_blocks,
            uint                    total_m,
            uint                    total_n,
            uint                    total_k,
            bool                    trans_a,
            bool                    trans_b
) {
    __shared__ float tile_a[TILE_32x32_WIDTH][TILE_32x32_WIDTH + 1];
    __shared__ float tile_b[TILE_32x32_WIDTH][TILE_32x32_WIDTH + 1];

    uint offset_m = blockIdx.y * TILE_32x32_WIDTH;
    uint offset_n = blockIdx.z * TILE_32x32_WIDTH;
    uint block_stride = ((trans_a ? total_m : total_k)
                         + TILE_32x32_WIDTH - 1) / TILE_32x32_WIDTH;

    // Move to the current batch.
    matrix_a += blockIdx.x * total_blocks * TILE_32x32_SIZE;
    matrix_b += blockIdx.x * total_k * total_n;
    matrix_c += blockIdx.x * total_m * total_n;

    // Copy sub-matrices to the shared memory and accumulate tiled matrix
    // multiplication.
    uint block_ptr = sparse_table[threadIdx.y];
    uint end_block_ptr = sparse_table[threadIdx.y + 1];

    float accumulator = 0.0f;
    for (; block_ptr < end_block_ptr; block_ptr ++) {
        // Get current sparse-block in corresponding row.
        short2 sparse_block = *((short2 *) sparse_blocks + block_ptr);
        uint offset_k = (trans_a
                         ? sparse_block.y / block_stride
                         : sparse_block.y % block_stride) * TILE_32x32_WIDTH;

        load_matrix_32x32_tile_sync(
            matrix_a + sparse_block.x * TILE_32x32_SIZE, (float *) tile_a,
            0, 0, TILE_32x32_WIDTH, TILE_32x32_WIDTH,
            TILE_32x32_WIDTH, TILE_32x32_WIDTH + 1, trans_a
        );

        load_matrix_32x32_tile_sync(
            matrix_b, (float *) tile_b,
            trans_b ? offset_n : offset_k, trans_b ? offset_k : offset_n,
            trans_b ?  total_n :  total_k, trans_b ?  total_k :  total_n,
            trans_b ?  total_k :  total_n, TILE_32x32_WIDTH + 1, !trans_b
        );

        // Compute tiled matrix multiplication.
        for (uint k = 0; k < TILE_32x32_WIDTH; k ++)
            if (offset_k + k < total_k)
                accumulator += tile_a[threadIdx.y][k] * tile_b[threadIdx.x][k];
        __syncthreads();
    }
    
    // Assign the accumulation of tiled matrix multiplication.
    if (offset_m + threadIdx.y < total_m && offset_n + threadIdx.x < total_n)
        matrix_c[(offset_m + threadIdx.y) * total_n
                 + (offset_n + threadIdx.x)] = accumulator;
}

/**
 * Batched sparse matrix multiplication for DDS mode in single precision.
 * 
 * Multiply a dense matrix with a sparse matrix and create new dense matrix
 * according to the given sparse layout.
 * 
 * Blocks               : (Batches, Block Rows, Block Columns)
 * Threads per Block    : (32, 32)
 */
__global__ void batched_sparse_matmul_32x32_dds_kernel(
    const   float*  __restrict__    matrix_a,
    const   float*  __restrict__    matrix_b,
            float*  __restrict__    matrix_c,
    const   short*  __restrict__    sparse_blocks,
    const   int*    __restrict__    sparse_table,
            uint                    total_blocks,
            uint                    total_m,
            uint                    total_n,
            uint                    total_k,
            bool                    trans_a,
            bool                    trans_b
) {
    __shared__ float tile_a[TILE_32x32_WIDTH][TILE_32x32_WIDTH + 1];
    __shared__ float tile_b[TILE_32x32_WIDTH][TILE_32x32_WIDTH + 1];

    uint offset_m = blockIdx.y * TILE_32x32_WIDTH;
    uint offset_n = blockIdx.z * TILE_32x32_WIDTH;
    uint block_stride = ((trans_b ? total_k : total_n)
                         + TILE_32x32_WIDTH - 1) / TILE_32x32_WIDTH;

    // Move to the current batch.
    matrix_a += blockIdx.x * total_m * total_k;
    matrix_b += blockIdx.x * total_blocks * TILE_32x32_SIZE;
    matrix_c += blockIdx.x * total_m * total_n;

    // Copy sub-matrices to the shared memory and accumulate tiled matrix
    // multiplication.
    uint block_ptr = sparse_table[blockIdx.z];
    uint end_block_ptr = sparse_table[blockIdx.z + 1];

    float accumulator = 0.0f;
    for (; block_ptr < end_block_ptr; block_ptr ++) {
        // Get current sparse-block in corresponding row.
        short2 sparse_block = *((short2 *) sparse_blocks + block_ptr);
        uint offset_k = (trans_b
                         ? sparse_block.y % block_stride
                         : sparse_block.y / block_stride) * TILE_32x32_WIDTH;

        load_matrix_32x32_tile_sync(
            matrix_a, (float *) tile_a,
            trans_a ? offset_k : offset_m, trans_a ? offset_m : offset_k,
            trans_a ?  total_k :  total_m, trans_a ?  total_m :  total_k,
            trans_a ?  total_m :  total_k, TILE_32x32_WIDTH + 1, trans_a
        );

        load_matrix_32x32_tile_sync(
            matrix_b + sparse_block.x * TILE_32x32_SIZE, (float *) tile_b,
            0, 0, TILE_32x32_WIDTH, TILE_32x32_WIDTH,
            TILE_32x32_WIDTH, TILE_32x32_WIDTH + 1, !trans_b
        );

        // Compute tiled matrix multiplication.
        for (uint k = 0; k < TILE_32x32_WIDTH; k ++)
            if (offset_k + k < total_k)
                accumulator += tile_a[threadIdx.y][k] * tile_b[threadIdx.x][k];
        __syncthreads();
    }

    // Assign the accumulation of tiled matrix multiplication.
    if (offset_m + threadIdx.y < total_m && offset_n + threadIdx.x < total_n)
        matrix_c[(offset_m + threadIdx.y) * total_n
                 + (offset_n + threadIdx.x)] = accumulator;
}


torch::Tensor batched_sparse_matmul(torch::Tensor a,
                                    torch::Tensor b,
                                    const std::string& mode,
                                    torch::Tensor row_blocks,
                                    torch::Tensor row_table,
                                    torch::Tensor col_blocks,
                                    torch::Tensor col_table,
                                    bool trans_a,
                                    bool trans_b) {
    if (mode == "sdd") {
        // Create output sparse-tensor shape with preserving batch dimensions.
        std::vector<int64_t> output_shape = a.sizes().vec();
        output_shape.pop_back();
        output_shape.pop_back();
        output_shape.push_back(row_blocks.size(0) / 2);
        output_shape.push_back(TILE_32x32_WIDTH);
        output_shape.push_back(TILE_32x32_WIDTH);

        // Merge all batch dimensions to single one.
        a = a.flatten(0, -3);
        b = b.flatten(0, -3);

        // Get the dimension sizes and create the output sparse-tensor.
        int64_t total_batches = a.size(0);
        int64_t total_blocks = row_blocks.size(0) / 2;
        int64_t total_m = a.size(trans_a ? -1 : -2);
        int64_t total_n = b.size(trans_b ? -2 : -1);
        int64_t total_k = b.size(trans_b ? -1 : -2);

        auto c = a.new_empty({total_batches, total_blocks,
                              TILE_32x32_WIDTH, TILE_32x32_WIDTH});

        dim3 blocks(total_blocks, total_batches);
        dim3 threadsPerBlock(TILE_32x32_WIDTH, TILE_32x32_WIDTH);

        batched_sparse_matmul_32x32_sdd_kernel<<<blocks, threadsPerBlock>>>(
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
            row_blocks.data_ptr<short>(), total_blocks,
            total_m, total_n, total_k, trans_a, trans_b
        );

        return c.reshape(output_shape);
    } else if (mode == "dsd") {
        auto sparse_blocks = trans_a ? col_blocks : row_blocks;
        auto sparse_table = trans_a ? col_table : row_table;

        int64_t total_m = (sparse_table.size(0) - 1) * TILE_32x32_WIDTH;
        int64_t total_n = b.size(trans_b ? -2 : -1);
        int64_t total_k = b.size(trans_b ? -1 : -2);

        // Create output sparse-tensor shape with preserving batch dimensions.
        std::vector<int64_t> output_shape = b.sizes().vec();
        output_shape.pop_back();
        output_shape.pop_back();
        output_shape.push_back(total_m);
        output_shape.push_back(total_n);

        // Merge all batch dimensions to single one.
        a = a.flatten(0, -4);
        b = b.flatten(0, -3);

        // Get the dimension sizes and create empty output dense tensor.
        int64_t total_batches = a.size(0);
        int64_t total_blocks = sparse_blocks.size(0) / 2;

        auto c = a.new_empty({total_batches, total_m, total_n});

        dim3 blocks(
            total_batches,
            (total_m + TILE_32x32_WIDTH - 1) / TILE_32x32_WIDTH,
            (total_n + TILE_32x32_WIDTH - 1) / TILE_32x32_WIDTH
        );
        dim3 threadsPerBlock(TILE_32x32_WIDTH, TILE_32x32_WIDTH);

        batched_sparse_matmul_32x32_dsd_kernel<<<blocks, threadsPerBlock>>>(
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
            sparse_blocks.data_ptr<short>(), sparse_table.data_ptr<int>(),
            total_blocks, total_m, total_n, total_k, trans_a, trans_b
        );

        return c.reshape(output_shape);
    } else if (mode == "dds") {
        auto sparse_blocks = trans_b ? row_blocks : col_blocks;
        auto sparse_table = trans_b ? row_table : col_table;

        int64_t total_m = a.size(trans_a ? -1 : -2);
        int64_t total_n = (sparse_table.size(0) - 1) * TILE_32x32_WIDTH;
        int64_t total_k = a.size(trans_a ? -2 : -1);

        // Create output sparse-tensor shape with preserving batch dimensions.
        std::vector<int64_t> output_shape = a.sizes().vec();
        output_shape.pop_back();
        output_shape.pop_back();
        output_shape.push_back(total_m);
        output_shape.push_back(total_n);

        // Merge all batch dimensions to single one.
        a = a.flatten(0, -3);
        b = b.flatten(0, -4);

        // Get the dimension sizes and create the output dense tensor.
        int64_t total_batches = a.size(0);
        int64_t total_blocks = sparse_blocks.size(0) / 2;

        auto c = a.new_empty({total_batches, total_m, total_n});

        dim3 blocks(
            total_batches,
            (total_m + TILE_32x32_WIDTH - 1) / TILE_32x32_WIDTH,
            (total_n + TILE_32x32_WIDTH - 1) / TILE_32x32_WIDTH
        );
        dim3 threadsPerBlock(TILE_32x32_WIDTH, TILE_32x32_WIDTH);

        batched_sparse_matmul_32x32_dds_kernel<<<blocks, threadsPerBlock>>>(
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
            sparse_blocks.data_ptr<short>(), sparse_table.data_ptr<int>(),
            total_blocks, total_m, total_n, total_k, trans_a, trans_b
        );

        return c.reshape(output_shape);
    }
}
