#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

#include "sparse_smm_op.h"


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
__global__ void batched_sparse_smm_op_32x32_sdd_kernel(
    const   float*  __restrict__    matrix_a,
    const   float*  __restrict__    matrix_b,
            float*  __restrict__    matrix_c,
    const   short*  __restrict__    sparse_table,
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
    short2 sparse_block = *((short2 *) sparse_table + blockIdx.x);
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
 * Blocks               : (Batches, Rows, Columns)
 * Threads per Block    : (32, 32)
 */
__global__ void batched_sparse_smm_op_32x32_dsd_kernel(
    const   float*  __restrict__    matrix_a,
    const   float*  __restrict__    matrix_b,
            float*  __restrict__    matrix_c,
    const   short*  __restrict__    sparse_table,
    const   int*    __restrict__    sparse_table_ptr,
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
    uint block_ptr = sparse_table_ptr[blockIdx.y];
    uint end_block_ptr = sparse_table_ptr[blockIdx.y + 1];

    float accumulator = 0.0f;
    for (; block_ptr < end_block_ptr; block_ptr ++) {
        // Get current sparse-block in corresponding row.
        short2 sparse_block = *((short2 *) sparse_table + block_ptr);
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
 * Blocks               : (Batches, Rows, Columns)
 * Threads per Block    : (32, 32)
 */
__global__ void batched_sparse_smm_op_32x32_dds_kernel(
    const   float*  __restrict__    matrix_a,
    const   float*  __restrict__    matrix_b,
            float*  __restrict__    matrix_c,
    const   short*  __restrict__    sparse_table,
    const   int*    __restrict__    sparse_table_ptr,
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
    uint block_ptr = sparse_table_ptr[blockIdx.z];
    uint end_block_ptr = sparse_table_ptr[blockIdx.z + 1];

    float accumulator = 0.0f;
    for (; block_ptr < end_block_ptr; block_ptr ++) {
        // Get current sparse-block in corresponding row.
        short2 sparse_block = *((short2 *) sparse_table + block_ptr);
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


void batched_sparse_smm_op_32x32_sdd(
    const   float*      matrix_a,
    const   float*      matrix_b,
            float*      matrix_c,
    const   short*      sparse_table,
            uint        total_blocks,
            uint        total_batches,
            uint        total_m,
            uint        total_n,
            uint        total_k,
            bool        trans_a,
            bool        trans_b
) {
    dim3 blocks(total_blocks, total_batches);
    dim3 threadsPerBlock(TILE_32x32_WIDTH, TILE_32x32_WIDTH);

    batched_sparse_smm_op_32x32_sdd_kernel<<<blocks, threadsPerBlock>>>(
        matrix_a, matrix_b, matrix_c,
        sparse_table, total_blocks,
        total_m, total_n, total_k, trans_a, trans_b
    );
}

void batched_sparse_smm_op_32x32_dsd(
    const   float*      matrix_a,
    const   float*      matrix_b,
            float*      matrix_c,
    const   short*      sparse_table,
    const   int*        sparse_table_ptr,
            uint        total_blocks,
            uint        total_batches,
            uint        total_m,
            uint        total_n,
            uint        total_k,
            bool        trans_a,
            bool        trans_b
) {
    dim3 blocks(
        total_batches,
        (total_m + TILE_32x32_WIDTH - 1) / TILE_32x32_WIDTH,
        (total_n + TILE_32x32_WIDTH - 1) / TILE_32x32_WIDTH
    );
    dim3 threadsPerBlock(TILE_32x32_WIDTH, TILE_32x32_WIDTH);

    batched_sparse_smm_op_32x32_dsd_kernel<<<blocks, threadsPerBlock>>>(
        matrix_a, matrix_b, matrix_c, sparse_table, sparse_table_ptr,
        total_blocks, total_m, total_n, total_k, trans_a, trans_b
    );
}

void batched_sparse_smm_op_32x32_dds(
    const   float*      matrix_a,
    const   float*      matrix_b,
            float*      matrix_c,
    const   short*      sparse_table,
    const   int*        sparse_table_ptr,
            uint        total_blocks,
            uint        total_batches,
            uint        total_m,
            uint        total_n,
            uint        total_k,
            bool        trans_a,
            bool        trans_b
) {
    dim3 blocks(
        total_batches,
        (total_m + TILE_32x32_WIDTH - 1) / TILE_32x32_WIDTH,
        (total_n + TILE_32x32_WIDTH - 1) / TILE_32x32_WIDTH
    );
    dim3 threadsPerBlock(TILE_32x32_WIDTH, TILE_32x32_WIDTH);

    batched_sparse_smm_op_32x32_dds_kernel<<<blocks, threadsPerBlock>>>(
        matrix_a, matrix_b, matrix_c, sparse_table, sparse_table_ptr,
        total_blocks, total_m, total_n, total_k, trans_a, trans_b
    );
}
