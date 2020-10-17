#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include "sparse_smm_op.h"


/**
 * Load 32x32 2d tile of matrix to the shared memory efficiently.
 * 
 * Shared Memory        : (32, 32 + ?)
 * Threads per Block    : (32, 32)
 */
__device__ void load_matrix_32x32_tile_sync(
    const   float*      __restrict__    src,
            float*      __restrict__    dst,
            uint                        offset_row,
            uint                        offset_col,
            uint                        total_rows,
            uint                        total_cols,
            uint                        stride_src,
            uint                        stride_dst,
            bool                        transpose
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
 * Batched sparse matrix multiplication for SDD mode with single precision.
 * 
 * Multiply a dense matrix with another dense matrix and create a sparse matrix
 * according to the given sparse layout.
 * 
 * Blocks               : (Total Chunks, Total Batches)
 * Threads per Block    : (32, 32)
 */
__global__ void batched_sparse_smm_op_32x32_sdd_kernel(
    const   float*      __restrict__    matrix_a,
    const   float*      __restrict__    matrix_b,
            float*      __restrict__    matrix_c,
    const   ushort*     __restrict__    chunk_table,
            uint                        total_chunks,
            uint                        total_m,
            uint                        total_n,
            uint                        total_k,
            bool                        trans_a,
            bool                        trans_b
) {
    __shared__ float tile_a[TILE_32x32_WIDTH][TILE_32x32_WIDTH + 1];
    __shared__ float tile_b[TILE_32x32_WIDTH][TILE_32x32_WIDTH + 1];

    uint x = threadIdx.x;
    uint y = threadIdx.y;
    uint chunk_idx = blockIdx.x;
    uint batch_idx = blockIdx.y;

    // Load current chunk information.
    ushort2 chunk_offset = *((ushort2 *) chunk_table + chunk_idx);
    uint offset_m = chunk_offset.x * TILE_32x32_WIDTH;
    uint offset_n = chunk_offset.y * TILE_32x32_WIDTH;

    // Copy sub-matrices to the shared memory and accumulate tiled matrix
    // multiplication.
    float accumulation = 0.0f;
    for (uint offset_k = 0; offset_k < total_k; offset_k += TILE_32x32_WIDTH) {
        load_matrix_32x32_tile_sync(
            matrix_a + batch_idx * total_m * total_k, (float *) tile_a,
            trans_a ? offset_k : offset_m, trans_a ? offset_m : offset_k,
            trans_a ?  total_k :  total_m, trans_a ?  total_m :  total_k,
            trans_a ?  total_m :  total_k, TILE_32x32_WIDTH + 1, trans_a
        );

        load_matrix_32x32_tile_sync(
            matrix_b + batch_idx * total_k * total_n, (float *) tile_b,
            trans_b ? offset_n : offset_k, trans_b ? offset_k : offset_n,
            trans_b ? total_n : total_k, trans_b ? total_k : total_n,
            trans_b ? total_k : total_n, TILE_32x32_WIDTH + 1, !trans_b
        );

        // Compute tiled matrix multiplication.
        for (uint k = 0; k < TILE_32x32_WIDTH; k ++)
            if (offset_k + k < total_k)
                accumulation += tile_a[y][k] * tile_b[x][k];
        __syncthreads();
    }

    // Assign the accumulation of tiled matrix multiplication.
    if (offset_m + y < total_m && offset_n + x < total_n)
        matrix_c[batch_idx * total_chunks * TILE_32x32_SIZE
                 + chunk_idx * TILE_32x32_SIZE
                 + y * TILE_32x32_WIDTH + x] = accumulation;
}

/**
 * 
 * 
 * 
 */
__global__ void batched_sparse_smm_op_32x32_dsd_kernel(

) {

}

/**
 * 
 * 
 * 
 */
__global__ void batched_sparse_smm_op_32x32_dds_kernel(

) {

}


void batched_sparse_smm_op_32x32(
    const   float*          __restrict__    matrix_a,
    const   float*          __restrict__    matrix_b,
            float*          __restrict__    matrix_c,
    const   ushort*         __restrict__    chunk_table,
            uint                            total_chunks,
            uint                            total_batches,
            uint                            total_m,
            uint                            total_n,
            uint                            total_k,
            bool                            trans_a,
            bool                            trans_b,
            SparseMode                      mode
) {
    dim3 blocks;
    dim3 threadsPerBlock;

    switch (mode) {
    case SparseMode::SDD:
        blocks = dim3(total_chunks, total_batches);
        threadsPerBlock = dim3(TILE_32x32_WIDTH, TILE_32x32_WIDTH);

        batched_sparse_smm_op_32x32_sdd_kernel<<<blocks, threadsPerBlock>>>(
            matrix_a, matrix_b, matrix_c, chunk_table,
            total_chunks, total_m, total_n, total_k,
            trans_a, trans_b
        );

        break;
    }
}
