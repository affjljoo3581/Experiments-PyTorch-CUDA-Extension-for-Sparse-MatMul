#include <cuda.h>
#include <cuda_runtime.h>

#include "sparse_ops.h"

#define FULL_MASK       0xffffffff


/**
 * Compute the maximum value of elements across the row in a sparse matrix.
 * 
 * Threads per Block    : (32, 32)
 */
__device__ __forceinline__ float reduce_sparse_matrix_32x32_row_max(
    const   float*  __restrict__    matrix,
    const   short*  __restrict__    sparse_blocks,
            uint                    start_block_ptr,
            uint                    end_block_ptr,
            uint                    offset_row
) {
    __shared__ float shared[32];

    // Get values from strided sparse blocks and find the maximum value.
    float max = -1e5f;
    
    uint block_ptr = start_block_ptr + threadIdx.y;
    for (; block_ptr < end_block_ptr; block_ptr += blockDim.y) {
        short2 sparse_block = *((short2 *) sparse_blocks + block_ptr);

        max = fmaxf(max, matrix[sparse_block.x * TILE_32x32_SIZE
                                + offset_row * TILE_32x32_WIDTH
                                + threadIdx.x]);
    }

    // Reduce the values in each warp and get the maximum value among them.
    for (uint offset = 16; offset > 0; offset /= 2)
        max = fmaxf(max, __shfl_down_sync(FULL_MASK, max, offset));
    max = __shfl_sync(FULL_MASK, max, 0);

    // The first threads of warps write the locally-reduced maximum values to
    // the shared memory for reducing in thread-level.
    if (threadIdx.x == 0) shared[threadIdx.y] = max;
    __syncthreads();

    // The first warp in each block calculates the final maximum value from the
    // shared memory.
    if (threadIdx.y == 0) {
        max = shared[threadIdx.x];
        for (uint offset = 16; offset > 0; offset /= 2)
            max = fmaxf(max, __shfl_down_sync(FULL_MASK, max, offset));
        shared[threadIdx.x] = max;
    }
    __syncthreads();

    return shared[0];
}

/**
 * Compute the sum of elements across the row in a sparse matrix.
 * 
 * Threads per Block    : (32, 32)
 */
__device__ __forceinline__ float reduce_sparse_matrix_32x32_row_sum(
    const   float*  __restrict__    matrix,
    const   short*  __restrict__    sparse_blocks,
            uint                    start_block_ptr,
            uint                    end_block_ptr,
            uint                    offset_row
) {
    __shared__ float shared[32];

    // Get values from strided sparse blocks and calculate the sum.
    float sum = 0.0f;
    
    uint block_ptr = start_block_ptr + threadIdx.y;
    for (; block_ptr < end_block_ptr; block_ptr += blockDim.y) {
        short2 sparse_block = *((short2 *) sparse_blocks + block_ptr);

        sum += matrix[sparse_block.x * TILE_32x32_SIZE
                      + offset_row * TILE_32x32_WIDTH
                      + threadIdx.x];
    }

    // Reduce the values in each warp and compute local sum.
    for (uint offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    sum = __shfl_sync(FULL_MASK, sum, 0);

    // The first threads of warps write the locally-reduced summed values to
    // the shared memory for reducing in thread-level.
    if (threadIdx.x == 0) shared[threadIdx.y] = sum;
    __syncthreads();

    // The first warp in each block calculates the final sum from the shared
    // memory.
    if (threadIdx.y == 0) {
        sum = shared[threadIdx.x];
        for (uint offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(FULL_MASK, sum, offset);
        shared[threadIdx.x] = sum;
    }
    __syncthreads();

    return shared[0];
}

/**
 * Calculate a softmax probability from the sparse logits matrix.
 * 
 * Blocks               : (Batches, Total Rows)
 * Threads per Block    : (32, 32)
 */
__global__ void sparse_softmax_32x32_forward_kernel(
    const   float*  __restrict__    matrix_x,
            float*  __restrict__    matrix_y,
    const   short*  __restrict__    sparse_blocks,
    const   int*    __restrict__    sparse_table,
            uint                    total_blocks
) {
    uint offset_row = blockIdx.y % TILE_32x32_WIDTH;
    uint start_block_ptr = sparse_table[blockIdx.y / TILE_32x32_WIDTH];
    uint end_block_ptr = sparse_table[blockIdx.y / TILE_32x32_WIDTH + 1];

    // Move to the current batch.
    matrix_x += blockIdx.x * total_blocks * TILE_32x32_SIZE;
    matrix_y += blockIdx.x * total_blocks * TILE_32x32_SIZE;

    // Get maximum value across the corresponding row and calculate stable
    // exponential by subtracting maximum logit to each one.
    float max = reduce_sparse_matrix_32x32_row_max(
        matrix_x, sparse_blocks, start_block_ptr, end_block_ptr, offset_row);

    uint block_ptr = start_block_ptr + threadIdx.y;
    for (; block_ptr < end_block_ptr; block_ptr += blockDim.y) {
        short2 sparse_block = *((short2 *) sparse_blocks + block_ptr);
        uint idx = sparse_block.x * TILE_32x32_SIZE
                   + offset_row * TILE_32x32_WIDTH + threadIdx.x;

        matrix_y[idx] = __expf(matrix_x[idx] - max);
    }

    // Get total sum of exponential values and divide each value by the sum for
    // probability property (sum of probabilities is 1).
    float sum = reduce_sparse_matrix_32x32_row_sum(
        matrix_y, sparse_blocks, start_block_ptr, end_block_ptr, offset_row);

    block_ptr = start_block_ptr + threadIdx.y;
    for (; block_ptr < end_block_ptr; block_ptr += blockDim.y) {
        short2 sparse_block = *((short2 *) sparse_blocks + block_ptr);
        uint idx = sparse_block.x * TILE_32x32_SIZE
                   + offset_row * TILE_32x32_WIDTH + threadIdx.x;

        matrix_y[idx] /= sum;
    }
}

/**
 * Calculate a gradient of input sparse logits matrix from the gradient of
 * softmax probability matrix.
 * 
 * Blocks               : (Batches, Total Rows)
 * Threads per Block    : (32, 32)
 */

__global__ void sparse_softmax_32x32_backward_kernel(
    const   float*  __restrict__    matrix_y,
    const   float*  __restrict__    matrix_dy,
            float*  __restrict__    matrix_dx,
    const   short*  __restrict__    sparse_blocks,
    const   int*    __restrict__    sparse_table,
            uint                    total_blocks
) {
    uint offset_row = blockIdx.y % TILE_32x32_WIDTH;
    uint start_block_ptr = sparse_table[blockIdx.y / TILE_32x32_WIDTH];
    uint end_block_ptr = sparse_table[blockIdx.y / TILE_32x32_WIDTH + 1];

    // Move to the current batch.
    matrix_y += blockIdx.x * total_blocks * TILE_32x32_SIZE;
    matrix_dy += blockIdx.x * total_blocks * TILE_32x32_SIZE;
    matrix_dx += blockIdx.x * total_blocks * TILE_32x32_SIZE;

    // Calculate the multiplication of the softmax probability and its gradient
    // and get sum of them to compute a gradient of softmax input.
    uint block_ptr = start_block_ptr + threadIdx.y;
    for (; block_ptr < end_block_ptr; block_ptr += blockDim.y) {
        short2 sparse_block = *((short2 *) sparse_blocks + block_ptr);
        uint idx = sparse_block.x * TILE_32x32_SIZE
                   + offset_row * TILE_32x32_WIDTH + threadIdx.x;

        matrix_dx[idx] = matrix_dy[idx] * matrix_y[idx];
    }

    float sum = reduce_sparse_matrix_32x32_row_sum(
        matrix_dx, sparse_blocks, start_block_ptr, end_block_ptr, offset_row);

    // Compute the gradient of softmax input.
    block_ptr = start_block_ptr + threadIdx.y;
    for (; block_ptr < end_block_ptr; block_ptr += blockDim.y) {
        short2 sparse_block = *((short2 *) sparse_blocks + block_ptr);
        uint idx = sparse_block.x * TILE_32x32_SIZE
                   + offset_row * TILE_32x32_WIDTH + threadIdx.x;

        matrix_dx[idx] = (matrix_dy[idx] - sum) * matrix_y[idx];
    }
}


torch::Tensor sparse_softmax_forward(torch::Tensor x,
                                     torch::Tensor row_blocks,
                                     torch::Tensor row_table) {
    auto output_shape = x.sizes();

    // Merge all batch dimensions to single one and create empty output tensor.
    x = x.flatten(0, -4);
    auto y = torch::empty_like(x);

    // Get the dimension sizes.
    int64_t total_batches = x.size(0);
    int64_t total_blocks = row_blocks.size(0) / 2;
    int64_t total_rows = (row_table.size(0) - 1) * TILE_32x32_WIDTH;

    dim3 blocks(total_batches, total_rows);
    dim3 threadsPerBlock(TILE_32x32_WIDTH, TILE_32x32_WIDTH);

    sparse_softmax_32x32_forward_kernel<<<blocks, threadsPerBlock>>>(
        x.data_ptr<float>(), y.data_ptr<float>(),
        row_blocks.data_ptr<short>(), row_table.data_ptr<int>(), total_blocks
    );

    return y.reshape(output_shape);
}

torch::Tensor sparse_softmax_backward(torch::Tensor y,
                                      torch::Tensor dy,
                                      torch::Tensor row_blocks,
                                      torch::Tensor row_table) {
    auto output_shape = y.sizes();

    // Merge all batch dimensions to single one and create empty output tensor.
    y = y.flatten(0, -4);
    dy = dy.flatten(0, -4);
    auto dx = torch::empty_like(y);

    // Get the dimension sizes.
    int64_t total_batches = y.size(0);
    int64_t total_blocks = row_blocks.size(0) / 2;
    int64_t total_rows = (row_table.size(0) - 1) * TILE_32x32_WIDTH;

    dim3 blocks(total_batches, total_rows);
    dim3 threadsPerBlock(TILE_32x32_WIDTH, TILE_32x32_WIDTH);

    sparse_softmax_32x32_backward_kernel<<<blocks, threadsPerBlock>>>(
        y.data_ptr<float>(), dy.data_ptr<float>(), dx.data_ptr<float>(),
        row_blocks.data_ptr<short>(), row_table.data_ptr<int>(), total_blocks
    );

    return dx.reshape(output_shape);
}
