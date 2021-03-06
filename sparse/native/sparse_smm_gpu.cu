#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <string>
#include <torch/extension.h>

#include "sparse_kernels.h"
#include "sparse_layout.cuh"


/**
 * Compute single-precision sparse matrix multiplication with SDD mode.
 * 
 * It multiplies a dense matrix with another dense matrix and create a new
 * sparse matrix from the sparse layout.
 * 
 * Blocks               : (Sparse Blocks, Total Batches)
 * Threads per Block    : 256
 */
template <bool tr_a, bool tr_b>
__global__ void sparse_smm_sdd_32x32x32_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
          float* __restrict__ matrix_c,
    sparse_layout layout, int num_blocks, int size_m, int size_n, int size_k
) {
    __shared__ float shared_a[32 * 33], shared_b[32 * 33];
    float4 buffer_a, buffer_b;
    float accum[2][2] = { 0 };

    // Load current block and get corresponding row and column positions.
    auto block = layout.get(blockIdx.x);
    int m = block.row() * 32;
    int n = block.col() * 32;

    // Get an offset of each matrix and calculate mapping indices.
    int offset_a = blockIdx.y * size_m * size_k;
    int offset_b = blockIdx.y * size_k * size_n;
    int offset_c = (blockIdx.y * num_blocks + block.idx()) * 32 * 32;

    int p = threadIdx.x / 8;
    int q = threadIdx.x % 8 * 4;
    int r = threadIdx.x / 16 * 2;
    int s = threadIdx.x % 16 * 2;

    // Prefetch first tiles from matrices in global memory.
    buffer_a = *(float4 *) &matrix_a[offset_a + (tr_a ? ((0 + p) * size_m + (m + q)) : ((m + p) * size_k + (0 + q)))];
    buffer_b = *(float4 *) &matrix_b[offset_b + (tr_b ? ((n + p) * size_k + (0 + q)) : ((0 + p) * size_n + (n + q)))];

    #pragma unroll 1
    for (int k = 32; k <= size_k; k += 32) {
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
        if (k < size_k) {
            buffer_a = *(float4 *) &matrix_a[offset_a + (tr_a ? ((k + p) * size_m + (m + q)) : ((m + p) * size_k + (k + q)))];
            buffer_b = *(float4 *) &matrix_b[offset_b + (tr_b ? ((n + p) * size_k + (k + q)) : ((k + p) * size_n + (n + q)))];
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

/**
 * Compute single-precision sparse matrix multiplication with DSD mode.
 * 
 * It multiplies a sparse matrix with a dense matrix and create a new dense
 * matrix from the sparse layout.
 * 
 * Blocks               : (Total Batches, Sparse Rows, Sparse Columns)
 * Threads per Block    : 256
 */
template <bool tr_a, bool tr_b>
__global__ void sparse_smm_dsd_32x32x32_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
          float* __restrict__ matrix_c,
    sparse_layout layout, int num_blocks, int size_m, int size_n, int size_k
) {
    __shared__ float shared_a[32 * 33], shared_b[32 * 33];
    float4 buffer_a, buffer_b;
    float accum[2][2] = { 0 };

    // Get an offset of each matrix and calculate mapping indices.
    int offset_a = blockIdx.x * num_blocks * 32 * 32;
    int offset_b = blockIdx.x * size_k * size_n;
    int offset_c = blockIdx.x * size_m * size_n;

    int m = blockIdx.y * 32;
    int n = blockIdx.z * 32;

    int p = threadIdx.x / 8;
    int q = threadIdx.x % 8 * 4;
    int r = threadIdx.x / 16 * 2;
    int s = threadIdx.x % 16 * 2;

    // Prefetch first tiles from matrices in global memory.
    auto iter = layout.begin(blockIdx.y);
    auto block = *iter;
    buffer_a = *(float4 *) &matrix_a[offset_a + (block.idx() * 32 * 32) + p * 32 + q];
    buffer_b = *(float4 *) &matrix_b[offset_b + (tr_b ? ((n + p) * size_k + (block.col() * 32 + q)) : ((block.col() * 32 + p) * size_n + (n + q)))];

    #pragma unroll 1
    while (iter.valid()) {
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
        iter.next();
        if (iter.valid()) {
            block = *iter;
            buffer_a = *(float4 *) &matrix_a[offset_a + (block.idx() * 32 * 32) + p * 32 + q];
            buffer_b = *(float4 *) &matrix_b[offset_b + (tr_b ? ((n + p) * size_k + (block.col() * 32 + q)) : ((block.col() * 32 + p) * size_n + (n + q)))];
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
    matrix_c[offset_c + (m + r + 0) * size_n + (n + s + 0)] = accum[0][0];
    matrix_c[offset_c + (m + r + 0) * size_n + (n + s + 1)] = accum[0][1];
    matrix_c[offset_c + (m + r + 1) * size_n + (n + s + 0)] = accum[1][0];
    matrix_c[offset_c + (m + r + 1) * size_n + (n + s + 1)] = accum[1][1];
}

/**
 * Compute single-precision sparse matrix multiplication with DDS mode.
 * 
 * It multiplies a dense matrix with a sparse matrix and create a new dense
 * matrix from the sparse layout.
 * 
 * Blocks               : (Total Batches, Sparse Rows, Sparse Columns)
 * Threads per Block    : 256
 */
template <bool tr_a, bool tr_b>
__global__ void sparse_smm_dds_32x32x32_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
          float* __restrict__ matrix_c,
    sparse_layout layout, int num_blocks, int size_m, int size_n, int size_k
) {
    __shared__ float shared_a[32 * 33], shared_b[32 * 33];
    float4 buffer_a, buffer_b;
    float accum[2][2] = { 0 };

    // Get an offset of each matrix and calculate mapping indices.
    int offset_a = blockIdx.x * size_m * size_k;
    int offset_b = blockIdx.x * num_blocks * 32 * 32;
    int offset_c = blockIdx.x * size_m * size_n;

    int m = blockIdx.y * 32;
    int n = blockIdx.z * 32;

    int p = threadIdx.x / 8;
    int q = threadIdx.x % 8 * 4;
    int r = threadIdx.x / 16 * 2;
    int s = threadIdx.x % 16 * 2;

    // Prefetch first tiles from matrices in global memory.
    auto iter = layout.begin(blockIdx.z);
    auto block = *iter;
    buffer_a = *(float4 *) &matrix_a[offset_a + (tr_a ? ((block.row() * 32 + p) * size_m + (m + q)) : ((m + p) * size_k + (block.row() * 32 + q)))];
    buffer_b = *(float4 *) &matrix_b[offset_b + (block.idx() * 32 * 32) + p * 32 + q];

    #pragma unroll 1
    while (iter.valid()) {
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
        iter.next();
        if (iter.valid()) {
            block = *iter;
            buffer_a = *(float4 *) &matrix_a[offset_a + (tr_a ? ((block.row() * 32 + p) * size_m + (m + q)) : ((m + p) * size_k + (block.row() * 32 + q)))];
            buffer_b = *(float4 *) &matrix_b[offset_b + (block.idx() * 32 * 32) + p * 32 + q];
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
    matrix_c[offset_c + (m + r + 0) * size_n + (n + s + 0)] = accum[0][0];
    matrix_c[offset_c + (m + r + 0) * size_n + (n + s + 1)] = accum[0][1];
    matrix_c[offset_c + (m + r + 1) * size_n + (n + s + 0)] = accum[1][0];
    matrix_c[offset_c + (m + r + 1) * size_n + (n + s + 1)] = accum[1][1];
}


void sparse_smm_32x32x32_kernel_wrapper(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    const torch::Tensor &matrix_c,
    const std::string& mode, const layout_tensors& layout, int num_blocks,
    int num_batches, int size_m, int size_n, int size_k, bool tr_a, bool tr_b
) {
    dim3 blocks;
    if (mode == "sdd") blocks = dim3(num_blocks, num_batches);
    else blocks = dim3(num_batches, size_m / 32, size_n / 32);

    if      ( tr_a &&  tr_b && mode == "sdd") sparse_smm_sdd_32x32x32_kernel< true,  true><<<blocks, 256>>>(matrix_a.data_ptr<float>(), matrix_b.data_ptr<float>(), matrix_c.data_ptr<float>(), layout, num_blocks, size_m, size_n, size_k);
    else if ( tr_a &&  tr_b && mode == "dsd") sparse_smm_dsd_32x32x32_kernel< true,  true><<<blocks, 256>>>(matrix_a.data_ptr<float>(), matrix_b.data_ptr<float>(), matrix_c.data_ptr<float>(), layout, num_blocks, size_m, size_n, size_k);
    else if ( tr_a &&  tr_b && mode == "dds") sparse_smm_dds_32x32x32_kernel< true,  true><<<blocks, 256>>>(matrix_a.data_ptr<float>(), matrix_b.data_ptr<float>(), matrix_c.data_ptr<float>(), layout, num_blocks, size_m, size_n, size_k);
    else if (!tr_a &&  tr_b && mode == "sdd") sparse_smm_sdd_32x32x32_kernel<false,  true><<<blocks, 256>>>(matrix_a.data_ptr<float>(), matrix_b.data_ptr<float>(), matrix_c.data_ptr<float>(), layout, num_blocks, size_m, size_n, size_k);
    else if (!tr_a &&  tr_b && mode == "dsd") sparse_smm_dsd_32x32x32_kernel<false,  true><<<blocks, 256>>>(matrix_a.data_ptr<float>(), matrix_b.data_ptr<float>(), matrix_c.data_ptr<float>(), layout, num_blocks, size_m, size_n, size_k);
    else if (!tr_a &&  tr_b && mode == "dds") sparse_smm_dds_32x32x32_kernel<false,  true><<<blocks, 256>>>(matrix_a.data_ptr<float>(), matrix_b.data_ptr<float>(), matrix_c.data_ptr<float>(), layout, num_blocks, size_m, size_n, size_k);
    else if ( tr_a && !tr_b && mode == "ssd") sparse_smm_sdd_32x32x32_kernel< true, false><<<blocks, 256>>>(matrix_a.data_ptr<float>(), matrix_b.data_ptr<float>(), matrix_c.data_ptr<float>(), layout, num_blocks, size_m, size_n, size_k);
    else if ( tr_a && !tr_b && mode == "dsd") sparse_smm_dsd_32x32x32_kernel< true, false><<<blocks, 256>>>(matrix_a.data_ptr<float>(), matrix_b.data_ptr<float>(), matrix_c.data_ptr<float>(), layout, num_blocks, size_m, size_n, size_k);
    else if ( tr_a && !tr_b && mode == "dds") sparse_smm_dds_32x32x32_kernel< true, false><<<blocks, 256>>>(matrix_a.data_ptr<float>(), matrix_b.data_ptr<float>(), matrix_c.data_ptr<float>(), layout, num_blocks, size_m, size_n, size_k);
    else if (!tr_a && !tr_b && mode == "sdd") sparse_smm_sdd_32x32x32_kernel<false, false><<<blocks, 256>>>(matrix_a.data_ptr<float>(), matrix_b.data_ptr<float>(), matrix_c.data_ptr<float>(), layout, num_blocks, size_m, size_n, size_k);
    else if (!tr_a && !tr_b && mode == "dsd") sparse_smm_dsd_32x32x32_kernel<false, false><<<blocks, 256>>>(matrix_a.data_ptr<float>(), matrix_b.data_ptr<float>(), matrix_c.data_ptr<float>(), layout, num_blocks, size_m, size_n, size_k);
    else if (!tr_a && !tr_b && mode == "dds") sparse_smm_dds_32x32x32_kernel<false, false><<<blocks, 256>>>(matrix_a.data_ptr<float>(), matrix_b.data_ptr<float>(), matrix_c.data_ptr<float>(), layout, num_blocks, size_m, size_n, size_k);
}
