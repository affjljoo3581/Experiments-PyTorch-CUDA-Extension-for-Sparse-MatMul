#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <string>
#include <torch/extension.h>

#include "sparse_kernels.h"
#include "sparse_layout.cuh"


struct __align__(16) half8 { half2 x, y, z, w; };


/**
 * Compute half-precision sparse matrix multiplication with SDD mode.
 * 
 * It multiplies a dense matrix with another dense matrix and create a new
 * sparse matrix from the sparse layout.
 * 
 * Blocks               : (Sparse Blocks, Total Batches)
 * Threads per Block    : 256
 */
template <bool tr_a, bool tr_b>
__global__ void sparse_hmm_sdd_32x32x32_kernel(
    const half* __restrict__ matrix_a,
    const half* __restrict__ matrix_b,
          half* __restrict__ matrix_c,
    sparse_layout layout, int num_blocks, int size_m, int size_n, int size_k
) {
    __shared__ half2 shared_a[32 * 16 + 32], shared_b[32 * 16 + 32];
    half8 buffer_a, buffer_b, neighbor;
    half2 accum[4][2] = {{{ 0, 0 }}};

    // Load current block and get corresponding row and column positions.
    auto block = layout.get(blockIdx.x);
    int m = block.row() * 32;
    int n = block.col() * 32;

    // Get an offset of each matrix and calculate mapping indices.
    int offset_a = blockIdx.y * size_m * size_k;
    int offset_b = blockIdx.y * size_k * size_n;
    int offset_c = (blockIdx.y * num_blocks + block.idx()) * 32 * 32;

    int p = threadIdx.x / 4;
    int q = threadIdx.x % 4 * 8;
    int r = threadIdx.x / 16 * 4;
    int s = threadIdx.x % 16 * 2;

    // Prefetch first tiles from matrices in global memory.
    buffer_a = *(half8 *) &matrix_a[0];//offset_a + (tr_a ? ((0 + p) * size_m + (m + q)) : ((m + p) * size_k + (0 + q)))];
    buffer_b = *(half8 *) &matrix_b[0];//offset_b + (tr_b ? ((n + p) * size_k + (0 + q)) : ((0 + p) * size_n + (n + q)))];

    #pragma unroll 1
    for (int k = 32; k <= size_k; k += 32) {
        if (tr_a) {
            neighbor.x = __shfl_xor_sync(0xffffffff, buffer_a.x, 16, warpSize);
            neighbor.y = __shfl_xor_sync(0xffffffff, buffer_a.y, 16, warpSize);
            neighbor.z = __shfl_xor_sync(0xffffffff, buffer_a.z, 16, warpSize);
            neighbor.w = __shfl_xor_sync(0xffffffff, buffer_a.w, 16, warpSize);

            buffer_a.x = (p % 2 == 0) ? __lows2half2(buffer_a.x, neighbor.x) : __highs2half2(neighbor.x, buffer_a.x);
            buffer_a.y = (p % 2 == 0) ? __lows2half2(buffer_a.y, neighbor.y) : __highs2half2(neighbor.y, buffer_a.y);
            buffer_a.z = (p % 2 == 0) ? __lows2half2(buffer_a.z, neighbor.z) : __highs2half2(neighbor.z, buffer_a.z);
            buffer_a.w = (p % 2 == 0) ? __lows2half2(buffer_a.w, neighbor.w) : __highs2half2(neighbor.w, buffer_a.w);
        }

        if (!tr_b) {
            neighbor.x = __shfl_xor_sync(0xffffffff, buffer_b.x, 16, warpSize);
            neighbor.y = __shfl_xor_sync(0xffffffff, buffer_b.y, 16, warpSize);
            neighbor.z = __shfl_xor_sync(0xffffffff, buffer_b.z, 16, warpSize);
            neighbor.w = __shfl_xor_sync(0xffffffff, buffer_b.w, 16, warpSize);

            buffer_b.x = (p % 2 == 0) ? __lows2half2(buffer_b.x, neighbor.x) : __highs2half2(neighbor.x, buffer_b.x);
            buffer_b.y = (p % 2 == 0) ? __lows2half2(buffer_b.y, neighbor.y) : __highs2half2(neighbor.y, buffer_b.y);
            buffer_b.z = (p % 2 == 0) ? __lows2half2(buffer_b.z, neighbor.z) : __highs2half2(neighbor.z, buffer_b.z);
            buffer_b.w = (p % 2 == 0) ? __lows2half2(buffer_b.w, neighbor.w) : __highs2half2(neighbor.w, buffer_b.w);
        }

        // Commit the prefetched tiles to the shared memory storage.
        __syncthreads();
        shared_a[tr_a ? ((q + p % 2 + 0) * 16 + (p + q + 0) / 2) : (p * 16 + (0 + q + 0) / 2)] = buffer_a.x;
        shared_a[tr_a ? ((q + p % 2 + 2) * 16 + (p + q + 2) / 2) : (p * 16 + (0 + q + 2) / 2)] = buffer_a.y;
        shared_a[tr_a ? ((q + p % 2 + 4) * 16 + (p + q + 4) / 2) : (p * 16 + (0 + q + 4) / 2)] = buffer_a.z;
        shared_a[tr_a ? ((q + p % 2 + 6) * 16 + (p + q + 6) / 2) : (p * 16 + (0 + q + 6) / 2)] = buffer_a.w;
        shared_b[tr_b ? (p * 16 + (0 + q + 0) / 2) : ((q + p % 2 + 0) * 16 + (p + q + 0) / 2)] = buffer_b.x;
        shared_b[tr_b ? (p * 16 + (0 + q + 2) / 2) : ((q + p % 2 + 2) * 16 + (p + q + 2) / 2)] = buffer_b.y;
        shared_a[tr_b ? (p * 16 + (0 + q + 4) / 2) : ((q + p % 2 + 4) * 16 + (p + q + 4) / 2)] = buffer_b.z;
        shared_a[tr_b ? (p * 16 + (0 + q + 6) / 2) : ((q + p % 2 + 6) * 16 + (p + q + 6) / 2)] = buffer_b.w;
        __syncthreads();

        // Prefetch next tiles from matrices in global memory.
        if (k < size_k) {
            buffer_a = *(half8 *) &matrix_a[0];//offset_a + (tr_a ? ((k + p) * size_m + (m + q)) : ((m + p) * size_k + (k + q)))];
            buffer_b = *(half8 *) &matrix_b[0];//offset_b + (tr_b ? ((n + p) * size_k + (k + q)) : ((k + p) * size_n + (n + q)))];
        }

        // Accumulate the tiled matrix multiplications by loading sliced vectors
        // from the shared memory to local register file.
        #pragma unroll
        for (int i = 0; i < 16; ++ i) {
            half2 reg_a[4], reg_b[2];

            reg_a[0] = shared_a[(r + 0) * 16 + i + (r / 2 + 0) * 0];
            reg_a[1] = shared_a[(r + 1) * 16 + i + (r / 2 + 0) * 0];
            reg_a[2] = shared_a[(r + 2) * 16 + i + (r / 2 + 1) * 0];
            reg_a[3] = shared_a[(r + 3) * 16 + i + (r / 2 + 1) * 0];
            reg_b[0] = shared_b[(s + 0) * 16 + i + s / 2 * 0];
            reg_b[1] = shared_b[(s + 1) * 16 + i + s / 2 * 0];
            /*
            accum[0][0] += reg_a[0] * reg_b[0];
            accum[0][1] += reg_a[0] * reg_b[1];
            accum[1][0] += reg_a[1] * reg_b[0];
            accum[1][1] += reg_a[1] * reg_b[1];
            accum[2][0] += reg_a[2] * reg_b[0];
            accum[2][1] += reg_a[2] * reg_b[1];
            accum[3][0] += reg_a[3] * reg_b[0];
            accum[3][1] += reg_a[3] * reg_b[1];
            */
        }
    }

    // Reduce the accumulated `half2` array to `half` by summing the low and
    // high halves.
    half result[4][2];

    result[0][0] = __low2half(accum[0][0]) + __high2half(accum[0][0]);
    result[0][1] = __low2half(accum[0][1]) + __high2half(accum[0][1]);
    result[1][0] = __low2half(accum[1][0]) + __high2half(accum[1][0]);
    result[1][1] = __low2half(accum[1][1]) + __high2half(accum[1][1]);
    result[2][0] = __low2half(accum[2][0]) + __high2half(accum[2][0]);
    result[2][1] = __low2half(accum[2][1]) + __high2half(accum[2][1]);
    result[3][0] = __low2half(accum[3][0]) + __high2half(accum[3][0]);
    result[3][1] = __low2half(accum[3][1]) + __high2half(accum[3][1]);

    // Write the accumulated results to the output matrix.
    *(half2 *) &matrix_c[offset_c + (r + 0) * 32 + s] = *(half2 *) result[0];
    *(half2 *) &matrix_c[offset_c + (r + 1) * 32 + s] = *(half2 *) result[1];
    *(half2 *) &matrix_c[offset_c + (r + 2) * 32 + s] = *(half2 *) result[2];
    *(half2 *) &matrix_c[offset_c + (r + 3) * 32 + s] = *(half2 *) result[3];
}

/**
 * Compute half-precision sparse matrix multiplication with DSD mode.
 * 
 * It multiplies a sparse matrix with a dense matrix and create a new dense
 * matrix from the sparse layout.
 * 
 * Blocks               : (Total Batches, Sparse Rows, Sparse Columns)
 * Threads per Block    : 128
 */
/*
template <bool tr_a, bool tr_b>
__global__ void sparse_hmm_dsd_32x32x32_kernel(
    const half* __restrict__ matrix_a,
    const half* __restrict__ matrix_b,
          half* __restrict__ matrix_c,
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
*/

/**
 * Compute half-precision sparse matrix multiplication with DDS mode.
 * 
 * It multiplies a dense matrix with a sparse matrix and create a new dense
 * matrix from the sparse layout.
 * 
 * Blocks               : (Total Batches, Sparse Rows, Sparse Columns)
 * Threads per Block    : 128
 */
/*
template <bool tr_a, bool tr_b>
__global__ void sparse_hmm_dds_32x32x32_kernel(
    const half* __restrict__ matrix_a,
    const half* __restrict__ matrix_b,
          half* __restrict__ matrix_c,
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
}*/


void sparse_hmm_32x32x32_kernel_wrapper(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    const torch::Tensor &matrix_c,
    const std::string& mode, const layout_tensors& layout, int num_blocks,
    int num_batches, int size_m, int size_n, int size_k, bool tr_a, bool tr_b
) {
    dim3 blocks;
    if (mode == "sdd") blocks = dim3(num_blocks, num_batches);
    else blocks = dim3(num_batches, size_m / 32, size_n / 32);

    if      ( tr_a &&  tr_b && mode == "sdd") sparse_hmm_sdd_32x32x32_kernel< true,  true><<<blocks, 128>>>((half *) matrix_a.data_ptr<at::Half>(), (half *) matrix_b.data_ptr<at::Half>(), (half *) matrix_c.data_ptr<at::Half>(), layout, num_blocks, size_m, size_n, size_k);
    //else if ( tr_a &&  tr_b && mode == "dsd") sparse_hmm_dsd_32x32x32_kernel< true,  true><<<blocks, 128>>>((half *) matrix_a.data_ptr<at::Half>(), (half *) matrix_b.data_ptr<at::Half>(), (half *) matrix_c.data_ptr<at::Half>(), layout, num_blocks, size_m, size_n, size_k);
    //else if ( tr_a &&  tr_b && mode == "dds") sparse_hmm_dds_32x32x32_kernel< true,  true><<<blocks, 128>>>((half *) matrix_a.data_ptr<at::Half>(), (half *) matrix_b.data_ptr<at::Half>(), (half *) matrix_c.data_ptr<at::Half>(), layout, num_blocks, size_m, size_n, size_k);
    else if (!tr_a &&  tr_b && mode == "sdd") sparse_hmm_sdd_32x32x32_kernel<false,  true><<<blocks, 128>>>((half *) matrix_a.data_ptr<at::Half>(), (half *) matrix_b.data_ptr<at::Half>(), (half *) matrix_c.data_ptr<at::Half>(), layout, num_blocks, size_m, size_n, size_k);
    //else if (!tr_a &&  tr_b && mode == "dsd") sparse_hmm_dsd_32x32x32_kernel<false,  true><<<blocks, 128>>>((half *) matrix_a.data_ptr<at::Half>(), (half *) matrix_b.data_ptr<at::Half>(), (half *) matrix_c.data_ptr<at::Half>(), layout, num_blocks, size_m, size_n, size_k);
    //else if (!tr_a &&  tr_b && mode == "dds") sparse_hmm_dds_32x32x32_kernel<false,  true><<<blocks, 128>>>((half *) matrix_a.data_ptr<at::Half>(), (half *) matrix_b.data_ptr<at::Half>(), (half *) matrix_c.data_ptr<at::Half>(), layout, num_blocks, size_m, size_n, size_k);
    else if ( tr_a && !tr_b && mode == "ssd") sparse_hmm_sdd_32x32x32_kernel< true, false><<<blocks, 128>>>((half *) matrix_a.data_ptr<at::Half>(), (half *) matrix_b.data_ptr<at::Half>(), (half *) matrix_c.data_ptr<at::Half>(), layout, num_blocks, size_m, size_n, size_k);
    //else if ( tr_a && !tr_b && mode == "dsd") sparse_hmm_dsd_32x32x32_kernel< true, false><<<blocks, 128>>>((half *) matrix_a.data_ptr<at::Half>(), (half *) matrix_b.data_ptr<at::Half>(), (half *) matrix_c.data_ptr<at::Half>(), layout, num_blocks, size_m, size_n, size_k);
    //else if ( tr_a && !tr_b && mode == "dds") sparse_hmm_dds_32x32x32_kernel< true, false><<<blocks, 128>>>((half *) matrix_a.data_ptr<at::Half>(), (half *) matrix_b.data_ptr<at::Half>(), (half *) matrix_c.data_ptr<at::Half>(), layout, num_blocks, size_m, size_n, size_k);
    else if (!tr_a && !tr_b && mode == "sdd") sparse_hmm_sdd_32x32x32_kernel<false, false><<<blocks, 128>>>((half *) matrix_a.data_ptr<at::Half>(), (half *) matrix_b.data_ptr<at::Half>(), (half *) matrix_c.data_ptr<at::Half>(), layout, num_blocks, size_m, size_n, size_k);
    //else if (!tr_a && !tr_b && mode == "dsd") sparse_hmm_dsd_32x32x32_kernel<false, false><<<blocks, 128>>>((half *) matrix_a.data_ptr<at::Half>(), (half *) matrix_b.data_ptr<at::Half>(), (half *) matrix_c.data_ptr<at::Half>(), layout, num_blocks, size_m, size_n, size_k);
    //else if (!tr_a && !tr_b && mode == "dds") sparse_hmm_dds_32x32x32_kernel<false, false><<<blocks, 128>>>((half *) matrix_a.data_ptr<at::Half>(), (half *) matrix_b.data_ptr<at::Half>(), (half *) matrix_c.data_ptr<at::Half>(), layout, num_blocks, size_m, size_n, size_k);
}
