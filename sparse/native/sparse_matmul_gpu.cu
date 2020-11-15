#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <torch/extension.h>

#include "sparse_kernels.h"
#include "sparse_layout.cuh"


/**
 * Compute sparse matrix multiplication with SDD mode.
 * 
 * It multiplies a dense matrix with other dense matrix and create a new sparse
 * matrix through corresponding sparse layout.
 * 
 * Blocks               : (Sparse Blocks, Total Batches)
 * Threads per Block    : 256
 */
template <bool tr_a, bool tr_b>
__global__ void sparse_matmul_sdd_32x32x32_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
          float* __restrict__ matrix_c,
    sparse_layout layout, int num_blocks,
    int size_m, int size_n, int size_k
) {
    __shared__ float shared_a[32 * 33], shared_b[32 * 33];
    float4 buffer_a, buffer_b;
    float accum[2][2] = { 0 };

    // Get a stride and offset of each matrix and calculate mapping indices.
    int offset_a = blockIdx.y * size_m * size_k + (tr_a ? m : m * size_k);
    int offset_b = blockIdx.y * size_k * size_n + (tr_b ? n * size_k : n);
    int stride_a = tr_a ? size_m : size_k;
    int stride_b = tr_b ? size_k : size_n;

    int p = threadIdx.x / 8;
    int q = threadIdx.x % 8 * 4;
    int r = threadIdx.x / 16 * 2;
    int s = threadIdx.x % 16 * 2;

    // Fetch current block and get corresponding row and column positions.
    auto block = layout.get(blockIdx.x);
    int m = block.row() * 32;
    int n = block.col() * 32;

    // Prefetch first tiles from matrices in global memory.
    buffer_a = *(float4 *) &matrix_a[offset_a + p * stride_a + q];
    buffer_b = *(float4 *) &matrix_b[offset_b + p * stride_b + q];

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
            buffer_a = *(float4 *) &matrix_a[
                offset_a + (tr_a ? k * size_m : k) + p * stride_a + q
            ];
            buffer_b = *(float4 *) &matrix_b[
                offset_b + (tr_b ? k : k * size_n) + p * stride_b + q
            ];
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
    int offset_c = (blockIdx.y * num_blocks + block.idx()) * 32 * 32;

    matrix_c[offset_c + (r + 0) * 32 + (s + 0)] = accum[0][0];
    matrix_c[offset_c + (r + 0) * 32 + (s + 1)] = accum[0][1];
    matrix_c[offset_c + (r + 1) * 32 + (s + 0)] = accum[1][0];
    matrix_c[offset_c + (r + 1) * 32 + (s + 1)] = accum[1][1];
}


torch::Tensor sparse_matmul(
    torch::Tensor a, torch::Tensor b, const std::string& mode,
    const layout_tensors& row_layout, const layout_tensors& col_layout,
    bool tr_a, bool tr_b
) {
    // Select current sparse layout by the given sparse mode.
    auto layout = (mode == "sdd"
                   || mode == "dsd" && !tr_a
                   || mode == "dds" && tr_b) ? row_layout : col_layout;
    uint num_blocks = std::get<0>(layout).size(0) / 2;
    uint sparse_width = (std::get<1>(layout).size(0) - 1) * 32;

    // Get the dimension sizes from the tensors.
    uint size_m = mode.at(1) == 'd' ? a.size(tr_a ? -1 : -2) : sparse_width;
    uint size_n = mode.at(2) == 'd' ? b.size(tr_b ? -2 : -1) : sparse_width;
    uint size_k = mode.at(2) == 'd' ? b.size(tr_b ? -1 : -2)
                                    : a.size(tr_a ? -2 : -1);

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
    kernel<<<blocks, 256>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
        layout, num_blocks, size_m, size_n, size_k //,
    );

    // Return the output tensor with multiple batch dimensions.
    return c.reshape(shape);
}
