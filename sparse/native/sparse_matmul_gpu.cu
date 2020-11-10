#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <torch/extension.h>

#include "sparse_ops.h"
#include "layout_utils.cuh"
#include "tiling_utils.cuh"


#define LAUNCH_BOUNDS_TILE(T, ROWS, COLUMNS)                        \
    __launch_bounds__(tile<T, ROWS, COLUMNS>::THREADS,              \
                      1024 / tile<T, ROWS, COLUMNS>::THREADS)

#define DISPATCH_KERNEL_WITH_TYPE(TYPE, ...)                        \
    [&] {   if (TYPE == at::ScalarType::Float) {                    \
                using T = float; using U = float; __VA_ARGS__();    \
            } else if (TYPE == at::ScalarType::Half) {              \
                using T = at::Half; using U = half; __VA_ARGS__();  \
            }                                                       }()


/**
 * Compute sparse matrix multiplication with SDD mode.
 * 
 * It computes a multiplication with a dense matrix with other dense matrix and
 * create a new sparse matrix through corresponding sparse layout.
 * 
 * Blocks               : (Sparse Blocks, Total Batches)
 * Threads per Block    : 256 - for single precision
 *                        128 - for half precision
 */
template <typename T>
__global__ void LAUNCH_BOUNDS_TILE(T, 32, 8) sparse_matmul_sdd_32x32x8_kernel(
    const T* __restrict__ matrix_a,
    const T* __restrict__ matrix_b,
          T* __restrict__ matrix_c,
    sparse_layout layout, uint num_blocks,
    uint size_m, uint size_n, uint size_k,
    bool trans_a, bool trans_b
) {
    // Fetch current block and get corresponding row and column indices.
    auto block = layout.get(blockIdx.x);
    uint m = block.row() * 32;
    uint n = block.col() * 32;

    // Define shared tile storages, loaders and accumulator.
    __shared__ tile<T, 32, 8>::storage storage_a, storage_b;

    tile<T, 32, 8>::loader loader_a(&matrix_a[blockIdx.y * size_m * size_k],
                                    storage_a, trans_a ? size_m : size_k,
                                    trans_a);
    tile<T, 32, 8>::loader loader_b(&matrix_b[blockIdx.y * size_k * size_n],
                                    storage_b, trans_b ? size_k : size_n,
                                    trans_b);

    tile<T, 32, 8>::accumulator accum(storage_a, storage_b);

    // Prefetch first tiles from the global memory.
    loader_a.prefetch(trans_a ? 0 : m, trans_a ? m : 0);
    loader_b.prefetch(trans_b ? n : 0, trans_b ? 0 : n);

    #pragma unroll 1
    for (uint next_k = 8; next_k < size_k; next_k += 8) {
        // Move the prefetched global memory data to the shared memory storage.
        loader_a.commit(k / 8 % 2);
        loader_b.commit(k / 8 % 2);
        __syncthreads();

        // Prefetch next tiles from the global memory if available.
        for (next_k < size_k) {
            loader_a.prefetch(trans_a ? next_k : m, trans_a ? m : next_k);
            loader_b.prefetch(trans_b ? n : next_k, trans_b ? next_k : n);
        }

        // Accumulate the tiled matrix multiplications by loading the sliced
        // vectors from the shared memory storage to local register files by
        // using the accumulator object.
        accum.product(k / 8 % 2);
    }

    // Write the accumulated matrix multiplication results to the global memory.
    accum.apply(&matrix_c[(blockIdx.y * num_blocks + block.idx()) * 32 * 32],
                0, 0, 32);
}


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

    DISPATCH_KERNEL_WITH_TYPE(a.scalar_type(), ([&] {
        auto kernel = mode == "sdd" ? sparse_matmul_sdd_32x32x8_kernel<T> :
                      mode == "dsd" ? sparse_matmul_sdd_32x32x8_kernel<T> :
                                      sparse_matmul_sdd_32x32x8_kernel<T>;
        kernel<<<blocks, tile<T, 32, 8>::THREADS>>>(
            (U*) a.data_ptr<T>(), (U*) b.data_ptr<T>(), (U*) c.data_ptr<T>(),
            layout, num_blocks, size_m, size_n, size_k,
            trans_a, trans_b
        );
    }));

    // Return the output tensor with multiple batch dimensions.
    return c.reshape(shape);
}
