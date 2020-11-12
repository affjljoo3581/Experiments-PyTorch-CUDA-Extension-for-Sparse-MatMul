#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <torch/extension.h>

#include "sparse_kernels.h"
#include "sparse_layout.cuh"
#include "tiling_mma.cuh"

/*
#define LAUNCH_BOUNDS(T, ROWS, COLUMNS)                                     \
    __launch_bounds__(tile<T, ROWS, COLUMNS>::THREADS,                      \
                      2048 / tile<T, ROWS, COLUMNS>::THREADS)

/**
 * Compute sparse matrix multiplication with SDD mode.
 * 
 * It computes a multiplication with a dense matrix with other dense matrix and
 * create a new sparse matrix through corresponding sparse layout.
 * 
 * Blocks               : (Sparse Blocks, Total Batches)
 * Threads per Block    : 256 - for single precision
 *                        128 - for half precision
 *
__global__ void LAUNCH_BOUNDS(float, 32, 8) sparse_matmul_sdd_32x32x8_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
          float* __restrict__ matrix_c,
    sparse_layout layout, uint num_blocks,
    uint size_m, uint size_n, uint size_k,
    bool trans_a, bool trans_b
) {
    float accumulator[4] = { 0.0f, };

    uint lane_idx = threadIdx.x % warpSize;
    uint warp_idx = threadIdx.x / warpSize;

    // Fetch current block and get corresponding row and column indices.
    auto block = layout.get(blockIdx.x);
    uint m = block.row() * 32;
    uint n = block.col() * 32;

    // Define shared tile storages, loaders and accumulator.
    __shared__ tile<float, 32, 8>::storage storage_a, storage_b;

    tile<float, 32, 8>::loader loader_a(
        &matrix_a[blockIdx.y * size_m * size_k],
        storage_a, trans_a ? size_m : size_k, trans_a
    );
    tile<float, 32, 8>::loader loader_b(
        &matrix_b[blockIdx.y * size_k * size_n],
        storage_b, trans_b ? size_k : size_n, !trans_b
    );

    // Prefetch first tiles from the global memory.
    loader_a.prefetch(trans_a ? 0 : m, trans_a ? m : 0);
    loader_b.prefetch(trans_b ? n : 0, trans_b ? 0 : n);

    #pragma unroll 1
    for (uint k = 0; k < size_k; k += 8) {
        // Move the prefetched global memory data to the shared memory storage.
        loader_a.commit(k / 8 % 2);
        loader_b.commit(k / 8 % 2);
        __syncthreads();

        // Prefetch next tiles from the global memory if available.
        if (k + 8 < size_k) {
            loader_a.prefetch(trans_a ? k + 8 : m, trans_a ? m : k + 8);
            loader_b.prefetch(trans_b ? n : k + 8, trans_b ? k + 8 : n);
        }

        // Accumulate the tiled matrix multiplications by loading the sliced
        // vectors from the shared memory storage to local register files.
        #pragma unroll
        for (uint i = 0; i < 8; ++ i) {
            float local_a[4], local_b;

            #pragma unroll
            for (uint j = 0; j < 4; ++ j)
                local_a[j] = storage_a.get(k / 8 % 2, warp_idx * 4 + j, i);
            local_b = storage_b.get(k / 8 % 2, lane_idx, i);

            #pragma unroll
            for (uint j = 0; j < 4; ++ j)
                accumulator[j] += local_a[j] * local_b;
        }
    }

    // Write the accumulated matrix multiplication results to the global memory.
    for (uint i = 0; i < 4; ++ i)
        matrix_c[(blockIdx.y * num_blocks + block.idx()) * 32 * 32
                 + (warp_idx * 4 + i) * 32 + lane_idx] = accumulator[i];
}
/*/

class tile_storage {
public:
    constexpr static uint ROWS      = 32;
    constexpr static uint COLUMNS   = 8;

    constexpr static uint SKEW      = 1;
    constexpr static uint STRIDE    = COLUMNS;

    constexpr static uint SIZE      = (ROWS * STRIDE + COLUMNS + 32 - 1) / 32 * 32;

    __device__ __forceinline__ float& get(uint page, uint i, uint j) {
        return buffers[page][i * STRIDE + j + i * STRIDE / 32];
    }
private:
    float buffers[2][SIZE];
};


class tile_loader {
public:
    __device__ __forceinline__ tile_loader(const float* __restrict__ src,
                                           tile_storage& storage,
                                           uint stride, bool trans)
        : src(src), storage(storage), stride(stride)
    {
        /*
        uint x = threadIdx.x % tile_storage::COLUMNS;
        uint y = threadIdx.x / tile_storage::COLUMNS;

        if (trans) {
            from.x = to.y = x % tile_storage::ROWS;
            from.y = to.x = x / tile_storage::ROWS * tile_storage::ROWS + y;
        } else {
            from = to = { x, y };
        }*/
        //if (trans) {
            from.x = to.y = threadIdx.x % tile_storage::ROWS;
            from.y = to.x = threadIdx.x / tile_storage::ROWS;
        //} else {
        //    from = to = { threadIdx.x % tile_storage::COLUMNS, threadIdx.x / tile_storage::COLUMNS };
        //}
    }

    __device__ __forceinline__ void prefetch(uint row, uint col) {
        buffer = src[(row + from.y) * stride + (col + from.x)];
    }

    __device__ __forceinline__ void commit(uint page) {
        storage.get(page, to.y, to.x) = buffer;
    }
private:
    const float* __restrict__ src;
    uint stride;

    tile_storage& storage;
    float buffer;

    uint2 from, to;
};


/**
 * Compute sparse matrix multiplication with SDD mode.
 * 
 * It multiplies a dense matrix with other dense matrix and create a new sparse
 * matrix through corresponding sparse layout.
 * 
 * Blocks               : (Sparse Blocks, Total Batches)
 * Threads per Block    : 256
 */
__global__ void __launch_bounds__(256, 8) sparse_matmul_sdd_32x32x8_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
          float* __restrict__ matrix_c,
    sparse_layout layout, uint num_blocks,
    uint size_m, uint size_n, uint size_k,
    bool trans_a, bool trans_b
) {
    uint lane_idx = threadIdx.x % 32;
    uint warp_idx = threadIdx.x / 32;

    // Define shared tile storages and tile loaders.
    __shared__ tile_storage tile_a, tile_b;

    tile_loader loader_a(matrix_a + blockIdx.y * size_m * size_k,
                         tile_a, trans_a ? size_m : size_k, trans_a);
    tile_loader loader_b(matrix_b + blockIdx.y * size_k * size_n,
                         tile_b, trans_b ? size_k : size_n, !trans_b);

    // Fetch current block and get corresponding row and column indices.
    auto block = layout.get(blockIdx.x);
    uint m = block.row() * 32;
    uint n = block.col() * 32;

    // Prefetch first tiles from the global memory.
    loader_a.prefetch(trans_a ? 0 : m, trans_a ? m : 0);
    loader_b.prefetch(trans_b ? n : 0, trans_b ? 0 : n);

    float accumulator[4] = { 0.0f, };

    #pragma unroll 1
    for (uint k = 0; k < size_k; k += 8) {
        uint page = k / 8 % 2;
        uint next_k = k + 8;

        // Move the prefetched global memory values to the shared memory.
        loader_a.commit(page);
        loader_b.commit(page);
        __syncthreads();

        // Prefetch the next tiles from the global memory.
        if (next_k < size_k) {
            //loader_a.prefetch(trans_a ? next_k : m, trans_a ? m : next_k);
            loader_a.prefetch(trans_a ? m : next_k, trans_a ? next_k : m);
            loader_b.prefetch(trans_b ? n : next_k, trans_b ? next_k : n);
        }

        // Accumulate the tiled matrix multiplications by loading the sliced
        // vectors from shared memory to local register files.
        #pragma unroll
        for (uint i = 0; i < 8; ++ i) {
            float local_a[4], local_b;

            #pragma unroll
            for (uint j = 0; j < 4; ++ j)
                local_a[j] = tile_a.get(page, warp_idx * 4 + j, i);
            local_b = tile_b.get(page, lane_idx, i);

            #pragma unroll
            for (uint j = 0; j < 4; ++ j)
                accumulator[j] += local_a[j] * local_b;
        }
    }

    #pragma unroll
    for (uint i = 0; i < 4; ++ i)
        matrix_c[(blockIdx.y * num_blocks + block.idx()) * 32 * 32
                 + (warp_idx * 4 + i) * 32
                 + lane_idx] = accumulator[i];
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

    auto kernel = mode == "sdd" ? sparse_matmul_sdd_32x32x8_kernel :
                  mode == "dsd" ? sparse_matmul_sdd_32x32x8_kernel :
                                  sparse_matmul_sdd_32x32x8_kernel;
    kernel<<<blocks, tile<float, 32, 8>::THREADS>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
        layout, num_blocks, size_m, size_n, size_k,
        trans_a, trans_b
    );

    // Return the output tensor with multiple batch dimensions.
    return c.reshape(shape);
}
