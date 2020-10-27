#include <cuda.h>
#include <cuda_runtime.h>

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <cuda_fp16.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include "utils.cuh"


__device__ void load_matrix_tile(
    const   float*      __restrict__    matrix,
            float*      __restrict__    local_tile,
            int64_t                     offset,
            int64_t                     start_row,
            int64_t                     start_col,
            int64_t                     total_rows,
            int64_t                     total_cols,
            int64_t                     stride_matrix,
            int64_t                     stride_local_tile,
            bool                        transpose
) {
    int64_t local_tile_idx =
        transpose ? threadIdx.x * stride_local_tile + threadIdx.y
                  : threadIdx.y * stride_local_tile + threadIdx.x;
    int64_t row_idx = start_row + threadIdx.y;
    int64_t col_idx = start_col + threadIdx.x;

    local_tile[local_tile_idx] =
        (row_idx < total_rows && col_idx < total_cols)
        ? matrix[offset + row_idx * stride_matrix + col_idx]
        : 0.0;

    __syncthreads();
}


__device__ void load_matrix_tile(
    const   half*      __restrict__     matrix,
            half*      __restrict__     local_tile,
            int64_t                     offset,
            int64_t                     start_row,
            int64_t                     start_col,
            int64_t                     total_rows,
            int64_t                     total_cols,
            int64_t                     stride_matrix,
            int64_t                     stride_local_tile,
            bool                        transpose
) {
    int64_t row_idx = start_row + threadIdx.y;
    int64_t col_idx = start_col + threadIdx.x;

    // Instead of reading a single `half`, loading a couple of `half`s
    // (i.e. `half2`) increases performance because accessing 4-bytes
    // simultaneously is more efficient than accessing 2-bytes separately.
    half2 coupled =
        (row_idx < total_rows && col_idx < total_cols)
        ? *((half2*) matrix + (offset + row_idx * stride_matrix + col_idx) / 2)
        : __float2half2_rn(0);

    if (transpose) {
        // To transpose the coupled matrix, we need to decompose the tangled
        // `half2`s and recombine `half`s horizontally. Get a coupled `half2`
        // from other row in same warp and swap the corresponding `half`s.
        half2 neighbor = __shfl_sync(
            0xffffffff,
            coupled,
            ((threadIdx.y + 1) * blockDim.x * threadIdx.x) % (2 * blockDim.x),
            warpSize
        );

        if (threadIdx.y % 2 == 0)
            coupled = __lows2half2(coupled, neighbor);
        else
            coupled = __highs2half2(neighbor, coupled);

        // After recombining `half2`s, rearrange the elements.
        *((half2*) local_tile
          + (threadIdx.x * 2 + threadIdx.y % 2) * stride_local_tile
          + threadIdx.y / 2) = coupled;
    }
    else
        *((half2*) local_tile
          + threadIdx.y * stride_local_tile
          + threadIdx.x) = coupled;

    __syncthreads();
}
