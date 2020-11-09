#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <torch/extension.h>

#include "sparse_ops.h"
#include "sparse_layout.cuh"


class tile_storage {
public:
    constexpr static uint ROWS      = 8;
    constexpr static uint COLUMNS   = 32;

    constexpr static uint SKEW      = 1;
    constexpr static uint STRIDE    = COLUMNS + SKEW;

    constexpr static uint SIZE      = (ROWS * STRIDE + 32 - 1) / 32 * 32;

    __device__ __forceinline__ half& get(uint page, uint i, uint j) {
        return buffers[page][i * STRIDE + j];
    }
private:
    half buffers[2][SIZE];
};


class tile_loader {
public:
    __device__ __forceinline__ tile_loader(const half* __restrict__ src,
                                           tile_storage& storage,
                                           uint stride, bool trans)
        : src(src), storage(storage), stride(stride)
    {
        uint x = threadIdx.x % tile_storage::COLUMNS;
        uint y = threadIdx.y / tile_storage::COLUMNS;

        if (trans) {

        } else {
            from = to = { x % 16 * 2 };
        }

    }
private:
    const half* __restrict__ src;
    uint stride;

    tile_storage& storage;
    float buffer;

    uint2 from, to;
}