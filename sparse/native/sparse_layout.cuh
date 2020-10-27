#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <tuple>
#include <sys/types.h>
#include <torch/extension.h>

#include "sparse_ops.h"

#define MAX_ROW_BLOCKS          2048


struct block_desc {
    int32_t idx;
    int32_t packed_pos;

    __device__ __forceinline__ uint row() { return this->packed_pos >> 16; }
    __device__ __forceinline__ uint col() { return this->packed_pos & 0xFF; }
};

struct sparse_iterator {
    block_desc *block;
    block_desc *end;

    __device__ __forceinline__ void next() { this->block ++; }
    __device__ __forceinline__ bool valid() { return this->block < this->end; }
    __device__ __forceinline__ block_desc& operator*() { return *this->block; }
};

struct sparse_layout {
    block_desc *blocks;
    int32_t *offset_table;

    sparse_layout(const torch::Tensor& blocks,
                  const torch::Tensor& offset_table)
        : blocks((block_desc *) blocks.data_ptr<int>()),
          offset_table(offset_table.data_ptr<int>()) {}

    sparse_layout(const layout_tensors& layout)
        : sparse_layout(std::get<0>(layout), std::get<1>(layout)) {}

    __device__ __forceinline__ sparse_iterator begin(uint i) {
        return { this->blocks + this->offset_table[i],
                 this->blocks + this->offset_table[i + 1] };
    }

    __device__ __forceinline__ sparse_iterator begin(uint i, int32_t *shared) {
        int32_t start = this->offset_table[i] * 2;
        int32_t end = this->offset_table[i + 1] * 2;

        // Preload the block descriptors to the shared memory in sync.
        uint i = threadIdx.x + threadIdx.y * gridDim.x;
        for (; i < end - start; i += gridDim.x * gridDim.y)
            shared[i] = *((int32_t *) blocks + start + i)
        __syncthreads();

        return { (block_desc *) shared, (block_desc *) (shared + end - start) };
    }
};
