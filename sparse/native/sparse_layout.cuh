#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include "sparse_ops.h"


class block_desc {
public:
    __device__ __forceinline__ uint idx() { return block_idx; }
    __device__ __forceinline__ uint row() { return packed_pos >> 16; }
    __device__ __forceinline__ uint col() { return packed_pos & 0xFF; }
private:
    int32_t block_idx;
    int32_t packed_pos;
};


class sparse_iterator {
public:
    __device__ __forceinline__ sparse_iterator(
        const block_desc* __restrict__ base, uint current, uint end
    ) : base(base), current(current), end(end) {}

    __device__ __forceinline__ void next() { ++ current; }
    __device__ __forceinline__ bool valid() { return current < end; }
    __device__ __forceinline__ const block_desc& operator*() {
        return __ldg(&base[current]);
    }
private:
    const block_desc* __restrict__ base;
    uint current, end;
};


class sparse_layout {
public:
    sparse_layout(const torch::Tensor& blocks,
                  const torch::Tensor& offset_table)
        : blocks((block_desc *) blocks.data_ptr<int>()),
          offset_table(offset_table.data_ptr<int>()) {}

    sparse_layout(const layout_tensors& layout)
        : sparse_layout(std::get<0>(layout), std::get<1>(layout)) {}

    __device__ __forceinline__ sparse_iterator begin(uint i) {
        return { blocks, __ldg(&offset_table[i]), __ldg(&offset_table[i + 1]) };
    }

    __device__ __forceinline__ const block_desc& get(uint i) {
        return __ldg(&blocks[i]);
    }
private:
    const block_desc* __restrict__ blocks;
    const int32_t* __restrict__ offset_table;
};
