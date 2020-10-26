#pragma once

#include <tuple>
#include <string>
#include <torch/extension.h>

#define TILE_32x32_WIDTH        32
#define TILE_32x32_SIZE         (TILE_32x32_WIDTH * TILE_32x32_WIDTH)


using layout_tensors = std::tuple<torch::Tensor, torch::Tensor>;


torch::Tensor sparse_matmul(
    torch::Tensor a, torch::Tensor b, const std::string& mode,
    const layout_tensors& row_layout, const layout_tensors& col_layout,
    bool trans_a, bool trans_b
);
