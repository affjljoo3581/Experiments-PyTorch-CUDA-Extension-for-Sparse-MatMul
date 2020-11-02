#pragma once

#include <tuple>
#include <string>
#include <torch/extension.h>

using layout_tensors = std::tuple<torch::Tensor, torch::Tensor>;

constexpr uint TILE_32x32_WIDTH = 32;
constexpr uint TILE_32x32_SIZE = TILE_32x32_WIDTH * TILE_32x32_WIDTH;


torch::Tensor sparse_matmul_single(
    torch::Tensor a, torch::Tensor b, const std::string& mode,
    const layout_tensors& row_layout, const layout_tensors& column_layout,
    bool trans_a, bool trans_b
);
