#pragma once

#include <tuple>
#include <string>
#include <torch/extension.h>

#define BLK_32_LEN                  32
#define BLK_32_SIZE                 (BLK_32_LEN * BLK_32_LEN)

using layout_tensors = std::tuple<torch::Tensor, torch::Tensor>;


torch::Tensor sparse_matmul(
    torch::Tensor a, torch::Tensor b, const std::string& mode,
    const layout_tensors& row_layout, const layout_tensors& col_layout,
    bool trans_a, bool trans_b
);
