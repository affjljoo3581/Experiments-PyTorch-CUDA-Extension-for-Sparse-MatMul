#pragma once

#include <tuple>
#include <string>
#include <torch/extension.h>

using layout_tensors = std::tuple<torch::Tensor, torch::Tensor>;


void sparse_smm_32x32x32_wrapper(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    const torch::Tensor &matrix_c,
    const std::string& mode, const layout_tensors& layout, int num_blocks,
    int num_batches, int size_m, int size_n, int size_k, bool tr_a, bool tr_b
);
