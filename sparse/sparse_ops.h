#pragma once

#include <string>
#include <torch/extension.h>

#define TILE_32x32_WIDTH        32
#define TILE_32x32_SIZE         (TILE_32x32_WIDTH * TILE_32x32_WIDTH)


torch::Tensor batched_sparse_matmul(torch::Tensor a,
                                    torch::Tensor b,
                                    const std::string& mode,
                                    torch::Tensor row_blocks,
                                    torch::Tensor row_table,
                                    torch::Tensor col_blocks,
                                    torch::Tensor col_table,
                                    bool trans_a,
                                    bool trans_b);


torch::Tensor sparse_softmax_forward(torch::Tensor x,
                                     torch::Tensor row_blocks,
                                     torch::Tensor row_table);


torch::Tensor sparse_softmax_backward(torch::Tensor y,
                                      torch::Tensor dy,
                                      torch::Tensor row_blocks,
                                      torch::Tensor row_table);
