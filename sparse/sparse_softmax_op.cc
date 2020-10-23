#include <sys/types.h>
#include <torch/extension.h>

#include "sparse_matrix.h"
#include "sparse_softmax_op.h"


torch::Tensor sparse_softmax_op_forward(torch::Tensor x,
                                        torch::Tensor row_blocks,
                                        torch::Tensor row_table) {
    auto output_shape = x.sizes();

    // Merge all batch dimensions to single one and create empty output tensor.
    x = x.flatten(0, -4);
    y = torch::empty_like(x);

    // Get the dimension sizes.
    uint total_batches = x.size(0);
    uint total_blocks = row_blocks.size(0) / 2;
    uint total_rows = (row_table.size(0) - 1) * TILE_32x32_WIDTH;

    sparse_softmax_op_32x32_forward(
        x.data_ptr<float>(), y.data_ptr<float>(),
        row_blocks.data_ptr<short>(), row_table.data_ptr<int>(),
        total_blocks, total_batches, total_rows
    );

    return y.reshape(output_shape);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_softmax_op_forward",
          &sparse_softmax_op_forward,
          "Sparse Softmax Activation");
}
