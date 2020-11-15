#include <tuple>
#include <torch/extension.h>

#include "sparse_kernels.h"


torch::Tensor sparse_matmul(
    torch::Tensor a, torch::Tensor b, const std::string &mode,
    const layout_tensors &row_layout, const layout_tensors &col_layout,
    bool tr_a, bool tr_b
) {
    auto layout = (mode == "sdd" || mode == "dsd" && !tr_a || mode == "dds" && tr_b) ? row_layout : col_layout;
    int num_blocks = std::get<0>(layout).size(0) / 2;
    int sparse_width = (std::get<1>(layout).size(0) - 1) * 32;

    // Get the dimension sizes from the tensors.
    int size_m = mode.at(1) == 'd' ? a.size(tr_a ? -1 : -2) : sparse_width;
    int size_n = mode.at(2) == 'd' ? b.size(tr_b ? -2 : -1) : sparse_width;
    int size_k = mode.at(2) == 'd' ? b.size(tr_b ? -1 : -2) : a.size(tr_a ? -2 : -1);

    // Construct output tensor shape with preserving multiple batch dimensions.
    auto dense = mode.at(1) == 'd' ? a : b;
    auto shape = dense.sizes().slice(0, a.dim() - 2).vec();
    if (mode.at(0) == 'd') shape.insert(shape.end(), { size_m, size_n });
    else shape.insert(shape.end(), { num_blocks, 32, 32 });

    // Merge the batch dimensions to one.
    a = a.flatten(0, mode.at(1) == 'd' ? -3 : -4);
    b = b.flatten(0, mode.at(2) == 'd' ? -3 : -4);
    int num_batches = a.size(0);

    // Create an empty output tensor and launch CUDA kernel.
    torch::Tensor c;
    if (mode.at(0) == 'd') c = a.new_empty( {num_batches, size_m, size_n });
    else c = a.new_empty({ num_batches, num_blocks, 32, 32 });

    if (a.scalar_type() == at::kFloat)
        sparse_smm_32x32x32_wrapper(a, b, c, mode, layout, num_blocks, num_batches, size_m, size_n, size_k, tr_a, tr_b);

    // Return the output tensor with multiple batch dimensions.
    return c.reshape(shape);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_matmul",
          &sparse_matmul,
          "Sparse matrix multiplication.");
}
