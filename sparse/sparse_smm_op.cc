#include <torch/extension.h>

#include <vector>
#include <string>
#include <sys/types.h>
#include "sparse_smm_op.h"


torch::Tensor batched_sparse_smm_op(
    torch::Tensor a, torch::Tensor b, torch::Tensor chunk_table,
    bool trans_a, bool trans_b, const std::string& mode
) {
    if (mode == "sdd") {
        // Create output sparse-tensor shape with preserving extra-batch
        // dimensions.
        std::vector<int64_t> return_shape = a.sizes().slice(a.dim() - 2).vec();
        return_shape.push_back(a.size(trans_a ? -1 : -2));
        return_shape.push_back(b.size(trans_b ? -2 : -1));

        // Merge all batch dimensions to single one.
        a = a.view({-1, a.size(-2), a.size(-1)});
        b = b.view({-1, b.size(-2), b.size(-1)});

        // Get the dimension sizes and create the output sparse-tensor.
        int64_t total_batches = a.size(0);
        int64_t total_chunks = chunk_table.size(0);
        int64_t total_m = a.size(trans_a ? -1 : -2);
        int64_t total_n = b.size(trans_b ? -2 : -1);
        int64_t total_k = b.size(trans_b ? -1 : -2);

        auto c = a.new_empty({total_batches, total_chunks,
                              TILE_32x32_WIDTH, TILE_32x32_WIDTH});

        // Note that the chunk table consists of 2-bytes of row or column
        // indices.
        if (chunk_table.scalar_type() != torch::kShort)
            chunk_table = chunk_table.toType(torch::kShort);

        batched_sparse_smm_op_32x32(
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
            chunk_table.data_ptr<ushort>(), total_chunks, total_batches,
            total_m, total_n, total_k, trans_a, trans_b, SparseMode::SDD
        );

        return c.reshape(return_shape);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_sparse_smm_op",
          &batched_sparse_smm_op,
          "Batched Sparse MatMul for Single Precision");
}
