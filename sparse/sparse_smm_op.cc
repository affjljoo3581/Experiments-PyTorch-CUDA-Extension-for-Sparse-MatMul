#include <vector>
#include <string>
#include <sys/types.h>
#include <torch/extension.h>

#include "sparse_smm_op.h"


torch::Tensor batched_sparse_smm_op(torch::Tensor a,
                                    torch::Tensor b,
                                    const std::string& mode,
                                    torch::Tensor row_table,
                                    torch::Tensor row_table_ptr,
                                    torch::Tensor col_table,
                                    torch::Tensor col_table_ptr,
                                    bool trans_a,
                                    bool trans_b) {
    if (mode == "sdd") {
        // Create output sparse-tensor shape with preserving extra-batch
        // dimensions.
        std::vector<int64_t> output_shape = a.sizes().vec();
        output_shape.pop_back();
        output_shape.pop_back();
        output_shape.push_back(row_table.size(0) / 2);
        output_shape.push_back(TILE_32x32_WIDTH);
        output_shape.push_back(TILE_32x32_WIDTH);

        // Merge all batch dimensions to single one.
        a = a.flatten(0, -3);
        b = b.flatten(0, -3);

        // Get the dimension sizes and create the output sparse-tensor.
        int64_t total_batches = a.size(0);
        int64_t total_blocks = row_table.size(0) / 2;
        int64_t total_m = a.size(trans_a ? -1 : -2);
        int64_t total_n = b.size(trans_b ? -2 : -1);
        int64_t total_k = b.size(trans_b ? -1 : -2);

        auto c = a.new_empty({total_batches, total_blocks,
                              TILE_32x32_WIDTH, TILE_32x32_WIDTH});

        batched_sparse_smm_op_32x32_sdd(
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
            row_table.data_ptr<short>(), total_blocks, total_batches,
            total_m, total_n, total_k, trans_a, trans_b
        );

        return c.reshape(output_shape);
    } else if (mode == "dsd") {
        auto sparse_table = trans_a ? col_table : row_table;
        auto sparse_table_ptr = trans_a ? col_table_ptr : row_table_ptr;

        int64_t total_m = (sparse_table_ptr.size(0) - 1) * TILE_32x32_WIDTH;
        int64_t total_n = b.size(trans_b ? -2 : -1);
        int64_t total_k = b.size(trans_b ? -1 : -2);

        // Create output dense tensor shape with preserving extra-batch
        // dimensions.
        std::vector<int64_t> output_shape = b.sizes().vec();
        output_shape.pop_back();
        output_shape.pop_back();
        output_shape.push_back(total_m);
        output_shape.push_back(total_n);

        // Merge all batch dimensions to single one.
        a = a.flatten(0, -4);
        b = b.flatten(0, -3);

        // Get the dimension sizes and create the output dense tensor.
        int64_t total_batches = a.size(0);
        int64_t total_blocks = sparse_table.size(0) / 2;

        auto c = a.new_empty({total_batches, total_m, total_n});

        batched_sparse_smm_op_32x32_dsd(
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
            sparse_table.data_ptr<short>(), sparse_table_ptr.data_ptr<int>(),
            total_blocks, total_batches, total_m, total_n, total_k,
            trans_a, trans_b
        );

        return c.reshape(output_shape);
    } else if (mode == "dds") {
        auto sparse_table = trans_b ? row_table : col_table;
        auto sparse_table_ptr = trans_b ? row_table_ptr : col_table_ptr;

        int64_t total_m = a.size(trans_a ? -1 : -2);
        int64_t total_n = (sparse_table_ptr.size(0) - 1) * TILE_32x32_WIDTH;
        int64_t total_k = a.size(trans_a ? -2 : -1);

        // Create output dense tensor shape with preserving extra-batch
        // dimensions.
        std::vector<int64_t> output_shape = a.sizes().vec();
        output_shape.pop_back();
        output_shape.pop_back();
        output_shape.push_back(total_m);
        output_shape.push_back(total_n);

        // Merge all batch dimensions to single one.
        a = a.flatten(0, -3);
        b = b.flatten(0, -4);

        // Get the dimension sizes and create the output dense tensor.
        int64_t total_batches = a.size(0);
        int64_t total_blocks = sparse_table.size(0) / 2;

        auto c = a.new_empty({total_batches, total_m, total_n});

        batched_sparse_smm_op_32x32_dds(
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
            sparse_table.data_ptr<short>(), sparse_table_ptr.data_ptr<int>(),
            total_blocks, total_batches, total_m, total_n, total_k,
            trans_a, trans_b
        );

        return c.reshape(output_shape);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_sparse_smm_op",
          &batched_sparse_smm_op,
          "Batched Sparse MatMul for Single Precision");
}
