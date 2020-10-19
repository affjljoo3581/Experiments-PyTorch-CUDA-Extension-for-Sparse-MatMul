#include <torch/extension.h>

#include <vector>
#include <string>
#include <sys/types.h>
#include "sparse_smm_op.h"


torch::Tensor batched_sparse_smm_op(torch::Tensor a,
                                    torch::Tensor b,
                                    torch::Tensor sparse_table,
                                    torch::Tensor row_block_indices,
                                    torch::Tensor row_block_indptr,
                                    torch::Tensor col_block_indices,
                                    torch::Tensor col_block_indptr,
                                    bool trans_a,
                                    bool trans_b,
                                    const std::string& mode) {
    if (mode == "sdd") {
        // Create output sparse-tensor shape with preserving extra-batch
        // dimensions.
        std::vector<int64_t> output_shape = a.sizes().vec();
        output_shape.pop_back();
        output_shape.pop_back();
        output_shape.push_back(sparse_table.size(0));
        output_shape.push_back(TILE_32x32_WIDTH);
        output_shape.push_back(TILE_32x32_WIDTH);

        // Merge all batch dimensions to single one.
        a = a.flatten(0, -3);
        b = b.flatten(0, -3);

        // Get the dimension sizes and create the output sparse-tensor.
        int64_t total_batches = a.size(0);
        int64_t total_blocks = sparse_table.size(0);
        int64_t total_m = a.size(trans_a ? -1 : -2);
        int64_t total_n = b.size(trans_b ? -2 : -1);
        int64_t total_k = b.size(trans_b ? -1 : -2);

        auto c = a.new_empty({total_batches, total_blocks,
                              TILE_32x32_WIDTH, TILE_32x32_WIDTH});

        batched_sparse_smm_op_32x32_sdd(
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
            sparse_table.data_ptr<short>(), total_blocks, total_batches,
            total_m, total_n, total_k, trans_a, trans_b
        );

        return c.reshape(output_shape);
    } else if (mode == "dsd") {
        auto block_indices = trans_a ? row_block_indices : col_block_indices;
        auto block_indptr = trans_a ? row_block_indptr : col_block_indptr;

        int64_t total_m = block_indices.size(0) - 1;
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
        int64_t total_blocks = sparse_table.size(0);

        auto c = a.new_empty({total_batches, total_m, total_n});

        batched_sparse_smm_op_32x32_dsd(
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
            sparse_table.data_ptr<short>(), block_indices.data_ptr<int32_t>(),
            block_indptr.data_ptr<int32_t>(), total_blocks, total_batches,
            total_m, total_n, total_k, trans_a, trans_b
        );

        return c.reshape(output_shape);
    } else if (mode == "dds") {
        auto block_indices = trans_b ? col_block_indices : row_block_indices;
        auto block_indptr = trans_b ? col_block_indptr : row_block_indptr;

        int64_t total_m = a.size(trans_a ? -1 : -2);
        int64_t total_n = block_indices.size(0) - 1;
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
        int64_t total_blocks = sparse_table.size(0);

        auto c = a.new_empty({total_batches, total_m, total_n});

        batched_sparse_smm_op_32x32_dds(
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
            sparse_table.data_ptr<short>(), block_indices.data_ptr<int32_t>(),
            block_indptr.data_ptr<int32_t>(), total_blocks, total_batches,
            total_m, total_n, total_k, trans_a, trans_b
        );

        return c.reshape(output_shape);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_sparse_smm_op",
          &batched_sparse_smm_op,
          "Batched Sparse MatMul for Single Precision");
}
