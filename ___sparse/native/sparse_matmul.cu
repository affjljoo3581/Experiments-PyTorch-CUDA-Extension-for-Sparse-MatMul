#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <string>
#include <torch/extension.h>

#include "sparse_ops.h"
#include "sparse_layout.cuh"

#include <mma.h>
using namespace nvcuda;

torch::Tensor sparse_matmul(
    torch::Tensor a, torch::Tensor b, const std::string& mode,
    const layout_tensors& row_layout, const layout_tensors& col_layout,
    bool trans_a, bool trans_b
) {
    // Select current sparse layout.
    auto layout = (mode == "ssd"
                   || mode == "dsd" && !trans_a
                   || mode == "dds" && trans_b) ? row_layout : col_layout;
    int64_t num_blocks = std::get<0>(layout).size(0) / 2;
    int64_t sparse_width = (std::get<1>(layout).size(0) - 1) * BLK_32_LEN;

    // Get the dimension sizes from the tensors and corresponding sparse mode.
    int64_t size_m = mode.at(1) == 'd' ? a.size(trans_a ? -1 : -2)
                                       : sparse_width;
    int64_t size_n = mode.at(2) == 'd' ? b.size(trans_b ? -2 : -1)
                                       : sparse_width;
    int64_t size_k = mode.at(2) == 'd' ? b.size(trans_b ? -1 : -2)
                                       : a.size(trans_a ? -2 : -1);

    // Construct output tensor shape with preserving multiple batch dimensions.
    auto dense = mode.at(1) == 'd' ? a : b;
    auto shape = dense.sizes().slice(0, dense.dim() - 2).vec();

    if (mode.at(0) == 'd') shape.insert(shape.end(), { size_m, size_n });
    else shape.insert(shape.end(), { num_blocks, BLK_32_LEN, BLK_32_LEN });

    // Merge the batch dimensions to one.
    a = a.flatten(0, mode.at(1) == 'd' ? -3 : -4);
    b = b.flatten(0, mode.at(2) == 'd' ? -3 : -4);

    int64_t num_batches = a.size(0);

    // Create an empty output tensor to store the multiplication result.
    torch::Tensor c;
    if (mode.at(0) == 'd') c = a.new_empty({ num_batches, size_m, size_n });
    else c = a.new_empty({ num_batches, num_blocks, BLK_32_LEN, BLK_32_LEN });

    // Launch CUDA kernel with corresponding sparse mode and dimension sizes.
    dim3 blocks, threadsPerBlock(BLK_32_LEN, BLK_32_LEN);
    if (mode == "sdd") blocks = dim3(num_blocks, num_batches);
    else blocks = dim3(num_batches,
                       (size_m + BLK_32_LEN - 1) / BLK_32_LEN,
                       (size_n + BLK_32_LEN - 1) / BLK_32_LEN);

    auto kernel = mode == "sdd" ? sparse_matmul_sdd_32x32_kernel :
                  mode == "dsd" ? sparse_matmul_dsd_32x32_kernel :
                                  sparse_matmul_dds_32x32_kernel;
    kernel<<<blocks, threadsPerBlock>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
        layout, num_blocks, size_m, size_n, size_k, trans_a, trans_b
    );

    // Return the output tensor with the multiple batch dimensions.
    return c.reshape(shape);
}
