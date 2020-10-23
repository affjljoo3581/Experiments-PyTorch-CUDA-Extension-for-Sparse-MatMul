#include <torch/extension.h>

#include "sparse_ops.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_sparse_matmul_op",
          &batched_sparse_matmul_op,
          "Batched Sparse Matrix Multiplication");

    m.def("sparse_softmax_op_forward",
          &sparse_softmax_op_forward,
          "Sparse Softmax Activation");
}
