#include <torch/extension.h>

#include "sparse_ops.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_sparse_matmul",
          &batched_sparse_matmul,
          "Batched Sparse Matrix Multiplication");

    m.def("sparse_softmax_forward",
          &sparse_softmax_forward,
          "Sparse Softmax Activation");
}
