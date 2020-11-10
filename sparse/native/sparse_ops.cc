#include <torch/extension.h>

#include "sparse_ops.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_matmul",
          &sparse_matmul,
          "Sparse matrix multiplication.");
}
