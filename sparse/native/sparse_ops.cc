#include <torch/extension.h>

#include "sparse_ops.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_matmul",
          &sparse_matmul_single,
          "Sparse matrix multiplication for single precision.");
}
