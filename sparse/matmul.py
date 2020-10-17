import torch
from typing import Any
from . import sparse_ops
from .layout import SparseLayout


class SparseMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any,
                a: torch.Tensor,
                b: torch.Tensor,
                layout: SparseLayout,
                transpose_a: bool,
                transpose_b: bool):
        ctx.save_for_backward(a, b)

        ctx.layout = layout
        ctx.transpose_a = transpose_a
        ctx.transpose_b = transpose_b

        return sparse_ops.batched_sparse_smm_op(
            a, b, layout.chunk_table, transpose_a, transpose_b, "sdd")

    @staticmethod
    def backward():
        pass
