import torch
from typing import Any
from .kernel import sparse_ops
from .layout import SparseLayout


class SparseMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any,
                a: torch.Tensor,
                b: torch.Tensor,
                layout: SparseLayout,
                mode: str = 'sdd',
                transpose_a: bool = False,
                transpose_b: bool = False) -> torch.Tensor:
        ctx.save_for_backward(a, b)

        ctx.layout = layout
        ctx.mode = mode

        ctx.transpose_a = transpose_a
        ctx.transpose_b = transpose_b

        return sparse_ops.batched_sparse_smm_op(
            a, b,
            layout.row_table, layout.row_table_ptr,
            layout.col_table, layout.col_table_ptr,
            transpose_a, transpose_b, mode)

    @staticmethod
    def backward():
        pass


matmul = SparseMatMul.apply
