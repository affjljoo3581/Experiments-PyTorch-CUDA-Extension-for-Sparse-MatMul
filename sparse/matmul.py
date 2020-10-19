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
                transpose_a: bool,
                transpose_b: bool,
                mode: str = 'sdd'):
        ctx.save_for_backward(a, b)

        ctx.layout = layout
        ctx.transpose_a = transpose_a
        ctx.transpose_b = transpose_b

        ctx.mode = mode

        return sparse_ops.batched_sparse_smm_op(
            a, b, layout.sparse_table,
            layout.row_block_indices, layout.row_block_indptr,
            layout.col_block_indices, layout.col_block_indptr,
            transpose_a, transpose_b, mode)

    @staticmethod
    def backward():
        pass


matmul = SparseMatMul.apply
