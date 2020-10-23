import torch
from typing import Any
from .kernel import sparse_ops
from .layout import SparseLayout


class SparseMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any,
                a: torch.Tensor,
                b: torch.Tensor,
                mode: str,
                layout: SparseLayout,
                trans_a: bool = False,
                trans_b: bool = False) -> torch.Tensor:
        ctx.save_for_backward(a, b)

        ctx.mode, ctx.layout = mode, layout
        ctx.trans_a, ctx.trans_b = trans_a, trans_b

        return sparse_ops.batched_sparse_matmul_op(
            a, b, mode,
            layout.row_table, layout.row_table_ptr,
            layout.col_table, layout.col_table_ptr,
            trans_a, trans_b)

    @staticmethod
    def backward(ctx: Any, dc: torch.Tensor):
        a, b = ctx.saved_tensors
        da, db = None, None

        mode, layout = ctx.mode, ctx.layout
        trans_a, trans_b = ctx.trans_a, ctx.trans_b

        if ctx.needs_input_grad[0]:
            if trans_a:
                da = sparse_ops.batched_sparse_matmul_op(
                    b, dc, mode[1] + mode[2] + mode[0],
                    layout.row_table, layout.row_table_ptr,
                    layout.col_table, layout.col_table_ptr,
                    trans_b, True)
            else:
                da = sparse_ops.batched_sparse_matmul_op(
                    dc, b, mode[1] + mode[0] + mode[2],
                    layout.row_table, layout.row_table_ptr,
                    layout.col_table, layout.col_table_ptr,
                    False, not trans_b)

        if ctx.needs_input_grad[1]:
            if trans_b:
                db = sparse_ops.batched_sparse_matmul_op(
                    dc, a, mode[2] + mode[0] + mode[1],
                    layout.row_table, layout.row_table_ptr,
                    layout.col_table, layout.col_table_ptr,
                    True, trans_a)
            else:
                db = sparse_ops.batched_sparse_matmul_op(
                    a, dc, mode[2] + mode[1] + mode[0],
                    layout.row_table, layout.row_table_ptr,
                    layout.col_table, layout.col_table_ptr,
                    not trans_a, False)

        return da, db, None, None, None, None


matmul = SparseMatMul.apply
