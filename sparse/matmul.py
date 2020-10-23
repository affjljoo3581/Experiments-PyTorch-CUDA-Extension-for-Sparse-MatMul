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

        ctx.layout = layout
        ctx.mode = mode

        ctx.trans_a = trans_a
        ctx.trans_b = trans_b

        return sparse_ops.batched_sparse_matmul_op(
            a, b, mode,
            layout.row_table, layout.row_table_ptr,
            layout.col_table, layout.col_table_ptr,
            trans_a, trans_b)

    @staticmethod
    def backward(ctx: Any, dc: torch.Tensor):
        a, b = torch.saved_tensors

        if ctx.needs_input_grad[0]:
            if ctx.trans_a:
                da = sparse_ops.batched_sparse_matmul_op(
                    b, dc, ctx.mode[1] + ctx.mode[2] + ctx.mode[0],
                    ctx.layout.row_table, ctx.layout.row_table_ptr,
                    ctx.layout.col_table, ctx.layout.col_table_ptr,
                    ctx.trans_b, True)
            else:
                da = sparse_ops.batched_sparse_matmul_op(
                    dc, b, ctx.mode[1] + ctx.mode[0] + ctx.mode[2],
                    ctx.layout.row_table, ctx.layout.row_table_ptr,
                    ctx.layout.col_table, ctx.layout.col_table_ptr,
                    False, not ctx.trans_b)

        if ctx.needs_input_grad[1]:
            if ctx.trans_b:
                db = sparse_ops.batched_sparse_matmul_op(
                    dc, a, ctx.mode[2] + ctx.mode[0] + ctx.mode[1],
                    ctx.layout.row_table, ctx.layout.row_table_ptr,
                    ctx.layout.col_table, ctx.layout.col_table_ptr,
                    True, ctx.trans_a)
            else:
                db = sparse_ops.batched_sparse_matmul_op(
                    a, dc, ctx.mode[2] + ctx.mode[1] + ctx.mode[0],
                    ctx.layout.row_table, ctx.layout.row_table_ptr,
                    ctx.layout.col_table, ctx.layout.col_table_ptr,
                    not ctx.trans_a, False)

        return da, db, None, None, None, None


matmul = SparseMatMul.apply
