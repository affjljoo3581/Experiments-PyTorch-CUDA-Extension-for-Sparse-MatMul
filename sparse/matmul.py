import torch
from .kernel import sparse_ops
from .layout import SparseLayout
from typing import Any, Tuple, Optional


class SparseMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, a: torch.Tensor, b: torch.Tensor,
                mode: str, layout: SparseLayout,
                trans_a: bool = False, trans_b: bool = False) -> torch.Tensor:
        ctx.save_for_backward(a, b)
        ctx.mode, ctx.layout = mode, layout
        ctx.trans_a, ctx.trans_b = trans_a, trans_b

        return sparse_ops.sparse_matmul(a, b, mode,
                                        layout.row_layout, layout.col_layout,
                                        trans_a, trans_b)

    @staticmethod
    def backward(ctx: Any, dc: torch.Tensor
                 ) -> Tuple[Optional[torch.Tensor], ...]:
        # Note that all tensors in sparse matmul op should be contiguous.
        da, db, dc = None, None, dc.contiguous()

        a, b = ctx.saved_tensors
        mode, layout = ctx.mode, ctx.layout
        trans_a, trans_b = ctx.trans_a, ctx.trans_b

        if ctx.needs_input_grad[0]:
            if trans_a:
                da_mode = mode[1] + mode[2] + mode[0]
            else:
                da_mode = mode[1] + mode[0] + mode[2]

            da = sparse_ops.sparse_matmul(
                b if trans_a else dc, dc if trans_a else b, da_mode,
                layout.row_layout, layout.col_layout,
                trans_a and trans_b, trans_a or not trans_b)

        if ctx.needs_input_grad[1]:
            if trans_b:
                db_mode = mode[2] + mode[0] + mode[1]
            else:
                db_mode = mode[2] + mode[1] + mode[0]

            db = sparse_ops.sparse_matmul(
                dc if trans_b else a, a if trans_a else dc, db_mode,
                layout.row_layout, layout.col_layout,
                not trans_a or trans_b, trans_a and trans_b)

        return da, db, None, None, None, None


# Use sparse matmul op by calling this method, instead of using `SparseMatMul`
# class directly.
matmul = SparseMatMul.apply
