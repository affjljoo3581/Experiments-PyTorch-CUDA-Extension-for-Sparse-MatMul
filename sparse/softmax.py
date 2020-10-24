import torch
from .kernel import sparse_ops
from .layout import SparseLayout
from typing import Any, Tuple, Optional


class SparseSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, layout: SparseLayout
                ) -> torch.Tensor:
        y = sparse_ops.sparse_softmax_forward(
            x, layout.row_blocks, layout.row_table)

        ctx.save_for_backward(y)
        ctx.layout = layout

        return y


    @staticmethod
    def backward(ctx: Any, dy: torch.Tensor
                 ) -> Tuple[Optional[torch.Tensor], ...]:
        y = ctx.saved_tensors
        layout = ctx.layout

        # Note that all tensors in sparse operations must be contiguous.
        if not dy.is_contiguous():
            dy = dy.contiguous()

        return sparse_ops.sparse_softmax_backward(
            y, dy, layout.row_blocks, layout.row_table)


softmax = SparseSoftmax.apply
