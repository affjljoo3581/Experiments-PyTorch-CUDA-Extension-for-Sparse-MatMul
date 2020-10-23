import torch
from .kernel import sparse_ops
from .layout import SparseLayout
from typing import Any, Tuple, Optional


class SparseSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, layout: SparseLayout
                ) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.layout = layout

        return sparse_ops.sparse_softmax_forward(x,
                                                 layout.row_blocks,
                                                 layout.row_table)

    @staticmethod
    def backward(ctx: Any, dy: torch.Tensor
                 ) -> Tuple[Optional[torch.Tensor], ...]:
        return None, None


softmax = SparseSoftmax.apply
