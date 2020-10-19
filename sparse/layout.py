import torch
from typing import Tuple


class SparseLayout:
    def __init__(self, pattern: torch.Tensor):
        self.sparse_table = self._create_sparse_table(pattern)
        self.sparse_table = self.sparse_table.short().cuda()

        # Calculate block indices and their pointers for row-order.
        indices, indptr = self._create_block_indices(pattern)
        self.row_block_indices = indices.int().cuda()
        self.row_block_indptr = indptr.int().cuda()

        # Calculate block indices and their pointers for column-order.
        indices, indptr = self._create_block_indices(pattern, transpose=True)
        self.col_block_indices = indices.int().cuda()
        self.col_block_indptr = indptr.int().cuda()

    def _create_sparse_table(self, pattern: torch.Tensor) -> torch.Tensor:
        return torch.nonzero(pattern, as_tuple=False).flatten()

    def _create_block_indices(self,
                              pattern: torch.Tensor,
                              transpose: bool = False
                              ) -> Tuple[torch.Tensor, tuple.Tensor]:
        rows, cols = torch.nonzero(pattern.t() if transpose else pattern,
                                   as_tuple=True)
        block_indices = rows * pattern.size(0 if transpose else 1) + cols

        # Get start position of each row group by cumulative summation of the
        # number of blocks in each row.
        block_indptr = torch.zeros(rows.size(0), dtype=torch.int)
        block_indptr = block_indptr.index_add(
            0, rows, torch.ones(rows.size(0), dtype=torch.int))
        block_indptr = block_indptr.cumsum(0)

        return block_indices, block_indptr
