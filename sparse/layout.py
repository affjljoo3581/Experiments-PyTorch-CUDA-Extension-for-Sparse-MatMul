import torch
from typing import Tuple


class SparseLayout:
    def __init__(self, pattern: torch.Tensor):
        self.row_table, self.row_table_ptr = self._create_sparse_table(pattern)
        self.col_table, self.col_table_ptr = \
            self._create_sparse_table(pattern, transpose=True)

    def _create_sparse_table(self,
                             pattern: torch.Tensor,
                             transpose: bool = False
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        if transpose:
            rows, cols = torch.nonzero(pattern.t(), as_tuple=True)
            rows, cols = cols, rows
        else:
            rows, cols = torch.nonzero(pattern, as_tuple=True)

        # Construct a sparse table with block indices and positions.
        block_pos = rows * pattern.size(1) + cols
        block_idx = pattern.flatten().cumsum(0).index_select(0, block_pos) - 1

        sparse_table = torch.stack(
            (block_idx.short(), block_pos.short()), dim=1)
        sparse_table = sparse_table.flatten()

        # Create a table pointers which are start indices of rows.
        sparse_table_ptr = pattern.new_zeros(
            pattern.size(1 if transpose else 0) + 1, dtype=torch.int)

        sparse_table_ptr.index_add_(
            dim=0,
            index=(cols if transpose else rows) + 1,
            source=torch.ones(rows.size(0), dtype=torch.int))
        sparse_table_ptr = sparse_table_ptr.cumsum(0, dtype=torch.int)

        return sparse_table.cuda(), sparse_table_ptr.cuda()
