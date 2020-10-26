import torch
from typing import Tuple


class SparseLayout:
    def __init__(self, pattern: torch.Tensor):
        self.pattern = pattern
        self.sparse_pos = torch.nonzero(pattern, as_tuple=False)

        self.row_blocks, self.row_table = \
            self._create_sparse_info(pattern, transpose=False)
        self.col_blocks, self.col_table = \
            self._create_sparse_info(pattern, transpose=True)

    def _create_sparse_info(self,
                            pattern: torch.Tensor,
                            transpose: bool = False
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        if transpose:
            # Get non-zero element indices sorted by column.
            rows, cols = torch.nonzero(pattern.t(), as_tuple=True)
            rows, cols = cols, rows
        else:
            rows, cols = torch.nonzero(pattern, as_tuple=True)

        # Construct a sparse block information with their indices and
        # positions.
        block_pos = rows * pattern.size(1) + cols
        block_idx = pattern.flatten().cumsum(0).index_select(0, block_pos) - 1

        sparse_blocks = torch.stack((block_idx, block_pos), dim=1)
        sparse_blocks = sparse_blocks.short().flatten()

        # Create a sparse table which maps each row to sparse blocks.
        sparse_table = pattern.new_zeros(
            pattern.size(1 if transpose else 0) + 1, dtype=torch.int)

        sparse_table.index_add_(
            dim=0,
            index=(cols if transpose else rows) + 1,
            source=torch.ones(rows.size(0), dtype=torch.int))
        sparse_table = sparse_table.cumsum(0, dtype=torch.int)

        return sparse_blocks.cuda(), sparse_table.cuda()

    def make_sparse(self, x: torch.Tensor, block_size: int = 32
                    ) -> torch.Tensor:
        return torch.stack([x[..., r:r+block_size, c:c+block_size]
                            for r, c in self.sparse_pos], dim=-3)
