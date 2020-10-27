import torch
from typing import Tuple


class SparseLayout:
    def __init__(self, pattern: torch.Tensor):
        self.pattern = pattern
        self.sparse_positions = torch.nonzero(pattern, as_tuple=False)

        # Create sparse layout tensors for CUDA kernel.
        self.row_layout = self._create_layout(pattern)
        self.col_layout = self._create_layout(pattern, transpose=True)

    def _create_layout(self, pattern: torch.Tensor, transpose: bool = False
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        if transpose:
            cols, rows = torch.nonzero(pattern.t(), as_tuple=True)
        else:
            rows, cols = torch.nonzero(pattern, as_tuple=True)

        # Construct sparse block descriptors with their indices and positions.
        idx = pattern.flatten().cumsum(0)
        idx = idx.index_select(0, rows * pattern.size(-1) + cols) - 1

        packed_pos = (rows << 16) + cols
        blocks = torch.stack((idx, packed_pos), dim=1).int().flatten()

        # Create an offset table containing the offsets of the first block in
        # each group.
        table_keys = (cols if transpose else rows) + 1
        table_values = pattern.new_ones(table_keys.size(0))

        table_size = pattern.size(1 if transpose else 0) + 1
        offset_table = (pattern.new_zeros(table_size)
                        .index_add(0, table_keys, table_values)
                        .cumsum(0, dtype=torch.int))

        return blocks.cuda(), offset_table.cuda()

    def to_sparse(self, x: torch.Tensor) -> torch.Tensor:
        # Split the dense matrix to the sparse blocks and concatenate them.
        return torch.stack([x[...,
                              i * 32: (i + 1) * 32,
                              j * 32: (j + 1) * 32]
                            for i, j in self.sparse_positions], dim=-3)

    def to_dense(self, x: torch.Tensor) -> torch.Tensor:
        output = x.new_empty(x.shape[:-3] + (self.pattern.size(-2) * 32,
                                             self.pattern.size(-1) * 32))

        # Copy the sparse block data to the dense tensor.
        for k, (i, j) in enumerate(self.sparse_positions):
            output[...,
                   i * 32: (i + 1) * 32,
                   j * 32: (j + 1) * 32] = x[..., k, :, :]

        return output
