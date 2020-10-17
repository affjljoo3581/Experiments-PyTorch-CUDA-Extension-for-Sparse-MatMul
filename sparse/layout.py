import torch


class SparseLayout:
    def __init__(self, pattern: torch.Tensor):
        self.chunk_table = pattern.nonzero().view(-1).type(torch.short)
