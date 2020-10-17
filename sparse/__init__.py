import os
import torch.utils.cpp_extension


_source_files = [file for file in os.listdir(os.path.basename(__file__))
                 if file.endswith(('.cu', '.cc'))]
sparse_ops = torch.utils.cpp_extension.load('sparse_ops', _source_files)
