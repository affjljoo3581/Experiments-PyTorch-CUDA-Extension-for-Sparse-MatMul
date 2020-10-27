import os
import torch.utils.cpp_extension


# Find files with `.cc` and `.cu` extensions in `native` directory.
_current_dir = os.path.join(os.path.dirname(__file__), 'native')
_source_files = [os.path.join(_current_dir, file)
                 for file in os.listdir(_current_dir)
                 if file.endswith(('.cu', '.cc'))]

sparse_ops = torch.utils.cpp_extension.load('sparse_ops', _source_files)
