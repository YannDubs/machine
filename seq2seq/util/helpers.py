import torch
import torch.nn as nn
import torch.nn.functional as F

import sys


def check_import(module, to_use=None):
    """Check whether the given module is imported."""
    if module not in sys.modules:
        if to_use is None:
            error = '{} module not imported. Try "pip install {}".'.format(module, module)
            raise ImportError(error)
        else:
            error = 'You need {} to use {}. Try "pip install {}".'.format(module, to_use, module)
            raise ImportError(error)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation=F.relu):
        super(MLP, self).__init__()
        self.activation = activation
