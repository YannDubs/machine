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
    """General MLP class.

    Args:
        input_size (int): size of the input
        hidden_size (int): number of hidden neurones
        output_size (int): output size
        activation (function, optional): activation function
    """

    def __init__(self, input_size, hidden_size, output_size, activation=F.relu):
        super(MLP, self).__init__()
        self.mlp = nn.Linear(input_size, hidden_size)
        self.activation = activation
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        y = self.mlp(x)
        y = self.activation(y)
        y = self.out(y)
        return y
