import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from seq2seq.util.initialization import linear_init


def check_import(module, to_use=None):
    """Check whether the given module is imported."""
    if module not in sys.modules:
        if to_use is None:
            error = '{} module not imported. Try "pip install {}".'.format(module, module)
            raise ImportError(error)
        else:
            error = 'You need {} to use {}. Try "pip install {}".'.format(module, to_use, module)
            raise ImportError(error)


def renormalize_input_length(x, input_lengths, max_len):
    if input_lengths is None:
        return x
    else:
        lengths = torch.FloatTensor(input_lengths)
        while lengths.dim() < x.dim():
            lengths = lengths.unsqueeze(-1)
        return (x * max_len) / lengths


class MLP(nn.Module):
    """General MLP class.

    Args:
        input_size (int): size of the input
        hidden_size (int): number of hidden neurones
        output_size (int): output size
        activation (function, optional): activation function
    """

    def __init__(self, input_size, hidden_size, output_size, activation=nn.LeakyReLU):
        super(MLP, self).__init__()
        self.mlp = nn.Linear(input_size, hidden_size)
        self.activation = activation()  # cannot be a function from Functional but class
        self.out = nn.Linear(hidden_size, output_size)

        self.reset_parameters()

    def forward(self, x):
        y = self.mlp(x)
        y = self.activation(y)
        y = self.out(y)
        return y

    def reset_parameters(self):
        linear_init(self.mlp, activation=self.activation)
        linear_init(self.out)


def generate_probabilities(x, min_p=0, activation="sigmoid", temperature=1, bias=0):
    if activation == "sigmoid":
        full_p = F.sigmoid(x * temperature + bias)
    elif activation == "hard-sigmoid":
        full_p = max(0, min(1, 0.2 * x * temperature + 0.5 + bias))  # makes the default similar to hard sigmoid
    else:
        raise ValueError("Unkown activation : {}".format(activation))

    range_p = 1 - min_p * 2
    p = full_p * range_p + min_p
    return p


def get_kwargs(**kwargs):
    return kwargs
