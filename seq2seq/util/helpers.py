import sys

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn import RNNBase

from seq2seq.util.initialization import linear_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_import(module, to_use=None):
    """Check whether the given module is imported."""
    if module not in sys.modules:
        if to_use is None:
            error = '{} module not imported. Try "pip install {}".'.format(module, module)
            raise ImportError(error)
        else:
            error = 'You need {} to use {}. Try "pip install {}".'.format(module, to_use, module)
            raise ImportError(error)


def renormalize_input_length(x, input_lengths, max_len=1):
    """Given a tensor that was normalized by a constant value across the whole batch, normalizes it by a diferent value for
    each example in the batch.

    Args:
        x (torch.tensor) tensor to normalize of any dimension and size as long as the batch dimension is the first one.
        input_lengths (list or torch.tensor) values used for normalizing the input, length should be `batch_size`.
        mac_len (float, optional) previous constant value that was used to normalize the input.
    """
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
        dropout_input (float, optional): dropout probability to apply on the input of the generator.
        dropout_hidden (float, optional): dropout probability to apply on the hidden layer of the generator.
        noise_sigma_input (float, optional): standard deviation of the noise to apply on the input of the generator.
        noise_sigma_hidden (float, optional): standard deviation of the noise to apply on the hidden layer of the generator.
    """

    def __init__(self, input_size, hidden_size, output_size,
                 activation=nn.ReLU,
                 bias=True,
                 dropout_input=0,
                 dropout_hidden=0,
                 noise_sigma_input=0,
                 noise_sigma_hidden=0):
        super(MLP, self).__init__()
        self.dropout_input = nn.Dropout(p=dropout_input)
        self.noise_sigma_input = GaussianNoise(noise_sigma_input)
        self.mlp = nn.Linear(input_size, hidden_size, bias=bias)
        self.dropout_hidden = nn.Dropout(p=dropout_input)
        self.noise_sigma_hidden = GaussianNoise(noise_sigma_input)
        self.activation = activation()  # cannot be a function from Functional but class
        self.out = nn.Linear(hidden_size, output_size, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        x = self.dropout_input(x)
        x = self.noise_sigma_input(x)
        y = self.mlp(x)
        y = self.dropout_hidden(y)
        y = self.noise_sigma_hidden(y)
        y = self.activation(y)
        y = self.out(y)
        return y

    def reset_parameters(self):
        linear_init(self.mlp, activation=self.activation)
        linear_init(self.out)


class ProbabilityConverter(nn.Module):
    """Maps floats to probabilites (between 0 and 1), element-wise.

    Args:
        min_p (int, optional): minimum probability, can be useful to set greater than 0 in order to keep gradient
            flowing if the probability is used for convex combinations of different parts of the model.
            Note that maximum probability is `1-min_p`.
        activation ({"sigmoid", "hard-sigmoid"}, optional): name of the activation to use to generate the probabilities.
            `sigmoid` has the advantage of being smooth and never exactly 0 or 1, which helps gradient flows.
            `hard-sigmoid` has the advantage of making all values between min_p and max_p equiprobable.
        temperature (bool, optional): whether to add a paremeter controling the steapness of the activation.
            This is useful when x is used for multiple tasks, and you don't want to constraint it's magnitude.
        bias (bool, optional): bias used to shift the activation. This is useful when x is used for multiple tasks,
            and you don't want to constraint it's scale.
        initial_temperature (int, optional): initial temperature, a higher temperature makes the activation steaper.
        initial_probability (float, optional): initial probability you want to start with.
        initial_x (float, optional): first value that will be given to the function, important to make
            initial_probability work correctly.
    """

    def __init__(self,
                 min_p=0.01,
                 activation="sigmoid",
                 is_temperature=False,
                 is_bias=False,
                 initial_temperature=1.0,
                 initial_probability=0.5,
                 initial_x=0):
        super(ProbabilityConverter, self).__init__()
        self.min_p = min_p
        self.activation = activation

        if is_temperature:
            self.temperature = Parameter(torch.tensor(initial_temperature)).to(device)
        else:
            self.temperature = torch.tensor(initial_temperature).to(device)

        initial_bias = self._probability_to_bias(initial_probability, initial_x=initial_x)
        if is_bias:
            self.bias = Parameter(torch.tensor(initial_bias)).to(device)
        else:
            self.bias = torch.tensor(initial_bias).to(device)

    def forward(self, x, transform_bias=lambda x: x, transform_temperature=lambda x: x):
        if self.activation == "sigmoid":
            full_p = F.sigmoid(x * transform_temperature(self.temperature) + transform_bias(self.bias))
        elif self.activation == "hard-sigmoid":
            # makes the default similar to hard sigmoid
            full_p = max(0, min(1, 0.2 * ((x * transform_temperature(self.temperature)) + transform_bias(self.bias)) + 0.5))
        else:
            raise ValueError("Unkown activation : {}".format(self.activation))

        range_p = 1 - self.min_p * 2
        p = full_p * range_p + self.min_p
        return p

    def _probability_to_bias(self, p, initial_x=0):
        assert p > self.min_p and p < 1 - self.min_p
        range_p = 1 - self.min_p * 2
        p = (p - self.min_p) / range_p
        p = torch.tensor(p, dtype=torch.float)
        if self.activation == "sigmoid":
            bias = torch.log(p / (1 - p)) - initial_x * self.temperature
        elif self.activation == "hard-sigmoid":
            bias = (p - 0.5) / 0.2 - initial_x * self.temperature
        else:
            raise ValueError("Unkown activation : {}".format(self.activation))

        return bias


def get_kwargs(**kwargs):
    """Helper function to go from parameters to dictionnary."""
    return kwargs


def get_rnn_cell(rnn_name):
    """Return the correct rnn cell."""
    if rnn_name.lower() == 'lstm':
        return nn.LSTM
    elif rnn_name.lower() == 'gru':
        return nn.GRU
    else:
        raise ValueError("Unsupported RNN Cell: {0}".format(rnn_name))


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the noise. Relative means that it will be
            multiplied by the magnitude of the value your are adding the noise to. This means that sigma can be the same
            regardless of the scale of the vector.
    """

    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training and self.sigma != 0:
            # detaching because don't want to biais the network to generate vectors
            # with smaller median, but still want the noise to be proportional to the norm
            sampled_noise = torch.randn_like(x) * self.sigma * x.detach()
            sampled_noise.to(device)
            x = x + sampled_noise
        return x


def log_sum_exp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp(i.e 'differentiable max)'.

    credits: https: // github.com / pytorch / pytorch / issues / 2591.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs
