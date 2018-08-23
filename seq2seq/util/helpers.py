import sys
import inspect

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn import RNNBase

from seq2seq.util.initialization import linear_init, get_hidden0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def indentity(x):
    """simple identity function"""
    return x


def mean(l):
    """Return mean of list."""
    return sum(l) / len(l)


def check_import(module, to_use=None):
    """Check whether the given module is imported."""
    if module not in sys.modules:
        if to_use is None:
            error = '{} module not imported. Try "pip install {}".'.format(module, module)
            raise ImportError(error)
        else:
            error = 'You need {} to use {}. Try "pip install {}".'.format(module, to_use, module)
            raise ImportError(error)


def rm_prefix(s, prefix):
    """Removes the prefix of a string if it exists."""
    if s.startswith(prefix):
        s = s[len(prefix):]
    return s


def get_default_args(func):
    """Get the default arguments of a function as a dictionary."""
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def renormalize_input_length(x, input_lengths, max_len=1):
    """Given a tensor that was normalized by a constant value across the whole
        batch, normalizes it by a diferent value for each example in the batch.

    Should preallocate the lengths only once on GPU to speed up.

    Args:
        x (torch.tensor) tensor to normalize of any dimension and size as long
            as the batch dimension is the first one.
        input_lengths (list or torch.tensor) values used for normalizing the
            input, length should be `batch_size`.
        max_len (float, optional) previous constant value that was used to
            normalize the input.
    """
    if input_lengths is None:
        return x
    else:
        if not isinstance(input_lengths, torch.Tensor):
            input_lengths = torch.FloatTensor(input_lengths).to(device)
        while input_lengths.dim() < x.dim():
            input_lengths = input_lengths.unsqueeze(-1)
        return (x * max_len) / input_lengths

        def rm_prefix(s, prefix):
            """Removes the prefix of a string if it exists."""
            if s.startswith(prefix):
                s = s[len(prefix):]
            return s


def get_extra_repr(module, always_shows=[], conditional_shows=dict()):
    """Gets the `extra_repr` for a module.

    Args:
        module (nn.Module): Module for which to get `extra_repr`.
        always_show (list of str): list of variables to always show.
        conditional_show (dictionary or list): variables to show depending on
            their values. Keys are the names, and variables the values
            they should not take to be shown. If a list then the condition
            is that the value is different form the default one in the constructor.
    """
    extra_repr = ""
    for show in always_shows:
        extra_repr += ", {0}={{{0}}}".format(show)

    if isinstance(conditional_shows, list):
        default_args = get_default_args(module.__class__)
        conditional_shows = {show: default_args[show] for show in conditional_shows}

    for show, condition in conditional_shows.items():
        if condition is None:
            if module.__dict__[show] is not None:
                extra_repr += ", {0}={{{0}}}".format(show)
        else:
            if module.__dict__[show] != condition:
                extra_repr += ", {0}={{{0}}}".format(show)

    extra_repr = rm_prefix(extra_repr, ", ")
    return extra_repr.format(**module.__dict__)


class MLP(nn.Module):
    """General MLP class.

    Args:
        input_size (int): size of the input
        hidden_size (int): number of hidden neurones
        output_size (int): output size
        activation (function, optional): activation function
        dropout_input (float, optional): dropout probability to apply on the
            input of the generator.
        dropout_hidden (float, optional): dropout probability to apply on the
            hidden layer of the generator.
        noise_sigma_input (float, optional): standard deviation of the noise to
            apply on the input of the generator.
        noise_sigma_hidden (float, optional): standard deviation of the noise to
            apply on the hidden layer of the generator.
    """

    def __init__(self, input_size, hidden_size, output_size,
                 activation=nn.ReLU,
                 bias=True,
                 dropout_input=0,
                 dropout_hidden=0,
                 noise_sigma_input=0,
                 noise_sigma_hidden=0):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.dropout_input = (nn.Dropout(p=dropout_input)
                              if dropout_input > 0 else indentity)
        self.noise_sigma_input = (GaussianNoise(noise_sigma_input)
                                  if noise_sigma_input > 0 else indentity)
        hidden_size = min(self.input_size, self.hidden_size)
        self.mlp = nn.Linear(self.input_size, self.hidden_size, bias=bias)
        self.dropout_hidden = (nn.Dropout(p=dropout_hidden)
                               if dropout_hidden > 0 else indentity)
        self.noise_sigma_hidden = (GaussianNoise(noise_sigma_hidden)
                                   if noise_sigma_hidden > 0 else indentity)
        self.activation = activation()  # cannot be a function from Functional but class
        self.out = nn.Linear(self.hidden_size, self.output_size, bias=bias)

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

    def extra_repr(self):
        return get_extra_repr(self, always_shows=["input_size", "hidden_size", "output_size"])


class ProbabilityConverter(nn.Module):
    """Maps floats to probabilites (between 0 and 1), element-wise.

    Args:
        min_p (int, optional): minimum probability, can be useful to set greater
            than 0 in order to keep gradient flowing if the probability is used
            for convex combinations of different parts of the model. Note that
            maximum probability is `1-min_p`.
        activation ({"sigmoid", "hard-sigmoid"}, optional): name of the activation
            to use to generate the probabilities. `sigmoid` has the advantage of
            being smooth and never exactly 0 or 1, which helps gradient flows.
            `hard-sigmoid` has the advantage of making all values between min_p
            and max_p equiprobable.
        temperature (bool, optional): whether to add a paremeter controling the
            steapness of the activation. This is useful when x is used for multiple
            tasks, and you don't want to constraint it's magnitude.
        bias (bool, optional): bias used to shift the activation. This is useful
            when x is used for multiple tasks, and you don't want to constraint
            it's scale.
        initial_temperature (int, optional): initial temperature, a higher
            temperature makes the activation steaper.
        initial_probability (float, optional): initial probability you want to
            start with.
        initial_x (float, optional): first value that will be given to the function,
            important to make initial_probability work correctly.
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
        self.is_temperature = is_temperature
        self.is_bias = is_bias
        self.initial_temperature = initial_temperature
        self.initial_probability = initial_probability
        self.initial_x = initial_x

        self.reset_parameters()

    def reset_parameters(self):
        if self.is_temperature:
            self.temperature = Parameter(torch.tensor(self.initial_temperature))
        else:
            self.temperature = torch.tensor(self.initial_temperature).to(device)

        initial_bias = self._probability_to_bias(self.initial_probability,
                                                 initial_x=self.initial_x)
        if self.is_bias:
            self.bias = Parameter(torch.tensor(initial_bias))
        else:
            self.bias = torch.tensor(initial_bias).to(device)

    def forward(self, x, transform_bias=lambda x: x, transform_temperature=lambda x: x):
        temperature = transform_temperature(self.temperature)
        bias = transform_bias(self.bias)

        if self.activation == "sigmoid":
            full_p = torch.sigmoid(x * temperature + bias)
        elif self.activation == "hard-sigmoid":
            # makes the default similar to hard sigmoid
            x = 0.2 * ((x * temperature) + bias) + 0.5
            full_p = torch.max(torch.tensor(0.0),
                               torch.min(torch.tensor(1.0), x))
        elif self.activation == "leaky-hard-sigmoid":
            negative_slope = 0.01
            x = 0.2 * ((x * temperature) + bias) + 0.5
            full_p = torch.min(F.leaky_relu(x, negative_slope=negative_slope),
                               1 + x * negative_slope)
        else:
            raise ValueError("Unkown activation : {}".format(self.activation))

        range_p = 1 - self.min_p * 2
        p = full_p * range_p + self.min_p
        return p

    def extra_repr(self):
        return get_extra_repr(self,
                              always_shows=["min_p", "activation"],
                              conditional_shows=["is_temperature", "is_bias",
                                                 "initial_temperature",
                                                 "initial_probability",
                                                 "initial_x"])

    def _probability_to_bias(self, p, initial_x=0):
        assert p > self.min_p and p < 1 - self.min_p
        range_p = 1 - self.min_p * 2
        p = (p - self.min_p) / range_p
        p = torch.tensor(p, dtype=torch.float)
        if self.activation == "sigmoid":
            bias = torch.log(p / (1 - p)) - initial_x * self.initial_temperature
        elif self.activation == "hard-sigmoid" or self.activation == "leaky-hard-sigmoid":
            bias = (p - 0.5) / 0.2 - initial_x * self.initial_temperature
        else:
            raise ValueError("Unkown activation : {}".format(self.activation))

        return bias


def get_rnn_cell(rnn_name):
    """Return the correct rnn cell."""
    if rnn_name.lower() == 'lstm':
        return nn.LSTM
    elif rnn_name.lower() == 'gru':
        return nn.GRU
    else:
        raise ValueError("Unsupported RNN Cell: {0}".format(rnn_name))


def apply_weight_norm(module):
    """Recursively apply weight norm to children of given module

    copied from : https://github.com/j-min/Adversarial_Video_Summary/blob/master/layers/weight_norm.py
    """
    if isinstance(module, nn.Linear):
        weight_norm(module, 'weight')
    if isinstance(module, (nn.RNNCell, nn.GRUCell, nn.LSTMCell)):
        weight_norm(module, 'weight_ih')
        weight_norm(module, 'weight_hh')
    if isinstance(module, (nn.RNN, nn.GRU, nn.LSTM)):
        for i in range(module.num_layers):
            weight_norm(module, f'weight_ih_l{i}')
            weight_norm(module, f'weight_hh_l{i}')
            if module.bidirectional:
                weight_norm(module, f'weight_ih_l{i}_reverse')
                weight_norm(module, f'weight_hh_l{i}_reverse')


def get_rnn(rnn_name, input_size, hidden_size,
            is_weight_norm=False,
            is_get_hidden0=True,
            **kwargs):
    """Return an initialized rnn."""
    Rnn = get_rnn_cell(rnn_name)
    rnn = Rnn(input_size, hidden_size, **kwargs)
    if is_weight_norm:
        apply_weight_norm(rnn)
    if is_get_hidden0:
        return rnn, get_hidden0(rnn)
    return rnn


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0.0).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x

    def extra_repr(self):
        return get_extra_repr(self,
                              always_shows=["sigma"],
                              conditional_shows=["is_relative_detach"])


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


class HyperparameterInterpolator:
    """Helper class to compute the value of a hyperparameter at each training step.

    initial_value (float): initial value of the hyperparameter.
    final_value (float): final value of the hyperparameter.
    n_steps_interpolate (int): number of training steps before reaching the
        `final_value`.
    mode (str, optional): interpolation mode. One of {"linear", "geometric"}.
    """

    def __init__(self, initial_value, final_value, n_steps_interpolate,
                 mode="linear"):
        if n_steps_interpolate < 0:
            # quick trick to swith final / initial
            n_steps_interpolate *= -1
            initial_value, final_value = final_value, initial_value

        self.initial_value = initial_value
        self.final_value = final_value
        self.n_steps_interpolate = n_steps_interpolate
        self.mode = mode.lower()
        self.is_interpolate = not (self.initial_value == self.final_value or
                                   self.n_steps_interpolate == 0)

        self.n_training_calls = 0

        if self.is_interpolate:
            if self.mode == "linear":
                delta = (self.final_value - self.initial_value)
                self.factor = delta / self.n_steps_interpolate
            elif self.mode == "geometric":
                delta = (self.final_value / self.initial_value)
                self.factor = delta ** (1 / self.n_steps_interpolate)
            else:
                raise ValueError("Unkown mode : {}.".format(mode))

    def reset_parameters(self):
        """Reset the interpolator."""
        self.n_training_calls = 0

    def extra_repr(self, value_name="value"):
        """
        Return a a string that can be used by `extra_repr` of a parent `nn.Module`.
        """
        if self.is_interpolate:
            txt = 'initial_{0}={1}, final_{0}={2}, n_steps_interpolate={3}, {4}'
            txt = txt.format(value_name,
                             self.initial_value,
                             self.final_value,
                             self.n_steps_interpolate, self.mode)
        else:
            txt = "{}={}".format(value_name, self.final_value)
        return txt

    def __call__(self, is_update):
        if not self.is_interpolate:
            return self.final_value

        if self.n_training_calls < self.n_steps_interpolate:
            current = self.initial_value
            if self.mode == "geometric":
                current *= (self.factor ** self.n_training_calls)
            elif self.mode == "linear":
                current += self.factor * self.n_training_calls
        else:
            current = self.final_value

        if is_update:
            self.n_training_calls += 1

        return current


class AnnealedGaussianNoise(GaussianNoise):
    """Gaussian noise regularizer with annealing.

    Args:
        initial_sigma (float): initial sigma.
        final_sigma (float): final relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        n_steps_interpolate (int): number of training steps before reaching the
            `final_sigma`.
        mode (str, optional): interpolation mode. One of {"linear", "geometric"}.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self,
                 initial_sigma=0.3,
                 final_sigma=0,
                 n_steps_interpolate=0,
                 mode="linear",
                 is_relative_detach=True):
        super().__init__(sigma=initial_sigma, is_relative_detach=is_relative_detach)

        self.get_sigma = HyperparameterInterpolator(initial_sigma,
                                                    final_sigma,
                                                    n_steps_interpolate,
                                                    mode=mode)

    def reset_parameters(self):
        self.get_sigma.reset_parameters()

    def extra_repr(self):
        detached_str = '' if self.is_relative_detach else ', not_detached'
        txt = self.get_sigma.extra_repr(value_name="sigma")
        return txt + detached_str

    def forward(self, x, is_update=True):
        self.sigma = self.get_sigma(is_update and self.training)
        return super().forward(x)


class AnnealedDropout(nn.Dropout):
    """Dropout regularizer with annealing.

    Args:
        initial_dropout (float): initial dropout probability.
        final_dropout (float): final dropout probability. Default is 0 if
            no interpolate and 0.1 if interpolating.
        n_steps_interpolate (int): number of training steps before reaching the
            `final_dropout`.
        mode (str, optional): interpolation mode. One of {"linear", "geometric"}.
    """

    def __init__(self,
                 initial_dropout=0.7,
                 final_dropout=None,
                 n_steps_interpolate=0,
                 mode="geometric"):
        super().__init__(p=initial_dropout)

        if final_dropout is None:
            final_dropout = 0 if n_steps_interpolate == 0 else 0.1

        self.get_dropout_p = HyperparameterInterpolator(initial_dropout,
                                                        final_dropout,
                                                        n_steps_interpolate,
                                                        mode=mode)

    def reset_parameters(self):
        self.get_dropout_p.reset_parameters()

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        txt = self.get_dropout_p.extra_repr(value_name="dropout")
        return txt + inplace_str

    def forward(self, x, is_update=True):
        self.p = self.get_dropout_p(is_update and self.training)
        return super().forward(x)


def format_source_lengths(source_lengths):
    if isinstance(source_lengths, tuple):
        source_lengths_list, source_lengths_tensor = source_lengths
    else:
        source_lengths_list, source_lengths_tensor = None, None

    return source_lengths_list, source_lengths_tensor
