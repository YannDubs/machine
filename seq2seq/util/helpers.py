import sys
import inspect
import math
import collections

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from seq2seq.util.initialization import get_hidden0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Clamper:
    """Clamp wrapper class. To bypass the lambda pickling issue."""

    def __init__(self, minimum=0, maximum=1., is_leaky=False, negative_slope=0.01):
        self.minimum = minimum
        self.maximum = maximum
        self.is_leaky = is_leaky
        self.negative_slope = negative_slope

    def __call__(self, x):
        return clamp(x, self.minimum, self.maximum, self.is_leaky, self.negative_slope)


def clamp(x, minimum=0., maximum=1., is_leaky=False, negative_slope=0.01):
    """Clamps a tensor to the given [minimum, maximum] (leaky) bound."""
    lower_bound = (minimum + negative_slope * x) if is_leaky else torch.zeros_like(x) + minimum
    upper_bound = (maximum + negative_slope * x) if is_leaky else torch.zeros_like(x) + maximum
    return torch.max(lower_bound, torch.min(x, upper_bound))


def identity(x):
    """simple identity function"""
    return x


def mean(l):
    """Return mean of list."""
    return sum(l) / len(l)


def recursive_update(d, u):
    """Recursively update a dicstionary `d` with `u`."""
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


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
            weight_norm(module, 'weight_ih_l{}'.format(i))
            weight_norm(module, 'weight_hh_l{}'.format(i))
            if module.bidirectional:
                weight_norm(module, 'weight_ih_l{}_reverse'.format(i))
                weight_norm(module, 'weight_hh_l{}_reverse'.format(i))


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


def format_source_lengths(source_lengths):
    if isinstance(source_lengths, tuple):
        source_lengths_list, source_lengths_tensor = source_lengths
    else:
        source_lengths_list, source_lengths_tensor = None, None

    return source_lengths_list, source_lengths_tensor


def apply_along_dim(f, X, dim=0, **kwargs):
    """
    Applies a function along the given dimension.
    Might be slow because list comprehension.
    """
    tensors = [f(x, **kwargs) for x in torch.unbind(X, dim=dim)]
    out = torch.stack(tensors, dim=dim)
    return out


def get_indices(l, keys):
    """Returns a list of the indices. SLow as O(K*N)"""
    out = []
    for k in keys:
        try:
            out.append(l.index(k))
        except ValueError:
            pass
    return out


### CLASSES ###

class HyperparameterInterpolator:
    """Helper class to compute the value of a hyperparameter at each training step.

    initial_value (float): initial value of the hyperparameter.
    final_value (float): final value of the hyperparameter.
    n_steps_interpolate (int): number of training steps before reaching the
        `final_value`.
    start_step (int, optional): number of steps to wait for before starting annealing.
        During the waiting time, the hyperparameter will be `default`.
    default (float, optional): default hyperparameter value that will be used
        for the first `start_step`s. If `None` uses `initial_value`.
    mode (str, optional): interpolation mode. One of {"linear", "geometric"}.
    """

    def __init__(self, initial_value, final_value, n_steps_interpolate,
                 start_step=0,
                 default=None,
                 mode="linear"):
        if n_steps_interpolate < 0:
            # quick trick to swith final / initial
            n_steps_interpolate *= -1
            initial_value, final_value = final_value, initial_value

        self.initial_value = initial_value
        self.final_value = final_value
        self.n_steps_interpolate = n_steps_interpolate
        self.start_step = start_step
        self.default = default if default is not None else self.initial_value
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

    @property
    def is_annealing(self):
        return (self.is_interpolate) and (
            self.start_step < self.n_training_calls) and (
            self.n_training_calls < (self.n_steps_interpolate + self.start_step))

    def __call__(self, is_update):
        if not self.is_interpolate:
            return self.final_value

        if is_update:
            self.n_training_calls += 1

        if self.start_step > self.n_training_calls:
            return self.default

        if self.is_annealing:
            current = self.initial_value
            if self.mode == "geometric":
                current *= (self.factor ** self.n_training_calls)
            elif self.mode == "linear":
                current += self.factor * self.n_training_calls
        else:
            current = self.final_value

        return current


class Rate2Steps:
    """Converts interpolating rates to steps useful for annealing.

    Args:
        total_training_calls (int): total number of training steps.
    """

    def __init__(self, total_training_calls):
        self.total_training_calls = total_training_calls

    def __call__(self, rate):
        return math.ceil(rate * self.total_training_calls)


def l0_loss(x, temperature=10, is_leaky=True, negative_slope=0.01, dim=None):
    """Computes the approxmate differentiable l0 loss of a matrix."""
    norm = torch.abs(torch.tanh(temperature * x))
    if is_leaky:
        norm = norm + torch.abs(negative_slope * x)

    if dim is None:
        return norm.mean()
    else:
        return norm.mean(dim=dim)


"""def regularization_loss(x, p=2., is_normalize=True, dim=None, **kwargs):
    Compute the norm for regularization.

    Args:
        p (float, optional): p of the lp norm to use. `p>=0`, i.e pseudo norms
            can be used. All of those have been made differentiable (even `p=0`).
        is_normalize (bool, optional): whether to normalize the output such that
            the range of the values are similar regardless of p.
        dim (int, optional): the dimension to reduce. By default flattens `x`
            then reduces to a scalar.
        kwargs:
            Additional parameters to `l0_norm`.

    if p == 0:
        return l0_norm(x, dim=None, **kwargs)

    if dim is None:
        loss = torch.norm(x, p=p)
    else:
        loss = torch.norm(x, p=p, dim=dim)

    if is_normalize:
        loss = loss * (x.size(0)**(1 - 1 / p))
    return loss

"""


def regularization_loss(x, p=2., dim=None, lower_bound=1e-4, **kwargs):
    """Compute the regularization loss.

    Args:
        p (float, optional): element wise power to apply. All of those have been
            made differentiable (even `p=0`).
        dim (int, optional): the dimension to reduce. By default flattens `x`
            then reduces to a scalar.
        lower_bound (float, optional): lower bounds the absolute value of a entry
            of x when p<1 to avoid exploding gradients.
        kwargs:
            Additional parameters to `l0_norm`.
    """
    if p < 1:
        x = abs_clamp(x, lower_bound)

    if p == 0:
        return l0_loss(x, dim=None, **kwargs)

    if dim is None:
        loss = (torch.abs(x)**p).mean()
    else:
        loss = (torch.abs(x)**p).mean(dim=dim)

    return loss


def abs_clamp(x, lower_bound):
    """Lowerbounds the absolute value of a tensor."""
    sign = x.sign()
    lower_bounded = torch.max(x * sign, torch.ones_like(x) * lower_bound
                              ) * sign
    return lower_bounded
