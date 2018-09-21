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

    def __init__(self,
                 minimum=-float("Inf"),
                 maximum=float("Inf"),
                 is_leaky=False,
                 negative_slope=0.01,
                 hard_min=None,
                 hard_max=None):
        self.minimum = minimum
        self.maximum = maximum
        self.is_leaky = is_leaky
        self.negative_slope = negative_slope
        self.hard_min = hard_min
        self.hard_max = hard_max

    def __call__(self, x):
        return clamp(x, minimum=self.minimum, maximum=self.maximum,
                     is_leaky=self.is_leaky, negative_slope=self.negative_slope,
                     hard_min=self.hard_min, hard_max=self.hard_max)


def clamp(x,
          minimum=-float("Inf"),
          maximum=float("Inf"),
          is_leaky=False,
          negative_slope=0.01,
          hard_min=None,
          hard_max=None):
    """Clamps a tensor to the given [minimum, maximum] (leaky) bound, with
    an optional hard clamping.
    """
    lower_bound = (minimum + negative_slope * x) if is_leaky else torch.zeros_like(x) + minimum
    upper_bound = (maximum + negative_slope * x) if is_leaky else torch.zeros_like(x) + maximum
    clamped = torch.max(lower_bound, torch.min(x, upper_bound))

    if hard_min is not None or hard_max is not None:
        if hard_min is None:
            hard_min = -float("Inf")
        elif hard_max is None:
            hard_max = float("Inf")
        clamped = clamp(x, minimum=hard_min, maximum=hard_max, is_leaky=False)

    return clamped


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


def rm_dict_keys(dic, keys_to_rm):
    """remove a set of keys from a dictionary not in place."""
    return {k: v for k, v in dic.items() if k not in keys_to_rm}


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
        input_lengths = atleast_nd(input_lengths, x.dim())
        return (x * max_len) / input_lengths


def atleast_nd(x, n):
    """Adds dimensions to x until reaches n."""
    while x.dim() < n:
        x = x.unsqueeze(-1)
    return x


def get_extra_repr(module, always_shows=[], conditional_shows=dict()):
    """Gets the `extra_repr` for a module.

    Note:
        All variables that you want to show have to be attributes of `module` with
        the same name.The name of the param in the function definition is not enough.

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

    Args:
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
        """Return the current value of the hyperparameter.

        Args:
            is_update (bool): whether to update the hyperparameter.
        """
        if not self.is_interpolate:
            return self.final_value

        if is_update:
            self.n_training_calls += 1

        if self.start_step >= self.n_training_calls:
            return self.default

        n_actual_training_calls = self.n_training_calls - self.start_step

        if self.is_annealing:
            current = self.initial_value
            if self.mode == "geometric":
                current *= (self.factor ** n_actual_training_calls)
            elif self.mode == "linear":
                current += self.factor * n_actual_training_calls
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
        if rate > 1:
            return rate  # rate was actually the final numebr of steps.
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


def regularization_loss(x, p=2., dim=None, lower_bound=1e-4, is_no_mean=False, **kwargs):
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

    loss = (torch.abs(x)**p)

    if not is_no_mean:
        if dim is None:
            loss = loss.mean()
        else:
            loss = loss.mean(dim=dim)

    return loss


def abs_clamp(x, lower_bound):
    """Lowerbounds the absolute value of a tensor."""
    sign = x.sign()
    lower_bounded = torch.max(x * sign, torch.ones_like(x) * lower_bound
                              ) * sign
    return lower_bounded


def modify_optimizer_grads(optimizer, is_neg=True, mul=None):
    """Modifies in place the gradient of the parameetrs of the `optimizer`."""
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                p.grad.detach_()
                if is_neg:
                    p.grad.neg_()
                if mul is not None:
                    p.grad.mul_(mul)


def freeze(model):
    """Freezes a (sub)model."""
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(model):
    """Unfreezes a (sub)model."""
    model.train()
    for p in model.parameters():
        p.requires_grad = True


def batch_reduction_f(x, f, batch_first=True, **kwargs):
    """Applies a reduction function `fun` batchwise."""
    if x.dim() <= 1:
        return x
    if not batch_first:
        x = x.transpose(1, 0)
    return f(x.view(x.size(0), -1), dim=1, **kwargs)


def add_to_visualize(values, keys, additional, save_every_n_batches=15):
    """Every `save_every` batch, adds a certain variable to the `visualization`
    sub-dictionary of additional. Such variables should be the ones that are
    interpretable, and for which the size is independant of the source length.
    I.e avaregae over the source length if it is dependant.

    The variables will then be averaged over decoding step and over batch_size.
    """
    if "visualize" in additional and additional["training_step"] % save_every_n_batches == 0:
        if isinstance(keys, list):
            for k, v in zip(keys, values):
                add_to_visualize(v, k, additional, save_every_n_batches=save_every_n_batches)
        else:
            # averages over the batch size
            if isinstance(values, torch.Tensor):
                values = values.mean(0).detach().cpu()
            additional["visualize"][keys] = values


def add_to_test(values, keys, additional, is_dev_mode):
    """
    Save a variable to additional["test"] only if dev mode is on. The
    variables saved should be the interpretable ones for which you want to
    know the value of during test time.

    Batch size should always be 1 when predicting with dev mode !
    """
    if is_dev_mode:
        if isinstance(keys, list):
            for k, v in zip(keys, values):
                add_to_test(v, k, additional, is_dev_mode)
        else:
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu()
            additional["test"][keys] = values


def add_regularization(loss, loss_name, additional, is_visualize=True, **kwargs):
    """
    Adds a regularization loss.
    """
    additional["losses"][loss_name] = loss

    """
    if is_visualize:
        name = 'losses_{}'.format(loss_name)
        add_to_visualize(loss, name, additional, **kwargs)
    """
