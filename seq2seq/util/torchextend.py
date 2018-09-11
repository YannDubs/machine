import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from seq2seq.util.initialization import linear_init, weights_init
from seq2seq.util.helpers import (get_extra_repr, identity, clamp, Clamper,
                                  HyperparameterInterpolator)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """General MLP class.

    Args:
        input_size (int): size of the input
        hidden_size (int): number of hidden neurones. Force is to be between
            [input_size, output_size]
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
        self.output_size = output_size
        self.hidden_size = min(self.input_size, max(hidden_size, self.output_size))

        self.dropout_input = (nn.Dropout(p=dropout_input)
                              if dropout_input > 0 else identity)
        self.noise_sigma_input = (GaussianNoise(noise_sigma_input)
                                  if noise_sigma_input > 0 else identity)
        self.mlp = nn.Linear(self.input_size, self.hidden_size, bias=bias)
        self.dropout_hidden = (nn.Dropout(p=dropout_hidden)
                               if dropout_hidden > 0 else identity)
        self.noise_sigma_hidden = (GaussianNoise(noise_sigma_hidden)
                                   if noise_sigma_hidden > 0 else identity)
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
            tasks, and you don't want to constraint it's magnitude. By default
            doesn't let it change sign.
        bias (bool, optional): bias used to shift the activation. This is useful
            when x is used for multiple tasks, and you don't want to constraint
            it's scale.
        initial_temperature (int, optional): initial temperature, a higher
            temperature makes the activation steaper.
        initial_probability (float, optional): initial probability you want to
            start with.
        initial_x (float, optional): first value that will be given to the function,
            important to make initial_probability work correctly.
        bias_transformer (callable, optional): transformer function of the bias.
            This function should only take care of the boundaries (ex: leaky relu
            or relu). Note: cannot be a lambda function because of pickle.
            (default: identity)
        temperature_transformer (callable, optional): transformer function of the
            temperature. This function should only take care of the boundaries
            (ex: leaky relu  or relu). By default leakyclamp to [.1,10]. Note:
            cannot be a lambda function because of pickle. (default: relu)
    """

    def __init__(self,
                 min_p=0.01,
                 activation="sigmoid",
                 is_temperature=False,
                 is_bias=False,
                 initial_temperature=1.0,
                 initial_probability=0.5,
                 initial_x=0,
                 bias_transformer=identity,
                 # TO DOC + say that _probability_to_bias doesn't take into accoiuntr transform => needs to be boundary case transform
                 temperature_transformer=Clamper(minimum=0.1, maximum=10., is_leaky=True)):
        super(ProbabilityConverter, self).__init__()
        self.min_p = min_p
        self.activation = activation
        self.is_temperature = is_temperature
        self.is_bias = is_bias
        self.initial_temperature = initial_temperature
        self.initial_probability = initial_probability
        self.initial_x = initial_x
        self.bias_transformer = bias_transformer
        self.temperature_transformer = temperature_transformer

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

    def forward(self, x):
        temperature = self.temperature_transformer(self.temperature)
        bias = self.bias_transformer(self.bias)

        if self.activation == "sigmoid":
            full_p = torch.sigmoid((x + bias) * temperature)
        elif self.activation == "hard-sigmoid":
            # makes the default similar to hard sigmoid
            x = 0.2 * ((x + bias) * temperature) + 0.5
            full_p = clamp(x, minimum=0., maximum=1., is_leaky=False)

        elif self.activation == "leaky-hard-sigmoid":
            x = 0.2 * ((x + bias) * temperature) + 0.5
            full_p = clamp(x, minimum=0., maximum=1.,
                           is_leaky=True, negative_slope=0.01)
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
            bias = -(torch.log((1 - p) / p) / self.initial_temperature + initial_x)

        elif self.activation == "hard-sigmoid" or self.activation == "leaky-hard-sigmoid":
            bias = ((p - 0.5) / 0.2) / self.initial_temperature - initial_x
        else:
            raise ValueError("Unkown activation : {}".format(self.activation))

        return bias


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): standard deviation used to generate the noise.
        is_relative_sigma (bool, optional): whether to use relative standard
            deviation instead of absolute. Relative means that it will be
            multiplied by the magnitude of the value your are adding the noise
            to. This means that sigma can be the same regardless of the scale of
            the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise if `is_relative_sigma=True` . If
            `False` then the scale of the noise won't be seen as a constant but
            something to optimize: this will bias the network to generate vectors
            with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_sigma=True, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_sigma = is_relative_sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0.0).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma
            if self.is_relative_sigma:
                scale = scale * (x.detach() if self.is_relative_detach else x)
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x

    def extra_repr(self):
        return get_extra_repr(self,
                              always_shows=["sigma"],
                              conditional_shows=["is_relative_sigma",
                                                 "is_relative_detach"])


class AnnealedGaussianNoise(GaussianNoise):
    """Gaussian noise regularizer with annealing.

    Args:
        initial_sigma (float, optional): initial sigma.
        final_sigma (float, optional): final standard deviation used to generate
            the noise.
        n_steps_interpolate (int, optional): number of training steps before
            reaching the `final_sigma`.
        mode (str, optional): interpolation mode. One of {"linear", "geometric"}.
        is_relative_sigma (bool, optional): whether to use relative standard
            deviation instead of absolute. Relative means that it will be
            multiplied by the magnitude of the value your are adding the noise
            to. This means that sigma can be the same regardless of the scale of
            the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise if `is_relative_sigma=True` . If
            `False` then the scale of the noise won't be seen as a constant but
            something to optimize: this will bias the network to generate vectors
            with smaller values.
        kwargs: additional arguments to `HyperparameterInterpolator`.
    """

    def __init__(self,
                 initial_sigma=0.2,
                 final_sigma=0,
                 n_steps_interpolate=0,
                 mode="linear",
                 is_relative_sigma=True,
                 is_relative_detach=True,
                 **kwargs):
        super().__init__(sigma=initial_sigma,
                         is_relative_sigma=is_relative_sigma,
                         is_relative_detach=is_relative_detach)

        self.get_sigma = HyperparameterInterpolator(initial_sigma,
                                                    final_sigma,
                                                    n_steps_interpolate,
                                                    mode=mode,
                                                    **kwargs)

    def reset_parameters(self):
        self.get_sigma.reset_parameters()

    def extra_repr(self):
        detached_str = '' if self.is_relative_sigma else ', not_relative'
        detached_str += '' if self.is_relative_detach else ', not_detached'
        txt = self.get_sigma.extra_repr(value_name="sigma")
        return txt + detached_str

    def forward(self, x, is_update=True):
        self.sigma = self.get_sigma(is_update and self.training)
        return super().forward(x)


class AnnealedDropout(nn.Dropout):
    """Dropout regularizer with annealing.

    Args:
        initial_dropout (float, optional): initial dropout probability.
        final_dropout (float, optional): final dropout probability. Default is 0
            if no interpolate and 0.1 if interpolating.
        n_steps_interpolate (int, optional): number of training steps before
            reaching the `final_dropout`.
        mode (str, optional): interpolation mode. One of {"linear", "geometric"}.
        kwargs: additional arguments to `HyperparameterInterpolator`.
    """

    def __init__(self,
                 initial_dropout=0.7,
                 final_dropout=None,
                 n_steps_interpolate=0,
                 mode="geometric",
                 **kwargs):
        super().__init__(p=initial_dropout)

        if final_dropout is None:
            final_dropout = 0 if n_steps_interpolate == 0 else 0.1

        self.get_dropout_p = HyperparameterInterpolator(initial_dropout,
                                                        final_dropout,
                                                        n_steps_interpolate,
                                                        mode=mode,
                                                        **kwargs)

    def reset_parameters(self):
        self.get_dropout_p.reset_parameters()

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        txt = self.get_dropout_p.extra_repr(value_name="dropout")
        return txt + inplace_str

    def forward(self, x, is_update=True):
        self.p = self.get_dropout_p(is_update and self.training)
        if self.p == 0:
            return x
        return super().forward(x)


class StochasticRounding(nn.Module):
    """Applies differentiable stochastic rounding.

    Notes:
        The gradients are biased but it's a lot uicker to compute than the
    concrete one.

    Args:
        min_p (float, optional): minimum probability of rounding to the "wrong"
            number. Useful to keep exploring.
        start_step (int, optional): number of steps to wait for before starting rounding.
    """

    def __init__(self, min_p=0.01, start_step=0):
        super().__init__()
        self.min_p = min_p
        self.start_step = start_step
        self.n_training_calls = 0

    def reset_parameters(self):
        self.n_training_calls = 0

    def extra_repr(self):
        return get_extra_repr(self, always_shows=["start_step"])

    def forward(self, x, is_update=True):
        if not self.training:
            return x.round()

        if is_update and self.training:
            self.n_training_calls += 1

        if self.start_step > self.n_training_calls:
            return x

        x_detached = x.detach()
        x_floored = x_detached.floor()
        decimals = (x_detached - x_floored)
        p = (decimals * (1 - 2 * self.min_p)) + self.min_p
        x_hard = x_floored + torch.bernoulli(p)
        x_delta = x_hard - x_detached
        x_rounded = x_delta + x
        return x_rounded


class ConcreteRounding(nn.Module):
    """Applies rounding through gumbel/concrete softmax.

    Notes:
        - The gradients are unbiased but a lot slower than StochasticRounding.
        - The temperature variable follows the implementation in the paper,
            so it is the inverse of the temperature in `ProbabilityConverter`.
            I.e lower temperature means higher slope.

    Args:
        start_step (int, optional): number of steps to wait for before starting rounding.
        min_p (float, optional): minimum probability of rounding to the "wrong"
            number. Useful to keep exploring.
        initial_temperature (float, optional): initial softmax temperature.
        final_temperature (float, optional): final softmax temperature. Default:
            `2/3 if n_steps_interpolate==0 else 0.5`.
        n_steps_interpolate (int, optional): number of training steps before
            reaching the `final_temperature`.
        mode (str, optional): interpolation mode. One of {"linear", "geometric"}.
        kwargs: additional arguments to `HyperparameterInterpolator`.
    """

    def __init__(self,
                 start_step=0,
                 min_p=0.01,
                 initial_temperature=1,
                 final_temperature=None,
                 n_steps_interpolate=0,
                 mode="linear",
                 **kwargs):
        super().__init__()

        if final_temperature is None:
            final_temperature = 2 / 3 if n_steps_interpolate == 0 else 0.5

        self.min_p = min_p
        self.get_temperature = HyperparameterInterpolator(initial_temperature,
                                                          final_temperature,
                                                          n_steps_interpolate,
                                                          mode=mode,
                                                          **kwargs)

    def reset_parameters(self):
        self.get_temperature.reset_parameters()

    def extra_repr(self):

        txt = get_extra_repr(self, always_shows=["start_step"])
        interpolator_txt = self.get_temperature.extra_repr(value_name="temperature")

        txt += ", " + interpolator_txt
        return txt

    def forward(self, x, is_update=True):
        if not self.training:
            return x.round()

        temperature = self.get_temperature(is_update)

        x_detached = x.detach()
        x_floored = x_detached.floor()

        decimals = x - x_floored
        p = decimals * (1 - 2 * self.min_p) + self.min_p
        softBernouilli = torch.distributions.RelaxedBernoulli(temperature, p)
        soft_sample = softBernouilli.rsample()
        new_d_detached = soft_sample.detach()
        # removes a detached version of the soft X and adds the real X
        # to emulate the fact that we add some non differentaible noise which just
        # hapens to make the variable rounded. I.e the total is still differentiable
        new_decimals = new_d_detached.round() - new_d_detached + soft_sample
        x_rounded = x_floored + new_decimals - x_detached + x
        return x_rounded


class L0Gates(nn.Module):
    """Return gates for L0 regularization.

    Notes:
        Main idea taken from `Learning Sparse Neural Networks through L_0
        Regularization`, but modified using straight through Gumbel
        softmax estimator.

    Args:
        input_size (int): size of the input to the gate generator.
        output_size (int): length of the vectors to dot product.
        bias (bool, optional): whether to use a bias for the gate generation.
        is_mlp (bool, optional): whether to use a MLP for the gate generation.
        kwargs:
            Additional arguments to the gate generator.
    """

    def __init__(self,
                 input_size, output_size,
                 is_mlp=False,
                 **kwargs):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.is_mlp = is_mlp
        if self.is_mlp:
            self.gate_generator = MLP(self.input_size, self.output_size, self.output_size,
                                      **kwargs)
        else:
            self.gate_generator = nn.Linear(self.input_size, self.output_size,
                                            **kwargs)

        self.rounder = ConcreteRounding()

        self.reset_parameters()

    def reset_parameters(self):
        linear_init(self.gate_generator, "sigmoid")

    def extra_repr(self):
        return get_extra_repr(self,
                              always_shows=["input_size", "output_size"],
                              conditional_shows=["is_mlp"])

    def forward(self, x):
        gates = self.gate_generator(x)
        gates = torch.sigmoid(gates)
        gates = self.rounder(gates)

        loss = gates.mean()

        return gates, loss
