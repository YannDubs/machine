"""Key Value Query Generator Classes."""
import math
import warnings

import torch
import torch.nn as nn

from seq2seq.util.initialization import get_hidden0, weights_init, replicate_hidden0
from seq2seq.util.helpers import(MLP, ProbabilityConverter, renormalize_input_length,
                                 get_rnn, AnnealedGaussianNoise, AnnealedDropout,
                                 get_extra_repr)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_size(size, hidden_size, name=None):
    if size == -1:
        return hidden_size
    elif 0 < size < 1:
        return math.ceil(size * hidden_size)
    elif 0 < size <= hidden_size:
        return size
    else:
        raise ValueError("Invalid size for {} : {}".format(name, size))


def _get_counter_size(is_abscounter, is_relcounter, is_rotcounters):
    return int(is_abscounter) + int(is_relcounter) + int(is_rotcounters) * 2


def _get_counters(max_len, is_abscounter, is_relcounter, is_rotcounters,
                  input_lengths_tensor, batch_size):
    """Return the batch couters."""
    increments = torch.arange(max_len + 1).view(1, -1, 1).expand(batch_size, -1, 1)

    abs_counter = rel_counter = rot_counters = torch.Tensor([])
    if is_abscounter:
        abs_counter = increments

    if is_relcounter:
        rel_counter = increments / max_len
        rel_counter = renormalize_input_length(rel_counter, input_lengths_tensor, max_len)

    if is_rotcounters:
        angles = increments / max_len * math.pi
        angles = renormalize_input_length(angles, input_lengths_tensor, max_len)
        rot_counters = torch.cat([torch.cos(angles), torch.sin(angles)], dim=2)

    if any(c.nelement() != 0 for c in (abs_counter, rel_counter, rot_counters)):
        counters = torch.cat([abs_counter, rel_counter, rot_counters], dim=2).to(device)
    else:
        counters = None

    return counters


class BaseKeyValueQuery(nn.Module):
    """Base class for quey query value generators.

    Args:
        hidden_size (int): size of the hidden activations of the controller.
        output_size (int, optional): output size of the generator.
        is_contained_kv (bool, optional): whether or not to use different parts
            of the controller output as input for key and value generation.
        min_input_size (int, optional): minimum input size for the generator.
    """

    def __init__(self, hidden_size,
                 output_size=-1,
                 is_contained_kv=False,
                 min_input_size=32,
                 is_dev_mode=False):

        super(BaseKeyValueQuery, self).__init__()

        self.is_dev_mode = is_dev_mode

        self.min_input_size = min_input_size
        self.hidden_size = hidden_size
        self.output_size = _compute_size(output_size, self.hidden_size,
                                         name="output_size - {}".format(type(self).__name__))
        self.is_contained_kv = is_contained_kv
        self.input_size = self._compute_input_size()

    def set_dev_mode(self, value=True):
        self.is_dev_mode = value

    def _compute_input_size(self):
        return (max(self.min_input_size, self.output_size)
                if self.is_contained_kv else self.hidden_size)

    def reset_parameters(self):
        self.apply(weights_init)

    def extra_repr(self):
        return get_extra_repr(self,
                              always_shows=["hidden_size", "output_size"],
                              conditional_shows=dict(is_contained_kv=False))


class BaseKeyQuery(BaseKeyValueQuery):
    """Base class for quey query generators.

    Args:
        hidden_size (int): size of the hidden activations of the controller.
        max_len (int): maximum length of the source sentence.
        is_kqrnn (bool, optional): whether to use a rnn for the generator.
        is_abscounter (bool, optional): whether to use an absolute counter.
        is_relcounter (bool, optional): whether to use a relative counter,
            corresponding to an absolute counter normalized by the source length.
        is_rotcounters (bool, optional): whether to use a rotational counter,
            corresponding to 2 dimensional unit vectors where the angle is given
            by a relative counter between 0 and 180.
        is_postcounter (bool, optional): whether to append the counters to the
            output of the generator instead of the inputs.
        is_normalize_encoder (bool, optional): whether to normalize relative
            counters by the actual source length, rather than the maximum source
            length.
        rnn_cell (str, optional): type of RNN cell.
        is_mlps (bool, optional): whether to use MLPs for the generators instead
            of a linear layer.
        annealed_dropout_kwargs (float, optional): additional arguments to the
            annealed input dropout.
        annealed_noise_kwargs (float, optional): additional arguments to the
            annealed input noise.
        annealed_dropout_kwargs (float, optional): additional arguments to the
            annealed output dropout.
        annealed_noise_output_kwargs (float, optional): additional arguments to
            the annealed output noise.
        kwargs:
            Additional arguments for the `BaseKeyValueQuery` parent class.
    """

    def __init__(self, hidden_size, max_len,
                 is_kqrnn=False,
                 is_abscounter=False,
                 is_relcounter=False,
                 is_rotcounters=False,
                 is_postcounter=False,
                 is_normalize_encoder=True,
                 rnn_cell='gru',
                 is_mlps=True,
                 is_weight_norm_rnn=False,
                 annealed_dropout_kwargs={},
                 annealed_noise_kwargs={},
                 annealed_dropout_output_kwargs={},
                 annealed_noise_output_kwargs={},
                 **kwargs):

        super(BaseKeyQuery, self).__init__(hidden_size, **kwargs)

        self.max_len = max_len
        self.is_kqrnn = is_kqrnn
        self.is_weight_norm_rnn = is_weight_norm_rnn

        self.is_postcounter = is_postcounter
        self.is_rotcounters = is_rotcounters
        self.is_abscounter = is_abscounter
        self.is_relcounter = is_relcounter
        self.counter_size = _get_counter_size(self.is_abscounter,
                                              self.is_relcounter,
                                              self.is_rotcounters)
        self.is_normalize_encoder = is_normalize_encoder

        self.dropout_input = AnnealedDropout(**annealed_dropout_kwargs)
        self.noise_input = AnnealedGaussianNoise(**annealed_noise_kwargs)

        self.input_size_with_counters = self.input_size
        self.output_size_without_counters = self.output_size
        if self.is_postcounter:
            self.output_size += self.counter_size
        else:
            self.input_size_with_counters += self.counter_size

        if self.is_kqrnn:
            self.generator = get_rnn(rnn_cell,
                                     self.input_size_with_counters,
                                     self.output_size_without_counters,
                                     num_layers=1,
                                     batch_first=True,
                                     is_weight_norm=self.is_weight_norm_rnn,
                                     is_get_hidden0=False)
        else:
            if is_mlps:
                self.generator = MLP(self.input_size_with_counters,
                                     self.output_size_without_counters,
                                     self.output_size_without_counters)
            else:
                self.generator = nn.Linear(self.input_size_with_counters,
                                           self.output_size_without_counters)

        self.dropout_output = AnnealedDropout(**annealed_dropout_output_kwargs)
        self.noise_output = AnnealedGaussianNoise(**annealed_noise_output_kwargs)

        self.reset_parameters()

    def flatten_parameters(self):
        """Flattens the parameters of the rnn."""
        if self.is_kqrnn:
            self.generator.flatten_parameters()

    def extra_repr(self):
        parrent_repr = super().extra_repr()
        # must use dict in conditional_shows because this is a base class
        new_repr = get_extra_repr(self,
                                  conditional_shows={"is_kqrnn": False,
                                                     "is_postcounter": False,
                                                     "is_rotcounters": False,
                                                     "is_abscounter": False,
                                                     "is_relcounter": False,
                                                     "is_normalize_encoder": True,
                                                     "is_weight_norm_rnn": False})
        if new_repr != "":
            parrent_repr += ", " + new_repr

        return parrent_repr

    def _get_counters(self, input_lengths_tensor, batch_size):
        return _get_counters(self.max_len, self.is_abscounter, self.is_relcounter,
                             self.is_rotcounters, input_lengths_tensor, batch_size)

    def forward(self, controller_out, input_lengths_tensor, additional):
        """Generates the key or query.

        Args:
            controller_out (torch.tensor): tensor of size (batch_size, input_length,
                hidden_size) containing the outputs of the controller (encoder
                or decoder for key and query respectively).
            input_lengths_tensor (tensor.FloatTensor): list of the lengths of
                each input in the batch.
            additional (dictionary): dictionary containing additional variables
                that are necessary for some hyperparamets.
        """
        batch_size, max_input_lengths, _ = controller_out.size()
        step = additional.get("step", 0)

        if self.is_contained_kv:
            input_generator = controller_out[:, :, :self.input_size]
        else:
            input_generator = controller_out

        input_lengths_tensor = input_lengths_tensor if self.is_normalize_encoder else None
        counters = self._get_counters(input_lengths_tensor, batch_size)
        if counters is not None:
            counters = counters[:, step:step + max_input_lengths, :]

        input_generator = self.noise_input(input_generator, is_update=(step == 0))
        input_generator = self.dropout_input(input_generator, is_update=(step == 0))

        if not self.is_postcounter and counters is not None:
            input_generator = torch.cat([input_generator, counters], dim=2)

        if self.is_kqrnn:
            kq_hidden = additional.get("kq_hidden", None)
            if kq_hidden is None:
                kq_hidden = replicate_hidden0(self.hidden0, batch_size)
            kq, kq_hidden = self.generator(input_generator, kq_hidden)
            additional["kq_hidden"] = kq_hidden
        else:
            kq = self.generator(input_generator)

        kq = self.dropout_output(kq, is_update=(step == 0))
        kq = self.noise_output(kq, is_update=(step == 0))

        if self.is_postcounter:
            kq = torch.cat([kq, counters], dim=2)

        return kq, additional


class KeyGenerator(BaseKeyQuery):
    """Key generator class.

   Args:
        hidden_size (int): size of the hidden activations of the controller.
        max_len (int): maximum length of the source sentence.
        kwargs:
            Additional arguments for the `BaseKeyQuery` parent class.
   """

    def __init__(self, hidden_size, max_len, **kwargs):
        super(KeyGenerator, self).__init__(hidden_size, max_len, **kwargs)

        if self.is_kqrnn:
            self.hidden0 = get_hidden0(self.generator)

    def reset_parameters(self):
        super().reset_parameters()

        if self.is_kqrnn:
            self.hidden0 = get_hidden0(self.generator)


class QueryGenerator(BaseKeyQuery):
    """Query generator class.

    Args:
         hidden_size (int): size of the hidden activations of the controller.
         max_len (int): maximum length of the source sentence.
         key_generator (KeyGenerator, optional): if given, will use the key
            generator for the query generator as well.
         kwargs:
             Additional arguments for the `BaseKeyQuery` parent class.
    """

    def __init__(self, hidden_size, max_len, key_generator=None, **kwargs):

        super(QueryGenerator, self).__init__(hidden_size, max_len, **kwargs)

        if key_generator is not None:
            self.generator = key_generator


class ValueGenerator(BaseKeyValueQuery):
    """Query generator class.

    Args:
        hidden_size (int): size of the hidden activations of the controller.
        embedding_size (int): size of the embeddings.
        min_generator_hidden (int, optional): minimum number fof hidden neurons
            to use if using a MLP.
        is_highway (bool, optional): whether to use a highway between the
            embedding an the output.
        is_res (bool, optional): whether to use a residual connection between
            the embedding an the output.
        is_mlps (bool, optional): whether to use MLPs for the generators instead
            of a linear layer.
        initial_highway (float, optional): initial highway carry rate. This can
            be useful to make the network learn the attention even before the
            decoder converged.
        is_single_carry (bool, optional): whetehr to use a one dimension carry
            weight instead of n dimensional. If a n dimension then the network
            can learn to carry some dimensions but not others. The downside is that
            the number of parameters would be larger.
        kwargs:
            Additional arguments for the `BaseKeyValueQuery` parent class.
    """

    def __init__(self, hidden_size, embedding_size,
                 min_generator_hidden=32,
                 is_highway=False,
                 is_res=False,
                 is_mlps=True,
                 initial_highway=0.5,
                 is_single_carry=True,
                 **kwargs):

        super(ValueGenerator, self).__init__(hidden_size, **kwargs)

        if (is_highway or is_res) and embedding_size != self.output_size:
            warnings.warn("Using value_size == {} instead of {} bcause highway or res.".format(embedding_size, self.output_size))
            self.output_size = embedding_size
            self.input_size = self._compute_input_size()

        self.is_highway = is_highway
        self.is_res = is_res
        self.is_single_carry = is_single_carry

        if is_mlps:
            self.generator = MLP(self.input_size,
                                 max(self.output_size, min_generator_hidden),
                                 self.output_size)
        else:
            self.generator = nn.Linear(self.input_size, self.output_size)

        if self.is_highway:
            carry_size = 1 if self.is_single_carry else self.output_size
            self.carrier = MLP(self.input_size,
                               max(min_generator_hidden, carry_size),
                               carry_size)
            self.carry_to_prob = ProbabilityConverter(initial_probability=initial_highway)

        self.reset_parameters()

    def extra_repr(self):
        parrent_repr = super().extra_repr()
        # must use dict in conditional_shows because this is a base class
        new_repr = get_extra_repr(self,
                                  conditional_shows=["is_highway",
                                                     "is_res",
                                                     "is_single_carry"])
        if new_repr != "":
            parrent_repr += ", " + new_repr

        return parrent_repr

    def forward(self, encoder_out, embedded, additional):
        """Generates the value.

        Args:
            encoder_out (torch.tensor): tensor of size (batch_size, input_length,
                hidden_size) containing the hidden activations of the encoder.
            embedded (torch.tensor): tensor of size (batch_size, input_length,
                embedding_size) containing the input embeddings.
            additional (dictionary): dictionary containing additional variables
                that are necessary for some hyperparamets.
        """
        if self.is_contained_kv:
            input_generator = encoder_out[:, :, -self.input_size:]
        else:
            input_generator = encoder_out

        values = self.generator(input_generator)

        if self.is_highway:
            carry_rates = self.carrier(input_generator)
            carry_rates = self.carry_to_prob(carry_rates)
            values = (1 - carry_rates) * values + carry_rates * embedded
            additional["carry_rates"] = carry_rates

        if self.is_res:
            values += embedded

        return values, additional
