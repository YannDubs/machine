import math

import torch
import torch.nn as nn

from seq2seq.util.initialization import get_hidden0, weights_init, replicate_hidden0
from seq2seq.util.helpers import MLP, generate_probabilities, renormalize_input_length

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


def _get_counters(max_len, is_abscounter, is_relcounter, is_rotcounters, input_lengths):
    abs_counter = rel_counter = rot_counters = torch.Tensor([])
    if is_abscounter:
        abs_counter = torch.arange(max_len + 1).unsqueeze(1)

    if is_relcounter:
        rel_counter = torch.arange(max_len + 1).unsqueeze(1) / max_len
        rel_counter = renormalize_input_length(rel_counter, input_lengths, max_len)

    if is_rotcounters:
        angles = torch.arange(max_len + 1) / max_len * math.pi
        angles = renormalize_input_length(angles, input_lengths, max_len)
        rot_counters = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

    if any(c.nelement() != 0 for c in (abs_counter, rel_counter, rot_counters)):
        counters = torch.cat([abs_counter, rel_counter, rot_counters], dim=1).to(device)
    else:
        counters = None

    return counters


class BaseKeyValueQuery(nn.Module):
    def __init__(self, hidden_size,
                 output_size=-1,
                 is_contained_kv=False):

        super(BaseKeyValueQuery, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = _compute_size(output_size, self.hidden_size, name="output_size - {}".format(type(self).__name__))
        self.is_contained_kv = is_contained_kv
        self.input_size = self.output_size if self.is_contained_kv else self.hidden_size

    def reset_parameters(self):
        self.apply(weights_init)


class BaseKeyQuery(BaseKeyValueQuery):
    def __init__(self, hidden_size, max_len,
                 is_kqrnn=False,
                 is_abscounter=False,
                 is_relcounter=False,
                 is_postcounter=False,
                 is_rotcounters=False,
                 is_normalize_encoder=False,
                 **kwargs):

        super(BaseKeyQuery, self).__init__(hidden_size, **kwargs)

        self.max_len = max_len
        self.is_kqrnn = is_kqrnn

        self.is_postcounter = is_postcounter
        self.is_rotcounters = is_rotcounters
        self.is_abscounter = is_abscounter
        self.is_relcounter = is_relcounter
        self.counter_size = _get_counter_size(self.is_abscounter, self.is_relcounter, self.is_rotcounters)
        self.is_normalize_encoder = is_normalize_encoder

        if self.is_postcounter:
            self.input_size += self.counter_size

        if self.is_kqrnn:
            self.generator = self.rnn_cell(self.input_size, self.output_size, 1, batch_first=True)
        else:
            self.generator = MLP(self.input_size, self.output_size, self.output_size)

        self.reset_parameters()

    def _get_counters(self, input_lengths):
        return _get_counters(self.max_len, self.is_abscounter, self.is_relcounter, self.is_rotcounters, input_lengths)

    def forward(self, inputs, input_lengths, additional):
        batch_size, max_input_lengths, _ = inputs.size()
        step = additional.get("step", 0)

        if self.is_contained_kv:
            input_generator = inputs[:, :, :self.output_size]
        else:
            input_generator = inputs

        input_lengths = input_lengths if self.is_normalize_encoder else None
        counters = self._get_counters(input_lengths)
        if counters is not None:
            counters = counters[step:step + max_input_lengths, :].view(1, -1, self.counter_size).expand(batch_size, -1, self.counter_size)

        if not self.is_postcounter and counters is not None:
            input_generator = torch.cat([input_generator, counters], dim=2)

        if self.is_kqrnn:
            kq_hidden = additional.get("kq_hidden", None)
            if kq_hidden is None:
                kq_hidden = replicate_hidden0(self.hidden0, batch_size, self.generator.batch_first)
            kq, kq_hidden = self.generator(input_generator, kq_hidden)
            additional["kq_hidden"] = kq_hidden
        else:
            kq = self.generator(input_generator)

        if self.is_postcounter:
            kq = torch.cat([kq, counters], dim=2)

        return kq, additional


class KeyGenerator(BaseKeyQuery):

    def __init__(self, hidden_size, max_len, **kwargs):
        super(KeyGenerator, self).__init__(hidden_size, max_len, **kwargs)

        if self.is_kqrnn:
            self.hidden0 = get_hidden0(self.generator)


class QueryGenerator(BaseKeyQuery):

    def __init__(self, hidden_size, max_len, key_generator=None, **kwargs):

        super(QueryGenerator, self).__init__(hidden_size, max_len, **kwargs)

        if key_generator is not None:
            self.generator = key_generator


class ValueGenerator(BaseKeyValueQuery):

    def __init__(self, hidden_size,
                 min_generator_hidden=32,
                 is_highway=False,
                 is_res=False,
                 **kwargs):

        super(ValueGenerator, self).__init__(hidden_size, **kwargs)

        if is_highway:
            assert self.output_size in [1, self.hidden_size], "Can only work with value size in {-1,1} when using highway."

        self.is_highway = is_highway
        self.is_res = is_res
        self.generator = MLP(self.input_size, max(self.output_size, min_generator_hidden), self.output_size)

        self.reset_parameters()

    def forward(self, encoder_out, embedded):
        if self.is_contained_kv:
            input_generator = encoder_out[:, :, -self.output_size:]
        else:
            input_generator = encoder_out

        values = self.generator(input_generator)

        if self.is_highway:
            carry_rates = generate_probabilities(values)
            values = (1 - carry_rates) * encoder_out + carry_rates * embedded

        if self.is_res:
            values += embedded

        return values
