import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .baseRNN import BaseRNN

from seq2seq.util.helpers import MLP


def _compute_size(size, hidden_size, name=None):
    if size == -1:
        return hidden_size
    elif 0 < size < 1:
        return math.ceil(size * hidden_size)
    elif 0 < size <= hidden_size:
        return size
    else:
        raise ValueError("Invalid size for {} : {}".format(name, size))


class EncoderRNN(BaseRNN):
    """
    Applies a multi-layer RNN to an input sequence.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        embedding_size (int): the size of the embedding of input variables
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (default False)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`

    Examples::

         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)

    """

    def __init__(self, vocab_size, max_len, hidden_size, embedding_size,
                 input_dropout_p=0, dropout_p=0, n_layers=1, bidirectional=False,
                 rnn_cell='gru', variable_lengths=False, is_highway=False):
        super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                                         input_dropout_p, dropout_p, n_layers, rnn_cell)

        if is_highway and hidden_size != embedding_size:
            raise ValueError("hidden_size should be equal embedding_size when using highway.")

        self.is_highway = is_highway
        self.embedding_size = embedding_size
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = self.rnn_cell(embedding_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

        if self.is_highway:
            self.carry = Parameter(torch.Tensor(1))
            self.reset_parameters()

    def forward(self, input_var, input_lengths=None):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        if self.variable_lengths:
            if self.is_highway:
                embedded_unpacked = embedded
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)

        output, hidden = self.rnn(embedded)

        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            if self.is_highway:
                embedded = embedded_unpacked

        if self.is_highway:
            max_rate = 0.999
            min_rate = 1 - max_rate
            carry_rate = torch.sigmoid(self.carry) * max_rate
            output = (1 + min_rate - carry_rate) * output + (carry_rate + min_rate) * embedded

        return output, hidden

    def reset_parameters(self, start_rate=0.9):
        def logit(p):
            return torch.log(p / (1 - p))

        if self.is_highway:
            self.carry = Parameter(logit(torch.tensor(start_rate)))


class KVEncoderRnn(BaseRNN):
    def __init__(self, vocab_size, max_len, hidden_size, embedding_size, n_layers=1, rnn_cell='gru',
                 bidirectional=False, variable_lengths=False, input_dropout_p=0, dropout_p=0,
                 key_size=-1, value_size=-1, is_highway=False, is_abscounter=False, is_relcounter=False,
                 is_postcounter=False, is_rotcounters=False, is_contained_kv=False,
                 is_kqrnn=False, is_res=False):
        super(KVEncoderRnn, self).__init__(vocab_size, max_len, hidden_size,
                                           input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.embedding_size = embedding_size
        self.variable_lengths = variable_lengths
        self.key_size = _compute_size(key_size, self.hidden_size, name="key_size")
        self.value_size = _compute_size(value_size, self.hidden_size, name="value_size")
        self.is_postcounter = is_postcounter
        self.is_rotcounters = is_rotcounters
        self.is_abscounter = is_abscounter
        self.is_relcounter = is_relcounter
        self.is_highway = is_highway
        self.is_res = is_res
        self.is_contained_kv = is_contained_kv
        self.is_kqrnn = is_kqrnn

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = self.rnn_cell(embedding_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

        if self.is_contained_kv:
            value_input_size = self.value_size
            key_input_size = self.key_size
        else:
            value_input_size = key_input_size = hidden_size

        min_value_generator_hidden = 32
        self.value_generator = MLP(value_input_size, max(self.value_size, min_value_generator_hidden), self.value_size)

        self.counter_size = 0
        if self.is_abscounter:
            self.counter_size += 1
        if self.is_relcounter:
            self.counter_size += 1
        if self.is_rotcounters:
            self.counter_size += 2

        key_input_size = key_input_size + int(not self.is_postcounter) * self.counter_size

        if self.is_kqrnn:
            self.key_generator = self.rnn_cell(key_input_size, self.key_size, 1, batch_first=True)
        else:
            self.key_generator = MLP(key_input_size, self.key_size, self.key_size)

        if self.is_highway:
            assert value_size == 1 or value_size == -1, "Can only work with value size in {-1,1} when using highway."

        abs_counter = rel_counter = rot_counters = torch.Tensor([])
        if self.is_abscounter:
            abs_counter = torch.arange(max_len + 1).unsqueeze(1)

        if self.is_relcounter:
            rel_counter = torch.arange(max_len + 1).unsqueeze(1) / max_len

        if self.is_rotcounters:
            angles = torch.arange(max_len + 1) / max_len * math.pi
            rot_counters = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

        if any(c.nelement() != 0 for c in (abs_counter, rel_counter, rot_counters)):
            self.counters = torch.cat([abs_counter, rel_counter, rot_counters], dim=1)
            if torch.cuda.is_available():
                self.counters = self.counters.cuda()
        else:
            self.counters = None

    def forward(self, input_var, input_lengths=None):
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        if self.variable_lengths:
            embedded_unpacked = embedded
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)

        output, hidden = self.rnn(embedded)

        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            embedded = embedded_unpacked

        batch, seq_len, _ = output.size()

        if self.is_contained_kv:
            key_input = output[:, :, :self.key_size]
            value_input = output[:, :, -self.value_size:]
        else:
            key_input = value_input = output

        if self.counters is not None:
            counters = self.counters[:seq_len, :].view(1, -1, self.counter_size).expand(batch, -1, self.counter_size)

        if not self.is_postcounter and self.counters is not None:
            # precounter
            key_input = torch.cat([output, counters], dim=2)

        if self.is_kqrnn:
            keys, keys_hidden = self.key_generator(key_input)
            hidden = (hidden, keys_hidden)
        else:
            keys = self.key_generator(key_input)

        values = self.value_generator(value_input)

        if self.is_postcounter:
            keys = torch.cat([keys, counters], dim=2)

        if self.is_highway:
            carry_rates = torch.sigmoid(values)
            values = (1 - carry_rates) * output + carry_rates * embedded

        if self.is_res:
            values += embedded

        return (keys, values), hidden
