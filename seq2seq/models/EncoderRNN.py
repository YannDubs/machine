import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .baseRNN import BaseRNN

from seq2seq.util.initialization import get_hidden0, replicate_hidden0, init_param, weights_init
from seq2seq.util.helpers import generate_probabilities
from seq2seq.models.KVQ import KeyGenerator, ValueGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(BaseRNN):
    """
    Applies a multi-layer KV-RNN to an input sequence.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        embedding_size (int): the size of the embedding of input variables
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (default False)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
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
                 input_dropout_p=0,
                 rnn_cell='gru',
                 n_layers=1,
                 bidirectional=False,
                 dropout_p=0,
                 variable_lengths=False,
                 key_kwargs={},
                 value_kwargs={},
                 is_highway=False,
                 is_res=False,
                 is_kv=False,
                 is_decoupled_kv=False):
        super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                                         input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.embedding_size = embedding_size
        self.variable_lengths = variable_lengths

        # # # keeping for testing # # #
        self.is_kv = is_kv
        self.is_highway = is_highway
        self.is_res = is_res
        self.is_decoupled_kv = is_decoupled_kv
        # # # # # # # # # # # # # # # #

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = self.rnn_cell(embedding_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        self.hidden0 = get_hidden0(self.rnn)

        if self.is_kv:
            self.key_generator = KeyGenerator(self.hidden_size, self.max_len, **key_kwargs)
            self.key_size = self.key_generator.output_size

            self.value_generator = ValueGenerator(self.hidden_size, **value_kwargs)
            self.value_size = self.value_generator.output_size
        else:
            self.key_size = hidden_size
            self.value_size = hidden_size

        # # # keeping for testing # # #
        if not self.is_kv and self.is_highway:
            if self.hidden_size != self.embedding_size:
                raise ValueError("hidden_size should be equal embedding_size when using highway.")
            self.carry = Parameter(torch.Tensor(1.0)).to(device)
        # # # # # # # # # # # # # # # #

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init)

        # # # keeping for testing # # #
        if self.is_highway:
            init_param(self.carry)
        # # # # # # # # # # # # # # # #

    def forward(self, input_var, input_lengths=None, additional=None):
        """
        Applies a multi-layer KV-RNN to an input sequence.

        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        if additional is None:
            additional = dict()

        batch_size = input_var.size(0)
        hidden = replicate_hidden0(self.hidden0, batch_size, self.rnn.batch_first)

        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        if self.variable_lengths:
            embedded_unpacked = embedded
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)

        output, hidden = self.rnn(embedded, hidden)

        if self.variable_lengths:
            embedded = embedded_unpacked
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        if self.is_kv:
            keys, additional = self.key_generator(output, input_lengths, additional)
            values = self.value_generator(output, embedded)

        # # # keeping for testing # # #
        else:
            keys = values = output

            if self.is_highway:
                carry_rate = generate_probabilities(self.carry)
                values = (1 - carry_rate) * values + (carry_rate) * embedded

            if self.is_decoupled_kv:
                if self.is_highway:
                    raise ValueError("Cannot have both highway and decoupled KV at the same time")

                dim = values.size(2)
                n_value = dim // 2

                select_keys = torch.zeros(dim).to(device)
                select_values = torch.ones(dim).to(device)

                select_keys[:-n_value] = 1
                select_values = select_values - select_keys

                keys = output * select_keys
                values = output * select_values
        # # # # # # # # # # # # # # # #

        return (keys, values), hidden, additional
