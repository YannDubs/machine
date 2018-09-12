""" Encoder class for a seq2seq. """
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .baseRNN import BaseRNN

from seq2seq.util.initialization import replicate_hidden0, init_param, weights_init
from seq2seq.util.helpers import (get_rnn, get_extra_repr,
                                  format_source_lengths)
from seq2seq.util.torchextend import ProbabilityConverter
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
        input_dropout_p (float, optional): dropout probability for the input
            sequence (default: 0)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder
            (default False)
        dropout_p (float, optional): dropout probability for the output sequence
            (default: 0)
        variable_lengths (bool, optional): if use variable length RNN (default: False)
        key_kwargs (dict, optional): additional arguments to the key generator.
        value_kwargs (dict, optional): additional arguments to the value generator.
        is_dev_mode (bool, optional): whether to store many useful variables in
            `additional`. Useful when predicting with a trained model in dev mode
             to understand what the model is doing. Use with `dev_predict`.
        is_viz_train (bool, optional): whether to save how the averages of some
            intepretable variables change during training in "visualization"
            of `additional`.

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within
            which each sequence is a list of token IDs.
        - **input_lengths** (tuple(list of int, float.Tensor), optional): list
            that contains the lengths of sequences in the mini-batch, it must be
            provided when using variable length RNN. The Tensor has the same information
            but is preinitialized on teh correct device. (default: `None`)

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded
            features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor
            containing the features in the hidden state `h`

    Examples::

         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)

    """

    def __init__(self, vocab_size, max_len, hidden_size, embedding_size,
                 input_dropout_p=0,
                 rnn_cell='gru',
                 is_weight_norm_rnn=False,  # TO DOC
                 n_layers=1,
                 bidirectional=False,
                 dropout_p=0,
                 variable_lengths=False,
                 key_kwargs={},
                 value_kwargs={},
                 is_highway=False,
                 is_res=False,
                 is_key=True,
                 is_value=True,
                 is_decoupled_kv=False,
                 initial_highway=0.5,
                 is_dev_mode=False,
                 is_viz_train=False):
        super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                                         input_dropout_p, dropout_p, n_layers,
                                         rnn_cell)

        self.is_weight_norm_rnn = is_weight_norm_rnn
        self.bidirectional_hidden_size = (self.hidden_size * 2
                                          if bidirectional else self.hidden_size)
        self.embedding_size = embedding_size
        self.variable_lengths = variable_lengths

        self.is_key = is_key
        self.is_value = is_value
        self.is_decoupled_kv = is_decoupled_kv
        self.is_dev_mode = is_dev_mode
        self.is_viz_train = is_viz_train

        # # # keeping for testing # # #
        self.is_highway = is_highway
        self.is_res = is_res
        # # # # # # # # # # # # # # # #

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.controller, self.hidden0 = get_rnn(self.rnn_cell,
                                                self.embedding_size, self.hidden_size,
                                                num_layers=self.n_layers,
                                                batch_first=True,
                                                bidirectional=bidirectional,
                                                dropout=dropout_p,
                                                is_weight_norm=self.is_weight_norm_rnn,
                                                is_get_hidden0=True)

        if self.is_key:
            self.key_generator = KeyGenerator(self.bidirectional_hidden_size,
                                              self.max_len,
                                              **key_kwargs)
            self.key_size = self.key_generator.output_size
        else:
            self.key_size = self.bidirectional_hidden_size

        if self.is_value:
            self.value_generator = ValueGenerator(self.bidirectional_hidden_size,
                                                  embedding_size,
                                                  **value_kwargs)
            self.value_size = self.value_generator.output_size
        else:
            self.value_size = self.bidirectional_hidden_size

        # # # keeping for testing # # #
        if not self.is_value and self.is_highway:
            if self.bidirectional_hidden_size != self.embedding_size:
                raise ValueError(
                    "hidden_size should be equal embedding_size when using highway.")
            self.carry = Parameter(torch.tensor(1.0))
            self.carry_to_prob = ProbabilityConverter(
                initial_probability=initial_highway)
        # # # # # # # # # # # # # # # #

        self.reset_parameters()

    def set_dev_mode(self, value=True):
        self.is_dev_mode = value
        if self.is_key:
            self.key_generator.set_dev_mode(value=value)
        if self.is_value:
            self.value_generator.set_dev_mode(value=value)

    def reset_parameters(self):
        self.apply(weights_init)

        if self.is_key:
            self.key_generator.reset_parameters()
        if self.is_value:
            self.value_generator.reset_parameters()

        # # # keeping for testing # # #
        if self.is_highway and not self.is_value:
            init_param(self.carry)
        # # # # # # # # # # # # # # # #

    def flatten_parameters(self):
        self.controller.flatten_parameters()

        if self.is_key:
            self.key_generator.flatten_parameters()

    def extra_repr(self):
        return get_extra_repr(self,
                              conditional_shows=["variable_lengths",
                                                 "is_highway",
                                                 "is_res",
                                                 "is_key",
                                                 "is_value",
                                                 "is_decoupled_kv"])

    def forward(self, input_var, input_lengths=None, additional=None):
        """
        Applies a multi-layer KV-RNN to an input sequence.

        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (tuple(list of int, torch.FloatTesnor), optional): A
                list that contains the lengths of sequences in the mini-batch. The
                Tensor has the same information but is preinitialized on teh
                correct device.

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the
                encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size):
                variable containing the features in the hidden state h
        """
        input_lengths_list, input_lengths_tensor = format_source_lengths(
            input_lengths)

        additional = self._initialize_additional(additional)

        batch_size = input_var.size(0)
        hidden = replicate_hidden0(self.hidden0, batch_size)

        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        if self.variable_lengths:
            embedded_unpacked = embedded
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths_list,
                                                         batch_first=True)

        output, hidden = self.controller(embedded, hidden)

        if self.variable_lengths:
            embedded = embedded_unpacked
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True)

        if self.is_key:
            keys, additional = self.key_generator(
                output, input_lengths_tensor, additional)
        else:
            keys = output

        if self.is_value:
            values, additional = self.value_generator(
                output, embedded, additional)
        else:
            values = output

            # # # keeping for testing # # #
            if self.is_highway:
                carry_rates = self.carry_to_prob(self.carry)
                if self.is_dev_mode:
                    additional["test"]["carry_rates"] = carry_rates

                values = (1 - carry_rates) * values + (carry_rates) * embedded
            # # # # # # # # # # # # # # # #

        # # # keeping for testing # # #
        if not self.is_value and not self.is_key:
            if self.is_decoupled_kv:
                if self.is_highway:
                    raise ValueError(
                        "Cannot have both highway and decoupled KV at the same time")

                dim = values.size(2)
                n_value = dim // 2

                select_keys = torch.zeros(dim).to(device)
                select_values = torch.ones(dim).to(device)

                select_keys[:-n_value] = 1
                select_values = select_values - select_keys

                keys = output * select_keys
                values = output * select_values
        # # # # # # # # # # # # # # # #

        additional["last_enc_controller_out"] = output[:, -1:, :]

        return (keys, values), hidden, additional

    def _initialize_additional(self, additional):
        if additional is None:
            additional = dict()

        if self.is_dev_mode:
            additional["test"] = additional.get("test", dict())

        if self.is_viz_train:
            additional["visualize"] = additional.get("visualize", dict())

        return additional
