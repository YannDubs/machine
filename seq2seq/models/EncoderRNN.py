""" Encoder class for a seq2seq. """
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence

from .baseRNN import BaseRNN

from seq2seq.util.initialization import replicate_hidden0, init_param, weights_init
from seq2seq.util.helpers import (get_rnn, get_extra_repr, format_source_lengths)
from seq2seq.util.torchextend import ProbabilityConverter
from seq2seq.models.KVQ import KeyGenerator, ValueGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _precompute_max_loss(p, max_n=100):
    return torch.tensor([np.mean([np.abs(i - (n + 1) / 2)**p
                                  for i in range(1, n + 1)])
                         for n in range(0, max_n)],
                        dtype=torch.float,
                        device=device)


MAX_LOSSES_P05 = _precompute_max_loss(0.5)


def _get_max_loss_key_confuser(input_lengths_list, input_lengths_tensor,
                               p=2, factor=1):
    """
    Returns the expected maximum loss of the key confuser depending on p used.
    `max_loss = ∑_{i=1}^n (i-N/2)**p`

    Args:
        input_lengths_list (list): list containing the legnth of each sentence
            of the batch.
        input_lengths_list (tensor): Float tensor containing the legnth of each
            sentence of the batch. Should already be on the correc device.
        p (float, optional): p of the Lp pseudo-norm used as loss.
        factor (float, optional): by how much to decrease the maxmum loss. If factor
            is 2 it means that you consider that the maximum loss will be achieved
            if your prediction is 1/factor (i.e half) way between the correct i
            and the best worst case output N/2. Factor = 10 means it can be a lot
            closer to i. This is usefull as there will always be some noise, and you
            don't want to penalize the model for some noise.
    """

    # E[(i-N/2)**2] = VAR(i) = (n**2 - 1)/12
    if p == 2:
        max_losses = (input_lengths_tensor**2 - 1) / 12
    elif p == 1:
        # computed by hand and use modulo because different if odd
        max_losses = (input_lengths_tensor**2 - input_lengths_tensor % 2) / (4 * input_lengths_tensor)
    elif p == 0.5:
        max_losses = MAX_LOSSES_P05[input_lengths_list]
    else:
        raise ValueError("Unkown p={}".format(p))

    max_losses = max_losses / (factor**p)

    return max_losses


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

        self.enc_counter = torch.arange(1, self.max_len + 1,
                                        dtype=torch.float,
                                        device=device)

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

    def forward(self, input_var, input_lengths=None, additional=None, confusers=dict()):
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
        input_lengths_list, input_lengths_tensor = format_source_lengths(input_lengths)

        additional = self._initialize_additional(additional)

        batch_size = input_var.size(0)
        max_input_len = input_var.size(1)

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
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        if self.is_key:
            keys, additional = self.key_generator(output,
                                                  input_lengths_tensor,
                                                  additional)
        else:
            keys = output

        if "key_confuser" in confusers:

            counting_target_i = self.enc_counter.expand(batch_size, -1)[:, :max_input_len]

            # masks everything which finished decoding
            mask = counting_target_i > input_lengths_tensor.unsqueeze(1)

            # important to have an "admissible" heuristic, i.e an upper bound
            # that is never greater than the real bound. if not when the confuser
            # outputs some random noise, the key would be regularized in a random manner.

            # the best loss it can get without any infor of j is always predict N/2

            # factor should anneal because after many iterations the discriminator
            # should start outputting N/2(if it has no idea). In which case
            # the bound can get tighter and tighter. Do not use low factor at begining
            # as the discriminator will generate random numbers which could be correct
            # and we don't want to penalize the generator for that.
            max_losses = _get_max_loss_key_confuser(input_lengths_list, input_lengths_tensor,
                                                    p=0.5,
                                                    factor=confusers["key_confuser"
                                                                     ].get_factor(self.training))

            to_cat = input_lengths_tensor.view(-1, 1, 1).expand(-1, max_input_len, 1)
            key_confuse_input = torch.cat([keys, to_cat], dim=-1)
            confusers["key_confuser"].compute_loss(key_confuse_input,
                                                   targets=counting_target_i.unsqueeze(-1),
                                                   seq_len=input_lengths_tensor.unsqueeze(-1),
                                                   max_losses=max_losses.unsqueeze(-1),
                                                   mask=mask)

        # DEV MODE TO UNDERSTAND CONFUSERS
        self._add_to_test(keys, "keys", additional)

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

    def _add_to_test(self, values, keys, additional):
        """
        Save a variable to additional["test"] only if dev mode is on. The
        variables saved should be the interpretable ones for which you want to
        know the value of during test time.

        Batch size should always be 1 when predicting with dev mode !
        """
        if self.is_dev_mode:
            if isinstance(keys, list):
                for k, v in zip(keys, values):
                    self._add_to_test(v, k, additional)
            else:
                if isinstance(values, torch.Tensor):
                    values = values.detach().cpu()
                additional["test"][keys] = values
