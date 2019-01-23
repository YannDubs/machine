import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention
from .baseRNN import BaseRNN

<<<<<<< HEAD:seq2seq/models/DecoderRNN.py
from seq2seq.util.helpers import MLP

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device
=======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
>>>>>>> upstream/master:machine/models/DecoderRNN.py


def _discrete_truncated_gaussian(x, mu, sigma):
    """Samples a values from a gaussian pdf and normalizes those."""
    x = torch.exp(-((x - mu) / sigma)**2)
    x = F.normalize(x, p=1, dim=1)
    return x


def _discrete_truncated_laplace(x, mu, b):
    """Samples a values from a laplacian pdf and normalizes those."""
    x = torch.exp(-1 * torch.abs((x - mu) / b))
    x = F.normalize(x, p=1, dim=1)
    return x


def _get_positioner(name):
    if name == "gaussian":
        return _discrete_truncated_gaussian
    elif name == "laplace":
        return _discrete_truncated_laplace
    else:
        raise ValueError("Unkown positioner method {}".format(name))


class DecoderRNN(BaseRNN):
    """
    Provides functionality for decoding in a seq2seq framework, with an option for attention.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
<<<<<<< HEAD:seq2seq/models/DecoderRNN.py
        use_attention (str, optional): type of attention to use, possible values: {'pre-rnn','post-rnn',None}. (default: None)
        attention_method (str, optional): the method to compute the alignment, possible values : {"mlp","dot"}. (default: None)
=======
        use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)
        full_focus(bool, optional): flag indication whether to use full attention mechanism or not (default: false)
>>>>>>> upstream/master:machine/models/DecoderRNN.py

    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`

    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_ATTN_SCORE* : list of floats, indicating attention weights. }.
    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

<<<<<<< HEAD:seq2seq/models/DecoderRNN.py
    def __init__(self, vocab_size, max_len, hidden_size, sos_id, eos_id,
                 n_layers=1, rnn_cell='gru', bidirectional=False, input_dropout_p=0,
                 dropout_p=0, use_attention=None, attention_method=None,
                 is_decoupled_kv=False, is_kv=False, querry_size=None, value_size=None,
                 is_abscounter=False, is_relcounter=False, is_postcounter=False, is_rotcounters=False,
                 is_contained_kv=False, querry_generator=None, is_kqrnn=False, positioning_method=None,
                 is_posrnn=False, is_normalize_encoder=False, is_full_focus=False, min_sigma=0.05, min_location_importance=0):
=======
    def __init__(self, vocab_size, max_len, hidden_size,
                 sos_id, eos_id,
                 n_layers=1, rnn_cell='gru', bidirectional=False,
                 input_dropout_p=0, dropout_p=0, use_attention=False, attention_method=None, full_focus=False):
>>>>>>> upstream/master:machine/models/DecoderRNN.py
        super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                                         input_dropout_p, dropout_p,
                                         n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        input_size = hidden_size

<<<<<<< HEAD:seq2seq/models/DecoderRNN.py
        if use_attention != False and attention_method is None:
            raise ValueError("Method for computing attention should be provided")

        if use_attention == 'post-rnn' and attention_method == 'mlp':
            raise NotImplementedError("post-rnn attention with mlp alignment model not implemented")
=======
        if use_attention and attention_method is None:
            raise ValueError(
                "Method for computing attention should be provided")
>>>>>>> upstream/master:machine/models/DecoderRNN.py

        self.attention_method = attention_method
        self.full_focus = full_focus

<<<<<<< HEAD:seq2seq/models/DecoderRNN.py
        # increase input size decoder if attention is applied before decoder rnn
        if use_attention == 'pre-rnn' and not is_full_focus:
            input_size += value_size
=======
        # increase input size decoder if attention is applied before decoder
        # rnn
        if use_attention == 'pre-rnn' and not full_focus:
            input_size *= 2
>>>>>>> upstream/master:machine/models/DecoderRNN.py

        self.rnn = self.rnn_cell(input_size, hidden_size, n_layers,
                                 batch_first=True, dropout=dropout_p)

        self.is_abscounter = is_abscounter
        self.is_relcounter = is_relcounter
        self.is_postcounter = is_postcounter
        self.is_rotcounters = is_rotcounters
        self.querry_size = querry_size if querry_size != -1 else hidden_size
        self.is_querry = self.querry_size is not None
        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.is_contained_kv = is_contained_kv
        self.value_size = value_size
        self.querry_generator = querry_generator
        self.is_kqrnn = is_kqrnn
        self.is_positioning_generator = positioning_method is not None
        self.is_posrnn = is_posrnn
        self.is_full_focus = is_full_focus

        self.counter_size = 0
        if self.is_abscounter:
            self.counter_size += 1
        if self.is_relcounter:
            self.counter_size += 1
        if self.is_rotcounters:
            self.counter_size += 2

        self.init_input = None

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if use_attention:
<<<<<<< HEAD:seq2seq/models/DecoderRNN.py
            self.attention = Attention(self.querry_size, self.attention_method, is_decoupled_kv=is_decoupled_kv,
                                       kv_architecture=is_kv, is_postcounter=is_postcounter, counter_size=self.counter_size,
                                       is_positioning_generator=self.is_positioning_generator)

        if self.querry_generator is None and self.is_querry:
            querry_input_size = self.querry_size if self.is_contained_kv else hidden_size
            querry_input_size += int(not self.is_postcounter) * self.counter_size
            if self.is_kqrnn:
                self.querry_generator = self.rnn_cell(querry_input_size, self.querry_size, 1, batch_first=True)
            else:
                self.querry_generator = MLP(querry_input_size, self.querry_size, self.querry_size)

        if use_attention == 'post-rnn':
            self.out = nn.Linear(self.hidden_size + self.value_size, self.output_size)
        else:
            self.out = nn.Linear(self.hidden_size, self.output_size)
            if self.is_full_focus:
                self.ffocus_merge = nn.Linear(self.hidden_size + self.value_size, self.hidden_size)

        self.biais = torch.tensor(1.0).cuda if torch.cuda.is_available() else torch.tensor(1.0)

        if self.is_positioning_generator:
            self.positioning_method = _get_positioner(positioning_method)
            self.rel_counter = torch.arange(1, self.max_length + 1).unsqueeze(1) / (self.max_length)
            self.front_step = torch.tensor(1 / self.max_length)
            self.min_sigma = min_sigma
            self.min_location_importance = min_location_importance

            if torch.cuda.is_available():
                self.front_step = self.front_step.cuda()
                self.rel_counter = self.rel_counter.cuda()

            positioning_generator_input = self.hidden_size + 5
            positioning_generator_output = 8
            positioning_generator_hidden = 32  # test
            if self.is_posrnn:
                self.positioning_rnn = self.rnn_cell(positioning_generator_input, positioning_generator_hidden, 1, batch_first=True)
                self.mu_weights_generator = nn.Linear(positioning_generator_hidden, positioning_generator_output)
                self.sigma_weights_generator = nn.Linear(positioning_generator_hidden, positioning_generator_output)
            else:
                self.mu_weights_generator = MLP(positioning_generator_input, positioning_generator_hidden, positioning_generator_output)
                self.sigma_weights_generator = MLP(positioning_generator_input, positioning_generator_hidden, positioning_generator_output)
            self.location_percentage_generator = MLP(self.hidden_size, positioning_generator_hidden, 1)

        abs_counter = rel_counter = rot_counters = torch.Tensor([])
        if self.is_abscounter:
            abs_counter = torch.arange(1, max_len + 1).unsqueeze(1)

        if self.is_relcounter:
            rel_counter = torch.arange(1, max_len + 1).unsqueeze(1) / self.max_length

        if self.is_rotcounters:
            angles = torch.arange(1, max_len + 1) / self.max_length * math.pi
            rot_counters = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

        if any(c.nelement() != 0 for c in (abs_counter, rel_counter, rot_counters)):
            self.counters = torch.cat([abs_counter, rel_counter, rot_counters], dim=1)
            if torch.cuda.is_available():
                self.counters = self.counters.cuda()
        else:
            self.counters = None

    def forward_step(self, input_var, hidden, encoder_outputs, function, input_lengths=None, additional=None):

        additional = dict() if additional is None else additional
        additional["batch_size"] = batch_size = input_var.size(0)
        additional["output_size"] = output_size = input_var.size(1)
        additional["input_lengths"] = input_lengths

        additional = self._initialize_additional(additional)

        if self.is_kqrnn:
            hidden, keys_hidden = hidden

        attn = None

        if self.use_attention == 'pre-rnn':
            if additional.get('decoder_output', None) is None:
                h = hidden
                if isinstance(hidden, tuple):
                    h, c = hidden

                additional["decoder_output"] = h[-1:].transpose(0, 1)

            context, attn = self._compute_context(additional['decoder_output'], encoder_outputs, additional)
        else:
            context = None

        decoder_input = self._compute_decoder_input(input_var, context=context)

        decoder_output, hidden = self.rnn(decoder_input, hidden)

        if self.use_attention == 'pre-rnn':
            additional['decoder_output'] = decoder_output

        if self.use_attention == 'post-rnn':
            context, attn = self._compute_context(decoder_output, encoder_outputs, additional)
            prediction_input = torch.cat((context, decoder_output), dim=2)
        else:
            prediction_input = decoder_output
=======
            self.attention = Attention(self.hidden_size, self.attention_method)
        else:
            self.attention = None

        if use_attention == 'post-rnn':
            self.out = nn.Linear(2 * self.hidden_size, self.output_size)
        else:
            self.out = nn.Linear(self.hidden_size, self.output_size)
            if self.full_focus:
                self.ffocus_merge = nn.Linear(
                    2 * self.hidden_size, hidden_size)

    def forward_step(self, input_var, hidden,
                     encoder_outputs, function, **kwargs):
        """
        Performs one or multiple forward decoder steps.

        Args:
            input_var (torch.tensor): Variable containing the input(s) to the decoder RNN
            hidden (torch.tensor): Variable containing the previous decoder hidden state.
            encoder_outputs (torch.tensor): Variable containing the target outputs of the decoder RNN
            function (torch.tensor): Activation function over the last output of the decoder RNN at every time step.

        Returns:
            predicted_softmax: The output softmax distribution at every time step of the decoder RNN
            hidden: The hidden state at every time step of the decoder RNN
            attn: The attention distribution at every time step of the decoder RNN
        """
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        if self.use_attention == 'pre-rnn':
            h = hidden
            if isinstance(hidden, tuple):
                h, c = hidden
            # Apply the attention method to get the attention vector and weighted context vector. Provide decoder step for hard attention
            # transpose to get batch at the second index
            context, attn = self.attention(
                h[-1:].transpose(0, 1), encoder_outputs, **kwargs)
            combined_input = torch.cat((context, embedded), dim=2)
            if self.full_focus:
                merged_input = F.relu(self.ffocus_merge(combined_input))
                combined_input = torch.mul(context, merged_input)
            output, hidden = self.rnn(combined_input, hidden)

        elif self.use_attention == 'post-rnn':
            output, hidden = self.rnn(embedded, hidden)
            # Apply the attention method to get the attention vector and
            # weighted context vector. Provide decoder step for hard attention
            context, attn = self.attention(output, encoder_outputs, **kwargs)
            output = torch.cat((context, output), dim=2)
>>>>>>> upstream/master:machine/models/DecoderRNN.py

        prediction_input = prediction_input.contiguous().view(-1, self.out.in_features)

<<<<<<< HEAD:seq2seq/models/DecoderRNN.py
        predicted_softmax = function(self.out(prediction_input), dim=1).view(batch_size, output_size, -1)
=======
        predicted_softmax = function(self.out(
            output.contiguous().view(-1, self.out.in_features)), dim=1).view(batch_size, output_size, -1)
>>>>>>> upstream/master:machine/models/DecoderRNN.py

        if self.is_kqrnn:
            hidden = (hidden, additional.pop('keys_hidden'))

        return predicted_softmax, hidden, attn, additional

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
<<<<<<< HEAD:seq2seq/models/DecoderRNN.py
                function=F.log_softmax, teacher_forcing_ratio=0, input_lengths=None):
=======
                function=F.log_softmax, teacher_forcing_ratio=0):

>>>>>>> upstream/master:machine/models/DecoderRNN.py
        ret_dict = dict()
        additional = dict()
        if self.is_posrnn:
            additional['positioner_hidden'] = None

        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function, teacher_forcing_ratio)

        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn, additional=None):
            keys_to_store = ["inspect", "mean_attention", "mean_content", "mu_old", 'sigma_old']

            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            if additional is not None:
                for k in keys_to_store:
                    if k in additional:
                        v = additional[k]
                        if isinstance(v, dict):
                            ret_dict[k] = ret_dict.get(k, dict())
                            for sub_k, sub_v in v.items():
                                ret_dict[k][sub_k] = ret_dict[k].get(sub_k, list())
                                ret_dict[k][sub_k].append(sub_v)
                        else:
                            ret_dict[k] = ret_dict.get(k, list())
                            ret_dict[k].append(v)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

<<<<<<< HEAD:seq2seq/models/DecoderRNN.py
        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        if (use_teacher_forcing and not self.is_positioning_generator) and self.use_attention == 'post-rnn':
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, attn, additional = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                                 function=function, input_lengths=input_lengths,
                                                                                 additional=additional)
=======
        # When we use pre-rnn attention we must unroll the decoder. We need to calculate the attention based on
        # the previous hidden state, before we can calculate the next hidden state.
        # We also need to unroll when we don't use teacher forcing. We need perform the decoder steps
        # one-by-one since the output needs to be copied to the input of the
        # next step.
        if self.use_attention == 'pre-rnn' or not use_teacher_forcing:
            unrolling = True
        else:
            unrolling = False

        if unrolling:
            symbols = None
            for di in range(max_length):
                # We always start with the SOS symbol as input. We need to add extra dimension of length 1 for the number of decoder steps (1 in this case)
                # When we use teacher forcing, we always use the target input.
                if di == 0 or use_teacher_forcing:
                    decoder_input = inputs[:, di].unsqueeze(1)
                # If we don't use teacher forcing (and we are beyond the first
                # SOS step), we use the last output as new input
                else:
                    decoder_input = symbols

                # Perform one forward step
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                              function=function)
                # Remove the unnecessary dimension.
                step_output = decoder_output.squeeze(1)
                # Get the actual symbol
                symbols = decode(di, step_output, step_attn)

        else:
            # Remove last token of the longest output target in the batch. We don't have to run the last decoder step where the teacher forcing input is EOS (or the last output)
            # It still is run for shorter output targets in the batch
            decoder_input = inputs[:, :-1]

            # Forward step without unrolling
            decoder_output, decoder_hidden, attn = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, function=function)
>>>>>>> upstream/master:machine/models/DecoderRNN.py

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn, additional=additional)

<<<<<<< HEAD:seq2seq/models/DecoderRNN.py
        elif (use_teacher_forcing and not self.is_positioning_generator) and self.use_attention == 'pre-rnn':
            # unroll computation to apply attention before rnn layer
            for di in range(inputs.size(1) - 1):
                decoder_input = inputs[:, di].unsqueeze(1)
                additional["di"] = di
                decoder_output, decoder_hidden, step_attn, additional = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                                          function=function,
                                                                                          input_lengths=input_lengths,
                                                                                          additional=additional)

                step_output = decoder_output.squeeze(1)
                decode(di, step_output, step_attn, additional=additional)

        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                if use_teacher_forcing:
                    decoder_input = inputs[:, di].unsqueeze(1)

                decoder_output, decoder_hidden, step_attn, additional = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                                          function=function,
                                                                                          input_lengths=input_lengths,
                                                                                          additional=additional)

                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output, step_attn, additional=additional)

                if not use_teacher_forcing:
                    decoder_input = symbols

=======
>>>>>>> upstream/master:machine/models/DecoderRNN.py
        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h)
                                    for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden,
                       encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError(
                    "Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                hidden = encoder_hidden[0] if self.is_kqrnn else encoder_hidden

                if self.rnn_cell is nn.LSTM:
                    batch_size = hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
<<<<<<< HEAD:seq2seq/models/DecoderRNN.py
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol
=======
                raise ValueError(
                    "Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.tensor([self.sos_id] * batch_size, dtype=torch.long,
                                  device=device).view(batch_size, 1)

            max_length = self.max_length
        else:
            # minus the start of sequence symbol
            max_length = inputs.size(1) - 1
>>>>>>> upstream/master:machine/models/DecoderRNN.py

        return inputs, batch_size, max_length

    def _renormalize_input_length(self, x, input_lengths):
        if input_lengths is None:
            return x
        else:
            lengths = torch.FloatTensor(input_lengths)
            while lengths.dim() < x.dim():
                lengths = lengths.unsqueeze(-1)
            return (x * self.max_length) / lengths

    def _get_positioning_inputs_building_blocks(self,
                                                decoder_outputs,
                                                mean_attention,
                                                mean_content,
                                                mu_old,
                                                sigma_old,
                                                di,
                                                batch_size,
                                                output_size,
                                                input_lengths):
        # if here then output_size == 1 because step by step
        rel_counter_decoder = self._renormalize_input_length(self.rel_counter[di:di + output_size].expand(batch_size, 1), input_lengths)

        shared = [mean_attention, mean_content, mu_old, sigma_old, rel_counter_decoder]

        positioning_inputs = torch.cat([decoder_outputs.squeeze(1)] + shared, dim=1)

        single_step = self._renormalize_input_length(self.front_step.expand(batch_size, 1), input_lengths)
        bias = self.biais.expand(batch_size, 1)
        positioning_building_blocks = torch.cat(shared + [single_step, -1 * single_step, bias], dim=1)

        return positioning_inputs, positioning_building_blocks

    def _compute_parametric_weights(self, positioning_inputs, additional):
        if self.is_posrnn:
            positioning_inputs, positioner_hidden = self.positioning_rnn(positioning_inputs.unsqueeze(1),
                                                                         additional['positioner_hidden'])
            positioning_inputs = positioning_inputs.squeeze(1)
            additional['positioner_hidden'] = positioner_hidden

        mu_weights = self.mu_weights_generator(positioning_inputs)
        sigma_weights = self.sigma_weights_generator(positioning_inputs)

        return mu_weights, sigma_weights

    def _compute_positioners(self, mu_weights, sigma_weights, positioning_building_blocks, additional, rel_counter_encoder):
        # (batch, 1, 5) * (batch, 5, 1) -> (batch, 1, 1)
        mu = torch.bmm(mu_weights.unsqueeze(1), positioning_building_blocks.unsqueeze(2))

        sigma = torch.bmm(sigma_weights.unsqueeze(1), positioning_building_blocks.unsqueeze(2))
        sigma = torch.relu(sigma) + self.min_sigma

        positioners = self.positioning_method(rel_counter_encoder,
                                              mu.expand_as(rel_counter_encoder),
                                              sigma.expand_as(rel_counter_encoder))

        additional['mu_old'] = mu.squeeze(2)
        additional['sigma_old'] = sigma.squeeze(2)

        return positioners

    def _compute_location_importance(self, decoder_outputs):
        if not self.is_positioning_generator:
            return None

        location_percentage = torch.sigmoid(self.location_percentage_generator(decoder_outputs)) + self.min_location_importance
        location_percentage = location_percentage / (1 + self.min_location_importance * 2)

        return location_percentage

    def _location_attention_generator(self, decoder_outputs, additional, rel_counter_encoder=None):
        if not self.is_positioning_generator:
            return None

        positioning_inputs, positioning_building_blocks = self._get_positioning_inputs_building_blocks(decoder_outputs,
                                                                                                       additional["mean_attention"],
                                                                                                       additional["mean_content"],
                                                                                                       additional["mu_old"],
                                                                                                       additional["sigma_old"],
                                                                                                       additional["di"],
                                                                                                       additional["batch_size"],
                                                                                                       additional["output_size"],
                                                                                                       additional["input_lengths"])

        mu_weights, sigma_weights = self._compute_parametric_weights(positioning_inputs, additional)

        positioners = self._compute_positioners(mu_weights, sigma_weights, positioning_building_blocks, additional, rel_counter_encoder)

        return positioners

    def _get_querry(self, decoder_outputs, additional):
        # uses the same neurons as for the key_input so that initialized by it.
        batch_size = additional["batch_size"]
        output_size = additional["output_size"]

        querry_input = decoder_outputs[:, :, :self.querry_size] if self.is_contained_kv else decoder_outputs

        if self.counters is not None:
            counters = self.counters[di:di + output_size, :].view(1, -1, self.counter_size).expand(batch_size, -1, self.counter_size)

        if not self.is_postcounter and self.counters is not None:
            # precounter
            querry_input = torch.cat([querry_input, counters], dim=2)

        if self.is_kqrnn:
            querry, keys_hidden = self.querry_generator(querry_input, additional['keys_hidden'])
            additional['keys_hidden'] = keys_hidden
        elif self.is_querry:
            querry = self.querry_generator(querry_input)
        else:
            querry = querry_input

        if self.is_postcounter:
            querry = torch.cat([querry, counters], dim=2)

        return querry

    def _compute_context(self, decoder_outputs, encoder_outputs, additional):
        batch_size = additional["batch_size"]

        if self.is_positioning_generator:
            rel_counter_encoder = self._renormalize_input_length(self.rel_counter.expand(batch_size, -1, 1),
                                                                 additional["input_lengths"])
        else:
            rel_counter_encoder = None

        positioners = self._location_attention_generator(decoder_outputs, additional, rel_counter_encoder)

        location_percentage = self._compute_location_importance(decoder_outputs)

        querry = self._get_querry(decoder_outputs, additional)

        context, attn, additional_attention = self.attention(querry,
                                                             encoder_outputs,
                                                             positioners=positioners,
                                                             location_percentage=location_percentage)

        additional["inspect"] = additional_attention

        if self.is_positioning_generator:
            additional["mean_attention"] = torch.bmm(attn, rel_counter_encoder[:, :attn.size(2), :]).squeeze(2)
            content_attn = additional_attention["content_attention"]
            additional["mean_content"] = torch.bmm(content_attn, rel_counter_encoder[:, :content_attn.size(2), :]).squeeze(2)

        return context, attn

    def _initialize_additional(self, additional):
        batch_size = additional["batch_size"]

        if self.is_positioning_generator:
            for k in ["mu_old", "sigma_old", "mean_attention", "mean_content"]:
                if additional.get(k, None) is None:
                    v = torch.zeros(batch_size, 1)
                    if torch.cuda.is_available():
                        v = v.cuda()
                    additional[k] = v

        additional["di"] = additional.get("di", 0)

        return additional

    def _compute_decoder_input(self, input_var, context=None):
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        if context is not None:
            combined_input = torch.cat((context, embedded), dim=2)
            if self.is_full_focus:
                merged_input = F.relu(self.ffocus_merge(combined_input))
                combined_input = torch.mul(context, merged_input)
            decoder_input = combined_input
        else:
            decoder_input = embedded

        return decoder_input
