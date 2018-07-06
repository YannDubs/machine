import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import ContentAttention, HardGuidance
from .baseRNN import BaseRNN

from seq2seq.util.helpers import renormalize_input_length
from seq2seq.util.initialization import weights_init
from seq2seq.models.KVQ import QueryGenerator
from seq2seq.models.Positioner import AttentionMixer, PositionAttention


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)
        full_focus(bool, optional): flag indication whether to use full attention mechanism or not (default: false)

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
          (default is `torch.nn.functional.log_softmax`).
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
          predicted token IDs, *KEY_ATTN_SCORE* : list of floats, indicating attention weights.  }.
    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, max_len, hidden_size, embedding_size, sos_id, eos_id,
                 input_dropout_p=0,
                 rnn_cell='gru',
                 n_layers=1,
                 bidirectional=False,
                 dropout_p=0,
                 use_attention=False,
                 attention_method=None,
                 is_full_focus=False,
                 is_transform_controller=False,
                 value_size=None,
                 is_positioner=False,
                 is_query=False,
                 pag_kwargs={},
                 query_kwargs={},
                 attmix_kwargs={}):

        super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                                         input_dropout_p, dropout_p,
                                         n_layers, rnn_cell)

        if use_attention != False and attention_method == None:
            raise ValueError("Method for computing attention should be provided")

        self.embedding_size = embedding_size
        self.output_size = vocab_size
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.bidirectional_encoder = bidirectional
        self.attention_method = attention_method
        self.is_full_focus = is_full_focus

        self.is_transform_controller = is_transform_controller
        self.value_size = value_size
        self.is_positioner = is_positioner

        # increase input size decoder if attention is applied before decoder rnn
        input_rnn_size = self.embedding_size
        input_prediction_size = self.hidden_size
        if self.use_attention == 'pre-rnn':
            if is_full_focus:
                input_rnn_size = self.value_size
            else:
                input_rnn_size += self.value_size
        elif self.use_attention == 'post-rnn':
            if is_full_focus:
                input_prediction_size = self.value_size
            else:
                input_prediction_size += self.value_size

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.controller = self.rnn_cell(input_rnn_size, self.hidden_size, self.n_layers, batch_first=True, dropout=self.dropout_p)

        if self.use_attention == "pre-rnn" and self.is_transform_controller:
            self.transform_controller = nn.Linear(self.hidden_size, self.hidden_size)

        post_counter_size = None
        if is_query:
            self.query_generator = QueryGenerator(self.hidden_size, self.max_len, **query_kwargs)
            self.query_size = self.query_generator.output_size

            # # # keeping for testing # # #
            if self.query_generator.is_postcounter:
                post_counter_size = self.query_generator.counter_size
            # # # # # # # # # # # # # # # #
        else:
            self.query_generator = None
            self.query_size = self.hidden_size

        if self.use_attention:
            self.content_attention = ContentAttention(self.query_size, self.attention_method, post_counter_size=post_counter_size)
        else:
            self.content_attention = None

        if self.is_positioner:
            self.position_attention = PositionAttention(self.hidden_size, self.max_len, **pag_kwargs)
            self.mix_attention = AttentionMixer(self.hidden_size, **attmix_kwargs)

        if self.is_full_focus:
            if self.use_attention == 'pre-rnn':
                self.ffocus_merge = nn.Linear(self.embedding_size + self.value_size, input_rnn_size)
            elif self.use_attention == 'post-rnn':
                self.ffocus_merge = nn.Linear(self.hidden_size + self.value_size, input_prediction_size)

        self.out = nn.Linear(input_prediction_size, self.output_size)

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init)

    def flatten_parameters(self):
        self.controller.flatten_parameters()

        if self.is_positioner:
            self.position_attention.flatten_parameters()

        if self.query_generator is not None:
            self.query_generator.flatten_parameters()

    def forward(self,
                inputs=None,
                encoder_hidden=None,
                encoder_outputs=None,
                function=F.log_softmax,
                teacher_forcing_ratio=0,
                provided_attention=None,
                source_lengths=None,
                additional=None):

        def decode(step, step_output, step_attn, additional=None):
            keys_to_store = ["content_attention", "position_attention", "position_percentage", "mu", 'sigma']

            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            if additional is not None:
                for k in keys_to_store:
                    if k in additional:
                        ret_dict[k] = ret_dict.get(k, list())
                        ret_dict[k].append(additional[k])
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        additional = self._initialize_additional(additional)

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function, teacher_forcing_ratio)

        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)
        controller_output = additional["last_enc_controller_out"] if self.use_attention == 'pre-rnn' else None
        if self.use_attention == 'pre-rnn' and self.is_transform_controller:
            controller_output = self.transform_controller(controller_output)

        # Prepare extra arguments for attention method
        attention_method_kwargs = {}
        if self.content_attention and isinstance(self.content_attention.method, HardGuidance):
            attention_method_kwargs['provided_attention'] = provided_attention

        # When we use pre-rnn attention we must unroll the decoder. We need to calculate the attention based on
        # the previous hidden state, before we can calculate the next hidden state.
        # We also need to unroll when we don't use teacher forcing. We need perform the decoder steps
        # one-by-one since the output needs to be copied to the input of the next step.
        if self.use_attention == 'pre-rnn' or not use_teacher_forcing or self.is_positioner:
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
                # If we don't use teacher forcing (and we are beyond the first SOS step), we use the last output as new input
                else:
                    decoder_input = symbols

                # Perform one forward step
                if self.content_attention and isinstance(self.content_attention.method, HardGuidance):
                    attention_method_kwargs['step'] = di
                decoder_output, decoder_hidden, step_attn, controller_output, additional = self.forward_step(decoder_input,
                                                                                                             decoder_hidden,
                                                                                                             encoder_outputs,
                                                                                                             function,
                                                                                                             controller_output,
                                                                                                             di,
                                                                                                             additional=additional,
                                                                                                             source_lengths=source_lengths,
                                                                                                             attention_method_kwargs=attention_method_kwargs)

                # Remove the unnecessary dimension.
                step_output = decoder_output.squeeze(1)
                # Get the actual symbol
                symbols = decode(di, step_output, step_attn, additional=additional)

        else:
            # Remove last token of the longest output target in the batch. We don't have to run the last decoder step where the teacher forcing input is EOS (or the last output)
            # It still is run for shorter output targets in the batch
            decoder_input = inputs[:, :-1]

            # Forward step without unrolling
            if self.content_attention and isinstance(self.content_attention.method, HardGuidance):
                attention_method_kwargs['step'] = -1
            decoder_output, decoder_hidden, attn, controller_output, additional = self.forward_step(decoder_input,
                                                                                                    decoder_hidden,
                                                                                                    encoder_outputs,
                                                                                                    function,
                                                                                                    controller_output,
                                                                                                    0,
                                                                                                    additional=additional,
                                                                                                    source_lengths=source_lengths,
                                                                                                    attention_method_kwargs=attention_method_kwargs)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn, additional=additional)

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def forward_step(self, input_var, hidden, encoder_outputs, function, controller_output, step,
                     source_lengths=None,
                     additional=None,
                     attention_method_kwargs={}):
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
        batch_size, output_len = input_var.size()

        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        attn = None
        if self.use_attention == 'pre-rnn':
            context, attn = self._compute_context(controller_output,
                                                  encoder_outputs,
                                                  source_lengths,
                                                  step,
                                                  additional,
                                                  attention_method_kwargs=attention_method_kwargs)
            controller_input = self._combine_context(embedded, context)
        else:
            controller_input = embedded

        controller_output, hidden = self.controller(controller_input, hidden)

        if self.use_attention == 'post-rnn':
            context, attn = self._compute_context(controller_output,
                                                  encoder_outputs,
                                                  source_lengths,
                                                  step,
                                                  additional,
                                                  attention_method_kwargs=attention_method_kwargs)
            prediction_input = self._combine_context(controller_output, context)
        else:
            prediction_input = controller_output

        prediction_input = prediction_input.contiguous().view(-1, self.out.in_features)

        predicted_softmax = function(self.out(prediction_input), dim=1).view(batch_size, output_len, -1)

        return predicted_softmax, hidden, attn, controller_output, additional

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
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

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                hidden = encoder_hidden

                if self.rnn_cell is nn.LSTM:
                    batch_size = hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_len
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length

    def _compute_context(self, controller_output, encoder_outputs, source_lengths, step, additional, attention_method_kwargs={}):
        keys, values = encoder_outputs

        batch_size = keys.size(0)

        if self.query_generator is not None:
            additional["step"] = step
            query, additional = self.query_generator(controller_output, source_lengths, additional)
        else:
            query = controller_output

        content_attn, content_confidence = self.content_attention(query, keys, **attention_method_kwargs)

        if self.is_positioner:
            # controller should know his mu and sigma because depend on building blocks that he doesn't have access to
            # so unlike content he doesn't know anything !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            pos_attn, pos_confidence, mu, sigma = self.position_attention(controller_output,
                                                                          encoder_outputs,
                                                                          source_lengths,
                                                                          step,
                                                                          additional["mu"],
                                                                          additional["sigma"],
                                                                          additional["mean_content"],
                                                                          additional["mean_attn"],
                                                                          additional)

            attn, pos_perc = self.mix_attention(controller_output,
                                                step,
                                                content_attn,
                                                content_confidence,
                                                pos_attn,
                                                pos_confidence,
                                                additional["position_percentage"])  # the controller should also know that !!!!!!!!!!!!!

            rel_counter_encoder = renormalize_input_length(self.position_attention.rel_counter.expand(batch_size, -1, 1),
                                                           source_lengths,
                                                           self.max_len)

            additional["mu"] = mu
            additional["sigma"] = sigma
            additional["mean_content"] = torch.bmm(content_attn, rel_counter_encoder[:, :content_attn.size(2), :]).squeeze(2)
            additional["mean_attn"] = torch.bmm(attn, rel_counter_encoder[:, :attn.size(2), :]).squeeze(2)
            additional["content_attention"] = content_attn
            additional["position_attention"] = pos_attn
            additional["position_percentage"] = pos_perc

        else:
            attn = content_attn

        context = torch.bmm(attn, values)

        return context, attn

    def _initialize_additional(self, additional):
        if additional is None:
            additional = dict()

        if self.is_positioner:
            if self.position_attention.is_recursive:
                additional['positioner_hidden'] = None

            for k in ["mu", "sigma", "mean_attn", "mean_content", "position_percentage"]:
                additional[k] = None

        return additional

    def _combine_context(self, input_var, context):
        combined_input = torch.cat((context, input_var), dim=2)

        if self.is_full_focus:
            merged_input = F.relu(self.ffocus_merge(combined_input))
            combined_input = torch.mul(context, merged_input)

        return combined_input
