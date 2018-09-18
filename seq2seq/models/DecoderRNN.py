""" Decoder class for a seq2seq. """
import random
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .attention import ContentAttention, HardGuidance
from .baseRNN import BaseRNN

from seq2seq.util.helpers import (renormalize_input_length, get_rnn,
                                  get_extra_repr, format_source_lengths,
                                  recursive_update)
from seq2seq.util.torchextend import AnnealedGaussianNoise
from seq2seq.util.initialization import weights_init, init_param
from seq2seq.models.KVQ import QueryGenerator
from seq2seq.models.Positioner import AttentionMixer, PositionAttention
from seq2seq.util.confuser import get_max_loss_loc_confuser


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IS_X05 = True


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
        is_weight_norm_rnn (bool, optional): whether to use weight normalization
            for the RNN. Weight normalization is similar to batch norm or layer
            normalization and has been shown to help when learning large models.
        bidirectional (bool, optional): if the encoder is bidirectional
            (default False)
        input_dropout_p (float, optional): dropout probability for the input
            sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence
            (default: 0)
        use_attention(bool, optional): flag indication whether to use attention
            mechanism or not (default: false)
        full_focus(bool, optional): flag indication whether to use full attention
            mechanism or not (default: false)
        is_transform_controller (bool, optional): whether to pass the hidden
            activation of the encoder through a linear layer before using it as
            initialization of the decoder. This is useful when using `pre-rnn`
            attention, where the first input to the content and positional attention
            generator is the last hidden activation.
        is_add_all_controller (bool, optional): whether to add all computed features
            to the decoder in order to have a central model that "knows everything".
        values_size (int, optional): size of the generated value. -1 means same
            as hidden size. Can also give percentage of hidden size betwen 0 and 1.
        is_content_attn (bool, optional): whether to use content attention.
        is_position_attn (bool, optional): whether to use positional attention.
        is_query (bool, optional): whether to use a query generator.
        content_kwargs (dict, optional): additional arguments to the content
            attention generator.
        position_kwargs (dict, optional): additional arguments to the positional
            attention generator.
        query_kwargs (dict, optional): additional arguments to the query generator.
        attmix_kwargs (dict, optional): additional arguments to the attention mixer.
        embedding_noise_kwargs (dict, optional): additional arguments to embedding
            noise.
        is_dev_mode (bool, optional): whether to store many useful variables in
            `additional`. Useful when predicting with a trained model in dev mode
             to understand what the model is doing. Use with `dev_predict`.
        is_viz_train (bool, optional): whether to save how the averages of some
            intepretable variables change during training in "visualization"
            of `additional`.

    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of
            output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`

    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length
            is the batch size and within which each sequence is a list of token IDs.
            It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size):
            tensor containing the features in the hidden state `h` of encoder. Used
            as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing
            the outputs of the encoder. Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols
            from RNN hidden state (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing
            will be used. A random number is drawn uniformly from 0-1 for every
            decoding token, and if the sample is smaller than the given value,
            teacher forcing would be used (default is 0).

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with
            size (batch_size, vocab_size) containing the outputs of the decoding
            function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size):
            tensor containing the last hidden state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows
            {*KEY_LENGTH* : list of integers representing lengths of output sequences,
            *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
            predicted token IDs, *KEY_ATTN_SCORE* : list of floats, indicating
            attention weights.  }.
    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, max_len, hidden_size, embedding_size, sos_id, eos_id,
                 input_dropout_p=0,
                 rnn_cell='gru',
                 is_weight_norm_rnn=False,
                 n_layers=1,
                 bidirectional=False,
                 dropout_p=0,
                 use_attention=None,
                 is_full_focus=False,
                 is_transform_controller=False,
                 is_add_all_controller=True,
                 value_size=None,
                 is_content_attn=True,
                 is_position_attn=True,
                 is_query=True,
                 content_kwargs={},
                 position_kwargs={},
                 query_kwargs={},
                 attmix_kwargs={},
                 embedding_noise_kwargs={},
                 is_dev_mode=False,
                 is_viz_train=False,
                 is_mid_focus=True,
                 is_old_content=False):  # TO DOC : Tru should be used when using pondering

        super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                                         input_dropout_p, dropout_p,
                                         n_layers, rnn_cell)
        self.is_dev_mode = is_dev_mode
        self.is_viz_train = is_viz_train

        self.is_weight_norm_rnn = is_weight_norm_rnn
        self.embedding_size = embedding_size
        self.output_size = vocab_size
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.bidirectional_encoder = bidirectional
        self.is_full_focus = is_full_focus
        self.is_mid_focus = is_mid_focus
        self.is_old_content = is_old_content

        self.is_add_all_controller = is_add_all_controller
        self.is_transform_controller = is_transform_controller
        self.value_size = value_size
        self.is_query = is_query
        self.is_position_attn = is_position_attn
        self.is_content_attn = is_content_attn

        if not self.is_content_attn and not self.is_position_attn:
            warnings.warn("Setting `use_attention` to `None` as content_attn and position_attn not used.")
            self.use_attention = None

        # increase input size decoder if attention is applied before decoder rnn
        input_rnn_size = self.embedding_size
        input_prediction_size = self.hidden_size
        if self.use_attention == 'pre-rnn':
            if self.is_full_focus or self.is_mid_focus:
                input_rnn_size = self.value_size
            else:
                input_rnn_size += self.value_size
        elif self.use_attention == 'post-rnn':
            if self.is_full_focus or self.is_mid_focus:
                input_prediction_size = self.value_size
            else:
                input_prediction_size += self.value_size

        n_additional_controller_features = 0

        if self.is_add_all_controller:
            n_additional_controller_features += 3  # abs_counter_decoder / rel_counter_decoder / source_len
            if self.use_attention is not None:
                self.mean_attn0 = Parameter(torch.tensor(0.0))
                n_additional_controller_features += 1  # mean_attn_old
            if self.is_content_attn:
                self.mean_content0 = Parameter(torch.tensor(0.0))
                self.content_confidence0 = Parameter(torch.tensor(0.5))
                n_additional_controller_features += 2  # mean_content_old / content_confidence_old
            if self.is_position_attn:
                self.pos_confidence0 = Parameter(torch.tensor(0.5))
                n_additional_controller_features += 4  # mu_old / sigma_old / mean_attn_olds / pos_confidence_old
            if self.is_content_attn and self.is_position_attn:
                n_additional_controller_features += 1  # position_perc_old

        self.rel_counter = torch.arange(0, self.max_len,
                                        dtype=torch.float,
                                        device=device).unsqueeze(1) / (self.max_len - 1)
        # DEV for confuser
        self.dec_counter = torch.arange(1, self.max_len + 1,
                                        dtype=torch.float,
                                        device=device)

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.noise_input = AnnealedGaussianNoise(**embedding_noise_kwargs)
        self.controller = get_rnn(self.rnn_cell,
                                  input_rnn_size + n_additional_controller_features,
                                  self.hidden_size,
                                  num_layers=self.n_layers,
                                  batch_first=True,
                                  dropout=self.dropout_p,
                                  is_weight_norm=self.is_weight_norm_rnn,
                                  is_get_hidden0=False)

        self.query_generator = None
        if self.use_attention is not None:

            if self.use_attention == "pre-rnn" and self.is_transform_controller:
                self.transform_controller = nn.Linear(self.hidden_size, self.hidden_size)

            post_counter_size = None
            if self.is_query:
                self.query_generator = QueryGenerator(self.hidden_size,
                                                      self.max_len,
                                                      **query_kwargs)
                self.query_size = self.query_generator.output_size

                # # # keeping for testing # # #
                if self.query_generator.is_postcounter:
                    post_counter_size = self.query_generator.counter_size
                # # # # # # # # # # # # # # # #
            else:
                self.query_size = self.hidden_size

            if self.is_content_attn:
                self.content_attention = ContentAttention(self.query_size,
                                                          post_counter_size=post_counter_size,
                                                          **content_kwargs)

            if self.is_position_attn:
                self.position_attention = PositionAttention(self.hidden_size,
                                                            self.max_len,
                                                            is_content_attn=self.is_content_attn,
                                                            **position_kwargs)

            if self.is_content_attn and self.is_position_attn:
                self.mix_attention = AttentionMixer(self.hidden_size, **attmix_kwargs)

            if self.is_full_focus or self.is_mid_focus:
                if self.use_attention == 'pre-rnn':
                    self.ffocus_merge = nn.Linear(self.embedding_size + self.value_size,
                                                  input_rnn_size)
                elif self.use_attention == 'post-rnn':
                    self.ffocus_merge = nn.Linear(self.hidden_size + self.value_size,
                                                  input_prediction_size)

        self.out = nn.Linear(input_prediction_size, self.output_size)

        self.reset_parameters()

    def set_dev_mode(self, value=True):
        self.is_dev_mode = value
        if self.query_generator is not None:
            self.query_generator.set_dev_mode(value=value)
        if self.is_position_attn:
            self.position_attention.set_dev_mode(value=value)
        if self.is_content_attn:
            self.content_attention.set_dev_mode(value=value)
        if self.is_content_attn and self.is_position_attn:
            self.mix_attention.set_dev_mode(value=value)

    def reset_parameters(self):
        self.apply(weights_init)

        if self.query_generator is not None:
            self.query_generator.reset_parameters()
        if self.is_position_attn:
            self.position_attention.reset_parameters()
        if self.is_content_attn:
            self.content_attention.reset_parameters()
        if self.is_content_attn and self.is_position_attn:
            self.mix_attention.reset_parameters()

        if self.is_add_all_controller:
            if self.use_attention is not None:
                if IS_X05:
                    self.mean_attn0 = Parameter(torch.tensor(0.5))
                else:
                    init_param(self.mean_attn0, is_positive=True)
            if self.is_content_attn:
                if IS_X05:
                    self.mean_content0 = Parameter(torch.tensor(0.5))
                else:
                    init_param(self.mean_content0, is_positive=True)

                self.content_confidence0 = Parameter(torch.tensor(0.5))
            if self.is_position_attn:
                self.pos_confidence0 = Parameter(torch.tensor(0.5))

    def flatten_parameters(self):
        self.controller.flatten_parameters()

        if self.is_position_attn:
            self.position_attention.flatten_parameters()

        if self.query_generator is not None:
            self.query_generator.flatten_parameters()

    def extra_repr(self):
        return get_extra_repr(self,
                              always_shows=["use_attention"],
                              conditional_shows=["is_content_attn",
                                                 "is_position_attn",
                                                 "is_query",
                                                 "is_add_all_controller",
                                                 "is_weight_norm_rnn",
                                                 "is_full_focus",
                                                 "is_transform_controller"])

    def forward(self,
                inputs=None,
                encoder_hidden=None,
                encoder_outputs=None,
                function=F.log_softmax,
                teacher_forcing_ratio=0,
                provided_content_attn=None,
                source_lengths=None,
                additional=None,
                confusers=dict(),
                additional_to_store=["test", "visualize", "losses"]):

        def store_additional(additional, ret_dict, additional_to_store,
                             is_multiple_call=False):
            def append_to_list(dictionary, k, v):
                dictionary[k] = dictionary.get(k, list())
                dictionary[k].append(v)

            if additional is None:
                return

            filtered_additional = {k: v for k, v in additional.items()
                                   if k in additional_to_store}

            if not is_multiple_call:
                recursive_update(ret_dict, filtered_additional)
                # removes so that doesn't add them multiple time uselessly
                for k in filtered_additional.keys():
                    additional.pop(k, None)
            else:
                for k, v in filtered_additional.items():
                    if v is not None:
                        if isinstance(v, dict):
                            ret_dict[k] = ret_dict.get(k, dict())
                            for sub_k, sub_v in v.items():
                                append_to_list(ret_dict[k], sub_k, sub_v)
                        else:
                            append_to_list(ret_dict, k, v)

        def decode(step, step_output, step_attn, additional=None):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)

            store_additional(additional, ret_dict, additional_to_store,
                             is_multiple_call=True)

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

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden,
                                                             encoder_outputs, function,
                                                             teacher_forcing_ratio)

        store_additional(additional, ret_dict, additional_to_store)
        additional = self._initialize_additional(additional)

        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)
        controller_output = (additional["last_enc_controller_out"]
                             if self.use_attention == 'pre-rnn' else None)
        if self.use_attention == 'pre-rnn' and self.is_transform_controller:
            controller_output = self.transform_controller(controller_output)

        # Prepare extra arguments for attention method
        content_method_kwargs = {}
        if self.is_content_attn and isinstance(self.content_attention.method, HardGuidance):
            content_method_kwargs['provided_content_attn'] = provided_content_attn

        # When we use pre-rnn attention we must unroll the decoder. We need to
        # calculate the attention based on the previous hidden state, before we
        # can calculate the next hidden state. We also need to unroll when we don't
        # use teacher forcing. We need perform the decoder steps one-by-one since
        # the output needs to be copied to the input of the next step.
        if (self.use_attention == 'pre-rnn' or
            not use_teacher_forcing or
            self.is_position_attn or
                self.is_add_all_controller):
            unrolling = True
        else:
            unrolling = False

        if unrolling:
            symbols = None
            for di in range(max_length):
                # We always start with the SOS symbol as input. We need to add
                # extra dimension of length 1 for the number of decoder steps
                # (1 in this case) When we use teacher forcing, we always use the
                # target input.
                if di == 0 or use_teacher_forcing:
                    decoder_input = inputs[:, di].unsqueeze(1)
                # If we don't use teacher forcing (and we are beyond the first
                # SOS step), we use the last output as new input
                else:
                    decoder_input = symbols

                # Perform one forward step
                if self.is_content_attn and isinstance(self.content_attention.method,
                                                       HardGuidance):
                    content_method_kwargs['step'] = di

                (decoder_output,
                 decoder_hidden,
                 step_attn,
                 controller_output,
                 additional) = self.forward_step(decoder_input,
                                                 decoder_hidden,
                                                 encoder_outputs,
                                                 function,
                                                 controller_output,
                                                 di,
                                                 additional=additional,
                                                 source_lengths=source_lengths,
                                                 content_method_kwargs=content_method_kwargs,
                                                 confusers=confusers)

                # Remove the unnecessary dimension.
                step_output = decoder_output.squeeze(1)
                # Get the actual symbol
                symbols = decode(di, step_output, step_attn, additional=additional)

        else:
            # Remove last token of the longest output target in the batch.
            # We don't have to run the last decoder step where the teacher forcing
            # input is EOS (or the last output) It still is run for shorter output
            # targets in the batch
            decoder_input = inputs[:, :-1]

            # Forward step without unrolling
            if self.is_content_attn and isinstance(self.content_attention.method, HardGuidance):
                content_method_kwargs['step'] = -1

            (decoder_output,
             decoder_hidden,
             attn,
             controller_output,
             additional) = self.forward_step(decoder_input,
                                             decoder_hidden,
                                             encoder_outputs,
                                             function,
                                             controller_output,
                                             0,
                                             additional=additional,
                                             source_lengths=source_lengths,
                                             content_method_kwargs=content_method_kwargs,
                                             confusers=confusers)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn, additional=additional)

        if "query_confuser" in confusers:
            # SLOW FOR CUDA : 1 allocation per batch!!
            output_lengths_tensor = torch.from_numpy(lengths).float().to(device)

            queries = torch.cat(additional["queries"], dim=1)
            max_decode_len = queries.size(1)

            counting_target_j = self.dec_counter.expand(batch_size, -1)[:, :max_decode_len]

            # masks everything which finished decoding
            mask = counting_target_j > output_lengths_tensor.unsqueeze(1)

            max_losses = get_max_loss_loc_confuser(output_lengths_tensor,
                                                   p=0.5,
                                                   factor=confusers["query_confuser"
                                                                    ].get_factor(self.training))

            to_cat = output_lengths_tensor.view(-1, 1, 1).expand(-1, max_decode_len, 1)
            query_confuse_input = torch.cat([queries, to_cat], dim=-1)
            confusers["query_confuser"].compute_loss(query_confuse_input,
                                                     targets=counting_target_j.unsqueeze(-1),
                                                     seq_len=output_lengths_tensor.unsqueeze(-1),
                                                     max_losses=max_losses.unsqueeze(-1),
                                                     mask=mask)

        if self.is_dev_mode:
            queries = torch.cat(additional["queries"], dim=1)
            # DEV MODE TO UNDERSTAND CONFUSERS
            # don't store in additional becuase already all in ret_dict
            ret_dict["test"]["queries"] = queries.detach().cpu()

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def forward_step(self, input_var, hidden, encoder_outputs, function, controller_output, step,
                     source_lengths=None,
                     additional=None,
                     content_method_kwargs={},
                     confusers=dict()):
        """
        Performs one or multiple forward decoder steps.

        Args:
            input_var (torch.tensor): Variable containing the input(s) to the decoder RNN
            hidden (torch.tensor): Variable containing the previous decoder hidden state.
            encoder_outputs (torch.tensor): Variable containing the target outputs
                of the decoder RNN
            function (torch.tensor): Activation function over the last output of
                the decoder RNN at every time step.

        Returns:
            predicted_softmax: The output softmax distribution at every time step
                of the decoder RNN
            hidden: The hidden state at every time step of the decoder RNN
            attn: The attention distribution at every time step of the decoder RNN
        """
        source_lengths_list, source_lengths_tensor = format_source_lengths(source_lengths)

        batch_size, output_len = input_var.size()

        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        embedded = self.noise_input(embedded, is_update=(step == 0))

        attn = None
        if self.use_attention == 'pre-rnn':
            context, attn = self._compute_context(controller_output,
                                                  encoder_outputs,
                                                  source_lengths,
                                                  step,
                                                  additional,
                                                  content_method_kwargs=content_method_kwargs,
                                                  confusers=confusers)
            controller_input = self._combine_context(embedded, context)
        else:
            controller_input = embedded

        if self.is_add_all_controller:
            additional_controller_features = self._get_additional_controller_features(additional,
                                                                                      step,
                                                                                      source_lengths_tensor)

            controller_input = torch.cat([controller_input] +
                                         additional_controller_features,
                                         dim=2)

        controller_output, hidden = self.controller(controller_input, hidden)

        if "eos_confuser" in confusers:
            confusers["eos_confuser"].compute_loss(controller_output)

        if self.use_attention == 'post-rnn':
            context, attn = self._compute_context(controller_output,
                                                  encoder_outputs,
                                                  source_lengths,
                                                  step,
                                                  additional,
                                                  content_method_kwargs=content_method_kwargs,
                                                  confusers=confusers)

            prediction_input = self._combine_context(controller_output, context)
        else:
            prediction_input = controller_output

        prediction_input = prediction_input.contiguous().view(-1, self.out.in_features)

        predicted_softmax = function(self.out(prediction_input),
                                     dim=1).view(batch_size, output_len, -1)

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
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch,
            # directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs,
                       function, teacher_forcing_ratio):
        if self.use_attention is not None:
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

                if self.rnn_cell == "lstm":
                    batch_size = hidden[0].size(1)
                elif self.rnn_cell == "gru":
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

    def _compute_context(self, controller_output, encoder_outputs, source_lengths,
                         step, additional, content_method_kwargs={}, confusers=dict()):
        source_lengths_list, source_lengths_tensor = format_source_lengths(source_lengths)

        keys, values = encoder_outputs

        batch_size = keys.size(0)

        if self.query_generator is not None:
            additional["step"] = step
            query, additional = self.query_generator(controller_output,
                                                     source_lengths_tensor,
                                                     additional)
        else:
            query = controller_output

        if "query_confuser" in confusers or self.is_dev_mode:
            additional["queries"] = additional.get("queries", []) + [query]

        unormalized_counter = self.rel_counter.expand(batch_size, -1, 1)
        rel_counter_encoder = renormalize_input_length(unormalized_counter,
                                                       source_lengths_tensor - 1,
                                                       self.max_len - 1)

        if self.is_content_attn:
            if step > 0:
                mean_content_old = additional["mean_content"]

            content_attn, content_confidence = self.content_attention(query,
                                                                      keys,
                                                                      additional,
                                                                      **content_method_kwargs)
            attn = content_attn

            additional["content_confidence"] = content_confidence
            additional["mean_content"] = torch.bmm(content_attn,
                                                   rel_counter_encoder[:,
                                                                       :content_attn.size(2), :]
                                                   ).squeeze(2)

            self._add_to_visualize([content_confidence, additional["mean_content"]],
                                   ["content_confidence", "mean_content"],
                                   additional)
            self._add_to_test(content_attn, "content_attention", additional)

        if self.is_position_attn:
            if step == 0:
                mean_attn_old = self.mean_attn0.expand(batch_size, 1)
                if self.is_content_attn:
                    mean_content_old = self.mean_content0.expand(batch_size, 1)
                else:
                    mean_content_old = None
            else:
                mean_attn_old = additional["mean_attn"]

                if not self.is_old_content:
                    mean_content_old = additional["mean_content"]

            pos_attn, pos_confidence, mu, sigma = self.position_attention(controller_output,
                                                                          source_lengths,
                                                                          step,
                                                                          additional["mu"],
                                                                          additional["sigma"],
                                                                          mean_content_old,
                                                                          mean_attn_old,
                                                                          additional["mean_attn_olds"],
                                                                          additional)

            additional["mu"] = mu
            additional["sigma"] = sigma
            additional["pos_confidence"] = pos_confidence

            attn = pos_attn

            self._add_to_visualize([mu, sigma, pos_confidence],
                                   ["mu", "sigma", "pos_confidence"],
                                   additional)

            self._add_to_test([pos_attn, mu, sigma],
                              ["position_attention", "mu", "sigma"],
                              additional)

        if self.is_content_attn and self.is_position_attn:
            attn, pos_perc = self.mix_attention(controller_output,
                                                step,
                                                content_attn,
                                                content_confidence,
                                                pos_attn,
                                                pos_confidence,
                                                additional["position_percentage"],
                                                additional)

            additional["position_percentage"] = pos_perc
            self._add_to_visualize(pos_perc, "position_percentage", additional)

            self._add_to_test([pos_confidence], ["pos_confidence"], additional)
            if self.mix_attention.mode != "pos_conf":
                self._add_to_test(content_confidence, "content_confidence", additional)

        additional["mean_attn"] = torch.bmm(attn,
                                            rel_counter_encoder[:, :attn.size(2), :]
                                            ).squeeze(2)

        context = torch.bmm(attn, values)

        self._add_to_visualize([additional["mean_attn"], step],
                               ["mean_attn", "step"],
                               additional)

        return context, attn

    def _initialize_additional(self, additional):
        if additional is None:
            additional = dict()

        if self.is_dev_mode:
            additional["test"] = additional.get("test", dict())

        if self.is_viz_train:
            additional["visualize"] = additional.get("visualize", dict())

        if self.training:
            additional["losses"] = additional.get("losses", dict())

        if self.is_position_attn:
            if self.position_attention.is_recursive:
                additional['positioner_hidden'] = None

            for k in ["mu", "sigma", "mean_attn", "mean_content", "position_percentage", "mean_attn_olds"]:
                additional[k] = None

        return additional

    def _combine_context(self, input_var, context):
        if self.is_mid_focus:
            combined_input = torch.cat((F.relu(context), input_var), dim=2)
            combined_input = self.ffocus_merge(combined_input)

        else:
            combined_input = torch.cat((context, input_var), dim=2)

            if self.is_full_focus:
                merged_input = F.relu(self.ffocus_merge(combined_input))
                combined_input = torch.mul(context, merged_input)

        return combined_input

    def _get_additional_controller_features(self, additional, step, source_lengths_tensor):
        batch_size = len(source_lengths_tensor)
        additional_features = []

        unormalized_counter = self.rel_counter[step:step + 1].expand(batch_size, 1)
        rel_counter_decoder = renormalize_input_length(unormalized_counter,
                                                       source_lengths_tensor - 1,
                                                       self.max_len - 1
                                                       ).unsqueeze(1)

        abs_counter_decoder = self.rel_counter[step:step + 1].expand(batch_size, 1
                                                                     ).unsqueeze(1)
        abs_counter_decoder = abs_counter_decoder * (self.max_len - 1)

        source_len = source_lengths_tensor.unsqueeze(-1).unsqueeze(-1)

        additional_features.extend([source_len, rel_counter_decoder, abs_counter_decoder])

        if self.use_attention is not None:
            if step != 0:
                mean_attn_old = additional["mean_attn"].unsqueeze(1)
            else:
                mean_attn_old = self.mean_attn0.expand(batch_size, 1).unsqueeze(1)

            additional_features.append(mean_attn_old)

            if self.is_content_attn:
                if step != 0:
                    mean_content_old = additional["mean_content"].unsqueeze(1)
                    content_confidence_old = additional["content_confidence"].unsqueeze(1)
                else:
                    mean_content_old = self.mean_content0.expand(batch_size, 1
                                                                 ).unsqueeze(1)
                    content_confidence_old = self.content_confidence0.expand(batch_size, 1
                                                                             ).unsqueeze(1)

                additional_features.extend([mean_content_old, content_confidence_old])

            if self.is_position_attn:
                if step != 0:
                    mu_old = additional["mu"]
                    sigma_old = additional["sigma"]
                    mean_attn_olds = additional["mean_attn_olds"]
                    pos_confidence_old = additional["pos_confidence"].unsqueeze(1)
                else:
                    mu_old = self.position_attention.mu0.expand(batch_size, 1
                                                                ).unsqueeze(1)
                    sigma_old = self.position_attention.sigma0.expand(batch_size, 1
                                                                      ).unsqueeze(1)
                    mean_attn_olds = mean_attn_old
                    pos_confidence_old = self.pos_confidence0.expand(batch_size, 1
                                                                     ).unsqueeze(1)

                additional_features.extend([mu_old, sigma_old, mean_attn_olds, pos_confidence_old])

            if self.is_content_attn and self.is_position_attn:
                if step != 0:
                    position_perc_old = additional["position_percentage"].unsqueeze(1)
                else:
                    position_perc_old = self.mix_attention.position_perc0.expand(batch_size, 1
                                                                                 ).unsqueeze(1)

                additional_features.extend([position_perc_old])

        return additional_features

    def _add_to_visualize(self, values, keys, additional, save_every_n_batches=15):
        """Every `save_every` batch, adds a certain variable to the `visualization`
        sub-dictionary of additional. Such variables should be the ones that are
        interpretable, and for which the size is independant of the source length.
        I.e avaregae over the source length if it is dependant.

        The variables will then be averaged over decoding step and over batch_size.
        """
        if "visualize" in additional and additional["training_step"] % save_every_n_batches == 0:
            if isinstance(keys, list):
                for k, v in zip(keys, values):
                    self._add_to_visualize(v, k, additional)
            else:
                # averages over the batch size
                if isinstance(values, torch.Tensor):
                    values = values.mean(0).detach().cpu()
                additional["visualize"][keys] = values

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
