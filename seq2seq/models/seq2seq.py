"""
Seq2seq class.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq.util.helpers import get_extra_repr
from seq2seq.util.torchextend import AnnealedGaussianNoise, AnnealedDropout
from seq2seq.util.base import Module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Seq2seq(Module):
    """Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)
        mid_dropout_kwargs (dictionary, optonal): additional arguments to mid dropout.
        mid_noise_kwargs (dictionary, optonal): additional arguments to mid noise.
        is_dev_mode (bool, optional): whether to store many useful variables in
            `additional`. Useful when predicting with a trained model in dev mode
             to understand what the model is doing. Use with `dev_predict`.

    Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio
        - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.

    """

    def __init__(self, encoder, decoder,
                 decode_function=F.log_softmax,
                 mid_dropout_kwargs={},
                 mid_noise_kwargs={}):
        super(Seq2seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function

        self.mid_dropout = AnnealedDropout(**mid_dropout_kwargs)
        self.is_update_mid_dropout = self.training
        self.mid_noise = AnnealedGaussianNoise(**mid_noise_kwargs)
        self.is_update_mid_noise = self.training

    def forward(self, input_variable,
                input_lengths=None,
                target_variables=None,
                teacher_forcing_ratio=0,
                confusers=dict()):

        self._update_n_training_calls()

        # precomputes a float tensor of the source lengths as it will be used a lot
        # removes the need of having to change the variable from CPU to GPU
        # multiple times at each iter
        input_lengths_tensor = torch.FloatTensor(input_lengths).to(device)
        input_lengths = (input_lengths, input_lengths_tensor)

        # Unpack target variables
        try:
            target_output = target_variables.get('decoder_output', None)
            # The attention target is preprended with an extra SOS step. We must remove this
            provided_content_attn = (target_variables['attention_target'][:, 1:]
                                     if 'attention_target' in target_variables else None)
        except AttributeError:
            target_output = None
            provided_content_attn = None

        encoder_outputs, encoder_hidden, additional = self.encoder(input_variable,
                                                                   input_lengths,
                                                                   confusers=confusers)

        self.is_update_mid_dropout = self.training
        self.is_update_mid_noise = self.training

        if "last_enc_controller_out" in additional:
            additional["last_enc_controller_out"
                       ] = self._mid_noise(additional["last_enc_controller_out"])
            additional["last_enc_controller_out"
                       ] = self._mid_dropout(additional["last_enc_controller_out"])

        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple(self._mid_noise(el) for el in encoder_hidden)
            encoder_hidden = tuple(self._mid_dropout(el) for el in encoder_hidden)
        else:
            encoder_hidden = self._mid_noise(encoder_hidden)
            encoder_hidden = self._mid_dropout(encoder_hidden)

        (decoder_outputs,
         decoder_hidden,
         ret_dict) = self.decoder(inputs=target_output,
                                  encoder_hidden=encoder_hidden,
                                  encoder_outputs=encoder_outputs,
                                  function=self.decode_function,
                                  teacher_forcing_ratio=teacher_forcing_ratio,
                                  provided_content_attn=provided_content_attn,
                                  source_lengths=input_lengths,
                                  additional=additional,
                                  confusers=confusers)

        ret_dict["test"].update(self.get_to_test())
        ret_dict["visualize"].update(self.get_to_visualize())
        ret_dict["losses"].update(self.get_regularization_losses())

        return decoder_outputs, decoder_hidden, ret_dict

    def _mid_dropout(self, x):
        x = self.mid_dropout(x, is_update=self.is_update_mid_dropout)
        self.is_update_mid_dropout = False  # makes sure that updates only once every forward
        return x

    def _mid_noise(self, x):
        x = self.mid_noise(x, is_update=self.is_update_mid_noise)
        self.is_update_mid_noise = False  # makes sure that updates only once every forward
        return x

    def extra_repr(self):
        pass
