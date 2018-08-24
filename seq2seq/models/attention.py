""" Content attention modules. """
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from seq2seq.util.initialization import weights_init, linear_init
from seq2seq.util.helpers import MLP, ProbabilityConverter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContentAttention(nn.Module):
    """
    Applies a content attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output
        method(str): The method to compute the alignment, mlp or dot

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.
        method (torch.nn.Module): layer that implements the method of computing the attention vector

    Examples::

         >>> attention = seq2seq.models.Attention(256)
         >>> context = torch.randn(5, 3, 256)
         >>> output = torch.randn(5, 5, 256)
         >>> output, attn = attention(output, context)

    """

    def __init__(self, dim, method="dot", post_counter_size=None, is_dev_mode=False):

        super(ContentAttention, self).__init__()
        # # # keeping for testing # # #
        self.is_dev_mode = is_dev_mode
        self.is_postcounter = post_counter_size is not None
        if self.is_postcounter:
            self.counter_size = post_counter_size
            self.context_count_mixer = MLP(2, 2, 1)
            self.count_localizer = MLP(self.counter_size * 2, self.counter_size * 2, 1)
            dim -= self.counter_size
        # # # # # # # # # # # # # # # #

        self.mask = None
        self.method = self.get_method(method, dim)

        self.maxlogit_to_conf = ProbabilityConverter(is_temperature=True,
                                                     is_bias=True)

        self.reset_parameters()

    def set_dev_mode(self, value=True):
        self.is_dev_mode = value

    def reset_parameters(self):
        self.apply(weights_init)

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, queries, keys, additional, **attention_method_kwargs):
        """Compute the content attention.

        Args:
            queries (torch.tensor): tensor of size (batch_size, n_queries, kq_size) containing the queries.
            keys (torch.tensor): tensor of size (batch_size, n_keys, kq_size) containing the keys.
            attention_method_kwargs:
                Additional arguments to the attention method.
        """
        batch_size, n_queries, kq_size = queries.size()
        n_keys = keys.size(1)

        mask = keys.eq(0.)[:, :, :1].transpose(1, 2)

        # Compute attention vals
        if self.is_postcounter:
            content_output = self.method(queries[:, :, :-self.counter_size], keys[:, :, :-self.counter_size])

            #(batch_size, n_queries, n_keys, counter_size)
            diff = torch.stack([keys[:, :, -self.counter_size:] - queries[:, i, -self.counter_size:].unsqueeze(1)
                                for i in range(n_queries)], dim=1)

            count_input = torch.cat((diff**2, diff), dim=3)
            count_output = self.count_localizer(count_input).squeeze(-1)

            localizer_input = torch.stack((content_output, count_output), dim=-1)
            logits = self.context_count_mixer(localizer_input).squeeze(-1)
        else:
            logits = self.method(queries, keys, **attention_method_kwargs)

        if self.mask is not None:
            logits.masked_fill_(self.mask, -float('inf'))

        # apply local mask
        logits.masked_fill_(mask, -float('inf'))

        self._add_to_test(logits, "logits", additional)

        approx_max_logit = logits.logsumexp(dim=-1)

        # SHOULD TRY THIS AT SOME POINT
        # indeed max < logsumexp < max + log(n)
        # so if want to be sure than never too far from real max can use
        # max - log(n)/2 < logsumexp - log(n)/2 < max + log(n)/2
        #approx_max_logit = approx_max_logit - math.log(logits.size(-1)) / 2

        confidence = self.maxlogit_to_conf(approx_max_logit)

        attn = F.softmax(logits.view(-1, n_keys), dim=1).view(batch_size, -1, n_keys)

        return attn, confidence

    def get_method(self, method, dim):
        """
        Set method to compute attention
        """
        if method == 'mlp':
            method = MLPAttn(dim)
        elif method == 'concat':
            method = Concat(dim)
        elif method == 'dot':
            method = Dot()
        elif method == 'hard':
            method = HardGuidance()
        else:
            raise ValueError("Unknown attention method {}".format(method))

        return method

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
                additional["test"][keys] = values


class Concat(nn.Module):
    """
    Implements the computation of attention by applying an
    MLP to the concatenation of the decoder and encoder
    hidden states.
    """

    def __init__(self, dim):
        super(Concat, self).__init__()
        self.mlp = nn.Linear(dim * 2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        linear_init(self.mlp)

    def forward(self, decoder_states, encoder_states):
        # apply mlp to all encoder states for current decoder

        # decoder_states --> (batch, dec_seqlen, hl_size)
        # encoder_states --> (batch, enc_seqlen, hl_size)
        batch_size, enc_seqlen, hl_size = encoder_states.size()
        _, dec_seqlen, _ = decoder_states.size()

        # (batch, enc_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
        encoder_states_exp = encoder_states.unsqueeze(1)
        encoder_states_exp = encoder_states_exp.expand(batch_size, dec_seqlen, enc_seqlen, hl_size)

        # (batch, dec_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
        decoder_states_exp = decoder_states.unsqueeze(2)
        decoder_states_exp = decoder_states_exp.expand(batch_size, dec_seqlen, enc_seqlen, hl_size)

        # reshape encoder and decoder states to allow batchwise computation. We will have
        # batch_size x enc_seqlen x dec_seqlen batches. So we apply the Linear
        # layer for each of them
        decoder_states_tr = decoder_states_exp.contiguous().view(-1, hl_size)
        encoder_states_tr = encoder_states_exp.contiguous().view(-1, hl_size)

        mlp_input = torch.cat((encoder_states_tr, decoder_states_tr), dim=1)

        # apply mlp and respape to get in correct form
        mlp_output = self.mlp(mlp_input)
        attn = mlp_output.view(batch_size, dec_seqlen, enc_seqlen)

        return attn


class Dot(nn.Module):

    def __init__(self):
        super(Dot, self).__init__()

    def forward(self, decoder_states, encoder_states):
        attn = torch.bmm(decoder_states, encoder_states.transpose(1, 2))
        return attn


class MLPAttn(nn.Module):

    def __init__(self, dim):
        super(MLPAttn, self).__init__()
        self.mlp = nn.Linear(dim * 2, dim)
        self.activation = nn.ReLU()
        self.out = nn.Linear(dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        linear_init(self.mlp, activation="relu")
        linear_init(self.out)

    def forward(self, decoder_states, encoder_states):
        # apply mlp to all encoder states for current decoder

        # decoder_states --> (batch, dec_seqlen, hl_size)
        # encoder_states --> (batch, enc_seqlen, hl_size)
        batch_size, enc_seqlen, hl_size = encoder_states.size()
        _, dec_seqlen, _ = decoder_states.size()

        # (batch, enc_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
        encoder_states_exp = encoder_states.unsqueeze(1)
        encoder_states_exp = encoder_states_exp.expand(batch_size, dec_seqlen, enc_seqlen, hl_size)

        # (batch, dec_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
        decoder_states_exp = decoder_states.unsqueeze(2)
        decoder_states_exp = decoder_states_exp.expand(batch_size, dec_seqlen, enc_seqlen, hl_size)

        # reshape encoder and decoder states to allow batchwise computation. We will have
        # batch_size x enc_seqlen x dec_seqlen batches. So we apply the Linear
        # layer for each of them
        decoder_states_tr = decoder_states_exp.contiguous().view(-1, hl_size)
        encoder_states_tr = encoder_states_exp.contiguous().view(-1, hl_size)

        mlp_input = torch.cat((encoder_states_tr, decoder_states_tr), dim=1)

        # apply mlp and reshape to get in correct form
        mlp_output = self.mlp(mlp_input)
        mlp_output = self.activation(mlp_output)
        out = self.out(mlp_output)
        attn = out.view(batch_size, dec_seqlen, enc_seqlen)

        return attn


class HardGuidance(nn.Module):
    """
    Attention method / attentive guidance method for data sets that are annotated with attentive guidance.
    """

    def forward(self, decoder_states, encoder_states, step, provided_content_attn):
        """
        Forward method that receives provided attentive guidance indices and returns proper
        attention scores vectors.

        Args:
            decoder_states (torch.FloatTensor): Hidden layer of all decoder states (batch, dec_seqlen, hl_size)
            encoder_states (torch.FloatTensor): Output layer of all encoder states (batch, dec_seqlen, hl_size)
            step (int): The current decoder step for unrolled RNN. Set to -1 for rolled RNN
            provided_content_attn (torch.LongTensor): Variable containing the provided attentive guidance indices (batch, max_provided_content_attn_length)

        Returns:
            torch.tensor: Attention score vectors (batch, dec_seqlen, hl_size)
        """

        # decoder_states --> (batch, dec_seqlen, hl_size)
        # encoder_states --> (batch, enc_seqlen, hl_size)
        batch_size, enc_seqlen, _ = encoder_states.size()
        _, dec_seqlen, _ = decoder_states.size()

        attention_indices = provided_content_attn.detach()
        # If we have shorter examples in a batch, attend the PAD outputs to the first encoder state
        attention_indices.masked_fill_(attention_indices.eq(-1), 0)

        # In the case of unrolled RNN, select only one column
        if step != -1:
            attention_indices = attention_indices[:, step]

        # Add a (second and) third dimension
        # In the case of rolled RNN: (batch_size x dec_seqlen) -> (batch_size x dec_seqlen x 1)
        # In the case of unrolled:   (batch_size)              -> (batch_size x 1          x 1)
        attention_indices = attention_indices.contiguous().view(batch_size, -1, 1)
        # Initialize attention vectors. These are the pre-softmax scores, so any
        # -inf will become 0 (if there is at least one value not -inf)
        attention_scores = torch.zeros(batch_size, dec_seqlen, enc_seqlen).fill_(-float('inf'))
        attention_scores = attention_scores.scatter_(dim=2, index=attention_indices, value=1)
        attention_scores = attention_scores

        return attention_scores
