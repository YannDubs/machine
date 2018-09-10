""" Content attention modules. """
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq.util.initialization import weights_init, linear_init
from seq2seq.util.torchextend import MLP, ProbabilityConverter
from seq2seq.util.helpers import Clamper

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

        # low initial temperature because logits can take a very high range of values
        # so don't want to have vanishing gradients from the start
        self.maxlogit_to_conf = ProbabilityConverter(is_temperature=True,
                                                     is_bias=True,
                                                     initial_temperature=0.1,
                                                     temperature_transformer=Clamper(minimum=0.05,
                                                                                     maximum=10,
                                                                                     is_leaky=True))

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

        approx_max_logit = logits.logsumexp(dim=-1)

        self._add_to_test([logits, approx_max_logit],
                          ["logits", "approx_max_logit"],
                          additional)
        self._add_to_visualize(approx_max_logit,
                               ["approx_max_logit"],
                               additional)

        # SHOULD TRY THIS AT SOME POINT
        # indeed max < logsumexp < max + log(n)
        # so if want to be sure than never too far from real max can use
        # max - log(n)/2 < logsumexp - log(n)/2 < max + log(n)/2
        # approx_max_logit = approx_max_logit - math.log(logits.size(-1)) / 2

        confidence = self.maxlogit_to_conf(approx_max_logit)

        attn = F.softmax(logits.view(-1, n_keys), dim=1).view(batch_size, -1, n_keys)

        return attn, confidence

    def get_method(self, method, dim):
        """
        Set method to compute attention
        """
        if method == 'multiplicative':
            method = MultiplicativeAttn(dim, is_scale=False)
        elif method == 'additive':
            method = AdditiveAttn(dim)
        elif method == 'scaledot':
            method = DotAttn(is_scale=True)
        elif method == 'dot':
            method = DotAttn(is_scale=False)
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
                    values = values.mean(0).cpu()
                additional["visualize"][keys] = values


class DotAttn(nn.Module):
    """
    Implements the computation of attention by using a scaled attention just liek
    in "attention is all you need". Scaling can help when dimension is large :
    making sure that there are no  extremely small gradients
    """

    def __init__(self, is_scale=True):
        super().__init__()
        self.is_scale = is_scale

    def forward(self, queries, keys):
        logits = torch.bmm(queries, keys.transpose(1, 2))
        if self.is_scale:
            logits = logits / math.sqrt(queries.size(-1))
        return logits


class MultiplicativeAttn(nn.Module):
    """
    Implements the computation of attention by using a scaled attention just liek
    in "attention is all you need". Scaling can help when dimension is large :
    making sure that there are no  extremely small gradients
    """

    def __init__(self, dim, is_scale=True):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.scaled_dot = DotAttn(is_scale=is_scale)

        self.reset_parameters()

    def reset_parameters(self):
        linear_init(self.linear)

    def forward(self, queries, keys):
        transformed_queries = self.linear(queries)
        logits = self.scaled_dot(transformed_queries, keys)
        return logits


class AdditiveAttn(nn.Module):
    """
    Implements additive attention as seen in Bahdanau et al. :
    "Neural Machine Translation by Jointly Learning to Align and Translate"
    """

    def __init__(self, dim):
        super().__init__()
        self.mlp = MLP(dim * 2, dim, 1, activation=nn.ReLU)
        # should try nn.Tanh

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init)

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
        logits = self.mlp(mlp_input)
        logits = logits.view(batch_size, dec_seqlen, enc_seqlen)

        return logits


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
