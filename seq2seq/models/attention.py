import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Attention(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.

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

    def __init__(self, dim, method, is_decoupled_kv=False, kv_architecture=False, is_postcounter=False, counter_size=None,
                 is_positioning_generator=False):
        super(Attention, self).__init__()
        self.mask = None
        self.method = self.get_method(method, dim)
        self.is_decoupled_kv = is_decoupled_kv
        self.kv_architecture = kv_architecture
        self.is_postcounter = is_postcounter
        self.counter_size = counter_size
        self.is_positioning_generator = is_positioning_generator

        if self.is_postcounter:
            self.semantic_count_localizer = nn.Linear(2, 1)
            self.count_localizer = nn.Linear(self.counter_size * 2, 1)

        self.count = 0

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, querries, encoder_states, positioners=None, location_percentage=None):

        additional = dict()

        batch_size = querries.size(0)
        n_querries = querries.size(1)

        if self.kv_architecture:
            keys, values = encoder_states

        elif self.is_decoupled_kv:
            dim = encoder_states.size(2)
            n_value = dim // 2

            select_keys = torch.zeros(dim)
            select_values = torch.ones(dim)

            if torch.cuda.is_available():
                select_keys = select_keys.cuda()
                select_values = select_values.cuda()

            select_keys[:-n_value] = 1
            select_values = select_values - select_keys

            keys = encoder_states * select_keys
            values = encoder_states * select_values

        else:
            keys = values = encoder_states

        input_size = values.size(1)
        # compute attention vals
        if self.is_postcounter:
            content_output = self.method(querries[:, :, :-self.counter_size], keys[:, :, :-self.counter_size])
            diff = torch.stack([keys[:, :, -self.counter_size:] - querries[:, i, -self.counter_size:].unsqueeze(1)
                                for i in range(n_querries)], dim=2)
            diff = diff.view(batch_size, input_size, n_querries, self.counter_size)

            count_input = torch.cat((diff**2, diff), dim=3)
            count_output = self.count_localizer(count_input).squeeze(-1)
            content_output = content_output.view(batch_size, input_size, n_querries)
            localizer_input = F.relu(torch.stack((content_output, count_output), dim=-1))
            attn = self.semantic_count_localizer(localizer_input).squeeze(-1)
        else:
            attn = self.method(querries, keys)

        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))

        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        if self.is_positioning_generator:
            positioners = positioners[:, :input_size, :].transpose(1, 2)
            positioners = F.normalize(positioners, p=1, dim=-1)
            additional["positional_attention"] = positioners
            additional["content_attention"] = attn
            additional["location_percentage"] = location_percentage
            attn = positioners * location_percentage + (1 - location_percentage) * attn

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        context = torch.bmm(attn, values)

        return context, attn, additional

    def get_method(self, method, dim):
        """
        Set method to compute attention
        """
        if method == 'mlp':
            method = Concat(dim)
        elif method == 'dot':
            method = Dot()
        else:
            return ValueError("Unknown attention method")
        return method


class Concat(nn.Module):
    """
    Implements the computation of attention by applying an
    MLP to the concatenation of the decoder and encoder
    hidden states.
    """

    def __init__(self, dim):
        super(Concat, self).__init__()
        self.mlp = nn.Linear(dim * 2, 1)

    def forward(self, decoder_states, encoder_states):
        # apply mlp to all encoder states for current decoder

        # decoder_states --> (batch, 1, hl_size)
        # encoder_states --> (batch, seqlen, hl_size)
        batch_size, seqlen, hl_size = encoder_states.size()

        # expand decoder states and transpose
        decoder_states_exp = decoder_states.expand(batch_size, seqlen, hl_size)
        decoder_states_tr = decoder_states_exp.contiguous().view(-1, hl_size)

        # reshape encoder states to allow batchwise computation
        encoder_states_tr = encoder_states.contiguous().view(-1, hl_size)

        mlp_input = torch.cat((encoder_states_tr, decoder_states_tr), dim=1)

        # apply mlp and respape to get in correct form
        mlp_output = self.mlp(mlp_input)
        attn = mlp_output.view(batch_size, seqlen)

        return attn


class Dot(nn.Module):
    def __init__(self):
        super(Dot, self).__init__()

    def forward(self, decoder_states, encoder_states):
        attn = torch.bmm(decoder_states, encoder_states.transpose(1, 2))
        return attn
