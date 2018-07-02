import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from seq2seq.util.helpers import MLP, renormalize_input_length, generate_probabilities
from seq2seq.util.initialization import get_hidden0, replicate_hidden0, init_param, weights_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _discrete_truncated_gaussian(x, mu, sigma):
    """Samples a values from a gaussian pdf and normalizes those."""
    x = torch.exp(-((x - mu) / sigma)**2)
    x = F.normalize(x, p=1, dim=0)
    return x


def _discrete_truncated_laplace(x, mu, b):
    """Samples a values from a laplacian pdf and normalizes those."""
    x = torch.exp(-1 * torch.abs((x - mu) / b))
    x = F.normalize(x, p=1, dim=0)
    return x


def _get_positioner(name):
    if name == "gaussian":
        return _discrete_truncated_gaussian
    elif name == "laplace":
        return _discrete_truncated_laplace
    else:
        raise ValueError("Unkown positioner method {}".format(name))


class PositionAttention(nn.Module):
    """Position Attention Generator."""

    def __init__(self, decoder_output_size,
                 is_recursive=False,
                 positioning_method="gaussian",
                 min_sigma=0.05,
                 hidden_size=32,
                 is_relative_sigma=True):
        super(PositionAttention, self).__init__()

        n_additional_musigma_input = 5
        input_size = decoder_output_size + n_additional_musigma_input
        self.positioning_method = _get_positioner(positioning_method)
        self.min_sigma = min_sigma
        self.is_relative_sigma = is_relative_sigma

        # Building blocks
        n_building_blocks = 8
        self.single_step = torch.tensor(1 / self.max_len).to(device)
        self.rel_counter = torch.arange(1, self.max_len + 1).unsqueeze(1).to(device) / (self.max_len)
        self.bias = torch.tensor(1.0).to(device)

        if self.is_recursive:
            self.rnn = self.rnn_cell(input_size, hidden_size, 1, batch_first=True)
            self.hidden0 = get_hidden0(self.rnn)
            self.mu_weights_generator = nn.Linear(hidden_size, n_building_blocks)
            self.sigma_weights_generator = nn.Linear(hidden_size, n_building_blocks)
        else:
            self.mu_weights_generator = MLP(input_size, hidden_size, n_building_blocks)
            self.sigma_weights_generator = MLP(input_size, hidden_size, n_building_blocks)

        self.confidence_temperature = Parameter(torch.tensor(1.0)).to(device)
        self.confidence_bias = Parameter(torch.tensor(0.0)).to(device)

        self.mu0 = Parameter(torch.tensor(0.0)).to(device)
        self.sigma0 = Parameter(torch.tensor(0.0)).to(device)
        self.mean_attn0 = Parameter(torch.tensor(0.0)).to(device)
        self.mean_content0 = Parameter(torch.tensor(0.0)).to(device)

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init)
        init_param(self.mu0, is_positive=True)
        init_param(self.sigma0, is_positive=True)
        init_param(self.mean_attn0, is_positive=True)
        init_param(self.mean_content0, is_positive=True)

    def forward(self,
                decoder_outputs,
                encoder_outputs,
                source_lengths,
                step,
                mu_old,
                sigma_old,
                mean_content_old,
                mean_attn_old,
                additional):
        batch_size, max_source_lengths, _ = decoder_outputs.size()

        positioning_inputs, building_blocks = self._get_features(decoder_outputs,
                                                                 source_lengths,
                                                                 step,
                                                                 mu_old,
                                                                 sigma_old,
                                                                 mean_content_old,
                                                                 mean_attn_old)

        mu, sigma = self._compute_parameters(positioning_inputs, building_blocks, step, additional)

        # smaller sigma means more confident => - sigma
        # -sigma can only be negative but you still want confidence between 0 and 1 so need to shift to right => add a positive bias
        pos_confidence = generate_probabilities(-sigma, temperature=self.confidence_temperature, bias=F.relu(self.confidence_bias))

        rel_counter_encoder = renormalize_input_length(self.rel_counter.expand(batch_size, -1, 1),
                                                       source_lengths,
                                                       self.max_length)

        pos_attn = torch.stack([self.positioning_method(rel_counter_encoder[b, :length, :], mu[b].squeeze(), sigma[b].squeeze())
                                for b, length in enumerate(source_lengths)], dim=0)

        # new size = (batch, n_queries, n_keys)
        pos_attn = pos_attn.transpose(2, 1)

        return pos_attn, pos_confidence, mu, sigma

    def _get_features(self, decoder_outputs, source_lengths, step, mu_old, sigma_old, mean_content_old, mean_attn_old):
        batch_size = decoder_outputs.size(0)

        rel_counter_decoder = renormalize_input_length(self.rel_counter[step:step + 1].expand(batch_size, 1),
                                                       source_lengths,
                                                       self.max_length)

        if step == 0:
            mu_old = self.mu0.expand(batch_size, 1)
            sigma_old = self.sigma0.expand(batch_size, 1)
            mean_content_old = self.mean_attn0.expand(batch_size, 1)
            mean_attn_old = self.mean_content0.expand(batch_size, 1)

        if self.is_relative_sigma:
            sigma_old = renormalize_input_length(sigma_old, source_lengths, 1)

        shared = [mu_old, sigma_old, mean_content_old, mean_attn_old, rel_counter_decoder]

        positioning_inputs = torch.cat([decoder_outputs.squeeze(1)] + shared, dim=1)

        single_step = renormalize_input_length(self.front_step.expand(batch_size, 1),
                                               source_lengths,
                                               self.max_length)
        bias = self.biais.expand(batch_size, 1)
        building_blocks = torch.cat(shared + [single_step, -1 * single_step, bias], dim=1)

        return positioning_inputs, building_blocks

    def _compute_parameters(self, positioning_inputs, building_blocks, step, additional):
        batch_size = positioning_inputs.size(0)

        if self.is_recursive:
            if step == 0:
                additional["positioner_hidden"] = replicate_hidden0(self.hidden0, batch_size, self.rnn.batch_first)
            positioning_inputs, positioner_hidden = self.rnn(positioning_inputs.unsqueeze(1), additional['positioner_hidden'])
            positioning_inputs = positioning_inputs.squeeze(1)
            additional['positioner_hidden'] = positioner_hidden

        mu_weights = self.mu_weights_generator(positioning_inputs)
        sigma_weights = self.sigma_weights_generator(positioning_inputs)

        # (batch, 1, 8) * (batch, 8, 1) -> (batch, 1, 1)
        mu = torch.bmm(mu_weights.unsqueeze(1), building_blocks.unsqueeze(2))

        sigma = torch.bmm(sigma_weights.unsqueeze(1), building_blocks.unsqueeze(2))
        sigma = torch.relu(sigma) + self.min_sigma

        return mu, sigma


class AttentionMixer(nn.Module):

    def __init__(self, decoder_output_size, hidden_size=32, min_position_perc=0.01):
        self.min_position_perc = min_position_perc

        n_additional_pos_perc_inputs = 3
        self.pos_perc_generator = MLP(decoder_output_size + n_additional_pos_perc_inputs, hidden_size, 1)

        self.position_perc0 = Parameter(torch.tensor(0.5)).to(device)

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init)

    def forward(self,
                decoder_output,
                step,
                content_attn,
                content_confidence,
                pos_attn,
                pos_confidence,
                position_perc_old):
        batch_size = decoder_output.size(0)

        if step == 0:
            position_perc_old = self.position_perc0.expand(batch_size, 1)

        additional_pos_perc_inputs = [pos_confidence, content_confidence, position_perc_old]

        position_perc_inputs = torch.cat([decoder_output.squeeze(1)] + additional_pos_perc_inputs, dim=1)

        position_perc = self.pos_perc_generator(position_perc_inputs)
        position_perc = generate_probabilities(position_perc, min_p=self.min_position_importance)

        attn = pos_attn * position_perc.squeeze(1) + (1 - position_perc) * content_attn.squeeze(1)
        attn = attn.unsqueeze(1)

        return attn, position_perc
