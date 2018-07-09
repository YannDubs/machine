import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence

from seq2seq.util.helpers import MLP, renormalize_input_length, ProbabilityConverter, get_rnn_cell
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
    """Get the correct positioner method."""
    if name == "gaussian":
        return _discrete_truncated_gaussian
    elif name == "laplace":
        return _discrete_truncated_laplace
    else:
        raise ValueError("Unkown positioner method {}".format(name))


class PositionAttention(nn.Module):
    """Position Attention Generator.

    Args:
        decoder_output_size (int): size of the hidden activations of the decoder.
        max_len (int): a maximum allowed length for the sequence to be processed
        is_recursive (bool, optional) whether to use a rnn.
        positioning_method (str, optional): name of the psotioner function to use
        min_sigma (float, optional): minimum value of the standard deviation in order not to have division by 0.
        hidden_size (int, optional): number of neurones to use in hidden layers.
        is_relative_sigma (bool, optional): whether to use a relative varaince, i.e normalized by encoder steps.
        rnn_cell (str, optional): type of RNN cell
        is_mlps (bool, optional): whether to use MLP's instead of linear function for the weight generators.
    """

    def __init__(self, decoder_output_size, max_len,
                 is_recursive=False,
                 positioning_method="gaussian",
                 min_sigma=0.05,
                 hidden_size=32,
                 is_relative_sigma=True,
                 rnn_cell='gru',
                 is_mlps=True,
                 is_clamp_mu=False):
        super(PositionAttention, self).__init__()

        n_additional_musigma_input = 5
        input_size = decoder_output_size + n_additional_musigma_input
        self.is_recursive = is_recursive
        self.positioning_method = _get_positioner(positioning_method)
        self.min_sigma = min_sigma
        self.is_relative_sigma = is_relative_sigma
        self.max_len = max_len
        self.is_clamp_mu = is_clamp_mu

        # Building blocks
        n_building_blocks = 8
        self.single_step = torch.tensor(1 / self.max_len).to(device)
        self.rel_counter = torch.arange(1, self.max_len + 1).unsqueeze(1).to(device) / (self.max_len)
        self.bias = torch.tensor(1.0).to(device)

        if self.is_recursive:
            self.rnn_cell = get_rnn_cell(rnn_cell)
            self.rnn = self.rnn_cell(input_size, hidden_size, 1, batch_first=True)
            self.hidden0 = get_hidden0(self.rnn)
            self.mu_weights_generator = nn.Linear(hidden_size, n_building_blocks)
            self.sigma_weights_generator = nn.Linear(hidden_size, n_building_blocks)
        else:
            if is_mlps:
                self.mu_weights_generator = MLP(input_size, hidden_size, n_building_blocks)
                self.sigma_weights_generator = MLP(input_size, hidden_size, n_building_blocks)
            else:
                self.mu_weights_generator = nn.Linear(input_size, n_building_blocks)
                self.sigma_weights_generator = nn.Linear(input_size, n_building_blocks)

        # inital sigma will not be 0 so have to change that value : I use the expectation of the initialization of sigma0
        # although what we really care about is sigma1 which could be very different from sigma0
        self.probabilty_converter = ProbabilityConverter(is_temperature=True, is_bias=True, initial_x=1)

        self.mu0 = Parameter(torch.tensor(0.0)).to(device)
        # start with unit standard deviation because starting with 0 would strongly bias network
        self.sigma0 = Parameter(torch.tensor(1.0)).to(device)
        self.mean_attn0 = Parameter(torch.tensor(0.0)).to(device)
        self.mean_content0 = Parameter(torch.tensor(0.0)).to(device)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset and initialize the module parameters."""
        self.apply(weights_init)
        init_param(self.mu0, is_positive=True)
        init_param(self.mean_attn0, is_positive=True)
        init_param(self.mean_content0, is_positive=True)

    def flatten_parameters(self):
        """Flattens the module parameters."""
        if self.is_recursive:
            self.rnn.flatten_parameters()

    def forward(self,
                decoder_outputs,
                source_lengths,
                step,
                mu_old,
                sigma_old,
                mean_content_old,
                mean_attn_old,
                additional):
        """Compute and return the positional attention, confidence, parameters.

        Args:
            decoder_outputs (torch.tensor): tensor of size (batch_size, n_steps, hidden_size) containing
                the hidden activations of the coder.
            source_lengths (list): list of the lengths of each source sentence in the batch.
            step (int): current decoding step.
            mu_old (torch.tensor): tensor of size (batch_size, n_steps, 1) containing last means of the positional attention.
                `None` for step == 0.
            sigma_old (torch.tensor): tensor of size (batch_size, n_steps, 1) containing last standard deviations of the
                positional attention. `None` for step == 0.
            mean_content_old (torch.tensor): tensor of size (batch_size, n_steps) containing the mean position of
                the last attention. `None` for step == 0.
            mean_attn_old (torch.tensor): tensor of size (batch_size, n_steps) containing the mean position of
                the last content attention. `None` for step == 0.
            additional (dictionary): dictionnary containing additional variables that are necessary for some hyperparamets.
        """
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
        # -sigma can only be negative but you still want confidence between 0 and 1 so need to shift to right => add only a positive bias
        pos_confidence = self.probabilty_converter(-sigma, transform_bias=F.relu)
        pos_confidence = pos_confidence.mean(dim=-1)

        rel_counter_encoder = renormalize_input_length(self.rel_counter.expand(batch_size, -1, 1),
                                                       source_lengths,
                                                       self.max_len)

        pos_attn = pad_sequence([self.positioning_method(rel_counter_encoder[b, :length, :], mu[b].squeeze(), sigma[b].squeeze())
                                 for b, length in enumerate(source_lengths)], batch_first=True)

        # new size = (batch, n_queries, n_keys)
        pos_attn = pos_attn.transpose(2, 1)

        return pos_attn, pos_confidence, mu, sigma

    def _get_features(self, decoder_outputs, source_lengths, step, mu_old, sigma_old, mean_content_old, mean_attn_old):
        """Gets the inputs and the buillding blocks for positioning. Together those will eb used to compute the parameters
        of the positioning function."""
        batch_size = decoder_outputs.size(0)

        rel_counter_decoder = renormalize_input_length(self.rel_counter[step:step + 1].expand(batch_size, 1),
                                                       source_lengths,
                                                       self.max_len)

        if step == 0:
            mu_old = self.mu0.expand(batch_size, 1)
            sigma_old = self.sigma0.expand(batch_size, 1)
            mean_content_old = self.mean_attn0.expand(batch_size, 1)
            mean_attn_old = self.mean_content0.expand(batch_size, 1)
        else:
            mu_old = mu_old.squeeze(2)
            sigma_old = sigma_old.squeeze(2)

        if self.is_relative_sigma:
            sigma_old = renormalize_input_length(sigma_old, source_lengths, 1)

        shared = [mu_old, sigma_old, mean_content_old, mean_attn_old, rel_counter_decoder]

        positioning_inputs = torch.cat([decoder_outputs.squeeze(1)] + shared, dim=1)

        single_step = renormalize_input_length(self.single_step.expand(batch_size, 1),
                                               source_lengths,
                                               self.max_len)
        bias = self.bias.expand(batch_size, 1)
        building_blocks = torch.cat(shared + [single_step, -1 * single_step, bias], dim=1)

        return positioning_inputs, building_blocks

    def _compute_parameters(self, positioning_inputs, building_blocks, step, additional):
        """Compute the parameters of the positioning function."""
        batch_size = positioning_inputs.size(0)

        if self.is_recursive:
            if step == 0:
                additional["positioner_hidden"] = replicate_hidden0(self.hidden0, batch_size)
            positioning_inputs, positioner_hidden = self.rnn(positioning_inputs.unsqueeze(1), additional['positioner_hidden'])
            positioning_inputs = positioning_inputs.squeeze(1)
            additional['positioner_hidden'] = positioner_hidden

        mu_weights = self.mu_weights_generator(positioning_inputs)
        sigma_weights = self.sigma_weights_generator(positioning_inputs)

        # (batch, 1, 8) * (batch, 8, 1) -> (batch, 1, 1)
        mu = torch.bmm(mu_weights.unsqueeze(1), building_blocks.unsqueeze(2))
        if self.is_clamp_mu:
            mu = torch.clamp(mu, 0.0, 1.0)

        sigma = torch.bmm(sigma_weights.unsqueeze(1), building_blocks.unsqueeze(2))
        sigma = torch.relu(sigma) + self.min_sigma

        return mu, sigma


class AttentionMixer(nn.Module):
    """Mixes content and positional attention.

    Args:
        decoder_output_size (int): size of the hidden activations of the decoder.
        hidden_size (int, optional): number of hidden neurons in the MLP.
        is_mlps (bool, optional): whether to use MLP's instead of linear function for the weight generators.
        is_predict_conf (bool, optional) whether to force the model to generate meaningfull confidence, by making the positonal
            percentange generator depend only on the content and the position confidence.
    """

    def __init__(self, decoder_output_size, hidden_size=32, is_mlps=True, is_predict_conf=False):
        super(AttentionMixer, self).__init__()
        self.is_predict_conf = is_predict_conf

        n_additional_pos_perc_inputs = 3

        if is_predict_conf:
            if is_mlps:
                self.pos_perc_generator = MLP(2, 2, 1)
            else:
                self.pos_perc_generator = nn.Linear(2, 1)
        else:
            self.position_perc0 = Parameter(torch.tensor(0.5)).to(device)

            if is_mlps:
                self.pos_perc_generator = MLP(decoder_output_size + n_additional_pos_perc_inputs, hidden_size, 1)
            else:
                self.pos_perc_generator = nn.Linear(decoder_output_size + n_additional_pos_perc_inputs, 1)

        self.probabilty_converter = ProbabilityConverter()

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
        """Compute and return the final attention and percentage of positional attention.

        Args:
            decoder_output (torch.tensor): tensor of size (batch_size, n_steps, hidden_size) containing
                the hidden activations of the decoder.
            step (int): current decoding step.
            content_attn (torch.tensor): tensor of size (batch_size, n_steps, source_length) containing the content attentions.
            content_confidence (torch.tensor): tensor of size (batch_size, n_steps) containing the confidence for the content attentions.
            pos_attn (torch.tensor): tensor of size (batch_size, n_steps, source_length) containing the positional attentions.
            pos_confidence (torch.tensor): tensor of size (batch_size, n_steps) containing the confidence for the positional attentions.
            position_perc_old (torch.tensor): tensor of size (batch_size, 1) containing the last positional percentage.
        """
        batch_size = decoder_output.size(0)

        if self.is_predict_conf:
            position_perc_inputs = torch.cat([pos_confidence, content_confidence], dim=1)
        else:
            if step == 0:
                position_perc_old = self.position_perc0.expand(batch_size, 1)

            additional_pos_perc_inputs = [pos_confidence, content_confidence, position_perc_old]
            position_perc_inputs = torch.cat([decoder_output.squeeze(1)] + additional_pos_perc_inputs, dim=1)

        position_perc = self.pos_perc_generator(position_perc_inputs)
        position_perc = self.probabilty_converter(position_perc)

        attn = pos_attn * position_perc.unsqueeze(-1) + (1 - position_perc.unsqueeze(-1)) * content_attn

        return attn, position_perc
