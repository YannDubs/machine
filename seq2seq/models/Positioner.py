""" Positioning attention classes. """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence

from seq2seq.util.helpers import (MLP, renormalize_input_length, get_rnn,
                                  ProbabilityConverter, AnnealedGaussianNoise,
                                  HyperparameterInterpolator, get_extra_repr,
                                  clamp, format_source_lengths)

from seq2seq.util.initialization import replicate_hidden0, init_param, weights_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IS_X05 = False
IS_SIGMA0_MINSIGMA = False


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
        positioning_method (str, optional): name of the psotioner function to use
        is_mlps (bool, optional): whether to use MLP's instead of linear function
            for the weight generators.
        hidden_size (int, optional): number of neurones to use in hidden layers.
        is_recursive (bool, optional) whether to use a rnn.
        rnn_cell (str, optional): type of RNN cell
        is_weight_norm_rnn (bool, optional): whether to use weight normalization
             for the RNN. Weight normalization is similar to batch norm or layer
             normalization and has been shown to help when learning large models.
        is_building_blocks_mu (bool, optional): whether to use building blocks to
            generate the positional mu rather than using a normal MLP.
        is_content_attn (bool, optional): whether you are using content attention.
        is_l1_bb_weights (bool, optional): whether to use a l1 regularization
            on the postional attention mu's building block weights. This can
            be usefull if the building blocks are gihly correlyted.

     bb_weights_annealed_noise_kwargs={},
     bb_annealed_noise_kwargs={},

        is_clamp_mu (bool, optional): whether to clamp the positioning `mu` in a
            range ~[0,1] using a leaky ReLu.
        is_relative_sigma (bool, optional): whether to use a relative varaince,
            i.e normalized by encoder steps.
        min_sigma (float, optional): minimum value of the standard deviation in
             order not to have division by 0. This is also important to let the network
             continue learning by always attending to multiple words. It should be
             in the range ~[0.2,1], 0.5 being a good general default.
        initial_sigma (float, optional): initial sigma the network should use. It
             should be in the range ~[2,6], 5 being a good general default.
        n_steps_interpolate_min_sigma (int, optional): if not 0 , it will force
            the network to keep a higher sigma while it's learning. min_sigma will
            actually start at `initial_sigma` and will linearly decrease at each
            training calls, until it reaches the given `min_sigma`. This parameter
            defines the number of training calls before the mdoel should reach
            the final `min_sigma`. As a rule of thumb it should be a percentage
            (example 0.1*n_epochs) of the total number of training calls so that
            the network can have some time to train with the final min_sigma.
        is_dev_mode (bool, optional): whether to store many useful variables in
            `additional`. Useful when predicting with a trained model in dev mode
             to understand what the model is doing. Use with `dev_predict`.
    """

    def __init__(self, decoder_output_size, max_len,
                 positioning_method="gaussian",
                 is_mlps=True,
                 hidden_size=32,
                 is_recursive=True,
                 rnn_cell='gru',
                 is_weight_norm_rnn=False,
                 is_building_blocks_mu=True,
                 is_bb_bias=False,  # TO DOC
                 is_content_attn=True,
                 is_l1_bb_weights=False,
                 bb_weights_annealed_noise_kwargs={},
                 bb_annealed_noise_kwargs={},
                 is_clamp_mu=True,
                 is_relative_sigma=True,
                 # with min_sigma=0.5 the max attention you can have is 0.9647
                 min_sigma=0.5,
                 # intial_sigma=5 chosen so that high sigma but up to length 50
                 # can still  have different attention => can learn
                 initial_sigma=5.0,
                 n_steps_interpolate_min_sigma=0,
                 is_dev_mode=False):
        super(PositionAttention, self).__init__()

        self.positioning_method = positioning_method
        self.is_weight_norm_rnn = is_weight_norm_rnn
        self.is_content_attn = is_content_attn
        self.is_dev_mode = is_dev_mode
        self.is_l1_bb_weights = is_l1_bb_weights

        n_additional_musigma_input = 8 - int(not self.is_content_attn)

        input_size = decoder_output_size + n_additional_musigma_input
        self.is_recursive = is_recursive
        self.positioner = _get_positioner(self.positioning_method)
        self.min_sigma = min_sigma
        self.initial_sigma = initial_sigma
        self.get_sigma = HyperparameterInterpolator(self.initial_sigma,
                                                    self.min_sigma,
                                                    n_steps_interpolate_min_sigma,
                                                    mode="linear")
        self.is_relative_sigma = is_relative_sigma
        self.max_len = max_len
        self.is_clamp_mu = is_clamp_mu

        # Building blocks
        self.is_building_blocks_mu = is_building_blocks_mu
        self.is_bb_bias = is_bb_bias
        n_building_blocks_mu = (5 - int(not self.is_content_attn) + int(self.is_bb_bias)
                                if self.is_building_blocks_mu else 1)
        self.single_step = torch.tensor(1 / self.max_len).to(device)
        self.rel_counter = torch.arange(1, self.max_len + 1
                                        ).type(torch.FloatTensor
                                               ).unsqueeze(1).to(device) / (self.max_len)

        self.building_blocks_labels = ["mu_old",
                                       "mean_attn_old",
                                       "rel_counter_decoder",
                                       "single_step"]
        if self.is_content_attn:
            self.building_blocks_labels += ["mean_content_old"]

        if self.is_bb_bias:
            self.bias = torch.tensor(1.0).to(device)
            self.building_blocks_labels += ["bias"]

        if self.is_recursive:
            self.rnn, self.hidden0 = get_rnn(rnn_cell, input_size, hidden_size,
                                             batch_first=True,
                                             is_weight_norm=self.is_weight_norm_rnn,
                                             is_get_hidden0=True)
            self.mu_weights_generator = nn.Linear(hidden_size,
                                                  n_building_blocks_mu)
            if is_mlps:
                self.sigma_generator = MLP(hidden_size, hidden_size // 2, 1)
            else:
                self.sigma_generator = nn.Linear(hidden_size, 1)
        else:
            if is_mlps:
                self.mu_weights_generator = MLP(input_size,
                                                hidden_size,
                                                n_building_blocks_mu)
                self.sigma_generator = MLP(input_size,
                                           hidden_size,
                                           1)
            else:
                self.mu_weights_generator = nn.Linear(input_size,
                                                      n_building_blocks_mu)
                self.sigma_generator = nn.Linear(input_size, 1)

        self.bb_noise = AnnealedGaussianNoise(**bb_annealed_noise_kwargs)
        self.bb_weights_noise = AnnealedGaussianNoise(**bb_weights_annealed_noise_kwargs)

        # inital sigma will not be 0 so have to change that value : I use the
        # expectation of the initialization of sigma0 although what we really
        # care about is sigma1 which could be very different from sigma0 note that
        # sigma0 is 1 but we give it in as -sigma, so it's as if start at -1
        sigma0 = self.initial_sigma if IS_SIGMA0_MINSIGMA else 1.0
        self.sigma_to_conf = ProbabilityConverter(is_temperature=True,
                                                  is_bias=True,
                                                  initial_x=-1 * sigma0)

        self.mu0 = Parameter(torch.tensor(0.0))
        # starting with sigma = 0 would strongly bias network
        self.sigma0 = Parameter(torch.tensor(self.initial_sigma))

        self.reset_parameters()

    def set_dev_mode(self, value=True):
        self.is_dev_mode = value

    def reset_parameters(self):
        """Reset and initialize the module parameters."""
        self.apply(weights_init)

        if IS_X05:
            self.mu0 = Parameter(torch.tensor(0.5))
        else:
            init_param(self.mu0, is_positive=True)

        sigma0 = self.initial_sigma if IS_SIGMA0_MINSIGMA else 1.0
        self.sigma0 = Parameter(torch.tensor(sigma0)).to(device)

        self.get_sigma.reset_parameters()

    def flatten_parameters(self):
        """Flattens the module parameters."""
        if self.is_recursive:
            self.rnn.flatten_parameters()

    def extra_repr(self):
        txt = self.get_sigma.extra_repr(value_name="sigma")
        return txt + ", " + get_extra_repr(self,
                                           conditional_shows=["positioning_method",
                                                              "is_clamp_mu",
                                                              "is_building_blocks_mu",
                                                              "is_relative_sigma",
                                                              "is_weight_norm_rnn",
                                                              "is_l1_bb_weights"])

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
            decoder_outputs (torch.tensor): tensor of size (batch_size, n_steps,
                hidden_size) containing the hidden activations of the coder.
            source_lengths (tuple(list of int, torch.FloatTesnor), optional): A
                list that contains the lengths of sequences in the mini-batch. The
                Tensor has the same information but is preinitialized on teh
                correct device.
            step (int): current decoding step.
            mu_old (torch.tensor): tensor of size (batch_size, n_steps, 1)
                containing last means of the positional attention. `None` for
                step == 0.
            sigma_old (torch.tensor): tensor of size (batch_size, n_steps, 1)
                containing last standard deviations of the positional attention.
                `None` for step == 0.
            mean_content_old (torch.tensor): tensor of size (batch_size, n_steps)
                containing the mean position of the last attention.
            mean_attn_old (torch.tensor): tensor of size (batch_size, n_steps)
                containing the mean position of the last content attention.
            additional (dictionary): dictionary containing additional variables
                that are necessary for some hyperparamets.
        """
        batch_size, max_source_lengths, _ = decoder_outputs.size()
        source_lengths_list, source_lengths_tensor = format_source_lengths(source_lengths)

        positioning_inputs, building_blocks = self._get_features(decoder_outputs,
                                                                 source_lengths_tensor,
                                                                 step,
                                                                 mu_old,
                                                                 sigma_old,
                                                                 mean_content_old,
                                                                 mean_attn_old)

        mu, sigma = self._compute_parameters(positioning_inputs,
                                             building_blocks,
                                             step,
                                             source_lengths_tensor,
                                             additional)

        # smaller sigma means more confident => - sigma
        # -sigma can only be negative but you still want confidence between 0
        # and 1 so need to shift to right => add only a positive bias
        # could use relu but for gradient flow use leaky relu
        pos_confidence = self.sigma_to_conf(-sigma, transform_bias=F.leaky_relu)
        pos_confidence = pos_confidence.mean(dim=-1)

        rel_counter_encoder = renormalize_input_length(self.rel_counter.expand(batch_size, -1, 1),
                                                       source_lengths_tensor,
                                                       self.max_len)

        pos_attn = pad_sequence([self.positioner(rel_counter_encoder[b, :length, :],
                                                 mu[b].squeeze(),
                                                 sigma[b].squeeze())
                                 for b, length in enumerate(source_lengths_list)],
                                batch_first=True)

        # new size = (batch, n_queries, n_keys)
        pos_attn = pos_attn.transpose(2, 1)

        return pos_attn, pos_confidence, mu, sigma

    def _get_features(self, decoder_outputs, source_lengths_tensor, step, mu_old,
                      sigma_old, mean_content_old, mean_attn_old):
        """Gets the inputs and the buillding blocks for positioning. Together
        those will eb used to compute the parameters of the positioning function.
        """
        batch_size = decoder_outputs.size(0)

        rel_counter_decoder = renormalize_input_length(self.rel_counter[step:step + 1
                                                                        ].expand(batch_size, 1),
                                                       source_lengths_tensor,
                                                       self.max_len)
        abs_counter_decoder = (self.rel_counter[step:step + 1].expand(batch_size, 1) *
                               self.max_len)

        if step == 0:
            mu_old = self.mu0.expand(batch_size, 1)
            sigma_old = self.sigma0.expand(batch_size, 1)
        else:
            mu_old = mu_old.squeeze(2)
            sigma_old = sigma_old.squeeze(2)

        single_step = renormalize_input_length(self.single_step.expand(batch_size, 1),
                                               source_lengths_tensor,
                                               self.max_len)

        shared = [mu_old, mean_attn_old, rel_counter_decoder, single_step]
        if self.is_content_attn:
            shared = shared + [mean_content_old]

        not_shared = [sigma_old,
                      abs_counter_decoder,
                      source_lengths_tensor.unsqueeze(-1)]
        additional_positioning_features = shared + not_shared

        positioning_inputs = torch.cat([decoder_outputs.squeeze(1)] +
                                       additional_positioning_features,
                                       dim=1)

        if self.is_bb_bias:
            bias = self.bias.expand(batch_size, 1)
            shared += [bias]

        assert_text = "building_blocks and their labels should map to each other."
        assert len(shared) == len(self.building_blocks_labels), assert_text
        building_blocks = torch.cat(shared, dim=1)

        return positioning_inputs, building_blocks

    def _compute_parameters(self, positioning_inputs, building_blocks, step,
                            source_lengths_tensor, additional):
        """Compute the parameters of the positioning function."""
        batch_size = positioning_inputs.size(0)

        if self.is_recursive:
            if step == 0:
                additional["positioner_hidden"] = replicate_hidden0(self.hidden0,
                                                                    batch_size)
            positioning_inputs, positioner_hidden = self.rnn(positioning_inputs.unsqueeze(1),
                                                             additional['positioner_hidden'])
            positioning_inputs = positioning_inputs.squeeze(1)
            additional['positioner_hidden'] = positioner_hidden

        mu_weights = self.mu_weights_generator(positioning_inputs)
        sigma = self.sigma_generator(positioning_inputs)

        sigma = self.min_sigma + sigma.unsqueeze(1)

        if self.is_building_blocks_mu:

            if self.is_dev_mode:
                additional['test']['mu_weights'] = mu_weights
                additional['test']['building_blocks'] = building_blocks
                additional['test']['step'] = step

            self._add_to_visualize([mu_weights, building_blocks],
                                   ['mu_weights', 'building_blocks'],
                                   additional)

            if self.is_l1_bb_weights:
                additional["losses"]["mu_weights"] = torch.abs(mu_weights).mean()
                # additional["losses"]["mu_weights0"] = torch.abs(mu_weights[:, :3]).mean()
                # additional["losses"]["mu_weights1"] = torch.abs(mu_weights[:, 4:]).mean()

            building_blocks = self.bb_noise(building_blocks, is_update=(step == 0))
            mu_weights = self.bb_weights_noise(mu_weights, is_update=(step == 0))
            # (batch, 1, 5) * (batch, 5, 1) -> (batch, 1, 1)
            mu = torch.bmm(mu_weights.unsqueeze(1), building_blocks.unsqueeze(2))
        else:
            mu = torch.sigmoid(mu_weights.unsqueeze(1))

        if self.is_clamp_mu:
            mu = clamp(mu, minimum=0, maximum=1, is_leaky=True)

        is_update_sigma = self.training and step == 0
        sigma = torch.max(sigma,
                          torch.zeros_like(sigma) + self.get_sigma(is_update_sigma))

        if self.is_relative_sigma:
            sigma = renormalize_input_length(sigma, source_lengths_tensor, 1)

        return mu, sigma

    def _add_to_visualize(self, values, keys, additional, save_every_n_batches=10):
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
                    values = values.mean(0)
                additional["visualize"][keys] = values


class AttentionMixer(nn.Module):
    """Mixes content and positional attention.

    Args:
        decoder_output_size (int): size of the hidden activations of the decoder.
        hidden_size (int, optional): number of hidden neurons in the MLP.
        is_mlps (bool, optional): whether to use MLP's instead of linear
            function for the weight generators.
        is_pos_perc_weight_conf (bool, optional): whether to force the model to
            generate meaningfull confidence, by making the positonal percentange
            be the `position_confidence / (position_confidence + content_confidence)`.
    """

    def __init__(self, decoder_output_size,
                 hidden_size=32,
                 is_mlps=True,
                 is_pos_perc_weight_conf=True,
                 is_dev_mode=False):
        super(AttentionMixer, self).__init__()

        self.is_dev_mode = is_dev_mode

        self.is_pos_perc_weight_conf = is_pos_perc_weight_conf

        n_additional_pos_perc_inputs = 3

        # should be under if not is_predict_conf (i.e net else)
        # but keeping while testing `additional_controller_features`
        self.position_perc0 = Parameter(torch.tensor(0.5))

        if not self.is_pos_perc_weight_conf:
            if is_mlps:
                self.pos_perc_generator = MLP(decoder_output_size +
                                              n_additional_pos_perc_inputs,
                                              hidden_size, 1)
            else:
                self.pos_perc_generator = nn.Linear(decoder_output_size +
                                                    n_additional_pos_perc_inputs,
                                                    1)

            self.posperc_to_prob = ProbabilityConverter()

        self.reset_parameters()

    def set_dev_mode(self, value=True):
        self.is_dev_mode = value

    def reset_parameters(self):
        self.apply(weights_init)
        self.position_perc0 = Parameter(torch.tensor(0.5))

    def extra_repr(self):
        return get_extra_repr(self,
                              conditional_shows=["is_pos_perc_weight_conf"])

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
            decoder_output (torch.tensor): tensor of size (batch_size, n_steps,
                hidden_size) containing the hidden activations of the decoder.
            step (int): current decoding step.
            content_attn (torch.tensor): tensor of size (batch_size, n_steps,
                 source_length) containing the content attentions.
            content_confidence (torch.tensor): tensor of size (batch_size, n_steps)
                 containing the confidence for the content attentions.
            pos_attn (torch.tensor): tensor of size (batch_size, n_steps, source_length)
                containing the positional attentions.
            pos_confidence (torch.tensor): tensor of size (batch_size, n_steps)
                containing the confidence for the positional attentions.
            position_perc_old (torch.tensor): tensor of size (batch_size, 1)
                containing the last positional percentage.
        """
        batch_size = decoder_output.size(0)

        if self.is_pos_perc_weight_conf:
            position_perc = pos_confidence / (pos_confidence + content_confidence)
        else:
            if step == 0:
                position_perc_old = self.position_perc0.expand(batch_size, 1)

            additional_pos_perc_inputs = [pos_confidence,
                                          content_confidence,
                                          position_perc_old]
            position_perc_inputs = torch.cat([decoder_output.squeeze(1)] +
                                             additional_pos_perc_inputs,
                                             dim=1)

            position_perc = self.pos_perc_generator(position_perc_inputs)
            position_perc = self.posperc_to_prob(position_perc)

        # COnvex combination
        attn = (pos_attn * position_perc.unsqueeze(-1) +
                (1 - position_perc.unsqueeze(-1)) * content_attn)

        return attn, position_perc
