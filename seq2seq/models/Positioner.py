""" Positioning attention classes. """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence

from seq2seq.util.helpers import (renormalize_input_length, get_rnn,
                                  HyperparameterInterpolator, get_extra_repr,
                                  clamp, format_source_lengths, Rate2Steps,
                                  get_indices, Clamper, regularization_loss,
                                  batch_reduction_f, add_to_test, add_to_visualize,
                                  add_regularization)
from seq2seq.util.torchextend import (MLP, StochasticRounding, ConcreteRounding,
                                      ProbabilityConverter, AnnealedGaussianNoise,
                                      L0Gates)
from seq2seq.util.initialization import replicate_hidden0, init_param, weights_init
from seq2seq.util.l0 import L0Dense

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_regularizers_positioner(total_training_calls, n_steps_prepare_pos=None):
    """Return the interpolators of the regularizers of the positioner.

    Args:
        total_training_calls (int): total planned number of calls to train the model.
        n_steps_prepare_pos (int, optional): number of steps that are considered
            as preparation for better convergence of the positioner.

    Returns:
        max_p_interpolators (dict of float):
            Dictionary mapping the name of the additional loss to maximum percentage
            of the loss that it can take. I.e if `max_p=1e-2`, for loss L then
            this loss will be reduced such that it can only reach 1e-2 of the
            NLL loss.
    """
    rate2steps = Rate2Steps(total_training_calls)
    max_p_interpolators = dict()
    is_prepare_pos = n_steps_prepare_pos is not None

    n_steps_interpolate = rate2steps(0.3)
    start_step = n_steps_prepare_pos if is_prepare_pos else rate2steps(0.05)
    max_p_interpolators["pos_mu_weights"
                        ] = HyperparameterInterpolator(3e-2, 5e-3, n_steps_interpolate,
                                                       start_step=start_step,
                                                       default=0,
                                                       mode="linear")
    print()
    print("pos_mu_weights:", max_p_interpolators["pos_mu_weights"].extra_repr())

    n_steps_interpolate = rate2steps(0.05)
    start_step = n_steps_prepare_pos if is_prepare_pos else rate2steps(0.05)
    max_p_interpolators["pos_const_weights"
                        ] = HyperparameterInterpolator(5e-2, 1e-2, n_steps_interpolate,
                                                       start_step=start_step,
                                                       default=5e-2,
                                                       mode="linear")

    print("pos_const_weights:", max_p_interpolators["pos_const_weights"].extra_repr())

    # wait until positioning converges
    n_steps_interpolate = n_steps_prepare_pos if is_prepare_pos else rate2steps(0.05)
    start_step = rate2steps(0)
    # n_steps_interpolate = rate2steps(0.05 if is_prepare_pos else n_steps_prepare_pos)
    # start_step = rate2steps(0)
    max_p_interpolators["pos_old_weights"
                        ] = HyperparameterInterpolator(5e-2, 0, n_steps_interpolate,
                                                       start_step=start_step,
                                                       default=1e-2,
                                                       mode="linear")

    print("pos_old_weights:", max_p_interpolators["pos_old_weights"].extra_repr())

    n_steps_interpolate = rate2steps(0)
    start_step = rate2steps(0)
    max_p_interpolators["pos_clamp_mu"
                        ] = HyperparameterInterpolator(5e-2, 5e-2, n_steps_interpolate,
                                                       start_step=start_step,
                                                       default=0,
                                                       mode="linear")

    print("pos_clamp_mu:", max_p_interpolators["pos_clamp_mu"].extra_repr())

    n_steps_interpolate = n_steps_prepare_pos if is_prepare_pos else rate2steps(0.05)
    start_step = rate2steps(0)
    max_p_interpolators["pos_round_weights"
                        ] = HyperparameterInterpolator(0, 5e-2, n_steps_interpolate,
                                                       start_step=start_step,
                                                       default=0,
                                                       mode="linear")
    print("pos_round_weights:", max_p_interpolators["pos_round_weights"].extra_repr())

    n_steps_interpolate = rate2steps(0.3)
    start_step = n_steps_prepare_pos if is_prepare_pos else rate2steps(0.05)
    max_p_interpolators["pos_l0_weights"
                        ] = HyperparameterInterpolator(3e-2, 5e-3, n_steps_interpolate,
                                                       start_step=start_step,
                                                       default=0,
                                                       mode="linear")
    print("pos_l0_weights:", max_p_interpolators["pos_l0_weights"].extra_repr())

    n_steps_interpolate = n_steps_prepare_pos if is_prepare_pos else rate2steps(0.05)
    start_step = rate2steps(0)
    max_p_interpolators["pos_variance_weights"
                        ] = HyperparameterInterpolator(0., 1e-2, n_steps_interpolate,
                                                       start_step=start_step,
                                                       default=0,
                                                       mode="linear")
    print("pos_variance_weights:", max_p_interpolators["pos_variance_weights"].extra_repr())

    n_steps_interpolate = rate2steps(0.3)
    start_step = rate2steps(0)
    max_p_interpolators["pos%"
                        ] = HyperparameterInterpolator(1e-2, 5e-2, n_steps_interpolate,
                                                       start_step=start_step,
                                                       default=0,
                                                       mode="linear")
    print("pos%:", max_p_interpolators["pos%"].extra_repr())

    # used for balancing, don't rescale
    max_p_interpolators["balancing"] = None
    print("balancing%:", max_p_interpolators["balancing"])
    print()

    return max_p_interpolators


def _discrete_truncated_gaussian(x, mu, sigma):
    """Return normalized Gaussian_pdf(x)."""
    x = torch.exp(-(x - mu)**2 / (2 * sigma**2))
    x = F.normalize(x, p=1, dim=0)
    return x


def _discrete_truncated_laplace(x, mu, b):
    """Return normalized Laplacian_pdf(x)."""
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


def _get_rounder(name=None, **kwargs):
    if name is None:
        return None
    elif name == "concrete":
        return ConcreteRounding(**kwargs)
    elif name == "stochastic":
        return StochasticRounding(**kwargs)
    else:
        raise ValueError("Unkown rounder method {}".format(name))


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
        is_reg_bb_weights (bool, optional): whether to use a regularization
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
                 n_steps_prepare_pos=None,  # TO DOC
                 n_steps_init_help=0,  # TO DOC
                 positioning_method="gaussian",
                 is_mlps=True,
                 hidden_size=32,
                 is_recursive=True,
                 rnn_cell='gru',
                 is_weight_norm_rnn=False,
                 is_building_blocks_mu=True,
                 is_bb_bias=False,  # TO DOC
                 is_content_attn=True,
                 is_sequential_attn=False,  # TO DOC. Only use fi Pondering
                 is_reg_bb_weights=False,
                 is_reg_const_weights=False,  # TO DOC
                 is_reg_old_weights=False,  # TO DOC
                 is_reg_clamp_mu=True,  # TO DOC OR SIMPLY ENFORCE
                 is_reg_round_weights=False,  # TO DOC
                 is_reg_variance_weights=False,  # TO DOC
                 is_l0_bb_weights=False,  # TO DOC
                 l0_mode="basic",  # TO DOC / CHOSE BEST
                 lp_reg_weights=1,
                 is_clamp_weights=True,  # TO DOC
                 rounder_weights_kwargs={},  # TO DOC
                 rounder_mu_kwargs={},  # TO DOC
                 bb_weights_annealed_noise_kwargs={},
                 bb_annealed_noise_kwargs={},
                 bb_const_annealed_noise_kwargs={},  # TO DOC
                 is_clamp_mu=True,
                 is_relative_sigma=True,
                 is_force_sigma=False,  # TO DOC
                 # with min_sigma=0.41 the max attention you can have is 0.9073 (and it's prime ;)
                 min_sigma=0.41,
                 # intial_sigma=5 chosen so that high sigma but up to length 50
                 # can still  have different attention => can learn
                 # (min when len = 50 : 1.2548e-21)
                 initial_sigma=5.0,
                 n_steps_interpolate_min_sigma=0,
                 is_dev_mode=False):
        super(PositionAttention, self).__init__()

        self.n_steps_prepare_pos = n_steps_prepare_pos
        self.n_steps_init_help = n_steps_init_help
        self.positioning_method = positioning_method
        self.is_weight_norm_rnn = is_weight_norm_rnn
        self.is_content_attn = is_content_attn
        self.is_sequential_attn = is_sequential_attn
        self.is_dev_mode = is_dev_mode
        self.is_reg_bb_weights = is_reg_bb_weights
        self.is_reg_const_weights = is_reg_const_weights
        self.is_reg_old_weights = is_reg_old_weights
        self.is_reg_clamp_mu = is_reg_clamp_mu
        self.is_reg_round_weights = is_reg_round_weights
        self.is_l0_bb_weights = is_l0_bb_weights
        self.lp_reg_weights = lp_reg_weights
        self.is_clamp_weights = is_clamp_weights
        self.is_reg_variance_weights = is_reg_variance_weights

        n_additional_mu_input = 9 - int(not self.is_content_attn)

        input_size = decoder_output_size + n_additional_mu_input
        self.is_recursive = is_recursive
        self.positioner = _get_positioner(self.positioning_method)
        self.is_force_sigma = is_force_sigma
        self.min_sigma = min_sigma
        self.hard_min_sigma = self.min_sigma / 1.5  # Max mu will be 0.9975
        self.initial_sigma = initial_sigma
        if self.n_steps_prepare_pos is None:
            self.get_sigma = HyperparameterInterpolator(self.initial_sigma,
                                                        self.min_sigma,
                                                        n_steps_interpolate_min_sigma,
                                                        mode="linear")
        else:
            self.get_sigma = HyperparameterInterpolator(self.initial_sigma,
                                                        self.min_sigma * 2,
                                                        self.n_steps_prepare_pos,
                                                        mode="linear")

        self.is_relative_sigma = is_relative_sigma
        self.max_len = max_len
        self.is_clamp_mu = is_clamp_mu

        # Building blocks
        self.is_building_blocks_mu = is_building_blocks_mu
        self.is_bb_bias = is_bb_bias
        self.single_step = torch.tensor(1. / (self.max_len - 1)).to(device)
        self.rel_counter = torch.arange(0, self.max_len,
                                        dtype=torch.float,
                                        device=device).unsqueeze(1) / (self.max_len - 1)

        self.bb_labels = ["mean_attn_old",
                          "rel_counter_decoder",
                          "single_step"]  # should use dictionnary instead

        if not self.is_sequential_attn:
            # if uses only one attention at a time
            # don't let the network see what other attention chose
            self.bb_labels += ["mu_old"]

            if self.is_content_attn:
                self.bb_labels += ["mean_content_old"]

        if self.is_bb_bias:
            self.bias = torch.tensor(1.0).to(device)
            self.bb_labels += ["bias"]

        n_building_blocks_mu = len(self.bb_labels) if self.is_building_blocks_mu else 1

        if self.is_recursive:
            self.rnn, self.hidden0 = get_rnn(rnn_cell, input_size, hidden_size,
                                             batch_first=True,
                                             is_weight_norm=self.is_weight_norm_rnn,
                                             is_get_hidden0=True)
            self.mu_weights_generator = nn.Linear(hidden_size,
                                                  n_building_blocks_mu)

            if not self.is_force_sigma:
                # If recursive don't use MLP for sigma
                self.sigma_generator = nn.Linear(hidden_size, 1)
        else:
            if is_mlps:
                self.mu_weights_generator = MLP(input_size,
                                                hidden_size,
                                                n_building_blocks_mu)

                if not self.is_force_sigma:
                    self.sigma_generator = MLP(input_size,
                                               hidden_size // 2,
                                               1)
            else:
                self.mu_weights_generator = nn.Linear(input_size,
                                                      n_building_blocks_mu)

                if not self.is_force_sigma:
                    self.sigma_generator = nn.Linear(input_size, 1)

        self.bb_noise = AnnealedGaussianNoise(**bb_annealed_noise_kwargs)
        self.bb_weights_noise = AnnealedGaussianNoise(is_relative_sigma=False,
                                                      **bb_weights_annealed_noise_kwargs)
        self.bb_const_noise = AnnealedGaussianNoise(**bb_const_annealed_noise_kwargs)

        self.rounder_weights = _get_rounder(**rounder_weights_kwargs)
        self.rounder_mu = _get_rounder(**rounder_mu_kwargs)

        if self.is_building_blocks_mu and self.is_l0_bb_weights:
            self.l0_mode = l0_mode
            if self.l0_mode == "basic":
                self.linear_l0_weights = L0Dense(n_building_blocks_mu, 1,
                                                 is_give_weights=True,
                                                 bias=False,
                                                 weight_decay=0.,
                                                 lamba=1.)
            elif self.l0_mode == "rounding":
                self.linear_l0_weights = L0Gates(hidden_size, n_building_blocks_mu,
                                                 is_at_least_1=True)
            else:
                raise ValueError("Unkown `l0_mode = {}`".format(l0_mode))

        """
        sigma0 = ((self.initial_sigma + self.min_sigma) / 2
                  if self.n_steps_prepare_pos is None else self.get_sigma.final_value)


        self.sigma_to_conf = ProbabilityConverter(activation="hard-sigmoid",
                                                  fix_point=(- self.min_sigma / 1.5, 1),
                                                  initial_x=-sigma0)
        """

        self.mean_attn_olds_factor = Parameter(torch.tensor(0.0))

        self.reset_parameters()

    def set_dev_mode(self, value=True):
        self.is_dev_mode = value

    def reset_parameters(self):
        """Reset and initialize the module parameters."""
        self.apply(weights_init)

        # could start at 0 if want to bias to start reading from the begining
        self.mu0 = Parameter(torch.tensor(0.5))

        sigma0 = ((self.initial_sigma + self.min_sigma) / 2
                  if self.n_steps_prepare_pos is None else self.get_sigma.final_value)
        self.sigma0 = Parameter(torch.tensor(sigma0))

        init_param(self.mean_attn_olds_factor, is_positive=True)

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
                                                              "is_reg_bb_weights",
                                                              "is_reg_const_weights",
                                                              "is_reg_old_weights"])

    def forward(self,
                decoder_outputs,
                source_lengths,
                step,
                mu_old,
                sigma_old,
                mean_content_old,
                mean_attn_old,
                mean_attn_olds,  # TO DOC
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
            mean_attn_old (torch.tensor): tensor of size (batch_size, n_steps)
                containing the mean mu across all previous time steps.
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
                                                                 mean_attn_old,
                                                                 mean_attn_olds,
                                                                 additional)

        mu, sigma = self._compute_parameters(positioning_inputs,
                                             building_blocks,
                                             step,
                                             source_lengths_tensor,
                                             additional)

        """
        # smaller sigma means more confident => - sigma
        # -sigma can only be negative but you still want confidence between 0
        # and 1 so need to shift to right => add only a positive bias
        # could use relu but for gradient flow use leaky relu
        pos_confidence = self.sigma_to_conf(-sigma)
        pos_confidence = pos_confidence.mean(dim=-1)
        #pos_confidence, _ = pos_attn.max(dim=-1)
        """

        # was hesitating between max pos_attn and linear_sigma to conf. The former
        # never went to 0 (so pos% always). The latter went abrubtly to 0 very
        # quickly so hard to get out. Decided to go with a middle ground
        min_p = 0.001
        pos_confidence = torch.exp(-sigma**2 + self.hard_min_sigma**2) * (1 - min_p)
        pos_confidence = pos_confidence.squeeze(-1)

        # relative sigma after sigma to conf because not fair that cannot be as confident
        if self.is_relative_sigma:
            sigma = renormalize_input_length(sigma, source_lengths_tensor, 1)

        rel_counter_encoder = renormalize_input_length(self.rel_counter.expand(batch_size, -1, 1),
                                                       source_lengths_tensor - 1,
                                                       self.max_len - 1)

        pos_attn = pad_sequence([self.positioner(rel_counter_encoder[b, :length, :],
                                                 mu[b].squeeze(),
                                                 sigma[b].squeeze())
                                 for b, length in enumerate(source_lengths_list)],
                                batch_first=True)

        # new size = (batch, n_queries, n_keys)
        pos_attn = pos_attn.transpose(2, 1)

        return pos_attn, pos_confidence, mu, sigma

    def _get_features(self, decoder_outputs, source_lengths_tensor, step, mu_old,
                      sigma_old, mean_content_old, mean_attn_old, mean_attn_olds,
                      additional):
        """Gets the inputs and the buillding blocks for positioning. Together
        those will eb used to compute the parameters of the positioning function.
        """
        batch_size = decoder_outputs.size(0)

        rel_counter_decoder = renormalize_input_length(self.rel_counter[step:step + 1
                                                                        ].expand(batch_size, 1),
                                                       source_lengths_tensor - 1,
                                                       self.max_len - 1)
        abs_counter_decoder = (self.rel_counter[step:step + 1].expand(batch_size, 1) *
                               (self.max_len - 1))

        if step == 0:
            mu_old = self.mu0.expand(batch_size, 1)
            sigma_old = self.sigma0.expand(batch_size, 1)
            mean_attn_olds = mu_old
        else:
            mu_old = mu_old.squeeze(2)
            sigma_old = sigma_old.squeeze(2)
            mean_attn_olds_factor = torch.relu(self.mean_attn_olds_factor)
            mean_attn_olds = (mean_attn_olds.squeeze(2) * step * (1 - mean_attn_olds_factor) +
                              mean_attn_old * mean_attn_olds_factor) / (step + 1)
        additional["mean_attn_olds"] = mean_attn_olds.unsqueeze(2)

        single_step = renormalize_input_length(self.single_step.expand(batch_size, 1),
                                               source_lengths_tensor - 1,
                                               self.max_len - 1)

        dict_features = dict(mean_attn_old=mean_attn_old,
                             rel_counter_decoder=rel_counter_decoder,
                             abs_counter_decoder=abs_counter_decoder,
                             sigma_old=sigma_old,
                             mu_old=mu_old,
                             single_step=single_step,
                             source_lengths=source_lengths_tensor.unsqueeze(-1),
                             bias=self.bias.expand(batch_size, 1),
                             mean_content_old=mean_content_old,
                             mean_attn_olds=mean_attn_olds)

        # next line needed for python < 3.6 . for higher can use
        # list(dict_mu_weights.values())
        ordered_blocks = [dict_features[l] for l in self.bb_labels]
        building_blocks = torch.cat(ordered_blocks, dim=1)

        not_shared = ["sigma_old", "abs_counter_decoder", "source_lengths",
                      "mu_old", "mean_attn_olds"] + (["mean_content_old"]
                                                     if self.is_content_attn else [])
        pos_features_labels = set(l for l in (self.bb_labels + not_shared) if l != "bias")
        additional_pos_features = [dict_features[l] for l in pos_features_labels]

        positioning_inputs = torch.cat([decoder_outputs.squeeze(1)] + additional_pos_features,
                                       dim=1)

        return positioning_inputs, building_blocks

    def _compute_parameters(self, positioning_inputs, building_blocks, step,
                            source_lengths_tensor, additional):
        """Compute the parameters of the positioning function."""
        batch_size = positioning_inputs.size(0)

        if self.is_recursive:
            if step == 0:
                additional["positioner_hidden"] = replicate_hidden0(self.hidden0,
                                                                    batch_size)
            positioning_outputs, positioner_hidden = self.rnn(positioning_inputs.unsqueeze(1),
                                                              additional['positioner_hidden'])
            positioning_outputs = positioning_outputs.squeeze(1)
            additional['positioner_hidden'] = positioner_hidden
        else:
            positioning_outputs = positioning_inputs

        mu_weights = self.mu_weights_generator(positioning_outputs)
        # ["mu_old", "mean_attn_old", "rel_counter_decoder", "single_step"]
        # ["mean_content_old"] ["bias"]

        add_to_test(mu_weights, 'raw_mu_weights', additional, self.is_dev_mode)

        if self.is_building_blocks_mu:
            bb_labels_old = [l for l in ["mu_old", "mean_attn_old", "mean_content_old"]
                             if l in self.bb_labels]

            # REGULARIZATION
            if self.is_l0_bb_weights and self.l0_mode == "rounding":
                gates, loss = self.linear_l0_weights(positioning_outputs)
                if "losses" in additional:
                    add_regularization(loss, "pos_l0_weights", additional)

                add_to_test(gates, "bb_gates", additional, self.is_dev_mode)

            if "losses" in additional and self.is_reg_round_weights:
                # used to round mu weight without forcing
                loss = batch_reduction_f(torch.abs(mu_weights -
                                                   mu_weights.detach().round()),
                                         torch.mean)
                add_regularization(loss, "pos_round_weights", additional)

            if "losses" in additional and self.is_reg_bb_weights:
                # used because the building blocks are highly dependant
                # but maybe not important now that rounds + clamp
                loss = batch_reduction_f(regularization_loss(mu_weights,
                                                             p=self.lp_reg_weights,
                                                             dim=-1),
                                         torch.mean)
                add_regularization(loss, "pos_mu_weights", additional)

            if "losses" in additional and self.is_reg_const_weights:
                # regularizes the constant values that could be used by the network
                # to bypass the other buidling blocks by having the weights = mu

                # no need of regularizing the bias weight when it's rounded
                add_bias = (self.is_bb_bias and
                            self.rounder_weights is None and
                            not self.is_reg_round_weights)
                reg_labels_const = ["single_step"] + (["bias"] if add_bias else [])
                w_idcs_const = get_indices(self.bb_labels, reg_labels_const)

                loss = batch_reduction_f(regularization_loss(mu_weights[:, w_idcs_const],
                                                             p=self.lp_reg_weights,
                                                             dim=-1),
                                         torch.mean)
                add_regularization(loss, "pos_const_weights", additional)

            if "losses" in additional and self.is_reg_old_weights:
                # regularizes the weights of the building blocks that are not stable
                # yet (because they depend on positioning attention)
                idcs_pos_old = get_indices(self.bb_labels, ["mu_old", "mean_attn_old"])

                loss = batch_reduction_f(regularization_loss(mu_weights[:, idcs_pos_old],
                                                             p=self.lp_reg_weights,
                                                             dim=-1),
                                         torch.mean)
                add_regularization(loss, "pos_old_weights", additional)

            # TRANFORM
            # noising
            building_blocks = self.bb_noise(building_blocks, is_update=(step == 0))
            mu_weights = self.bb_weights_noise(mu_weights, is_update=(step == 0))

            dict_mu_weights = dict(zip(self.bb_labels, mu_weights.unbind(-1)))

            if self.bb_const_noise.sigma != 0:
                add_bias = (self.is_bb_bias and
                            self.rounder_weights is None and
                            not self.is_reg_round_weights)
                noise_labels_const = ["single_step"] + (["bias"] if add_bias else [])
                for i, l in enumerate(noise_labels_const):
                    dict_mu_weights[l] = self.bb_const_noise(dict_mu_weights[l],
                                                             is_update=(step == 0 and i == 0))

            # initialization helper
            if self.training and additional["training_step"] < self.n_steps_init_help:
                # adds either 0.5 or -0.5
                dict_mu_weights["rel_counter_decoder"] = (dict_mu_weights["rel_counter_decoder"] +
                                                          0.5 - (additional["training_step"] % 2))

            # clamping
            if self.is_clamp_weights:
                for l in bb_labels_old + (["bias"] if self.is_bb_bias else []):
                    dict_mu_weights[l] = clamp(dict_mu_weights[l],
                                               minimum=0., maximum=1., is_leaky=True)

                dict_mu_weights["rel_counter_decoder"] = clamp(dict_mu_weights["rel_counter_decoder"],
                                                               minimum=-1.,
                                                               maximum=1.,
                                                               is_leaky=True)

            # rounding
            if self.rounder_weights is not None:
                for i, l in enumerate(bb_labels_old + ["single_step",
                                                       "rel_counter_decoder"]):
                    dict_mu_weights[l] = self.rounder_weights(dict_mu_weights[l],
                                                              is_update=(step == 0 and i == 0))

                if self.is_bb_bias:
                    # rounds up to 0.25. IS IT NECESSARY ?????????????????
                    dict_mu_weights["bias"] = self.rounder_weights(dict_mu_weights["bias"] * 2.,
                                                                   is_update=False) / 2.

            # next line needed for python < 3.6 . for higher can use
            # list(dict_mu_weights.values())
            ordered_weights = [dict_mu_weights[l] for l in self.bb_labels]
            mu_weights = torch.stack(ordered_weights, dim=-1)

            if "losses" in additional and self.is_reg_variance_weights:
                # forces the weights to always be relatively similar
                # after rounding
                if step != 0:
                    loss = batch_reduction_f(regularization_loss(mu_weights -
                                                                 additional["mu_weights"],
                                                                 p=0.5,
                                                                 dim=-1),
                                             torch.mean)
                    add_regularization(loss, "pos_variance_weights", additional)

                additional["mu_weights"] = mu_weights

            # IF WANT TO PLOT AFTER ROUNDING
            add_to_test([mu_weights, building_blocks],
                        ['mu_weights', 'building_blocks'],
                        additional, self.is_dev_mode)

            add_to_visualize([mu_weights, building_blocks],
                             ['mu_weights', 'building_blocks'],
                             additional)

            if self.is_l0_bb_weights:
                if self.l0_mode == "basic":
                    # same as bm in the else statement but samples which parameters
                    # can use first
                    self.linear_l0_weights.set_weights(mu_weights.unsqueeze(2))
                    mu = self.linear_l0_weights(building_blocks.unsqueeze(1))
                    if "losses" in additional:
                        loss = self.linear_l0_weights.regularization()
                        # IS OUTPUT BATCHWISE ?
                        add_regularization(loss, "pos_l0_weights", additional)

                elif self.l0_mode == "rounding":
                    mu = torch.bmm((mu_weights * gates).unsqueeze(1),
                                   building_blocks.unsqueeze(2))
            else:
                # (batch, 1, 5) * (batch, 5, 1) -> (batch, 1, 1)
                mu = torch.bmm(mu_weights.unsqueeze(1), building_blocks.unsqueeze(2))

            if self.rounder_mu is not None:
                # rounding to words
                normalizer = (source_lengths_tensor - 1).unsqueeze(1).unsqueeze(1)
                mu = self.rounder_mu(mu * normalizer, is_update=(step == 0)
                                     ) / normalizer
        else:
            mu = torch.sigmoid(mu_weights.unsqueeze(1))

        if self.is_clamp_mu:
            mu_old = mu
            mu = clamp(mu, minimum=0, maximum=1, is_leaky=True)
            if "losses" in additional and self.is_reg_clamp_mu:
                loss = batch_reduction_f(mu - mu_old,
                                         torch.norm,
                                         p=2)
                add_regularization(loss, "pos_clamp_mu", additional)

        is_update_sigma = self.training and step == 0

        if self.is_force_sigma:
            sigma = torch.zeros_like(mu) + self.get_sigma(is_update_sigma)

        else:
            if self.n_steps_prepare_pos is None:
                # KEEPING FOR TESTING OLD STYLE !!!!!!!!!!!!
                sigma = self.sigma_generator(positioning_outputs)
                sigma = self.min_sigma + sigma.unsqueeze(1)
                sigma = torch.max(sigma,
                                  torch.zeros_like(sigma) + self.get_sigma(is_update_sigma))

            else:
                if self.get_sigma.is_annealing:
                    current_min_sigma = self.get_sigma(is_update_sigma)

                    # if you are still annealing min sigma then don't backprop
                    # to sigma generator
                    sigma = current_min_sigma + torch.zeros_like(mu)
                else:
                    unclamped_sigma = (self.get_sigma.final_value +
                                       self.sigma_generator(positioning_outputs))
                    sigma = clamp(unclamped_sigma.unsqueeze(1),
                                  minimum=self.min_sigma,
                                  is_leaky=True,
                                  negative_slope=0.1,
                                  hard_min=self.hard_min_sigma)

        return mu, sigma


class AttentionMixer(nn.Module):
    """Mixes content and positional attention.

    Args:
        decoder_output_size (int): size of the hidden activations of the decoder.
        hidden_size (int, optional): number of hidden neurons in the MLP.
        is_mlps (bool, optional): whether to use MLP's instead of linear
            function for the weight generators.
        mode ({"generated","normalized_pos_conf","pos_conf"}, optional) mode of
            the attention mixer. `generated` will generate one from the controller,
            this might give good results but is less interpretable. `mean_conf`
            will normalize the positional confidence by `(position_confidence
            + content_confidence)`, this will force meaningfull confidences for
            both attentions. The latter should not be used when not using sequential
            attention because pos% will always be 0.5 if both are confident, i.e
            content cannot just be used for position to help it.`pos_conf` will
            directly use the position cofidence, this will force meaningfull
            positioning confidence but not the content ones. This also says
            to the network that if position is confident use it regardless of content
            because it's more extrapolable.
    """

    def __init__(self, decoder_output_size,
                 hidden_size=32,
                 is_mlps=True,
                 mode="normalized_pos_conf",
                 is_dev_mode=False,
                 n_steps_wait=0,  # TO DOC
                 is_reg_pos_perc=False,  # TO DOC
                 rounder_perc_kwargs={}):    # TO DOC
        super(AttentionMixer, self).__init__()

        self.is_dev_mode = is_dev_mode
        self.mode = mode.lower()
        self.n_steps_wait = n_steps_wait
        self.is_reg_pos_perc = is_reg_pos_perc
        self.rounder_perc = _get_rounder(**rounder_perc_kwargs)

        n_additional_pos_perc_inputs = 3

        # should be under if not is_predict_conf (i.e net else)
        # but keeping while testing `additional_controller_features`
        self.position_perc0 = Parameter(torch.tensor(0.5))

        if self.mode == "generated":
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
                              always_shows=["mode"])

    def forward(self,
                decoder_output,
                step,
                content_attn,
                content_confidence,
                pos_attn,
                pos_confidence,
                position_perc_old,
                additional):
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
            additional (dictionary): dictionary containing additional variables
                that are necessary for some hyperparamets.
        """
        batch_size = decoder_output.size(0)

        if not self.training or additional["training_step"] >= self.n_steps_wait:
            if self.mode == "pos_conf":
                position_perc = pos_confidence
            elif self.mode == "normalized_pos_conf":
                position_perc = pos_confidence / (pos_confidence + content_confidence)
            elif self.mode == "generated":
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
            else:
                raise ValueError("Unkown mode={}".format(self.mode))
        else:
            position_perc = torch.tensor(0.5).to(device).expand(batch_size, 1)

        if self.rounder_perc is not None:
            position_perc = self.rounder_perc(position_perc)

        add_to_test(position_perc, "position_percentage", additional, self.is_dev_mode)

        if "losses" in additional:
            if self.is_reg_pos_perc:
                # if can solve with positioning pleas do
                loss = 1 - batch_reduction_f(position_perc, torch.mean)
                add_regularization(loss, "pos%", additional)

            self._rescale_losses(additional["losses"], position_perc)
            additional["losses"]["balancing"] = -self._balance_losses(additional["losses"],
                                                                      position_perc)

        # COnvex combination
        attn = (pos_attn * position_perc.unsqueeze(-1) +
                (1 - position_perc.unsqueeze(-1)) * content_attn)

        return attn, position_perc

    def _rescale_losses(self, losses, position_perc):
        """
        Rescale the content and positional regularization such that they are
        proportional to our use of them.
        """
        # don't broadcast multiplication : want vector output
        position_perc = position_perc.view(-1)

        for name in losses.keys():
            if name.startswith("pos_"):
                losses[name] = losses[name] * position_perc
            elif name.startswith("cont_"):
                losses[name] = losses[name] * (1 - position_perc)

    def _balance_losses(self, losses, position_perc):
        """
        Adds / Remove some pos_perc loss in order to compensate the regularization
        that has been added for the positional or content attention. I.e
        the positional or content regularization should only help for convergence of
        one of the attentions and should not push the network to use one type of
        attention.
        """
        # don't broadcast multiplication : want vector output
        position_perc = position_perc.view(-1)

        diff_pos_cont_loss = 0
        for name, loss in losses.items():
            if name.startswith("pos_"):
                diff_pos_cont_loss += loss.detach()
            elif name.startswith("cont_"):
                diff_pos_cont_loss -= loss.detach()

        # this represents how much the network will be penalized by using
        # psoitional attn (can be negative)
        penalty_use_pos = diff_pos_cont_loss * position_perc
        return penalty_use_pos
