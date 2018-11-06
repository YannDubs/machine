"""
Positioning attention.

TO DO:
- remove all the dependencies on `additional`. By removing the number of possible
    hyperparameters it will be a lot simple to simply write down the functions
    with specific inputs / outputs without dependencies on `additional`.
- many parameters that I keep for dev mode / comparasion here: you definitely
    don't have to accept all of these when refactoring. In case you are not sure
    which ones to keep : just ask me (but the ones that I know we should remove
    for sure are noted.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence

from seq2seq.util.helpers import (renormalize_input_length, get_rnn,
                                  HyperparameterInterpolator, get_extra_repr,
                                  clamp, format_source_lengths, Rate2Steps,
                                  get_indices, Clamper, regularization_loss,
                                  batch_reduction_f,  HyperparameterCurriculumInterpolator,
                                  add_regularization)
from seq2seq.util.torchextend import (MLP, StochasticRounding, ConcreteRounding,
                                      ProbabilityConverter, AnnealedGaussianNoise,
                                      L0Gates)
from seq2seq.util.initialization import replicate_hidden0, init_param, weights_init
from seq2seq.util.base import Module
from seq2seq.util.l0 import L0Dense

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_regularizers_positioner(total_training_calls, n_steps_prepare_pos=None, is_new_l0=False):
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

    if is_new_l0:
        # DEV MODE
        n_steps_interpolate = rate2steps(0.3)
        start_step = rate2steps(0.)
        max_p_interpolators["pos_l0_weights"
                            ] = HyperparameterInterpolator(0, 5e-3, n_steps_interpolate,
                                                           start_step=start_step,
                                                           default=0,
                                                           mode="linear")
        print("pos_l0_weights:", max_p_interpolators["pos_l0_weights"].extra_repr())
    else:
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
                        ] = HyperparameterInterpolator(5e-2, 1e-2, n_steps_interpolate,
                                                       start_step=start_step,
                                                       default=0,
                                                       mode="linear")
    print("pos%:", max_p_interpolators["pos%"].extra_repr())
    print()

    return max_p_interpolators

"""
def get_regularizers_positioner(total_training_calls,
                                n_steps_prepare_pos=None,
                                is_new_l0=False):
    def _initialize_regularizer(name, curriculum, **kwargs):
        max_p_interpolators[name] = HyperparameterCurriculumInterpolator(curriculum, **kwargs)

    rate2steps = Rate2Steps(total_training_calls)
    max_p_interpolators = dict()
    n_steps_prepare_pos = rate2steps(0.05)

    _initialize_regularizer("pos_const_weights",
                            [dict(step=n_steps_prepare_pos, value=5e-2, mode="geometric"),
                             dict(step=int(n_steps_prepare_pos * 3 / 2), value=0, mode="geometric"),
                             dict(step=n_steps_prepare_pos * 2, value=5e-3)])

    _initialize_regularizer("pos_old_weights",
                            [dict(step=0, value=5e-2, mode="linear"),
                             dict(step=n_steps_prepare_pos, value=0)])

    _initialize_regularizer("pos_clamp_weights",
                            [dict(step=0, value=5e-3, mode="linear"),
                             dict(step=n_steps_prepare_pos, value=5e-2)])

    ### DEV MODE ###
    if is_new_l0:
        _initialize_regularizer("pos_l0_weights",
                                [dict(step=n_steps_prepare_pos, value=1e-2, mode="geometric"),
                                 dict(step=n_steps_prepare_pos * 2, value=1e-3)])
    else:
        _initialize_regularizer("pos_l0_weights",
                                [dict(step=0, value=0),
                                 dict(step=n_steps_prepare_pos, value=3e-2, mode="geometric"),
                                 dict(step=n_steps_prepare_pos + rate2steps(0.3), value=5e-3)])

    _initialize_regularizer("pos_variance_weights",
                            [dict(step=n_steps_prepare_pos, value=5e-3, mode="geometric"),
                             dict(step=n_steps_prepare_pos * 2, value=0)])

    _initialize_regularizer("pos_clamp_mu",
                            [dict(step=0, value=5e-3, mode="linear"),
                             dict(step=n_steps_prepare_pos, value=5e-2)])

    # don't a name starting with `pos_%` because later I look for losses
    # starting with `pos_` and it means it only apply to positioing attention
    # while this ones applies to mixing the attention
    _initialize_regularizer("pos%",
                            [dict(step=0, value=5e-3, mode="geometric"),
                             dict(step=n_steps_prepare_pos, value=5e-2, mode="linear"),
                             dict(step=n_steps_prepare_pos * 2, value=5e-3)])

    return max_p_interpolators
"""


def _discrete_truncated_gaussian(x, mu, sigma):
    """Return normalized Gaussian_pdf(x)."""
    x=torch.exp(-(x - mu)**2 / (2 * sigma**2))
    x=F.normalize(x, p = 1, dim = 0)
    return x


def _discrete_truncated_laplace(x, mu, b):
    """Return normalized Laplacian_pdf(x)."""
    x=torch.exp(-1 * torch.abs((x - mu) / b))
    x=F.normalize(x, p = 1, dim = 0)
    return x


def _get_positioner(name):
    """Get the correct positioner method."""
    if name == "gaussian":
        return _discrete_truncated_gaussian
    elif name == "laplace":
        return _discrete_truncated_laplace
    else:
        raise ValueError("Unkown positioner method {}".format(name))


def _get_rounder(name = None, **kwargs):
    if name is None:
        return None
    elif name == "concrete":
        return ConcreteRounding(**kwargs)
    elif name == "stochastic":
        return StochasticRounding(**kwargs)
    else:
        raise ValueError("Unkown rounder method {}".format(name))


class PositionAttention(Module):
    """Position Attention Generator.

    Args:

        decoder_output_size (int): size of the hidden activations of the decoder.
        max_len (int): a maximum allowed length for the sequence to be processed
        n_steps_prepare_pos (int, optional): number of steps during which to consider
            the positioning as in a preparation mode. During preparation mode,
            the model have less parameters to tweak, it will focus on what I thought
            were the most crucial bits. For example it will have a fix
            sigma and won't have many of the regularization term, this is to
            help it start at a decent place in a lower dimensional space, before
            going to the hard task of tweaking all at the same time.
        n_steps_init_help (int, optional): number of training steps for which to
            us an initializer helper for the position attention. Currently the helper
            consists of alternating between values of 0.5 and -0.5 for the
            "rel_counter_decoder" weights.
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
        is_bb_bias (bool, optional): adding a bias term to the building blocks.
            THis has the advantage of letting the network go to absolut positions
            (ex: middle, end, ...). THe disadvantage being that the model will often
            stop using other building blocks and thus be less general.
        is_content_attn (bool, optional): whether you are using content attention.
        is_sequential_attn (bool, optional): whether to force the network to only
            look at content or position at each step.  Although this is desirable
            in the long run (first you look for content then positioning not both
            at the same time), it makes the model a lot less powerful until we
            start using pondering (i.e be able to look for something without
            outputing anything, just like humans would).
        is_reg_bb_weights (bool, optional): whether to use a regularization
            on the postional attention mu's building block weights. This can
            be usefull if the building blocks are gihly correlyted.
        is_reg_const_weights (bool, optional): whether to use a lp regularization
            on the constant position mu building block. This can be usefull in
            otrder to push the network to use non constant building blocks that are
            more extrapolable (i.e with constants, the network has to make varying
            weights which is not interpretable. If the blocks ae varying then
            the "hard" extrapolable output would already be done for the network).
        is_reg_old_weights (bool, optional): whether to use a lp norm regularisation
            on the building blocks that depend on previous positioning attention.
            This can be useful as these building blocks cannot be used correctly
            before positioning attention actually converged.
        is_reg_clamp_mu (bool, optional): whether to regularise with lp norm the
            clamping of mu. I.e push the network to not overshoot and really
            generate the desired mu rather than the clamped one. This can be
            useful as if the mu completely overshoots it will be hard for it to
            come back to normal values if it needs to. It also makes sense to
            output what you want rather than relying on postpropressing.
        is_reg_round_weighs (bool, optional): whether to regularise with lp norm
            the building block weights in order to push them towards integers.
            This is the soft version of `rounder_weights`.
        is_reg_variance_weights (bool, optional): whether to use lp norm
            regularisation to force the building blocks to have low variance
            across time steps. This can be useful as it forces the model to use
            simpler weight patterns that are more extrapolable. For example it
            would prefer giving a weight of `1` to `block_j/n`than using a weight
            of `j` to `block_1/n`.
        is_l0_bb_weights (bool, optional): whether to use l0 regularisation on
            the building block weights. This is achieved by reparametrizing the
            l0 loss as done in “Learning Sparse Neural Network through L_0
            Regularisation”.
        l0_mode ({“basic”, “rounding”}, optional): what type of l0 regularisation
            to use. If `basic` it will follow the method used in “Learning Sparse
            Neural Network through L_0 Regularisation”. If `rounding` it will
            round the gates in the forward pass, thus using gates that are either
            0 or 1 during the forward pass but using the approximative binary
            gates during the backward pass.
        lp_reg_weights (bool, optional): the p in the lp norm to use for all the
            regularisation above. p can be in [0,”inf”]. If `p=0` will use some
            approximation to the l0 norm. `is_l0_bb_weights` is preferred over`p=0`
            with `is_reg_bb_weights`.
        is_clamp_weights (bool, optional): whether to clamp the building block
            weights on some meaningful intervals.
         rounder_weights_kwargs (dictionary, optional): additional arguments to the
            rounder weights. Rounding is desirable to make the output more
            interpretable and extrapolable (as the building blocks were designed
            such that integer wights could be used to solve most positonal patterns).
         rounder_mu_kwargs (dictionary, optional): additional arguments to the
            rounder mu. Rounding is desirable to make the position attention
            look at the correct position even for sentences longer than it have
            ever seen.
         bb_weights_annealed_noise_kwargs (dictionary, optional): additional arguments
            to the the annealed noise to apply to the building block weights. This
            can be seen as a softer version of `is_reg_bb_weights`.
         bb_annealed_noise_kwargs (dictionary, optional): additional arguments
            to the the annealed noise to apply to the values of the building block.
            This can be seen as a softer version of `is_reg_bb_weights`.
         bb_const_annealed_noise_kwargs (dictionary, optional): additional arguments
            to apply to the value of the constant building blocks. This can be seen as
            a softer version of `is_reg_const_weights`.
        is_clamp_mu (bool, optional): whether to clamp the positioning `mu` in a
            range ~[0,1] using a leaky ReLu.
        is_relative_sigma (bool, optional): whether to use a relative varaince,
            i.e normalized by encoder steps.
        is_force_sigma (bool, optional): whether to us the annealed sigma instead
            of learning the value.
        is_learn_sigma_to_conf (bool, optional): whether to use a trainable
            sigma_to_conf converter function. If `True` will use a reverse sigmoid
            with learnable temperature.
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
    """

    def __init__(self, decoder_output_size, max_len,
                 n_steps_prepare_pos=None,
                 n_steps_init_help=0,
                 positioning_method="gaussian",
                 is_mlps=True,
                 hidden_size=32,
                 is_recursive=True,
                 rnn_cell='gru',
                 is_weight_norm_rnn=False,  # TO DO - medium: chose best and remove parameter
                  is_building_blocks_mu=True,  # TO DO: remove this parameter (i.e force True)
                  is_bb_bias=True,  # TO DO: remove this parameter (i.e force True)
                 is_content_attn=True,
                 is_sequential_attn=False,
                 is_reg_bb_weights=False,  # TO DO: remove this parameter (i.e force True)
                 is_reg_const_weights=False,  # TO DO - medium: chose best and remove parameter
                 is_reg_old_weights=False,  # TO DO - medium: chose best and remove parameter
                 is_reg_clamp_mu=True,  # TO DO - medium: chose best and remove parameter
                 is_reg_round_weights=False,  # TO DO: remove this parameter (i.e force True)
                 is_reg_variance_weights=False,  # TO DO - medium: chose best and remove parameter
                 is_l0_bb_weights=False,
                 l0_mode="rounding",  # TO DO: remove this parameter (i.e force rounding)
                 lp_reg_weights=1,  # TO DO - medium: chose best and remove parameter
                 is_clamp_weights=True,  # TO DO - medium: chose best and remove parameter
                 rounder_weights_kwargs={},
                 rounder_mu_kwargs={},
                 bb_weights_annealed_noise_kwargs={}, # TO DO: remove this parameter if not useful
                 bb_annealed_noise_kwargs={},  # TO DO: remove this parameter if not useful
                 bb_const_annealed_noise_kwargs={},  # TO DO: remove this parameter if not useful
                 is_clamp_mu=True,  # TO DO: remove this parameter (i.e force True)
                 is_relative_sigma=True,  # TO DO: remove this parameter (i.e force True)
                 is_force_sigma=False,  # TO DO: remove this parameter (i.e force True)
                 is_learn_sigma_to_conf=False,  # TO DO: remove this parameter (i.e force True)
                 # with min_sigma=0.41 the max attention you can have is 0.9073 (and it's prime ;)
                 min_sigma=0.41,
                 # intial_sigma=5 chosen so that high sigma but up to length 50
                 # can still  have different attention => can learn
                 # (min when len = 50 : 1.2548e-21)
                 initial_sigma=5.0,
                 n_steps_interpolate_min_sigma=0):
        super(PositionAttention, self).__init__()

        self.n_steps_prepare_pos = n_steps_prepare_pos
        self.n_steps_init_help = n_steps_init_help
        self.positioning_method = positioning_method
        self.is_weight_norm_rnn = is_weight_norm_rnn
        self.is_content_attn = is_content_attn
        self.is_sequential_attn = is_sequential_attn
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
        self.is_learn_sigma_to_conf = is_learn_sigma_to_conf
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
                if self.n_steps_prepare_pos is not None:
                    # should always be given not only with self.n_steps_prepare_pos
                    # but the positioner doesn't know the the number of training
                    # steps
                    rounding_kwargs = dict(n_steps_interpolate=self.n_steps_prepare_pos)
                else:
                    rounding_kwargs = dict()
                self.linear_l0_weights = L0Gates(hidden_size, n_building_blocks_mu,
                                                 is_at_least_1=True,
                                                 rounding_kwargs=rounding_kwargs)
            else:
                raise ValueError("Unkown `l0_mode = {}`".format(l0_mode))

        if self.is_learn_sigma_to_conf:
            sigma0 = ((self.initial_sigma + self.min_sigma) / 2
                      if self.n_steps_prepare_pos is None else self.get_sigma.final_value)

            self.sigma_to_conf = ProbabilityConverter(activation="hard-sigmoid",
                                                      fix_point=(- self.min_sigma / 1.5, 1),
                                                      initial_x=-sigma0,
                                                      is_temperature=True)

        self.mean_attn_olds_factor = Parameter(torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self):
        """Reset and initialize the module parameters."""
        super().reset_parameters()

        self.get_sigma.reset_parameters()

        # could start at 0 if want to bias to start reading from the begining
        self.mu0 = Parameter(torch.tensor(0.5))

        sigma0 = ((self.initial_sigma + self.min_sigma) / 2
                   if self.n_steps_prepare_pos is None else self.get_sigma.final_value)
        self.sigma0 = Parameter(torch.tensor(sigma0))

        init_param(self.mean_attn_olds_factor, is_positive=True)

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
                mean_attn_olds,
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
            mean_attn_olds (torch.tensor): tensor of size (batch_size, n_steps)
                containing the (dscounted) mean mu across all previous time steps.
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

        if self.is_learn_sigma_to_conf:
            # smaller sigma means more confident => - sigma
            # -sigma can only be negative but you still want confidence between 0
            # and 1 so need to shift to right => add only a positive bias
            # could use relu but for gradient flow use leaky relu
            pos_confidence = self.sigma_to_conf(-sigma)
            pos_confidence = pos_confidence.mean(dim=-1)
            # pos_confidence, _ = pos_attn.max(dim=-1)
        else:
            # was hesitating between max pos_attn and linear_sigma to conf. The former
            # never went to 0 (so pos% always). The latter went abrubtly to 0 very
            # quickly so hard to get out. Decided to go with a middle ground
            min_p = 0.001
            pos_confidence = torch.exp(-sigma**2 + self.hard_min_sigma**2) * (1 - min_p)
            pos_confidence = pos_confidence.squeeze(-1)

        self.add_to_visualize([mu, sigma, pos_confidence],
                         ["mu", "sigma", "pos_confidence"])

        self.add_to_test([mu, sigma], ["mu", "sigma"])

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

        self.add_to_test(mu_weights, 'raw_mu_weights')

        if self.is_building_blocks_mu:
            bb_labels_old = [l for l in ["mu_old", "mean_attn_old", "mean_content_old"]
                             if l in self.bb_labels]

            IS_FORCING = False # Dev Mode
            if IS_FORCING:
                mu_weights = mu_weights / 10
                mu_weights[:, self.bb_labels.index("bias")] = mu_weights[:,self.bb_labels.index("bias")] + 0.7
                mu_weights[:, self.bb_labels.index("rel_counter_decoder")] = mu_weights[:,self.bb_labels.index("rel_counter_decoder")] - 0.7

            # REGULARIZATION
            if self.is_l0_bb_weights and self.l0_mode == "rounding":
                gates, loss = self.linear_l0_weights(positioning_outputs)
                if self.is_regularize:
                    self.add_regularization_loss("pos_l0_weights", loss)

                self.add_to_test(gates, "bb_gates")

            if self.is_regularize and self.is_reg_round_weights:
                # used to round mu weight without forcing
                loss = batch_reduction_f(torch.abs(mu_weights -
                                                   mu_weights.detach().round()),
                                         torch.mean)
                self.add_regularization_loss("pos_round_weights", loss)

            if self.is_regularize and self.is_reg_bb_weights:
                # used because the building blocks are highly dependant
                # but maybe not important now that rounds + clamp
                loss = batch_reduction_f(regularization_loss(mu_weights,
                                                             p=self.lp_reg_weights,
                                                             dim=-1),
                                         torch.mean)
                self.add_regularization_loss("pos_mu_weights", loss)

            if self.is_regularize and self.is_reg_const_weights:
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
                self.add_regularization_loss("pos_const_weights", loss)

            if self.is_regularize and self.is_reg_old_weights:
                # regularizes the weights of the building blocks that are not stable
                # yet (because they depend on positioning attention)
                idcs_pos_old = get_indices(self.bb_labels, ["mu_old", "mean_attn_old"])

                loss = batch_reduction_f(regularization_loss(mu_weights[:, idcs_pos_old],
                                                             p=self.lp_reg_weights,
                                                             dim=-1),
                                         torch.mean)
                self.add_regularization_loss("pos_old_weights", loss)

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
            if self.training and self.n_training_calls < self.n_steps_init_help:
                interpolating_factor = (self.n_training_calls-1)/self.n_steps_init_help

                def interpolate_help(start, end):
                    delta = (end-start)*interpolating_factor
                    return start+delta

                # adds either 0.75 or -0.75
                IS_ALTERNATE = False
                if IS_ALTERNATE:
                    dict_mu_weights["rel_counter_decoder"
                                    ] = (dict_mu_weights["rel_counter_decoder" ] *
                                         interpolate_help(0.25, 1) +
                                         interpolate_help((0.5 -
                                                          (self.n_training_calls % 2))*3/2,
                                                         0))
                    if self.is_bb_bias:
                        # adds either 0.125 or 0.875
                        dict_mu_weights["bias"
                                        ] = (dict_mu_weights["bias"] *
                                             interpolate_help(0.125, 1) +
                                             interpolate_help(0.5 +
                                                              (0.5 - self.n_training_calls%2)* 0.75,
                                                              0))
                else:
                    dict_mu_weights["rel_counter_decoder"
                                    ] = (dict_mu_weights["rel_counter_decoder"]*
                                         interpolate_help(0.25, 1) -
                                         interpolate_help(3/4, 0))
                    if self.is_bb_bias:
                        # adds either 0.125 or 0.875
                        dict_mu_weights["bias"
                                        ] = (dict_mu_weights["bias"] *
                                             interpolate_help(0.125, 1) +
                                             interpolate_help(0.5 + 3/8, 0))

                for l in self.bb_labels:
                    if l not in ["rel_counter_decoder", "bias"]:
                        dict_mu_weights[l] = dict_mu_weights[l] * interpolate_help(0.1, 1)

                # DEV MODE
                if self.is_l0_bb_weights:
                    new_gates = torch.zeros_like(gates)
                    new_gates[:, self.bb_labels.index("bias")] = 1.
                    new_gates[:, self.bb_labels.index("rel_counter_decoder")] = 1.

                    gates = new_gates * interpolate_help(0.5,0) + gates * interpolate_help(0.5,1)

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
                    # rounds up to 0.5. IS IT NECESSARY ?????????????????
                    dict_mu_weights["bias"] = self.rounder_weights(dict_mu_weights["bias"] * 2.,
                                                                   is_update=False) / 2.

            # next line needed for python < 3.6 . for higher can use
            # list(dict_mu_weights.values())
            ordered_weights = [dict_mu_weights[l] for l in self.bb_labels]
            mu_weights = torch.stack(ordered_weights, dim=-1)

            # DEV MODE / PRINT MODE !!!!!!!!
            PRINT_MODE=False
            if PRINT_MODE and self.training and step == 0 and self.n_training_calls % 100 == 0:
                for l in self.bb_labels:
                    print(l, dict_mu_weights[l])
                print()

            if self.is_regularize and self.is_reg_variance_weights:
                # forces the weights to always be relatively similar
                # after rounding
                if step != 0:
                    loss = batch_reduction_f(regularization_loss(mu_weights -
                                                                 additional["mu_weights"],
                                                                 p=0.5,
                                                                 dim=-1),
                                             torch.mean)
                    self.add_regularization_loss("pos_variance_weights", loss)

                additional["mu_weights"] = mu_weights

            # IF WANT TO PLOT AFTER ROUNDING
            self.add_to_test([mu_weights, building_blocks],
                        ['mu_weights', 'building_blocks'])

            self.add_to_visualize([mu_weights, building_blocks],
                             ['mu_weights', 'building_blocks'])

            if self.is_l0_bb_weights:
                if self.l0_mode == "basic":
                    # same as bm in the else statement but samples which parameters
                    # can use first
                    self.linear_l0_weights.set_weights(mu_weights.unsqueeze(2))
                    mu = self.linear_l0_weights(building_blocks.unsqueeze(1))
                    if self.is_regularize:
                        loss = self.linear_l0_weights.regularization()
                        # IS OUTPUT BATCHWISE ?
                        self.add_regularization_loss("pos_l0_weights", loss)

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
            mu_old = mu.clone()
            mu = clamp(mu, minimum=0, maximum=1, is_leaky=True)
            if self.is_regularize and self.is_reg_clamp_mu:
                loss = batch_reduction_f(mu - mu_old,
                                         torch.norm,
                                         p=2)
                self.add_regularization_loss("pos_clamp_mu", loss)
        else:
            if self.is_regularize and self.is_reg_clamp_mu:
                clamped_mu = clamp(mu, minimum=0, maximum=1, is_leaky=True)
                loss = batch_reduction_f(clamped_mu,
                                         torch.norm,
                                         p=2)
                self.add_regularization_loss("pos_clamp_mu", loss)

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
                current_min_sigma = self.get_sigma(is_update_sigma)

                if self.get_sigma.is_annealing:

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

        # DEV MODE
        if self.training and self.n_training_calls < self.n_steps_init_help:
            sigma = self.min_sigma * interpolate_help(0.9,0) + sigma * interpolate_help(0.1,1)

        return mu, sigma
