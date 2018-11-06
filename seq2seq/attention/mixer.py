"""
Content and positioning attention mixer.

"""

import pdb
from pdb import set_trace as bp

import torch
from torch import nn
from torch.nn.parameter import Parameter

from seq2seq.util.helpers import (batch_reduction_f, get_extra_repr, add_regularization)
from seq2seq.util.torchextend import (MLP, ConcreteRounding, ProbabilityConverter,
                                      StochasticRounding)
from seq2seq.util.initialization import weights_init
from seq2seq.util.base import Module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_rounder(name=None, **kwargs):
    if name is None:
        return None
    elif name == "concrete":
        return ConcreteRounding(**kwargs)
    elif name == "stochastic":
        return StochasticRounding(**kwargs)
    else:
        raise ValueError("Unkown rounder method {}".format(name))


class AttentionMixer(Module):
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
        n_steps_wait (float, optional): number of training steps to wait
            for before starting to generate the positional percentage. Until then
            will use `default_pos_perc`.
        default_pos_perc (float, optional): constant positional percentage to
            use while `rate_attnmix_wait`.
        is_reg_pos_perc (bool, optional): whether to use lp norm regularisation
            in order to push the network to use positional attention when it can.
            This is desirable as positional attention is tailored for location
            attention and is thus more interpretable and extrapolable. This is
            only needed if content attention is able to find some positional
            pattern, which shouldn’t be the case if it confused correctly.
        rounder_perc_kwargs (dictionary, optional): Additional arguments to
            the percentage rounder.
    """

    def __init__(self, decoder_output_size,
                 hidden_size=32,
                 is_mlps=True,
                 mode="pos_conf",
                 n_steps_wait=0,
                 default_pos_perc=0.5,
                 is_reg_pos_perc=False,
                 rounder_perc_kwargs={},
                 balance=None):
        super(AttentionMixer, self).__init__()

        # DEV MODE
        self.balance = balance

        self.mode = mode.lower()
        self.n_steps_wait = n_steps_wait
        self.default_pos_perc = default_pos_perc
        self.is_reg_pos_perc = is_reg_pos_perc
        self.rounder_perc = _get_rounder(**rounder_perc_kwargs)

        n_additional_pos_perc_inputs = 3

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

    def reset_parameters(self):
        super().reset_parameters()

        # should be under if not is_predict_conf (i.e net else)
        # but keeping while testing `additional_controller_features`
        self.position_perc0 = Parameter(torch.tensor(self.default_pos_perc))

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

        if not self.training or self.n_training_calls >= self.n_steps_wait:
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
            position_perc = torch.tensor(self.default_pos_perc).to(device).expand(batch_size, 1)

        if self.rounder_perc is not None:
            position_perc = self.rounder_perc(position_perc)

        self.add_to_test(position_perc, "position_percentage")
        self.add_to_visualize(position_perc, "position_percentage")

        mean_pos_perc = batch_reduction_f(position_perc, torch.mean)

        # self._rescale_losses(position_perc)
        self._rescale_losses(position_perc, additional)

        if self.is_regularize and self.is_reg_pos_perc:
            # if can solve with positioning pleas do
            loss = 1 - mean_pos_perc
            self.add_regularization_loss("pos%", loss)

        # will be used for balancing (i.e not pushing towards one type of attn
        # if components of other are regularized)
        additional["pos_perc"] = mean_pos_perc.view(-1)

        # COnvex combination
        attn = (pos_attn * position_perc.unsqueeze(-1) +
                (1 - position_perc.unsqueeze(-1)) * content_attn)

        return attn, position_perc

    def _rescale_losses(self, position_perc, additional):
        """
        Rescale the content and positional regularization such that they are
        proportional to our use of them.

        Note:
            - Detaches the scaling factor because don't want to push the network
            to use one type of attention because of some losses that you use on
            the other type.
        """
        if not self.is_regularize:
            return

        # don't broadcast multiplication : want vector output
        position_perc = position_perc.view(-1)

        losses = additional["losses"]

        if self.balance in ["math_s"]:
            for name in losses.keys():
                if name.startswith("pos_"):
                    self.add_regularization_loss(name, losses[name] * position_perc.detach())
                elif name.startswith("cont_"):
                    self.add_regularization_loss(name, losses[name] * (1 - position_perc).detach())
                else:
                    # just stores so no loss of losses
                    self.add_regularization_loss(name, losses[name])
        elif self.balance in ["old", "math_c"]:
            for name in losses.keys():
                if name.startswith("pos_"):
                    self.add_regularization_loss(name, losses[name] * position_perc)
                elif name.startswith("cont_"):
                    self.add_regularization_loss(name, losses[name] * (1 - position_perc))
                else:
                    # just stores so no loss of losses
                    self.add_regularization_loss(name, losses[name])
