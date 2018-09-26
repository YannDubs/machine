import numpy as np

import torch

import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR

from seq2seq.optim import Optimizer
from seq2seq.util.torchextend import MLP
from seq2seq.util.initialization import linear_init
from seq2seq.util.helpers import (modify_optimizer_grads, clamp, batch_reduction_f,
                                  HyperparameterInterpolator, add_to_visualize,
                                  SummaryStatistics)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Confuser(object):
    """Confuser object used to remove certain features from a generator. It forces
    the generator to maximize a certain criterion and a discrimiator to minimize it.

    Args:
        discriminator_criterion (callable) loss function of the discriminator.
            If the output is not a scalar, it will be averaged to get the final loss.
        input_size (int) dimension of the input.
        target_size (int) dimension of the target.
        generator (parameters, optional): part of the model to confuse.
        generator_criterion (callable, optional) oss function of the generator.
             if `None`, `discriminator_criterion` will be used. If the output is
             not a scalar, it will be averaged to get the final loss.
        hidden_size (int, optional): number of hidden neurones to use for the
            discriminator. In `None` uses a linear layer.
        default_targets (torch.tensor): default target if not given to the forward
            function.
        max_scale (float, optional): maximum percentage of the total loss that
            the confuser loss can reach. Note that this only limits the gradients
            for the generator not the discrimator.
        n_steps_discriminate_only (int, optional): Number of steps at the begining
            where you only train the discriminator.
        kwargs:
            Additional parameters for the discriminator.
    """

    def __init__(self, discriminator_criterion, input_size, target_size, generator,
                 generator_criterion=None,
                 hidden_size=32,
                 default_targets=None,
                 final_max_scale=5e-2,  # TO DOC annealing
                 n_steps_discriminate_only=10,
                 optim="adam",  # TO DOC
                 final_factor=1.5,  # TO DOC
                 n_steps_interpolate=0,  # TO DOC
                 is_anticyclic=True,  # TO DOC
                 factor_kwargs={},  # TO DOC
                 max_scale_kwargs={},  # TO DOC
                 **kwargs):
        self.is_anticyclic = is_anticyclic
        if self.is_anticyclic:
            self.summary_stats = SummaryStatistics(statistics_name="all")
            input_size = input_size + self.summary_stats.n_statistics

        self.discriminator_criterion = discriminator_criterion
        self.generator_criterion = (generator_criterion if generator_criterion is not None
                                    else discriminator_criterion)

        self.generator = generator

        if default_targets is not None:
            self.default_targets = default_targets

        if hidden_size is not None:
            self.discriminator = MLP(input_size, hidden_size, target_size, **kwargs)
        else:
            self.discriminator = nn.Linear(input_size, target_size, **kwargs)

        # intuitively geenrator should have very high momentum because you want
        # to minimize gradients that just pushing towards oscilattor behavior (i.e
        # just switching the order of the neurones), and maximize gradients
        # that are similar: i.e saying that should forget count
        #self.discriminator_optim = torch.optim.Adam(self.discriminator.parameters())

        self.optim = optim
        if self.optim == "adampre":
            self.discriminator_optim = AdamPre(self.discriminator.parameters())
            self.generator_optim = AdamPre(self.generator.parameters())
        elif self.optim == "sgd":
            self.discriminator_optim = Optimizer(torch.optim.SGD,
                                                 self.discriminator.parameters(),
                                                 lr=0.05,
                                                 momentum=0.3,
                                                 nesterov=True,
                                                 max_grad_value=1,
                                                 max_grad_norm=2,
                                                 scheduler=ExponentialLR,
                                                 scheduler_kwargs={"gamma": 0.95})
            self.generator_optim = Optimizer(torch.optim.SGD,
                                             self.generator.parameters(),
                                             lr=0.01,
                                             momentum=0.9,
                                             nesterov=True,
                                             max_grad_value=1,
                                             max_grad_norm=2,
                                             scheduler=ExponentialLR,
                                             scheduler_kwargs={"gamma": 0.95})
        elif self.optim == "adam":
            self.discriminator_optim = Optimizer(torch.optim.Adam,
                                                 self.discriminator.parameters(),
                                                 max_grad_value=1,
                                                 max_grad_norm=2,
                                                 scheduler=ExponentialLR,
                                                 scheduler_kwargs={"gamma": 0.95})
            self.generator_optim = Optimizer(torch.optim.Adam,
                                             self.generator.parameters(),
                                             max_grad_value=1,
                                             max_grad_norm=2,
                                             scheduler=ExponentialLR,
                                             scheduler_kwargs={"gamma": 0.95})

        self.n_steps_discriminate_only = n_steps_discriminate_only

        initial_factor = 10
        self.get_factor = HyperparameterInterpolator(initial_factor,
                                                     final_factor,
                                                     n_steps_interpolate,
                                                     mode="geometric",
                                                     start_step=self.n_steps_discriminate_only,
                                                     default=initial_factor,
                                                     **factor_kwargs)

        initial_max_scale = 0.5
        self.get_max_scale = HyperparameterInterpolator(initial_max_scale,
                                                        final_max_scale,
                                                        n_steps_interpolate,
                                                        mode="geometric",
                                                        start_step=self.n_steps_discriminate_only,
                                                        default=initial_max_scale,
                                                        **max_scale_kwargs)

        self.to_backprop_generator = None
        self.n_training_calls = 0

        self.reset_parameters()

    def to(self, device):
        self.discriminator.to(device)

    def reset_parameters(self):
        if isinstance(self.discriminator, MLP):
            self.discriminator.reset_parameters()
        else:
            linear_init(self.discriminator)

        self._prepare_for_new_batch()

        self.n_training_calls = 0

    def _prepare_for_new_batch(self):
        self.discriminator_losses = torch.tensor(0.,
                                                 requires_grad=True,
                                                 dtype=torch.float,
                                                 device=device)
        self.generator_losses = self.discriminator_losses.clone()
        self.to_backprop_generator = None

    def _scale_generator_loss(self, generator_loss, main_loss=None):
        max_scale = self.get_max_scale(True)
        if main_loss is not None:
            if generator_loss > max_scale * main_loss:
                scaling_factor = (max_scale * main_loss / generator_loss).detach()
            else:
                scaling_factor = 1
        else:
            scaling_factor = max_scale

        return generator_loss * scaling_factor

    def _compute_1_loss(self, criterion, inputs, targets, seq_len, mask,
                        is_multi_call, to_summarize_stats=None):
        """Computes one single loss."""
        if self.is_anticyclic:
            if to_summarize_stats is None:
                to_summarize_stats = inputs
            inputs = torch.cat((inputs, self.summary_stats(inputs)), dim=-1)
        outputs = self.discriminator(inputs)

        if targets is None:
            targets = self.default_targets.expand_as(outputs)

        losses = criterion(outputs, targets).squeeze(-1)

        if mask is not None:
            losses.masked_fill_(mask, 0.)

        if is_multi_call:
            # mean of all besides batch size
            losses = batch_reduction_f(losses, torch.mean)
            losses = losses + losses / seq_len
        else:
            losses = losses

        return losses

    def update(self, loss, epoch):
        """Updates the optimizers."""
        if self.optim == "adampre":
            pass
        else:
            self.discriminator_optim.update(loss, epoch)
            self.generator_optim.update(loss, epoch)

    def compute_loss(self, inputs,
                     targets=None,
                     seq_len=None,
                     max_losses=None,
                     mask=None,
                     is_multi_call=False,
                     to_summarize_stats=None):  # TO DOC
        """Computes the loss for the confuser.

        inputs (torch.tensor): inputs to the confuser. I.e where you want to remove
            the `targets` from.
        targets (torch.tensor, optional): targets of the confuser. I.e what you want to remove
            from the `inputs`. If `None` will use `default_targets`.
        seq_len (int or torch.tensor, optional): number of calls per batch / sequence
            length. This is used so that the loss is an average, not a sum of losses.
            I.e to make it independant of the number of calls. The shape of must
            be broadcastable with the shape of the output of the criterion.
        max_losses (float or torch.tensor, optional): max losses. This is used so
            that you don't add unnessessary noise when the generator is "confused"
            enough. The current losses will be clamped in a leaky manner, then
            hard manner f reaches `max_losses*2`. The shape of  max_losses must
            be broadcastable with the shape of the output of the criterion.
        mask (torch.tensor, optional): mask to apply to the output of the criterion.
            Should be used to mask when varying sequence lengths. The shape of mask
            must be broadcastable with the shape of the output of the criterion.
        is_multi_call (bool, optional): whether will make multiple calls of
            `compute_loss` between `backward`. In this case max_losses cannot
            clamp the loss at each call but only the final average loss over all
            calls.
        """
        # GENERATOR
        if self.n_training_calls > self.n_steps_discriminate_only:
            self.generator_losses = self._compute_1_loss(self.generator_criterion,
                                                         inputs, targets, seq_len,
                                                         mask, is_multi_call,
                                                         to_summarize_stats)

            if max_losses is not None:
                self.to_backprop_generator = self.generator_losses < max_losses

        # DISCRIMINATOR
        # detach inputs because no backprop fro discriminator
        self.discriminator_losses = self._compute_1_loss(self.discriminator_criterion,
                                                         inputs.detach(), targets,
                                                         seq_len, mask, is_multi_call,
                                                         to_summarize_stats)

    def __call__(self, main_loss=None, additional=None, name="", **kwargs):
        """
        Computes the gradient of the generator parameters to minimize the
        confuing loss and of the discriminator parameters to maximize the same
        loss.

        Note:
            Should call model.zero_grad() at the end to be sure that clean slate.
        """

        # GENERATOR
        # something to backprop ?
        if self.to_backprop_generator is not None and bool(self.to_backprop_generator.any()):
            if self.to_backprop_generator is not None:
                generator_losses = self.generator_losses[self.to_backprop_generator]
            else:
                generator_losses = self.generator_losses

            # generator should try maximizing loss so inverse sign
            generator_loss = -1 * generator_losses.mean()
            generator_loss = self._scale_generator_loss(generator_loss, main_loss)

            # # # # # DEV MODE # # # # #
            if additional is not None:
                add_to_visualize(generator_losses.mean().item(),
                                 "losses_generator_{}".format(name),
                                 additional, is_training=True)

                add_to_visualize(generator_loss.item(),
                                 "losses_weighted_generator_{}".format(name),
                                 additional, is_training=True)
            # # # # # # # # # # # # # # #

            # has to retain graph to not recompute all
            generator_loss.backward(retain_graph=True)
            self.generator_optim.step()
            self.generator.zero_grad()
            # retaining graph => has to zero the grad of the discriminator also
            self.discriminator.zero_grad()

        # DISCRIMINATOR
        discriminator_loss = self.discriminator_losses.mean()

        # # # # # DEV MODE # # # # #
        if additional is not None:
            add_to_visualize(discriminator_loss.item(),
                             "losses_discriminator_{}".format(name),
                             additional, is_training=True)
        # # # # # # # # # # # # # # #

        discriminator_loss.backward(**kwargs)
        self.discriminator_optim.step()
        self.discriminator.zero_grad()

        # RESET
        self._prepare_for_new_batch()
        self.n_training_calls += 1


def _precompute_max_loss(p, max_n=100):
    return torch.tensor(np.array([np.mean([np.abs(i - (n + 1) / 2)**p
                                           for i in range(1, n + 1)])
                                  for n in range(0, max_n)], dtype=np.float32)
                        ).float().to(device)


MAX_LOSSES_P05 = _precompute_max_loss(0.5)


def get_max_loss_loc_confuser(input_lengths_tensor, p=2, factor=1):
    """
    Returns the expected maximum loss of the key confuser depending on p used.
    `max_loss = ∑_{i=1}^n (i-N/2)**p`

    Args:
        input_lengths_list (tensor): Float tensor containing the legnth of each
            sentence of the batch. Should already be on the correc device.
        p (float, optional): p of the Lp pseudo-norm used as loss.
        factor (float, optional): by how much to decrease the maxmum loss. If factor
            is 2 it means that you consider that the maximum loss will be achieved
            if your prediction is 1/factor (i.e half) way between the correct i
            and the best worst case output N/2. Factor = 10 means it can be a lot
            closer to i. This is usefull as there will always be some noise, and you
            don't want to penalize the model for some noise.
    """

    # E[(i-N/2)**2] = VAR(i) = (n**2 - 1)/12
    if p == 2:
        max_losses = (input_lengths_tensor**2 - 1) / 12
    elif p == 1:
        # computed by hand and use modulo because different if odd
        max_losses = (input_lengths_tensor**2 - input_lengths_tensor % 2) / (4 * input_lengths_tensor)
    elif p == 0.5:
        max_losses = MAX_LOSSES_P05[input_lengths_tensor.long()]
    else:
        raise ValueError("Unkown p={}".format(p))

    max_losses = max_losses / (factor**p)

    return max_losses

# # # # TEST # # # #


from torch.optim.optimizer import Optimizer as BaseOptimizer
import math


class AdamPre(BaseOptimizer):
    """Implements Adam algorithm with prediction step.
    This class implements lookahead version of Adam Optimizer.
    The structure of class is similar to Adam class in Pytorch.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, name='NotGiven'):
        self.name = name
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AdamPre, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                    state['oldWeights'] = p.data.clone()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** min(state['step'], 1022)
                bias_correction2 = 1 - beta2 ** min(state['step'], 1022)
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)
        return loss

    def stepLookAhead(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                temp_grad = p.data.sub(state['oldWeights'])
                state['oldWeights'].copy_(p.data)
                p.data.add_(temp_grad)
        return loss

    def restoreStepLookAhead(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                p.data.copy_(state['oldWeights'])
        return loss
