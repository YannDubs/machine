import torch

import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR

from seq2seq.optim import Optimizer
from seq2seq.util.torchextend import MLP
from seq2seq.util.initialization import linear_init
from seq2seq.util.helpers import modify_optimizer_grads, clamp, batch_reduction_f

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
                 max_scale=0.1,
                 n_steps_discriminate_only=0,
                 **kwargs):
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
        """
        self.discriminator_optim = AdamPre(self.discriminator.parameters())
        self.generator_optim = AdamPre(self.generator.parameters())
        """

        self.max_scale = max_scale
        self.n_steps_discriminate_only = n_steps_discriminate_only

        self.to_backprop_generator = None
        self.n_training_calls = 0

        self.reset_parameters()

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
        if main_loss is not None:
            if generator_loss > self.max_scale * main_loss:
                scaling_factor = (self.max_scale * main_loss / generator_loss).detach()
            else:
                scaling_factor = 1
        else:
            scaling_factor = self.max_scale

        return generator_loss * scaling_factor

    def _compute_1_loss(self, criterion, inputs, targets, seq_len, max_losses,
                        mask, is_multi_call):
        """Computes one single loss."""
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
        #pass
        self.discriminator_optim.update(loss, epoch)
        self.generator_optim.update(loss, epoch)

    def compute_loss(self, inputs,
                     targets=None,
                     seq_len=None,
                     max_losses=None,
                     mask=None,
                     is_multi_call=False):
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
                                                         max_losses, mask,
                                                         is_multi_call)

            if max_losses is not None:
                self.to_backprop_generator = self.generator_losses < max_losses

        # DISCRIMINATOR
        # detach inputs because no backprop fro discriminator
        self.discriminator_losses = self._compute_1_loss(self.discriminator_criterion,
                                                         inputs.detach(), targets,
                                                         seq_len, max_losses, mask,
                                                         is_multi_call)

    def __call__(self, main_loss=None, **kwargs):
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

            # has to retain graph to not recompute all
            generator_loss.backward(retain_graph=True)
            self.generator_optim.step()
            self.generator.zero_grad()
            # retaining graph => has to zero the grad of the discriminator also
            self.discriminator.zero_grad()

        # DISCRIMINATOR
        discriminator_loss = self.discriminator_losses.mean()

        discriminator_loss.backward(**kwargs)
        self.discriminator_optim.step()
        self.discriminator.zero_grad()

        # RESET
        self._prepare_for_new_batch()
        self.n_training_calls += 1


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
