import torch

import torch.nn as nn


from seq2seq.util.torchextend import MLP
from seq2seq.util.initialization import linear_init
from seq2seq.util.helpers import modify_optimizer_grads, clamp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Confuser(object):
    """Confuser object used to remove certain features from a model. It forces
    the model to maximize a certain criterion and a discrimiator to minimize it.

    Args:
        criterion (callable) loss function. If the output is not a scalar, it will
            be averaged to get the final loss.
        input_size (int) dimension of the input.
        target_size (int) dimension of the target.
        hidden_size (int, optional): number of hidden neurones to use for the
            siimple model. In `None` uses a linear layer.
        bias (bool, optional): whether to use a bias in the model.
        default_targets (torch.tensor): default target if not given to the forward
            function.
        max_scale (float, optional): maximum percentage of the total loss that
            the confuser loss can reach. Note that this only limits the gradients
            for the model not the discrimator.
        n_steps_discriminate_only (int, optional): Number of steps at the begining
            where you only train the discriminator.
    """

    def __init__(self, criterion, input_size, target_size,
                 hidden_size=32,
                 bias=True,
                 default_targets=None,
                 max_scale=0.05,
                 n_steps_discriminate_only=0):
        self.criterion = criterion
        if default_targets is not None:
            self.default_targets = default_targets

        if hidden_size is not None:
            self.model = MLP(input_size, hidden_size, target_size, bias=bias)
        else:
            self.model = nn.Linear(input_size, target_size, bias=bias)

        self.optimizer = torch.optim.Adam(self.model.parameters())

        self.max_scale = max_scale
        self.n_steps_discriminate_only = n_steps_discriminate_only

        self.to_backprop_model = None
        self.n_training_calls = 0

        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.model, MLP):
            self.model.reset_parameters()
        else:
            linear_init(self.model)

        self._prepare_for_new_batch()

        self.n_training_calls = 0

    def _prepare_for_new_batch(self):
        self.losses = torch.tensor(0., requires_grad=True).float()
        self.to_backprop_model = None

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
            that you don't add unnessessary noise when the model is "confused" enough.
            The current losses will be clamped in a leaky manner, then hard manner
            if reaches `max_losses*2`. The shape of  max_losses must be broadcastable
            with the shape of the output of the criterion.
        mask (torch.tensor, optional): mask to apply to the output of the criterion.
            Should be used to mask when varying sequence lengths. The shape of mask
            must be broadcastable with the shape of the output of the criterion.
        is_multi_call (bool, optional): whether will make multiple calls of
            `compute_loss` between `backward`. In this case max_losses cannot
            clamp the loss at each call but only the final average loss over all
            calls.
        """
        if self.n_training_calls <= self.n_steps_discriminate_only:
            # masks only the maximization process
            max_losses = 0

        outputs = self.model(inputs)
        if targets is None:
            targets = self.default_targets.expand_as(outputs)

        current_losses = self.criterion(outputs, targets).squeeze(-1)

        if mask is not None:
            current_losses.masked_fill_(mask, 0.)

        if is_multi_call:
            current_losses = current_losses.view(current_losses.size(0), -1).mean(1)
            self.losses = self.losses + current_losses / seq_len

            if max_losses is not None:
                self.to_backprop_model = self.losses < max_losses
        else:
            if max_losses is not None:
                self.to_backprop_model = current_losses < max_losses

            self.losses = current_losses / seq_len

    def backward(self, retain_graph=False, main_loss=None):
        """
        Computes the gradient of the confuser parameters to minimize the
        confuing loss and of the model parameters to maximize the same loss. This
        has to be before the models `optimizer.step()`.
        """
        if self.to_backprop_model is not None:
            # if nothing to backprop
            if not bool(self.to_backprop_model.any()):
                return
            # only select examples where the average loss < max_loss
            # but only for the mdoel not the confuser!
            losses = self.losses[self.to_backprop_model]
        else:
            losses = self.losses

        loss = losses.mean()

        if main_loss is not None:
            if loss > self.max_scale * main_loss:
                self.weight = (self.max_scale * main_loss / loss).detach()
            else:
                self.weight = 1
        else:
            self.weight = self.max_scale

        # model should try maximizing loss so inverse sign
        loss = -loss

        loss = loss * self.weight

        loss.backward(retain_graph=retain_graph)

    def step_discriminator(self):
        """
        Takes a step to minimize the loss with respect to the confuser. This
        has to be called after `self.backward`,  but before calling
        `model.zero_grad()`.
        """
        if self.to_backprop_model is not None and not bool(self.to_backprop_model.all()):
            # if had to skip some batch examples
            self.optimizer.zero_grad()
            loss = self.losses.mean()
            # directly minimizing loss
            loss.backward(retain_graph=True)
        else:
            # minimize loss for disciminator
            # rescale by 1/weight to get back to real loss. I.e don't scale the discrimator
            modify_optimizer_grads(self.optimizer, is_neg=True, mul=(1 / self.weight))

        self.optimizer.step()
        self.optimizer.zero_grad()

        self._prepare_for_new_batch()

        self.n_training_calls += 1
