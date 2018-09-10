import torch

import torch.nn as nn
from torch.autograd import Variable


from seq2seq.util.torchextend import MLP
from seq2seq.util.initialization import linear_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Confuser(object):
    """Confuser object used to remove vertain features from a model.

    Args:
        criterion ()
        input_size (int)
        output_size (int)
        hidden_size (int, optional): number of hidden neurones to use for the
            siimple model. In `None` uses a linear layer.
        bias (bool, optional): whether to use a bias in the model.
        default_targets (torch.tensor): default target if not given to the forward
            function.
    """

    def __init__(self, criterion, input_size, output_size,
                 hidden_size=32, bias=True, default_targets=None,
                 scaler=0.05):
        self.criterion = criterion
        if default_targets is not None:
            self.default_targets = default_targets

        if hidden_size is not None:
            self.model = MLP(input_size, hidden_size, output_size, bias=bias)
        else:
            self.model = nn.Linear(input_size, output_size, bias=bias)

        self.optimizer = torch.optim.Adam(self.model.parameters())

        self.scaler = scaler

        self.reset_parameters()

    def reset_parameters(self):
        self.reset_loss()
        if isinstance(self.model, MLP):
            self.model.reset_parameters()
        else:
            linear_init(self.model)

    def reset_loss(self):
        self.loss = Variable(torch.tensor(0).float(), requires_grad=True)

    def compute_loss(self, inputs, targets=None):
        outputs = self.model(inputs)
        if targets is None:
            targets = self.default_targets.expand_as(outputs)
        self.loss = self.loss + self.criterion(outputs, targets).mean()

    def backward(self, retain_graph=False, main_loss=None):
        if main_loss is not None:
            if self.loss.detach() > self.scaler * main_loss:
                weight = self.scaler * main_loss / self.loss.detach()
        else:
            weight = self.scaler

        self.loss = self.loss * weight

        self.loss = -self.loss
        self.loss.backward(retain_graph=retain_graph)
        self.reset_loss()

    def step_discriminator(self):
        self._reverse_sign_grad()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _reverse_sign_grad(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.neg_()
