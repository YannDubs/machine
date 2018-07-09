import torch

import torch.nn as nn


from seq2seq.util.helpers import MLP
from seq2seq.util.initialization import linear_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Confuser(object):
    def __init__(self, criterion, input_size, output_size, hidden_size=32, bias=True, default_outputs=None):
        self.criterion = criterion
        if default_outputs is not None:
            self.default_outputs = default_outputs

        if hidden_size is not None:
            self.model = MLP(input_size, hidden_size, output_size, bias=bias)
        else:
            self.model = nn.Linear(input_size, output_size, bias=bias)

        self.optimizer = torch.optim.Adam(self.model.parameters())

        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.model, MLP):
            self.model.reset_parameters()
        else:
            linear_init(self.model)

    def compute_loss(self, inputs, outputs=None):
        if outputs is None:
            outputs = self.default_outputs.expand_as(inputs)
        self.loss = self.criterion(inputs, outputs).mean()

    def backward(self, retain_graph=False):
        self.loss = -self.loss
        self.loss.backward(retain_graph=retain_graph)

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
