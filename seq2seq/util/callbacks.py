from __future__ import print_function

import numpy as np
from seq2seq.util.helpers import check_import

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


def _mode_dependent_param(mode, min_delta=0):
    """Given the mode it returns the operator, best loss and delta respectively."""
    assert min_delta >= 0, "Min delta should be positive but {} given".format(min_delta)

    if mode == 'min':
        monitor_op = np.less
        best_loss = np.Inf
        min_delta *= -1
    elif mode == 'max':
        monitor_op = np.greater
        best_loss = -np.Inf
        min_delta *= 1
    else:
        raise ValueError('Unkown mode : {}'.format(mode))

    return monitor_op, best_loss, min_delta


class EarlyStopping(object):
    """
    Early stopping to terminate training early if stops improving.

    Args:
        mode ({'min', 'max'}): defines what is considered as a better. (default: "min")
        patience (int): number of epochs to wait for improvement before stopping.
            The counter will be reseted when improvement.
        min_delta (float): minimum change in monitored value to qualify as improvement.
            This number should be positive.
    """

    def __init__(self, mode="min", patience=5, min_delta=0):
        self.mode = mode
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0

        self.monitor_op, self.best_loss, self.min_delta = _mode_dependent_param(self.mode, min_delta=min_delta)

    def __call__(self, current_loss):
        """
        Args:
            current_loss (float): loss of the current epoch.
        Returns:
            out (bool): whether or not to stop training.
        """
        if self.monitor_op(current_loss - self.min_delta, self.best_loss):
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True
        return False


class Plotter(object):
    """
    Plots the train and validation loss.

    Args:
        n_epochs (int): total number of epochs.
    """

    def __init__(self):
        check_import('matplotlib.pyplot', to_use='Plotter')
        self.losses_train = []
        self.losses_valid = []

    def step(self, loss_train, loss_valid):
        """
        Stores the training and validation losses.
        """
        self.losses_train.append(loss_train)
        self.losses_valid.append(loss_valid)

    def __call__(self, start_epoch=0, delta_epoch=1):
        """
        Plots the trainign and validation history.
        """
        x = list(range(start_epoch, len(self.losses_train) * delta_epoch + start_epoch, delta_epoch))
        plt.plot(x, self.losses_train)
        plt.plot(x, self.losses_valid)
        plt.title("Model's loss")
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()

        print("valid loss : {}, train loss : {}, steps : {}".format(self.losses_valid[-1],
                                                                    self.losses_train[-1],
                                                                    x[-1]))


class History(object):
    def __init__(self, n_epochs):
        self.losses_train = [None] * n_epochs
        self.losses_valid = [None] * n_epochs
        self.i = 0
        self.names = ['losses_train', 'losses_valid']

    def step(self, loss_train, loss_valid):
        """
        Stores the training and validation losses.
        """
        self.losses_train[self.i] = loss_train
        self.losses_valid[self.i] = loss_valid
        self.i += 1

    def __iter__(self):
        return iter([self.losses_train,
                     self.losses_valid])
