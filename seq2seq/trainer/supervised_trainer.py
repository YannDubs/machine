from __future__ import division
import logging
import os
import random
import time
import shutil

import torch
import torchtext
from torch import optim

from collections import defaultdict

import seq2seq
from seq2seq.evaluator import Evaluator
from seq2seq.loss import NLLLoss
from seq2seq.metrics import WordAccuracy
from seq2seq.optim import Optimizer
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.util.callbacks import Plotter, History


class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (list, optional): list of seq2seq.loss.Loss objects for training (default: [seq2seq.loss.NLLLoss])
        metrics (list, optional): list of seq2seq.metric.metric objects to be computed during evaluation
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of epochs to checkpoint after, (default: 100)
        print_every (int, optional): number of iterations to print after, (default: 100)
        early_stopper (EarlyStopping, optional): Early stopper that will stop if no improvements
            for a certian amount of time. Only used if dev_data given. Should have mode="min". (default: None)
    """

    def __init__(self, expt_dir='experiment', loss=[NLLLoss()], loss_weights=None, metrics=[], batch_size=64, eval_batch_size=128,
                 random_seed=None, checkpoint_every=100, print_every=100, early_stopper=None, anneal_middropout=0, min_middropout=0.01):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        k = NLLLoss()
        self.loss = loss
        self.metrics = metrics
        self.loss_weights = loss_weights or len(loss) * [1.]
        self.evaluator = Evaluator(loss=self.loss, metrics=self.metrics, batch_size=eval_batch_size)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every
        self.anneal_middropout = anneal_middropout
        self.min_middropout = 0 if self.anneal_middropout == 0 else min_middropout

        self.early_stopper = early_stopper
        if early_stopper is not None:
            assert self.early_stopper.mode == "min", "Can currently only be used with the loss, please use mode='min'"

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)

    def _train_batch(self, input_variable, input_lengths, target_variable, model, teacher_forcing_ratio, mid_dropout_p=0):
        loss = self.loss

        # Forward propagation
        decoder_outputs, decoder_hidden, other = model(input_variable, input_lengths, target_variable['decoder_output'],
                                                       teacher_forcing_ratio=teacher_forcing_ratio, mid_dropout_p=mid_dropout_p)

        losses = self.evaluator.compute_batch_loss(decoder_outputs, decoder_hidden, other, target_variable)

        # Backward propagation
        for i, loss in enumerate(losses[:-1], 0):
            loss.scale_loss(self.loss_weights[i])
            loss.backward(retain_graph=True)

        losses[-1].scale_loss(self.loss_weights[-1])
        losses[-1].backward()

        self.optimizer.step()
        model.zero_grad()

        return losses

    def _train_epoches(self, data, model, n_epochs, start_epoch, start_step,
                       dev_data=None, monitor_data=[], teacher_forcing_ratio=0,
                       top_k=5):
        log = self.logger

        print_loss_total = defaultdict(float)  # Reset every print_every
        epoch_loss_total = defaultdict(float)  # Reset every epoch
        epoch_loss_avg = defaultdict(float)
        print_loss_avg = defaultdict(float)

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False, sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=device, repeat=False)

        steps_per_epoch = len(batch_iterator)
        total_steps = steps_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0

        # store initial model to be sure at least one model is stored
        val_data = dev_data or data

        losses, metrics = self.evaluator.evaluate(model, val_data, self.get_batch_data)

        total_loss, log_msg, model_name = self.get_losses(losses, metrics, step)

        loss_best = top_k * [total_loss]
        best_checkpoints = top_k * [None]
        best_checkpoints[0] = model_name

        Checkpoint(model=model,
                   optimizer=self.optimizer,
                   epoch=start_epoch, step=start_step,
                   input_vocab=data.fields[seq2seq.src_field_name].vocab,
                   output_vocab=data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir, name=model_name)

        for epoch in range(start_epoch, n_epochs + 1):
            mid_dropout_p = max(self.min_middropout, self.anneal_middropout**epoch)

            log.debug("Epoch: %d, Step: %d" % (epoch, step))

            batch_generator = batch_iterator.__iter__()

            # consuming seen batches from previous training
            for _ in range((epoch - 1) * steps_per_epoch, step):
                next(batch_generator)

            model.train(True)
            for batch in batch_generator:
                step += 1
                step_elapsed += 1

                input_variables, input_lengths, target_variables = self.get_batch_data(batch)

                # compute batch loss
                losses = self._train_batch(input_variables,
                                           input_lengths.tolist(),
                                           target_variables,
                                           model,
                                           teacher_forcing_ratio,
                                           mid_dropout_p=mid_dropout_p)

                # Record average loss
                for loss in losses:
                    name = loss.log_name
                    print_loss_total[name] += loss.get_loss()
                    epoch_loss_total[name] += loss.get_loss()

                # print log info according to print_every parm
                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    for loss in losses:
                        name = loss.log_name
                        print_loss_avg[name] = print_loss_total[name] / self.print_every
                        print_loss_total[name] = 0

                    train_log_msg = ' '.join(['%s: %.4f' % (loss.log_name, loss.get_loss()) for loss in losses])

                    m_logs = {}
                    # compute vals for all monitored sets
                    for m_data in monitor_data:
                        losses, metrics = self.evaluator.evaluate(model, monitor_data[m_data], self.get_batch_data)
                        total_loss, log_msg, model_name = self.get_losses(losses, metrics, step)
                        m_logs[m_data] = ' '.join(['%s: %.4f' % (loss.log_name, loss.get_loss()) for loss in losses])

                    all_losses = ' '.join(['%s %s' % (name, m_logs[name]) for name in m_logs])

                    log_msg = 'Progress %d%%, Train %s, %s' % (
                        step / total_steps * 100,
                        train_log_msg,
                        all_losses)

                    log.info(log_msg)

                # check if new model should be saved
                if step % self.checkpoint_every == 0 or step == total_steps:
                    # compute dev loss
                    losses, metrics = self.evaluator.evaluate(model, val_data, self.get_batch_data)
                    total_loss, log_msg, model_name = self.get_losses(losses, metrics, step)

                    max_eval_loss = max(loss_best)
                    if total_loss < max_eval_loss:
                        index_max = loss_best.index(max_eval_loss)
                        # rm prev model
                        if best_checkpoints[index_max] is not None:
                            shutil.rmtree(os.path.join(self.expt_dir, best_checkpoints[index_max]))
                        best_checkpoints[index_max] = model_name
                        loss_best[index_max] = total_loss

                        # save model
                        Checkpoint(model=model,
                                   optimizer=self.optimizer,
                                   epoch=epoch, step=step,
                                   input_vocab=data.fields[seq2seq.src_field_name].vocab,
                                   output_vocab=data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir, name=model_name)

            if step_elapsed == 0:
                continue

            for loss in losses:
                epoch_loss_avg[loss.log_name] = epoch_loss_total[loss.log_name] / min(steps_per_epoch, step - start_step)
                epoch_loss_total[loss.log_name] = 0

            loss_msg = ' '.join(['%s: %.4f' % (loss.log_name, loss.get_loss()) for loss in losses])
            log_msg = "Finished epoch %d: Train %s" % (epoch, loss_msg)

            train_losses, metrics = self.evaluator.evaluate(model, data, self.get_batch_data)
            loss_total_train, log_, model_name = self.get_losses(losses, metrics, step)

            if dev_data is not None:
                losses, metrics = self.evaluator.evaluate(model, dev_data, self.get_batch_data)
                loss_total_dev, log_, model_name = self.get_losses(losses, metrics, step)

                self.optimizer.update(loss_total_dev, epoch)
                log_msg += ", Dev " + log_
                model.train(mode=True)

                if self.early_stopper is not None and self.early_stopper(loss_total_dev):
                    log.info(log_msg)
                    log.info('Terminated Training due Early Stopping at Epoch {}'.format(epoch))
                    break

                self.history.step(loss_total_train, loss_total_dev)
                if self.plotter is not None:
                    self.plotter.step(loss_total_train, loss_total_dev)
            else:
                self.optimizer.update(sum(epoch_loss_avg.values()), epoch)  # TODO check if this makes sense!

            log.info(log_msg)

        if self.plotter is not None:
            self.plotter(start_epoch=start_epoch)

    def train(self, model, data, num_epochs=5,
              resume=False, dev_data=None, monitor_data={},
              optimizer=None, teacher_forcing_ratio=0,
              learning_rate=0.001, checkpoint_path=None, top_k=5, is_plot=False):
        """ Run training for a given model.

        Args:
            model (seq2seq.models): model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (seq2seq.dataset.dataset.Dataset): dataset object to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            resume(bool, optional): resume training with the latest checkpoint, (default False)
            dev_data (seq2seq.dataset.dataset.Dataset, optional): dev Dataset (default None)
            optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
            teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
            learing_rate (float, optional): learning rate used by the optimizer (default 0.001)
            checkpoint_path (str, optional): path to load checkpoint from in case training should be resumed
            top_k (int): how many models should be stored during training

        Returns:
            model (seq2seq.models): trained model.
        """
        # If training is set to resume

        if resume:
            resume_checkpoint = Checkpoint.load(checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0

            def get_optim(optim_name):
                optims = {'adam': optim.Adam, 'adagrad': optim.Adagrad,
                          'adadelta': optim.Adadelta, 'adamax': optim.Adamax,
                          'rmsprop': optim.RMSprop, 'sgd': optim.SGD,
                          None: optim.Adam}
                return optims[optim_name]

            self.optimizer = Optimizer(get_optim(optimizer)(model.parameters(), lr=learning_rate),
                                       max_grad_norm=5)

        self.history = History(num_epochs)
        self.plotter = Plotter() if is_plot and dev_data is not None else None
        self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

        self._train_epoches(data, model, num_epochs,
                            start_epoch, step, dev_data=dev_data,
                            monitor_data=monitor_data,
                            teacher_forcing_ratio=teacher_forcing_ratio,
                            top_k=top_k)
        return model, self.history

    @staticmethod
    def get_batch_data(batch):
        input_variables, input_lengths = getattr(batch, seq2seq.src_field_name)
        target_variables = {'decoder_output': getattr(batch, seq2seq.tgt_field_name)}
        return input_variables, input_lengths, target_variables

    @staticmethod
    def get_losses(losses, metrics, step):
        total_loss = 0
        model_name = ''
        log_msg = ''

        for metric in metrics:
            val = metric.get_val()
            log_msg += '%s %.4f ' % (metric.name, val)
            model_name += '%s_%.2f_' % (metric.log_name, val)

        for loss in losses:
            val = loss.get_loss()
            log_msg += '%s %.4f ' % (loss.name, val)
            model_name += '%s_%.2f_' % (loss.log_name, val)
            total_loss += val

        model_name += 's%d' % step

        return total_loss, log_msg, model_name
