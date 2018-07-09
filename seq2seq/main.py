import os
import glob
import logging

import torch
import torch.nn as nn

from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss.loss import get_losses
from seq2seq.metrics.metrics import get_metrics
from seq2seq.dataset.helpers import get_train_dev
from seq2seq.util.callbacks import EarlyStopping
from seq2seq.util.helpers import get_kwargs
from seq2seq.util.confuser import Confuser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
log_level = "warning"
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, log_level.upper()))
logger = logging.getLogger(__name__)


def get_latest(path):
    """Return the latest modified/added file in a path."""
    list_of_files = glob.glob(os.path.join(path, "*"))
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def _save_parameters(args, directory, filename="train_arguments.txt"):
    """Save arguments to a file given a dictionnary."""
    with open(os.path.join(directory, filename), 'w') as file:
        file.writelines('{}={}\n'.format(k, v) for k, v in args.items())


def _rename_latest(path, new_name):
    """Rename the latest modified/added file in a path."""
    latest_file = get_latest(path)
    os.rename(latest_file, os.path.join(path, new_name))


def get_seq2seq_model(src,
                      tgt,
                      max_len,
                      attention="post-rnn",
                      attention_method='dot',
                      variable_lengths=True,
                      rnn_cell='gru',
                      is_bidirectional=False,
                      embedding_size=128,
                      hidden_size=128,
                      n_layers=1,
                      dropout_layers_encoder=0.2,
                      dropout_layers_decoder=0.2,
                      dropout_input_encoder=0,
                      dropout_input_decoder=0,
                      mid_dropout=0,
                      mid_noise_sigma=0,
                      is_full_focus=False,
                      is_highway=False,
                      is_res=False,
                      is_single_carry=True,
                      initial_highway=0.5,
                      is_kv=False,
                      is_decoupled_kv=False,
                      is_normalize_encoder=True,
                      is_contained_kv=False,
                      key_size=-1,
                      value_size=-1,
                      is_transform_controller=False,
                      is_query=False,
                      is_sharekq=False,
                      is_kqrnn=False,
                      kq_dropout=0,
                      kq_noise_sigma=0,
                      kq_noise_sigma_output=0,
                      is_abscounter=False,
                      is_relcounter=False,
                      is_rotcounters=False,
                      is_postcounter=False,
                      is_positioner=False,
                      positioning_method="gaussian",
                      is_posrnn=False,
                      is_relative_sigma=True,
                      is_clamp_mu=False,
                      is_predict_conf=False,
                      is_mlps=True):
    """Return a initialized extrapolator model.

    Args:
        src (SourceField): source field.
        tgt (TargetField): target field.
        max_len (int): maximum possible length of any source sentence.
        attention ({"post-rnn", "pre-rnn", None}, optional): where to use attention.
        attention_method ({"dot", "hard", "mlp"}, optional): attention function.
        variable_lengths (bool, optional): whether to use treat each input in a batch as a sentence of different length.
        rnn_cell ({"gru", "lstm", optional): type of rnn.
        is_bidirectional (bool, optional): whether to use biderctional rnn for the encoder.
        embedding_size (int, optional): size of embedding for the decoder and encoder.
        hidden_size (int, optional): hidden size for unidirectional encoder.
        n_layers (int, optional): number of stacked RNN for the controller (encoder and decoder).
        dropout_layers_encoder (float, optional): dropout between different RNN layers of the encoder.
        dropout_layers_decoder (float, optional): dropout between different RNN layers of the decoder.
        dropout_input_encoder (float, optional): dropout after the embedding layer of the encoder.
        dropout_input_decoder (float, optional): dropout after the embedding layer of the decoder.
        mid_dropout (float, optional): dropout between the decoder and encoder.
        mid_noise_sigma (float, optional): relative standard deviation of the noise to apply between the encoder and decoder.
        is_full_focus (bool, optional): whether to use a trick that forces the network to focus on attention rather than only it's input.
        is_highway (bool, optional): whether to use a highway betwen the embedding and the value of the encoder.
        is_res (bool, optional): whether to use a residual connection betwen the embedding and the value of the encoder.
        is_single_carry (bool, optional): whetehr to use a one dimension carry weight instead of n dimensional. If
            a n dimension then the network can learn to carry some dimensions but not others. The downside is that
            the number of parameters would be larger.
        initial_highway (float, optional): initial highway carry rate. This can be useful to make the network learn the attention
            even before the decoder converged.
        is_kv (bool, optional): whether to use a key value generator.
        is_decoupled_kv (bool, optional): whether to use half of the hidden activation as key and other half as value. This is the same
            to the key-value paper. `Frustratingly short attention spans in neural language modeling`.
        is_normalize_encoder (bool, optional): whether to normalize relative counters by the actual source length, rather than
            the maximum source length.
        is_contained_kv (bool, optional): whether to use different parts of the controller output as input for key
            and value generation.
        key_size (int, optional): size of the generated key. -1 means same as hidden size. Can also give percentage of hidden size
            betwen 0 and 1.
        values_size (int, optional): size of the generated value. -1 means same as hidden size. Can also give percentage of
            hidden size betwen 0 and 1.
        is_transform_controller (bool, optional): whether to pass the hidden activation of the encoder through a linear layer before
            using it as initialization of the decoder. This is useful when using `pre-rnn` attention, where the first input
            to the content and positional attention generator is the last hidden activation.
        is_query (bool, optional): whether to use a query generator.
        is_sharekq (bool, optional): whether to use the same generator for the key and the query.
        is_kqrnn (bool, optional): whether to use a seq2seq model for the key and the query generator.
        kq_dropout (float, optional): dropout to the input of the key and query generator.
        kq_noise_sigma (float, optional): relative standard deviation of the noise to apply to the input of the key and query generator.
        kq_noise_sigma_output (float, optional): relative standard deviation of the noise to apply to the output of the key
            and query generator.
        is_abscounter (bool, optional): whether to use an absolute counter.
        is_relcounter (bool, optional): whether to use a relative counter, corresponding to an absolute counter normalized by
            the source length.
        is_rotcounters (bool, optional): whether to use a rotational counter, corresponding to 2 dimensional unit vectors where the
            angle is given by a relative counter between 0 and 180.
        is_postcounter (bool, optional): whether to append the counters to the output of the generator instead of the inputs.
        is_positioner (bool, optional): whether to use positional attention in addition to the content one.
        positioning_method ({"gaussian", "laplace"}, optional): name of the positional distribution. `laplace` is more human plausible
            but `gaussian` works best.
        is_posrnn (bool, optional) whether to use a rnn for the positional attention generator.
        is_relative_sigma (bool, optional): whether to use a relative varaince, i.e normalized by encoder steps.
        is_predict_conf (bool, optional) whether to force the model to generate meaningfull confidence, by making the positonal
            percentange generator depend only on the content and the position confidence.
        is_mlps (bool, optional): whether to use MLPs for the generators instead of a linear layer.
    """

    # Encoder
    key_kwargs = get_kwargs(output_size=key_size,
                            is_contained_kv=is_contained_kv,
                            is_kqrnn=is_kqrnn,
                            is_abscounter=is_abscounter,
                            is_relcounter=is_relcounter,
                            is_postcounter=is_postcounter,
                            is_rotcounters=is_rotcounters,
                            is_normalize_encoder=is_normalize_encoder,
                            rnn_cell=rnn_cell,
                            is_mlps=is_mlps,
                            dropout=kq_dropout,
                            kq_noise_sigma=kq_noise_sigma,
                            kq_noise_sigma_output=kq_noise_sigma_output)

    value_kwargs = get_kwargs(output_size=value_size,
                              is_contained_kv=is_contained_kv,
                              is_highway=is_highway,
                              is_res=is_res,
                              is_mlps=is_mlps,
                              initial_highway=initial_highway,
                              is_single_carry=is_single_carry)

    encoder = EncoderRNN(len(src.vocab),
                         max_len,
                         hidden_size,
                         embedding_size,
                         input_dropout_p=dropout_input_encoder,
                         rnn_cell=rnn_cell,
                         n_layers=n_layers,
                         bidirectional=is_bidirectional,
                         dropout_p=dropout_layers_encoder,
                         variable_lengths=variable_lengths,
                         key_kwargs=key_kwargs,
                         value_kwargs=value_kwargs,
                         is_highway=is_highway,
                         is_res=is_res,
                         is_kv=is_kv,
                         is_decoupled_kv=is_decoupled_kv,
                         initial_highway=initial_highway)

    value_size = encoder.value_size

    query_additional_kwargs = get_kwargs(key_generator=encoder.key_generator.generator if is_sharekq else None)
    key_kwargs.update(query_additional_kwargs)
    query_kwargs = key_kwargs

    pag_kwargs = get_kwargs(is_recursive=is_posrnn,
                            positioning_method=positioning_method,
                            is_relative_sigma=is_relative_sigma,
                            rnn_cell=rnn_cell,
                            is_mlps=is_mlps,
                            is_clamp_mu=is_clamp_mu)

    attmix_kwargs = get_kwargs(is_mlps=is_mlps, is_predict_conf=is_predict_conf)

    decoder_hidden_size = encoder.bidirectional_hidden_size

    decoder = DecoderRNN(len(tgt.vocab),
                         max_len,
                         decoder_hidden_size,
                         embedding_size,
                         tgt.sos_id,
                         tgt.eos_id,
                         input_dropout_p=dropout_input_decoder,
                         rnn_cell=rnn_cell,
                         n_layers=n_layers,
                         bidirectional=is_bidirectional,
                         dropout_p=dropout_layers_decoder,
                         use_attention=attention,
                         attention_method=attention_method,
                         is_full_focus=is_full_focus,
                         is_transform_controller=is_transform_controller,
                         value_size=value_size,
                         is_positioner=is_positioner,
                         is_query=is_query,
                         pag_kwargs=pag_kwargs,
                         query_kwargs=query_kwargs,
                         attmix_kwargs=attmix_kwargs)

    seq2seq = Seq2seq(encoder, decoder, mid_dropout=mid_dropout, mid_noise_sigma=mid_noise_sigma)

    return seq2seq


def train(train_path,
          dev_path,
          oneshot_path=None,
          metric_names=["word accuracy", "sequence accuracy", "final target accuracy"],
          loss_names=["nll"],
          max_len=50,
          epochs=100,
          output_dir="models/",
          src_vocab=50000,
          tgt_vocab=50000,
          is_predict_eos=True,
          teacher_forcing_ratio=0.2,
          batch_size=32,
          eval_batch_size=128,
          lr=0.001,
          save_every=100,
          print_every=100,
          log_level="info",
          cuda_device=0,
          optim=None,
          resume=False,
          checkpoint_path=None,
          patience=15,
          name_checkpoint=None,
          write_logs=None,
          is_attnloss=False,
          eos_weight=None,
          attention="post-rnn",
          attention_method='dot',
          anneal_middropout=0,
          is_basic_init=False,
          is_amsgrad=False,
          is_confuse_eos=False,
          is_confuse_query=False,
          **kwargs):
    """Trains the model given all parameters.

    Args:
        train_path (str): path to the training data.
        dev_path (str): path to the validation data.
        oneshot_path (str, optional): path to the data containing the new examples that should be learned in a few shot learning.
            If given, the model will be transfered on this data after having converged on the training data.
        metric_names (list of str, optional): names of the metrics to use. See `seq2seq.metrics.metrics.get_metrics` for more details.
        loss_names (list of str, optional): names of the metrics to use. See `seq2seq.loss.loss.get_losses` for more details.
        max_len (int, optional): maximum possible length of any source sentence.
        epochs (int, optional): maximum number of training epochs.
        output_dir (str, optional): path to the directory where the model checkpoints should be stored.
        src_vocab (int, optional): maximum source vocabulary size.
        tgt_vocab (int, optional): maximum target vocabulary size.
        is_predict_eos (bool, optional): whether the mdoel has to predict the <eos> token.
        teacher_forcing_ratio (float, optional): teacher forcing ratio.
        batch_size (int, optional): size of each training batch.
        eval_batch_size (int, optional): size of each evaluation batch.
        lr (float, optional): learning rate.
        save_every (int, optional): Every how many batches the model should be saved.
        print_every (int, optional): Every how many batches to print results.
        log_level (str, optional): Logging level.
        cuda_device (int, optional): Set cuda device to use .
        optim ({'adam', 'adadelta', 'adagrad', 'adamax', 'rmsprop', 'sgd'}, optional): Name of the optimizer to use.
        resume (bool, optional): Whether to resume training from the latest checkpoint.
        checkpoint_path (str, optional): path to load checkpoint from in case training should be resumed
        patience (int, optional): patience if using early stoping. If `None` doesn't use any early stoping.
        name_checkpoint (str, optional): name to give to the checkpoint to make it more human readable.
        write_logs (str, optional): Specify file to write logs to after training.
        is_attnloss (str, optional): Whether to add attention loss, to force the netwrok to learn the given attention, as seen in
            "Learning compositionally through attentive guidance".
        eos_weight (int, optional): weight of the loss that should be given to the <eos> token.
        attention ({"post-rnn", "pre-rnn", None}, optional): where to use attention.
        attention_method ({"dot", "hard", "mlp"}, optional): attention function.
        anneal_middropout (float, optional): annealed dropout between the decoder and encoder, at each epoch the middropout will
            be of `anneal_middropout^epoch`.
        is_basic_init (bool, optional): Whether to use the basic uniform initialization instead of the more "state of the art", layer
            dependent initialization.
        is_amsgrad (bool, optional): Whether to use amsgrad, which is supposed to make Adam more stable :
            "On the Convergence of Adam and Beyond".
        is_confuse_eos (bool, optional): ????? TEST MDOE ?????
        is_confuse_query (bool, optional): whether to remove the ability of the query to know what decoding step it is at. By
            doing so the network is forced to used the positional attention when counting is crucial.
        kwargs:
            Additional arguments to `get_seq2seq_model`.
    """
    saved_args = locals()
    logger.setLevel(log_level.upper())

    if not attention:
        attention_method = None

    if torch.cuda.is_available():
        print("Cuda device set to %i" % cuda_device)
        torch.cuda.set_device(cuda_device)

    train, dev, src, tgt, oneshot = get_train_dev(train_path, dev_path, max_len, src_vocab, tgt_vocab,
                                                  is_predict_eos=is_predict_eos,
                                                  is_attnloss=is_attnloss,
                                                  attention_method=attention_method,
                                                  oneshot_path=oneshot_path)

    seq2seq = get_seq2seq_model(src, tgt, max_len,
                                attention=attention,
                                attention_method=attention_method,
                                **kwargs)

    n_parameters = sum([p.numel() for p in seq2seq.parameters()])
    saved_args["n_parameters"] = n_parameters

    seq2seq.to(device)
    seq2seq.reset_parameters()

    if is_basic_init:
        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

    metrics = get_metrics(metric_names, src, tgt, is_predict_eos)
    losses, loss_weights = get_losses(loss_names, tgt, is_predict_eos, eos_weight=eos_weight)

    early_stopper = EarlyStopping(patience=patience) if (patience is not None) else None

    trainer = SupervisedTrainer(loss=losses,
                                metrics=metrics,
                                loss_weights=loss_weights,
                                batch_size=batch_size,
                                eval_batch_size=eval_batch_size,
                                checkpoint_every=save_every,
                                print_every=print_every,
                                expt_dir=output_dir,
                                early_stopper=early_stopper,
                                anneal_middropout=anneal_middropout)

    if optim is None and is_amsgrad:
        optimizer_kwargs = {"amsgrad": True}
    else:
        optimizer_kwargs = {}

    confusers = dict()
    if is_confuse_eos:
        confusers["eos_confuser"] = Confuser(nn.MSELoss(), seq2seq.decoder.hidden_size, 1,
                                             hidden_size=None,
                                             bias=False,
                                             default_outputs=torch.tensor(6.0))
    if is_confuse_query:
        confusers["query_confuser"] = Confuser(nn.MSELoss(), seq2seq.decoder.query_size, 1)

    seq2seq, logs, history = trainer.train(seq2seq,
                                           train,
                                           num_epochs=epochs,
                                           dev_data=dev,
                                           optimizer=optim,
                                           optimizer_kwargs=optimizer_kwargs,
                                           teacher_forcing_ratio=teacher_forcing_ratio,
                                           learning_rate=lr,
                                           resume=resume,
                                           checkpoint_path=checkpoint_path,
                                           top_k=1,
                                           confusers=confusers)

    if oneshot is not None:
        seq2seq, logs_oneshot, history_oneshot = trainer.train(seq2seq,
                                                               oneshot,
                                                               num_epochs=5,
                                                               dev_data=dev,
                                                               optimizer=optim,
                                                               optimizer_kwargs=optimizer_kwargs,
                                                               teacher_forcing_ratio=0,
                                                               learning_rate=lr,
                                                               is_oneshot=True,
                                                               checkpoint_path=get_latest(output_dir),
                                                               top_k=1)

    if name_checkpoint is not None:
        _rename_latest(output_dir, name_checkpoint)

    _save_parameters(saved_args, os.path.join(output_dir, name_checkpoint))

    if write_logs:
        output_path = os.path.join(output_dir, write_logs)
        logs.write_to_file(output_path)

    return seq2seq, history
