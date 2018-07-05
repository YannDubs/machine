import os
import glob
import logging

import torch

from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss.loss import get_losses
from seq2seq.metrics.metrics import get_metrics
from seq2seq.dataset.helpers import get_train_dev
from seq2seq.util.callbacks import EarlyStopping
from seq2seq.util.helpers import get_kwargs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
log_level = "warning"
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, log_level.upper()))
logger = logging.getLogger(__name__)


def _save_parameters(args, directory, filename="train_arguments.txt"):
    """Save arguments to a file given a dictionnary."""
    with open(os.path.join(directory, filename), 'w') as file:
        file.writelines('{}={}\n'.format(k, v) for k, v in args.items())


def _rename_latest(path, new_name):
    """Rename the latest modified/added file in a path."""
    list_of_files = glob.glob(os.path.join(path, "*"))
    latest_file = max(list_of_files, key=os.path.getctime)
    os.rename(latest_file, os.path.join(path, new_name))


def _get_seq2seq_model(src,
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
                       is_full_focus=False,
                       is_highway=False,
                       is_res=False,
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
                       is_abscounter=False,
                       is_relcounter=False,
                       is_postcounter=False,
                       is_rotcounters=False,
                       is_positioner=False,
                       positioning_method="gaussian",
                       is_posrnn=False,
                       is_relative_sigma=True,
                       is_mlps=True,
                       kq_dropout=0,
                       mid_noise_sigma=0):

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
                            dropout=kq_dropout)

    value_kwargs = get_kwargs(output_size=value_size,
                              is_contained_kv=is_contained_kv,
                              is_highway=is_highway,
                              is_res=is_res,
                              is_mlps=is_mlps)

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
                         is_decoupled_kv=is_decoupled_kv)

    value_size = encoder.value_size

    query_additional_kwargs = get_kwargs(key_generator=encoder.key_generator.generator if is_sharekq else None)
    key_kwargs.update(query_additional_kwargs)
    query_kwargs = key_kwargs

    pag_kwargs = get_kwargs(is_recursive=is_posrnn,
                            positioning_method=positioning_method,
                            is_relative_sigma=is_relative_sigma,
                            rnn_cell=rnn_cell,
                            is_mlps=is_mlps)

    attmix_kwargs = get_kwargs(is_mlps=is_mlps)

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
          **kwargs):
    """Trains the model given all parameters."""
    saved_args = locals()
    logger.setLevel(log_level.upper())

    if not attention:
        attention_method = None

    if torch.cuda.is_available():
        print("Cuda device set to %i" % cuda_device)
        torch.cuda.set_device(cuda_device)

    train, dev, src, tgt = get_train_dev(train_path, dev_path, max_len, src_vocab, tgt_vocab,
                                         is_predict_eos=is_predict_eos,
                                         is_attnloss=is_attnloss,
                                         attention_method=attention_method)

    seq2seq = _get_seq2seq_model(src, tgt, max_len,
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
                                           top_k=1)

    if name_checkpoint is not None:
        _rename_latest(output_dir, name_checkpoint)

    _save_parameters(saved_args, os.path.join(output_dir, name_checkpoint))

    if write_logs:
        output_path = os.path.join(output_dir, write_logs)
        logs.write_to_file(output_path)

    return seq2seq, history
