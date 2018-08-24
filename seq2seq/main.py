import os
import glob
import logging
import warnings
import math

import torch
import torch.nn as nn

from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss.loss import get_losses, LossWeightUpdater
from seq2seq.metrics.metrics import get_metrics
from seq2seq.dataset.helpers import get_train_dev
from seq2seq.util.callbacks import EarlyStopping
from seq2seq.util.confuser import Confuser
from seq2seq.util.helpers import Rate2Steps

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
    """Save arguments to a file given a dictionary."""
    with open(os.path.join(directory, filename), 'w') as file:
        file.writelines('{}={}\n'.format(k, v) for k, v in args.items())


def _rename_latest(path, new_name):
    """Rename the latest modified/added file in a path."""
    latest_file = get_latest(path)
    os.rename(latest_file, os.path.join(path, new_name))


def get_seq2seq_model(src,
                      tgt,
                      max_len,
                      total_training_calls,
                      variable_lengths=True,
                      is_mlps=True,
                      embedding_size=128,
                      rnn_cell='gru',
                      hidden_size=128,
                      is_bidirectional=False,
                      n_layers=1,
                      is_weight_norm_rnn=False,
                      dropout_layers_encoder=0.2,
                      dropout_layers_decoder=0.2,
                      dropout_input_encoder=0,
                      dropout_input_decoder=0,
                      anneal_decoder_noise_input=0,
                      anneal_mid_dropout=0.1,
                      anneal_mid_noise=0,
                      is_res=False,
                      is_highway=False,
                      initial_highway=0.5,
                      is_single_carry=True,
                      is_transform_controller=False,
                      is_add_all_controller=True,
                      use_attention="post-rnn",
                      is_full_focus=False,
                      content_method='dot',
                      is_content_attn=True,
                      is_key=True,
                      is_value=True,
                      key_size=-1,
                      value_size=-1,
                      is_decoupled_kv=False,
                      is_contained_kv=False,
                      is_query=True,
                      is_sharekq=False,
                      is_kqrnn=False,
                      anneal_kq_dropout_input=0,
                      anneal_kq_noise_input=0,
                      anneal_kq_dropout_output=0,
                      anneal_kq_noise_output=0,
                      is_normalize_encoder=True,
                      is_abscounter=False,
                      is_relcounter=False,
                      is_rotcounters=False,
                      is_postcounter=False,
                      is_position_attn=True,
                      positioning_method="gaussian",
                      is_posrnn=True,
                      is_relative_sigma=True,
                      is_clamp_mu=True,
                      anneal_min_sigma=0.1,
                      is_building_blocks_mu=True,
                      is_bb_bias=False,
                      is_l1_bb_weights=False,
                      is_l1_bias_weight=False,  # TO DOC / DEV MODE
                      anneal_bb_weights_noise=0,  # DEV MODE : which best
                      anneal_bb_noise=0,  # DEV MODE : which best
                      is_pos_perc_weight_conf=True,
                      is_dev_mode=False,
                      is_viz_train=False):
    """Return a initialized extrapolator model.

    Args:
        src (SourceField): source field.
        tgt (TargetField): target field.
        max_len (int): maximum possible length of any source sentence.
        total_training_calls (int): number of maximum training calls.
        variable_lengths (bool, optional): whether to use treat each input in a
            batch as a sentence of different length.
        is_mlps (bool, optional): whether to use MLPs for the generators instead
            of a linear layer.
        embedding_size (int, optional): size of embedding for the decoder and
            encoder.
        rnn_cell ({"gru", "lstm", optional): type of rnn.
        hidden_size (int, optional): hidden size for unidirectional encoder.
        is_bidirectional (bool, optional): whether to use biderctional rnn for
            the encoder.
        n_layers (int, optional): number of stacked RNN for the controller
            (encoder and decoder).
        is_weight_norm_rnn (bool, optional): whether to use weight normalization
            for the RNN. Weight normalization is similar to batch norm or layer
            normalization and has been shown to help when learning large models.
        dropout_layers_encoder (float, optional): dropout between different RNN
            layers of the encoder.
        dropout_layers_decoder (float, optional): dropout between different RNN
            layers of the decoder.
        dropout_input_encoder (float, optional): dropout after the embedding
            layer of the encoder.
        dropout_input_decoder (float, optional): dropout after the embedding
            layer of the decoder.
        anneal_decoder_noise_input (float, optional): if not `0`, adds
            annealed noise to the embedding of the decoder. This is used to replicate
            full_focus in a more egeneral way (by simply slowing down the learning
            process without attention). This parameter defines the percentage of
            training calls before the model should reach the final standard
            deviation of noise.
        anneal_mid_dropout (float, optional): annealed dropout between
            the decoder and encoder. `mid_dropout` will actually start at
            `initial_mid_dropout` and will geometrically decrease at each
            training calls, until it reaches `final_mid_dropout`. This
            parameter defines the percentage of training calls before the mdoel
            should reach `final_mid_dropout`.
        anneal_mid_noise (float, optional): annealed noise between
            the decoder and encoder. This parameter defines the percentage of
            training calls before the noise model should reach the final relative
            standard deviation.
        is_res (bool, optional): whether to use a residual connection betwen the
            embedding and the value of the encoder.
        is_highway (bool, optional): whether to use a highway betwen the embedding
            and the value of the encoder.
        initial_highway (float, optional): initial highway carry rate. This can be
            useful to make the network learn the attention even before the
            decoder converged.
        is_single_carry (bool, optional): whetehr to use a one dimension carry weight
            instead of n dimensional. If a n dimension then the network can learn
            to carry some dimensions but not others. The downside is that
            the number of parameters would be larger.
        is_transform_controller (bool, optional): whether to pass the hidden activation
            of the encoder through a linear layer before using it as initialization
            of the decoder. This could be useful when using `pre-rnn` attention,
            where the first input to the content and positional attention generator
            is the last hidden activation. In practice doesn't really make a difference.
        is_add_all_controller (bool, optional): whether to add all computed features
            to the decoder in order to have a central model that "knows everything".
        use_attention ({"post-rnn", "pre-rnn", None}, optional): where to use attention.
        is_full_focus (bool, optional): whether to use a trick that forces the
            network to focus on attention rather than only it's input.
        content_method ({"dot", "hard", "mlp"}, optional): content attention
            function to use.
        is_content_attn (bool, optional): whether to use content attention.
        is_key (bool, optional): whether to use a key generator.
        is_value (bool, optional): whether to use a value generator.
        key_size (int, optional): size of the generated key. -1 means same as hidden
            size. Can also give percentage of hidden size betwen 0 and 1.
        values_size (int, optional): size of the generated value. -1 means same
            as hidden size. Can also give percentage of hidden size betwen 0 and 1.
        is_decoupled_kv (bool, optional): whether to use half of the hidden
            activation as key and other half as value. This is the same to the
            key-value paper. `Frustratingly short attention spans in neural
            language modeling`.
        is_contained_kv (bool, optional): whether to use different parts of the
            controller output as input for key and value generation.
        is_query (bool, optional): whether to use a query generator.
        is_sharekq (bool, optional): whether to use the same generator for the
            key and the query.
        is_kqrnn (bool, optional): whether to use a seq2seq model for the key and
            the query generator.
        anneal_kq_dropout_input (float, optional): annealed dropout to
            the input of the key and query generator. This parameter
            defines the percentage of training calls before the model should reach
            the final dropout.
        anneal_kq_noise_input (float, optional): annealed noise to
            the input of the key and query generator. This parameter
            defines the percentage of training calls before the noise model should
            reach the final relative standard deviation.
        anneal_kq_dropout_output (float, optional): annealed dropout to
            the output of the key and query generator. This parameter
            defines the percentage of training calls before the model should reach
            the final dropout.
        anneal_kq_noise_output (float, optional): annealed noise to
            the output of the key and query generator. This parameter
            defines the percentage of training calls before the noise model should
            reach the final relative standard deviation.
        is_normalize_encoder (bool, optional): whether to normalize relative counters
            by the actual source length, rather than the maximum source length.
        is_abscounter (bool, optional): whether to use an absolute counter.
        is_relcounter (bool, optional): whether to use a relative counter,
            corresponding to an absolute counter normalized by the source length.
        is_rotcounters (bool, optional): whether to use a rotational counter,
            corresponding to 2 dimensional unit vectors where the angle is given
            by a relative counter between 0 and 180.
        is_postcounter (bool, optional): whether to append the counters to the
            output of the generator instead of the inputs.
        is_position_attn (bool, optional): whether to use positional attention.
        positioning_method ({"gaussian",
            "laplace"}, optional): name of the positional distribution.
            `laplace` is more human plausible but `gaussian` works best.
        is_posrnn (bool, optional): whether to use a rnn for the positional
            attention generator.
        is_relative_sigma (bool, optional): whether to use a relative varaince,
            i.e normalized by encoder steps.
        is_clamp_mu (bool, optional): whether to clamp the positioning `mu` in a
                    range ~[0,1] using a leaky ReLu.
        anneal_min_sigma (float, optional): if not 0 , it will force
            the network to keep a higher sigma while it's learning. min_sigma will
            actually start at `initial_sigma` and will linearly decrease at each
            training calls, until it reaches the given `min_sigma`. This parameter
            defines the percentage of training calls before the mdoel should reach
            the final `min_sigma`.
        is_building_blocks_mu (bool, optional): whether to use building blocks to
            generate the positional mu rather than using a normal MLP.
        is_bb_bias (bool, optional): adding a bias term to the building blocks.
            THis has the advantage of letting the network go to absolut positions
            (ex: middle, end, ...). THe disadvantage being that the model will often
            stop using other building blocks and thus be less general.
        is_l1_bb_weights (bool, optional): whether to use a l1 regularization
            on the postional attention mu's building block weights. This can
            be usefull if the building blocks are gihly correlyted.


        anneal_bb_weights_noise=0,  # TO DOC ###
        anneal_bb_noise=0,  # TO DOC ###

        is_pos_perc_weight_conf (bool, optional): whether to force the model to
            generate meaningfull confidence, by making the positonal percentange
            be the `position_confidence / (position_confidence + content_confidence)`.
        is_dev_mode (bool, optional): whether to store many useful variables in
            `additional`. Useful when predicting with a trained model in dev mode
             to understand what the model is doing. Use with `dev_predict`.
        is_viz_train (bool, optional): whether to save how the averages of some
            intepretable variables change during training in "visualization"
            of `additional`.
    """
    if not is_content_attn and (is_key or is_query):
        warnings.warn("`is_key` and `is_query` are useless when no content attention. Setting them to False.")
        is_key = False
        is_query = False

    # interpolating rates to interpolating steps
    rate2steps = Rate2Steps(total_training_calls)

    # Encoder
    kq_annealed_dropout_kwargs = dict(n_steps_interpolate=rate2steps(anneal_kq_dropout_input))
    kq_annealed_dropout_output_kwargs = dict(n_steps_interpolate=rate2steps(anneal_kq_dropout_output))
    kq_annealed_noise_kwargs = dict(n_steps_interpolate=rate2steps(anneal_kq_noise_input))
    kq_annealed_noise_output_kwargs = dict(n_steps_interpolate=rate2steps(anneal_kq_noise_output))

    key_kwargs = dict(output_size=key_size,
                      is_contained_kv=is_contained_kv,
                      is_kqrnn=is_kqrnn,
                      is_abscounter=is_abscounter,
                      is_relcounter=is_relcounter,
                      is_postcounter=is_postcounter,
                      is_rotcounters=is_rotcounters,
                      is_normalize_encoder=is_normalize_encoder,
                      rnn_cell=rnn_cell,
                      is_mlps=is_mlps,
                      is_weight_norm_rnn=is_weight_norm_rnn,
                      annealed_dropout_kwargs=kq_annealed_dropout_kwargs,
                      annealed_noise_kwargs=kq_annealed_noise_kwargs,
                      annealed_dropout_output_kwargs=kq_annealed_dropout_output_kwargs,
                      annealed_noise_output_kwargs=kq_annealed_noise_output_kwargs,
                      is_dev_mode=is_dev_mode)

    value_kwargs = dict(output_size=value_size,
                        is_contained_kv=is_contained_kv,
                        is_highway=is_highway,
                        is_res=is_res,
                        is_mlps=is_mlps,
                        initial_highway=initial_highway,
                        is_single_carry=is_single_carry,
                        is_dev_mode=is_dev_mode)

    encoder = EncoderRNN(len(src.vocab),
                         max_len,
                         hidden_size,
                         embedding_size,
                         input_dropout_p=dropout_input_encoder,
                         rnn_cell=rnn_cell,
                         is_weight_norm_rnn=is_weight_norm_rnn,
                         n_layers=n_layers,
                         bidirectional=is_bidirectional,
                         dropout_p=dropout_layers_encoder,
                         variable_lengths=variable_lengths,
                         key_kwargs=key_kwargs,
                         value_kwargs=value_kwargs,
                         is_highway=is_highway,
                         is_res=is_res,
                         is_key=is_key,
                         is_value=is_value,
                         is_decoupled_kv=is_decoupled_kv,
                         initial_highway=initial_highway,
                         is_dev_mode=is_dev_mode,
                         is_viz_train=is_viz_train)

    # Decoder
    query_additional_kwargs = dict(key_generator=(encoder.key_generator.generator
                                                  if is_sharekq else None))
    key_kwargs.update(query_additional_kwargs)
    query_kwargs = key_kwargs

    bb_weights_annealed_noise_kwargs = dict(n_steps_interpolate=rate2steps(anneal_bb_weights_noise),
                                            initial_sigma=0.1,
                                            final_sigma=0,
                                            mode="linear")
    bb_annealed_noise_kwargs = dict(n_steps_interpolate=rate2steps(anneal_bb_noise),
                                    initial_sigma=0.1,
                                    final_sigma=0,
                                    mode="linear")

    position_kwargs = dict(is_recursive=is_posrnn,
                           positioning_method=positioning_method,
                           is_relative_sigma=is_relative_sigma,
                           n_steps_interpolate_min_sigma=rate2steps(anneal_min_sigma),
                           rnn_cell=rnn_cell,
                           is_mlps=is_mlps,
                           is_weight_norm_rnn=is_weight_norm_rnn,
                           is_clamp_mu=is_clamp_mu,
                           is_dev_mode=is_dev_mode,
                           is_building_blocks_mu=is_building_blocks_mu,
                           is_bb_bias=is_bb_bias,
                           is_l1_bb_weights=is_l1_bb_weights,
                           is_l1_bias_weight=is_l1_bias_weight,
                           bb_weights_annealed_noise_kwargs=bb_weights_annealed_noise_kwargs,
                           bb_annealed_noise_kwargs=bb_annealed_noise_kwargs)

    content_kwargs = dict(method=content_method)

    attmix_kwargs = dict(is_mlps=is_mlps,
                         is_pos_perc_weight_conf=is_pos_perc_weight_conf,
                         is_dev_mode=is_dev_mode)

    embedding_noise_kwargs = dict(n_steps_interpolate=rate2steps(anneal_decoder_noise_input))

    decoder_hidden_size = encoder.bidirectional_hidden_size

    decoder = DecoderRNN(len(tgt.vocab),
                         max_len,
                         decoder_hidden_size,
                         embedding_size,
                         tgt.sos_id,
                         tgt.eos_id,
                         input_dropout_p=dropout_input_decoder,
                         rnn_cell=rnn_cell,
                         is_weight_norm_rnn=is_weight_norm_rnn,
                         n_layers=n_layers,
                         bidirectional=is_bidirectional,
                         dropout_p=dropout_layers_decoder,
                         use_attention=use_attention,
                         is_full_focus=is_full_focus,
                         is_transform_controller=is_transform_controller,
                         value_size=encoder.value_size,
                         is_content_attn=is_content_attn,
                         is_position_attn=is_position_attn,
                         is_query=is_query,
                         content_kwargs=content_kwargs,
                         position_kwargs=position_kwargs,
                         query_kwargs=query_kwargs,
                         attmix_kwargs=attmix_kwargs,
                         embedding_noise_kwargs=embedding_noise_kwargs,
                         is_dev_mode=is_dev_mode,
                         is_add_all_controller=is_add_all_controller,
                         is_viz_train=is_viz_train)

    mid_dropout_kwargs = dict(n_steps_interpolate=rate2steps(anneal_mid_dropout))
    mid_noise_kwargs = dict(n_steps_interpolate=rate2steps(anneal_mid_noise))

    seq2seq = Seq2seq(encoder, decoder,
                      mid_dropout_kwargs=mid_dropout_kwargs,
                      mid_noise_kwargs=mid_noise_kwargs,
                      is_dev_mode=is_dev_mode)

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
          eval_batch_size=256,
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
          eos_weight=1,
          anneal_eos_weight=0,  # to doc : not tested + to hyperparmeter optimize
          _initial_eos_weight=0.05,  # to doc
          use_attention="post-rnn",
          content_method='dot',
          is_basic_init=False,
          is_amsgrad=False,
          is_confuse_eos=False,  # DEV MODE : TO DOC
          is_confuse_query=False,
          **kwargs):
    """Trains the model given all parameters.

    Args:
        train_path (str): path to the training data.
        dev_path (str): path to the validation data.
        oneshot_path (str, optional): path to the data containing the new examples
            that should be learned in a few shot learning. If given, the model
            will be transfered on this data after having converged on the training
            data.
        metric_names (list of str, optional): names of the metrics to use. See
            `seq2seq.metrics.metrics.get_metrics` for more details.
        loss_names (list of str, optional): names of the metrics to use. See
            `seq2seq.loss.loss.get_losses` for more details.
        max_len (int, optional): maximum possible length of any source sentence.
        epochs (int, optional): maximum number of training epochs.
        output_dir (str, optional): path to the directory where the model
            checkpoints should be stored.
        src_vocab (int, optional): maximum source vocabulary size.
        tgt_vocab (int, optional): maximum target vocabulary size.
        is_predict_eos (bool, optional): whether the mdoel has to predict the <eos>
            token.
        teacher_forcing_ratio (float, optional): teacher forcing ratio.
        batch_size (int, optional): size of each training batch.
        eval_batch_size (int, optional): size of each evaluation batch.
        lr (float, optional): learning rate.
        save_every (int, optional): Every how many batches the model should be saved.
        print_every (int, optional): Every how many batches to print results.
        log_level (str, optional): Logging level.
        cuda_device (int, optional): Set cuda device to use .
        optim ({'adam', 'adadelta', 'adagrad', 'adamax', 'rmsprop', 'sgd'}, optional):
            Name of the optimizer to use.
        resume (bool, optional): Whether to resume training from the latest checkpoint.
        checkpoint_path (str, optional): path to load checkpoint from in case
            training should be resumed
        patience (int, optional): patience if using early stoping. If `None`
            doesn't use any early stoping.
        name_checkpoint (str, optional): name to give to the checkpoint to make
            it more human readable.
        write_logs (str, optional): Specify file to write logs to after training.
        is_attnloss (str, optional): Whether to add attention loss, to force the
            netwrok to learn the given attention, as seen in "Learning
            compositionally through attentive guidance".
        eos_weight (int, optional): weight of the loss that should be given to
            the <eos> token.
        anneal_eos_weight (float, optional): if not 0 , it will force
            the network to keep a lower eos weight while it's learning. eos_weight
            will actually start at `_initial_eos_weight` and will linearly
            increase or decrease (if `_initial_eos_weight` > `eos_weight`)
            at each training calls, until it reaches the given `eos_weight`. This
            parameter defines the percentage of training calls before the mdoel
            should reach the final `eos_weight`.
        attention ({"post-rnn", "pre-rnn", None}, optional): where to use attention.
        content_method ({"dot", "hard", "mlp"}, optional): attention function.
        is_basic_init (bool, optional): Whether to use the basic uniform initialization
            instead of the more "state of the art", layer dependent initialization.
            Should not be used, because many parameters are th probabilities,
            and you don't want to start a probability at 0 but 0.5.
        is_amsgrad (bool, optional): Whether to use amsgrad, which is supposed
            to make Adam more stable : "On the Convergence of Adam and Beyond".
        is_confuse_query (bool, optional): whether to remove the ability of the
            query to know what decoding step it is at. By doing so the network is
            forced to used the positional attention when counting is crucial.
        kwargs:
            Additional arguments to `get_seq2seq_model`.
    """
    saved_args = locals()
    logger.setLevel(log_level.upper())

    if torch.cuda.is_available():
        print("Cuda device set to %i" % cuda_device)
        torch.cuda.set_device(cuda_device)

    train, dev, src, tgt, oneshot = get_train_dev(train_path,
                                                  dev_path,
                                                  max_len,
                                                  src_vocab,
                                                  tgt_vocab,
                                                  is_predict_eos=is_predict_eos,
                                                  is_attnloss=is_attnloss,
                                                  content_method=content_method,
                                                  oneshot_path=oneshot_path)

    total_training_calls = math.ceil(epochs * len(train) / batch_size)
    rate2steps = Rate2Steps(total_training_calls)

    seq2seq = get_seq2seq_model(src, tgt, max_len, total_training_calls,
                                use_attention=use_attention,
                                content_method=content_method,
                                **kwargs)

    n_parameters = sum([p.numel() for p in seq2seq.parameters()])
    saved_args["n_parameters"] = n_parameters

    seq2seq.reset_parameters()
    seq2seq.to(device)

    if is_basic_init:
        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

    metrics = get_metrics(metric_names, src, tgt, is_predict_eos)
    losses, loss_weights = get_losses(loss_names, tgt, is_predict_eos,
                                      eos_weight=eos_weight,
                                      total_training_calls=total_training_calls)

    early_stopper = EarlyStopping(patience=patience) if (patience is not None) else None

    ### DEV MODE ###
    if anneal_eos_weight != 0:
        n_steps_interpolate_eos_weight = rate2steps(anneal_eos_weight)

        loss_weight_updater = LossWeightUpdater(indices=[tgt.eos_id],
                                                initial_weights=[_initial_eos_weight],
                                                final_weights=[eos_weight],
                                                n_steps_interpolates=[n_steps_interpolate_eos_weight],
                                                modes=["geometric"])
    else:
        loss_weight_updater = None
    ################

    trainer = SupervisedTrainer(loss=losses,
                                metrics=metrics,
                                loss_weights=loss_weights,
                                batch_size=batch_size,
                                eval_batch_size=eval_batch_size,
                                checkpoint_every=save_every,
                                print_every=print_every,
                                expt_dir=output_dir,
                                early_stopper=early_stopper,
                                loss_weight_updater=loss_weight_updater)

    if optim is None and is_amsgrad:
        optimizer_kwargs = {"amsgrad": True}
    else:
        optimizer_kwargs = {}

    confusers = dict()
    if is_confuse_eos:
        confusers["eos_confuser"] = Confuser(nn.MSELoss(),
                                             seq2seq.decoder.hidden_size,
                                             1,
                                             hidden_size=None,
                                             bias=False,
                                             default_targets=torch.tensor(max_len).float())
    if is_confuse_query:
        confusers["query_confuser"] = Confuser(nn.MSELoss(),
                                               seq2seq.decoder.query_size,
                                               1,
                                               scaler=0.05)

    seq2seq, logs, history, other = trainer.train(seq2seq,
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
        (seq2seq,
         logs_oneshot,
         history_oneshot,
         other) = trainer.train(seq2seq,
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

    return seq2seq, history, other
