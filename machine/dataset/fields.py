import logging

import torchtext


class SourceField(torchtext.data.Field):
    """Wrapper class of torchtext.data.Field that forces batch_first and include_lengths to be True.

    Attributes:
        eos_id: index of the end of sentence symbol.
    """

    def __init__(self, **kwargs):
<<<<<<< HEAD:seq2seq/dataset/fields.py
        """Initialize the datafield, but force batch_first and include_lengths to be True, which is required for correct functionality of pytorch-seq2seq.
=======
        """Initialize the datafield, but force batch_first and include_lengths to be True, which is required for correct functionality of machine.
>>>>>>> upstream/master:machine/dataset/fields.py
        Also allow to include SOS and EOS symbols for the source sequence.

        Args:
            **kwargs: Description
        """
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') is False:
            logger.warning(
                "Option batch_first has to be set to use machine.  Changed to True.")
        kwargs['batch_first'] = True

        if kwargs.get('include_lengths') is False:
            logger.warning(
                "Option include_lengths has to be set to use machine.  Changed to True.")
        kwargs['include_lengths'] = True

        super(SourceField, self).__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
        super(SourceField, self).build_vocab(*args, **kwargs)


class TargetField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first to be True and prepend <sos> and append <eos> to sequences in preprocessing step.
    Attributes:
        sos_id: index of the start of sentence symbol
        eos_id: index of the end of sentence symbol
    """

    SYM_SOS = '<sos>'
    SYM_EOS = '<eos>'
    include_eos = True

    def __init__(self, include_eos=True, **kwargs):
        logger = logging.getLogger(__name__)
        self.include_eos = include_eos

        if not kwargs.get('batch_first'):
            logger.warning(
                "Option batch_first has to be set to use machine.  Changed to True.")
        kwargs['batch_first'] = True

        if kwargs.get('preprocessing') is None:
            def func(seq): return seq
        else:
            func = kwargs['preprocessing']

        if self.include_eos:
            app_eos = [self.SYM_EOS]
        else:
            app_eos = []

<<<<<<< HEAD:seq2seq/dataset/fields.py
        kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + func(seq) + app_eos
=======
        kwargs['preprocessing'] = lambda seq: [
            self.SYM_SOS] + func(seq) + app_eos
>>>>>>> upstream/master:machine/dataset/fields.py

        self.sos_id = None
        self.eos_id = None
        super(TargetField, self).__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
        super(TargetField, self).build_vocab(*args, **kwargs)
        self.sos_id = self.vocab.stoi[self.SYM_SOS]
        self.eos_id = self.vocab.stoi[self.SYM_EOS]
