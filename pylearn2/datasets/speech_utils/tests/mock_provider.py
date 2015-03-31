__author__ = 'Pawel Swietojanski'

import numpy
from random import randint
from pylearn2.space import Space


class MockUttProvider(object):
    def __init__(self, num_utts, data_specs, utt_length=None):
        self.num_utts = num_utts
        assert isinstance(data_specs, tuple) ,(
            "Data specs expected to be tuple, "
            "got %s " % type(data_specs)
        )
        self.data_specs = data_specs
        self.num_utts_provided = 0
        self.utt_length = utt_length

    def reset(self):
        self.num_utts_provided = 0

    def __iter__(self):
        return self

    def next(self):
        if self.num_utts_provided >= self.num_utts:
            raise StopIteration
        self.num_utts_provided += 1
        return self._produce_data_specs_batch()

    def _produce_data_specs_batch(self):
        #utts can have different lengths
        #we do not know it until we read it
        if self.utt_length is None:
            utt_length = randint(2, 1200)
        else:
            utt_length = self.utt_length
        spaces, sources = self.data_specs
        assert isinstance(spaces, Space), (
            "Space expected ty be of *Space type, "
            "but got %s "%type(spaces)
        )
        return spaces.get_origin_batch(utt_length)

    def get_data_specs(self):
        return self.data_specs