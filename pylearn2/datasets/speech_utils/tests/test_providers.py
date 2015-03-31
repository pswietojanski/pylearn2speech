__author__ = 'Pawel Swietojanski'

import time
from pylearn2.datasets.speech_utils.tests.mock_provider import MockUttProvider
from pylearn2.space import CompositeSpace, VectorSpace
from pylearn2.datasets.speech_utils.providers import BufferedProviderDataSpec


def test_buffered_provider_2d():
    batch_size = 1024
    num_utts = 10**6
    utt_length = None #None will result in random lengths
    data_space = CompositeSpace((VectorSpace(dim=40), VectorSpace(dim=1)))
    data_source = ('features', 'targets')
    provider = MockUttProvider(num_utts, (data_space, data_source), utt_length)
    buffered_provider = BufferedProviderDataSpec(provider, batch_size)

    num_datapoints = 0
    start = time.clock()
    for batch in buffered_provider:
        data_space.np_validate(batch)
        num_datapoints += batch_size
    stop = time.clock()-start

    print "Converting (and producing by mock provider) %i utterances " \
          "(%i datapoints) by BufferedDataProvider took %f seconds" % \
          (num_utts, num_datapoints, stop)

def test_buffered_provider_3d():
    batch_size = 1024
    num_utts = 10**6
    utt_length = None #None will result in random lengths
    data_space = CompositeSpace((VectorSpace(dim=40), VectorSpace(dim=1), VectorSpace(dim=1)))
    data_source = ('features', 'targets1', 'targets2')
    provider = MockUttProvider(num_utts, (data_space, data_source), utt_length)
    buffered_provider = BufferedProviderDataSpec(provider, batch_size)

    num_datapoints = 0
    start = time.clock()
    for batch in buffered_provider:
        data_space.np_validate(batch)
        num_datapoints += batch_size
    stop = time.clock()-start

    print "Converting (and producing by mock provider) %i utterances " \
          "(%i datapoints) by BufferedDataProvider took %f seconds" % \
          (num_utts, num_datapoints, stop)

def test_buffered_provider_3d_1ofK():
    batch_size = 1024
    num_utts = 10**6
    utt_length = None #None will result in random lengths
    data_space = CompositeSpace((VectorSpace(dim=40), VectorSpace(dim=1000), VectorSpace(dim=200)))
    data_source = ('features', 'targets1', 'targets2')
    provider = MockUttProvider(num_utts, (data_space, data_source), utt_length)
    buffered_provider = BufferedProviderDataSpec(provider, batch_size)

    num_datapoints = 0
    start = time.clock()
    for batch in buffered_provider:
        data_space.np_validate(batch)
        num_datapoints += batch_size
    stop = time.clock()-start

    print "Converting (and producing by mock provider) %i utterances " \
          "(%i datapoints) by BufferedDataProvider took %f seconds" % \
          (num_utts, num_datapoints, stop)

if __name__ == "__main__":
    test_buffered_provider_2d()
    test_buffered_provider_3d()
    test_buffered_provider_3d_1ofK()