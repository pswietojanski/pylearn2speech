import random
import time
import thread
import sys
import numpy
import logging

from Queue import Empty as QueueEmptyException
from Queue import Full as QueueFullException

QUEUE_THREADS = 1
QUEUE_PROCESSES = 2

PYLEARN2_LOADING_MODE = QUEUE_THREADS
# PYLEARN2_LOADING_MODE = QUEUE_PROCESSES
if PYLEARN2_LOADING_MODE == QUEUE_THREADS:
    from threading import Thread as PYLEARN2_LOADING_MODE
    from Queue import Queue
elif PYLEARN2_LOADING_MODE == QUEUE_PROCESSES:
    from multiprocessing import Process as PYLEARN2_LOADING_MODE
    from multiprocessing import Queue
assert PYLEARN2_LOADING_MODE != None

from pylearn2.datasets.speech_utils.providers import BufferedProvider, BufferedProviderDataSpec
from pylearn2.datasets.preprocessing_speech import PipelineOnlinePreprocessor, SpliceFrames

class QueueCacheLastElem(object):
    """Special class denoting when StopIteration exception should be raised in Queue-based dataset interfaces."""

    def __init__(self):
        pass


class Pylearn2Cache(PYLEARN2_LOADING_MODE):
    def __init__(self, queue, provider, batch_size, preprocessor=None):
        super(Pylearn2Cache, self).__init__()
        self.queue = queue
        self.batch_size = batch_size
        self.provider = provider
        self.preprocessor = preprocessor
        self.deamon = True

    def run(self):
        raise NotImplementedError('Abstract class, not supposed to be run.')

    @staticmethod
    def make_queue(maxsize=5):
        return Queue(maxsize=maxsize)


class Pylearn2CacheSimple(Pylearn2Cache):
    def __init__(self,
                 queue,
                 provider,
                 batch_size,
                 preprocessor=None):

        super(Pylearn2CacheSimple, self).\
            __init__(queue=queue,
                     provider=provider,
                     batch_size=batch_size,
                     preprocessor=preprocessor)

        #TODO: add here an assert for SpliceFrames preprocessor in case
        #provider is utt based, not random...
        self.provider = BufferedProviderDataSpec(provider, batch_size)
        self.lfreq = 2 ** 20  # print progress/efficiency stats after 1M examples
        self.num_classes = self.provider.num_classes()

    def run(self):

        tstart = time.time()
        texamples = 0

        for data in self.provider:
            # TODO: do this working in both supervised and unsupervised modes

            if any([space is None for space in data]):
                continue

            xy = data
            if self.preprocessor is not None:
                xy = self.preprocessor.apply(xy)

            xy = list(xy)

            if len(xy) > 2:
                data = xy[0]
                y = xy[1:]
            else:
                data, y = xy

            if isinstance(y, (list, tuple)):
                assert isinstance(self.num_classes, (list, tuple)) and \
                       len(y) == len(self.num_classes), (
                    "Specified %i labels streams but provided " \
                    " only %i target dimensions. Specify n_classes" \
                    " as a list with one element per target stream." \
                    % (len(y), len(self.num_classes))
                )
                rval_y = [self.convert_to_one_hot(yel, self.num_classes[idx], 0) \
                          for yel, idx in zip(y, xrange(len(y)))]
                new_y = tuple(rval_y)
            else:
                rval_y = self.convert_to_one_hot(y, self.num_classes[0], 0)
                new_y = tuple([rval_y])

            rval = (data,) + new_y

            texamples += self.provider.batch_size
            if texamples % self.lfreq == 0:
                ttime = time.time() - tstart
                print 'Pylearn2CacheSimple : Consuming %i examples took %f minutes which gives ca. %f pres/second.' % (
                    texamples, ttime / 60., texamples / ttime)

            try:
                #the trainer calls iterator several times == creates separate queues which would become are eternally blocked
                #in this place as provider produce maxsize elems and then gets blocked, a quick solution is to wait for some time
                #and then stop the process (it is rather unlikely that computation code consumes less than a minibatch per 60s)
                self.queue.put(rval, block=True, timeout=60)
            except QueueFullException:
                return

        ttime = time.time() - tstart
        self.queue.put(QueueCacheLastElem(), block=True)  # all data has been loaded
        print 'Pylearn2CacheSimple : Consuming the entire set %i examples took %f minutes which gives ca. %f pres/second.' % (
            texamples, ttime / 60., texamples / ttime)

    def convert_to_one_hot(self, y, max_classes, min_class=0):

        if y.ndim != 1:
            raise ValueError("Called convert_to_one_hot on a DenseDesignMatrix whose labels aren't scalar.")

        if 'int' not in str(y.dtype):
            raise ValueError("Called convert_to_one_hot on a DenseDesignMatrix whose labels aren't integer-valued.")

        y = y - min_class
        yr = numpy.zeros((y.shape[0], max_classes), dtype=numpy.float32)

        for i in xrange(y.shape[0]):
            yr[i, y[i]] = 1

        return yr
