
import random
import time
import thread
import sys
import numpy

from Queue import Empty as QueueEmptyException
from Queue import Full as QueueFullException 

QUEUE_THREADS=1
QUEUE_PROCESSES=2

PYLEARN2_LOADING_MODE = QUEUE_THREADS
#PYLEARN2_LOADING_MODE = QUEUE_PROCESSES
if PYLEARN2_LOADING_MODE == QUEUE_THREADS:
    from threading import Thread as PYLEARN2_LOADING_MODE
    from Queue import Queue
elif PYLEARN2_LOADING_MODE == QUEUE_PROCESSES:
    from multiprocessing import Process as PYLEARN2_LOADING_MODE
    from multiprocessing import Queue
assert PYLEARN2_LOADING_MODE!=None

from pylearn2.datasets.speech_utils.providers import BufferedProvider

class QueueCacheLastElem(object):
    """Special class denoting when StopIteration exception should be raised in Queue-based dataset interfaces."""
    def __init__(self):
        pass

class Pylearn2Cache(PYLEARN2_LOADING_MODE):
    def __init__(self, queue, provider, batch_size, num_classes, preprocessor=None):
        
        super(Pylearn2Cache, self).__init__()
        self.queue = queue
        self.batch_size = batch_size
        self.provider = provider
        self.preprocessor = preprocessor
        self.num_classes = num_classes
        self.deamon = True
        
    def run(self):
        pass
        
    @staticmethod
    def make_queue(maxsize=5):
        return Queue(maxsize=maxsize)

class Pylearn2CacheSimple(Pylearn2Cache):

    def __init__(self, queue, provider, batch_size, num_classes, preprocessor=None, frame_shuffling_window=None):
        super(Pylearn2CacheSimple, self).__init__(queue, provider, batch_size, num_classes, preprocessor)
        self.provider = BufferedProvider(provider, batch_size)
        self.frame_shuffling_window = frame_shuffling_window
        self.lfreq=2**20 #print progress/efficiency stats after 1M examples
        
    def run(self): 
        tstart = time.time()    
        texamples = 0
        #supervised = (not isinstance (self.provider, pylearn2.datasets.speech_utils.kaldi_providers.KaldiFeatsProviderUtt))
        for item in self.provider:
            #TODO: do this working in both supervised and unsupervised modes
            X, y = item 
            if X is None or y is None:
                continue
            data = X
            if self.preprocessor != None:
                data = self.preprocessor.apply(data)

            y = self.convert_to_one_hot(y, 0)
            rval = (data, y)
            
            texamples += self.provider.batch_size
            if texamples%self.lfreq==0:
                ttime = time.time()-tstart
                print 'Pylearn2CacheSimple : Consuming %i examples took %f minutes which gives ca. %f pres/second.'%(texamples, ttime/60., texamples/ttime)
            
            try: 
                #the trainer calls iterator several times == creates separate queues which would become are eternally blocked
                #in this place as provider produce maxsize elems and then gets blocked, a quick solution is to wait for some time
                #and then stop the process (it is rather unlikely that computation code consumes less than a minibatch per 60s)
                self.queue.put(rval, block=True, timeout=60)
            except QueueFullException:
                return
            
        ttime = time.time()-tstart
        self.queue.put(QueueCacheLastElem(), block=True) #all data has been loaded
        print 'Pylearn2CacheSimple : Consuming the entire set %i examples took %f minutes which gives ca. %f pres/second.'%(texamples, ttime/60., texamples/ttime)

    def frame_shuffler(self, (data, y)):
        assert len(data) == len(y)
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(data)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(y)

        return (data, y)

    def convert_to_one_hot(self, y, min_class=0):
        
        if y.ndim != 1:
            raise ValueError("Called convert_to_one_hot on a DenseDesignMatrix whose labels aren't scalar.")

        if 'int' not in str(y.dtype):
            raise ValueError("Called convert_to_one_hot on a DenseDesignMatrix whose labels aren't integer-valued.")

        y = y - min_class
        yr = numpy.zeros((y.shape[0], self.num_classes), dtype=numpy.float32)

        for i in xrange(y.shape[0]):
            yr[i, y[i]] = 1

        return yr
