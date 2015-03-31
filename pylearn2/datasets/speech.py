
__authors__ = "Pawel Swietojanski"
__copyright__ = "Copyright 2013, University of Edinburgh"
__credits__ = ["Pawel Swietojanski"]
__license__ = "3-clause BSD"
__maintainer__ = "Pawel Swietojanski"
__email__ = "p.swietojanski@ed.ac.uk"

import functools
import warnings
import numpy as np

from theano import config
from pylearn2.datasets.dataset import Dataset
from pylearn2.datasets.speech_utils.kaldi_providers import KaldiAlignFeatsProviderUtt, KaldiFeatsProviderUtt, MultiStreamKaldiAlignFeatsProviderUtt
from pylearn2.datasets.speech_utils.cache import Pylearn2CacheSimple
from pylearn2.utils.iteration import QueuedDatasetIterator

def get_kaldi_provider(flist, aligns, template_call=None, shuffle_flists=True, subset=-1, max_time=-1, supervised=True):
    provider = None
    if supervised is False:
        provider = KaldiFeatsProviderUtt(files_paths_list = flist, 
                            template_shell_command = template_call, 
                            randomize = shuffle_flists, 
                            max_utt = subset,
                            max_time = max_time)
    else:
        if isinstance(flist, list):
            provider = MultiStreamKaldiAlignFeatsProviderUtt(files_paths_lists = flist,
                                                        align_file=aligns,
                                                        template_shell_command=template_call,
                                                        randomize = shuffle_flists, 
                                                        subset = subset)
        else:
            provider = KaldiAlignFeatsProviderUtt(files_paths_list = flist,
                                        template_shell_command = template_call, 
                                        align_file = aligns, 
                                        randomize = shuffle_flists, 
                                        max_utt = subset,
                                        max_time = max_time)
    return provider

class SpeechDataset(Dataset):
    """WRITEME:
    1) X and y keeps only portion of the whole dataset (many speech datasets are impossible to fit into (GPU) memory at once)
    2) The iterator communicates through queues only
    3) [Optional] preprocessor acts online, per mini-batch
    """
    def __init__(self, flist,
                       aligns=None,
                       template_shell_command='copy-feats-1file scp:\"echo ${SCP_ENTRY}|\" -',
                       num_classes = None,
                       shuffle_flists=False,
                       subset=-1,
                       max_time=-1,
                       preprocessor=None):
        
        #self.args = locals()
        
        self.preprocessor = preprocessor
        self.num_classes = num_classes
        
        #TODO: determine provider type based on flist content or explicitly provide and arg for that
        prov_type = 'kaldi'
        self.provider = None
        self.supervised = (aligns != None)
        if prov_type == 'kaldi':
            self.provider = get_kaldi_provider(flist, aligns, template_shell_command, shuffle_flists, subset, max_time, self.supervised)
        elif prov_type  == 'htk':
            raise NotImplementedError('HTK implementation not ready. Feel free to contribute.')
        else:
            raise NotImplementedError('Unkown feature list or forced-aligns. Currently supported are lists compatible with HTK and Kaldi tools.')
        
        
        # will be instantiated by self.iterator() when needed
        self._queue = None
        self._cache = None
            
    """Abstract interface for Datasets."""
    def get_batch_design(self, batch_size, include_labels=False):
        """
        Returns a randomly chosen batch of data formatted as a design
        matrix.

        Deprecated, use `iterator()`.
        """
        raise NotImplementedError()

    def get_batch_topo(self, batch_size):
        """
        Returns a topology-preserving batch of data.

        The first index is over different examples, and has length
        batch_size. The next indices are the topologically significant
        dimensions of the data, i.e. for images, image rows followed by
        image columns.  The last index is over separate channels.

        Deprecated, use `iterator()`.
        """
        raise NotImplementedError()

    def __iter__(self):
        return self.iterator()

    @functools.wraps(Dataset.iterator)
    def iterator(self,mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):
        """
        Return an iterator for this dataset with the specified
        behaviour. Unspecified values are filled-in by the default.

        Parameters
        ----------
        mode : str or object, optional
            One of 'sequential', 'random_slice', or 'random_uniform',
            *or* a class that instantiates an iterator that returns
            slices or index sequences on every call to next().
        batch_size : int, optional
            The size of an individual batch. Optional if `mode` is
            'sequential' and `num_batches` is specified (batch size
            will be calculated based on full dataset size).
        num_batches : int, optional
            The total number of batches. Unnecessary if `mode` is
            'sequential' and `batch_size` is specified (number of
            batches will be calculated based on full dataset size).
        topo : boolean, optional
            Whether batches returned by the iterator should present
            examples in a topological view or not. Defaults to
            `False`.
        rng : int, object or array_like, optional
            Either an instance of `numpy.random.RandomState` (or
            something with a compatible interface), or a seed value
            to be passed to the constructor to create a `RandomState`.
            See the docstring for `numpy.random.RandomState` for
            details on the accepted seed formats. If unspecified,
            defaults to using the dataset's own internal random
            number generator, which persists across iterations
            through the dataset and may potentially be shared by
            multiple iterator objects simultaneously (see "Notes"
            below).
        targets: TODO WRITEME: DWF or LD should fill this in, but
            IG thinks it is just a bool saying whether to include
            the targets or not

        Returns
        -------
        iter_obj : object
            An iterator object implementing the standard Python
            iterator protocol (i.e. it has an `__iter__` method that
            return the object itself, and a `next()` method that
            returns results until it raises `StopIteration`).

        Notes
        -----
        Arguments are passed as instantiation parameters to classes
        that derive from `pylearn2.utils.iteration.SubsetIterator`.

        Iterating simultaneously with multiple iterator objects
        sharing the same random number generator could lead to
        difficult-to-reproduce behaviour during training. It is
        therefore *strongly recommended* that each iterator be given
        its own random number generator with the `rng` pabrameter
        in such situations.
        """
        
        assert self.provider != None
        
        self.provider.reset()
        
        self._queue = None
        self._cache = None
        
        self._queue = Pylearn2CacheSimple.make_queue(15)
        self._cache = Pylearn2CacheSimple(queue=self._queue, provider=self.provider, batch_size=batch_size, \
                                          num_classes=self.num_classes, preprocessor=self.preprocessor) 
        self._cache.start()
        
        return QueuedDatasetIterator(queue=self._queue, dataset_size=self.provider.num_examples, batch_size=batch_size)

    def adjust_for_viewer(self, X):
        """
            X: a tensor in the same space as the data
            returns the same tensor shifted and scaled by a transformation
            that maps the data range to [-1, 1] so that it can be displayed
            with pylearn2.gui.patch_viewer tools

            for example, for MNIST X will lie in [0,1] and the return value
                should be X*2-1

            Default is to do nothing
        """

        return X

    def has_targets(self):
        """ Returns true if the dataset includes targets """

        return self.supervised

    def get_topo_batch_axis(self):
        """
        Returns the index of the axis that corresponds to different examples
        in a batch when using topological_view.
        """

        # Subclasses that support topological view must implement this to
        # specify how their data is formatted.
        return 0
    
    def cache(self):
        return self._cache
    
    def queue(self):
        return self._queue


class SpeechDatasetProvider(Dataset):
    """WRITEME:
    1) X and y keeps only portion of the whole dataset (many speech datasets are impossible to fit into (GPU) memory at once)
    2) The iterator communicates through queues only
    3) [Optional] preprocessor acts online, per mini-batch
    """
    def __init__(self,
                 provider,
                 preprocessor=None):

        #self.args = locals()
        self.provider = provider
        self.preprocessor = preprocessor

        # will be instantiated by self.iterator() when asked for
        # by training code
        self._queue = None
        self._cache = None

    """Abstract interface for Datasets."""
    def get_batch_design(self, batch_size, include_labels=False):
        """
        Returns a randomly chosen batch of data formatted as a design
        matrix.

        Deprecated, use `iterator()`.
        """
        raise NotImplementedError()

    def get_batch_topo(self, batch_size):
        """
        Returns a topology-preserving batch of data.

        The first index is over different examples, and has length
        batch_size. The next indices are the topologically significant
        dimensions of the data, i.e. for images, image rows followed by
        image columns.  The last index is over separate channels.

        Deprecated, use `iterator()`.
        """
        raise NotImplementedError()

    def __iter__(self):
        return self.iterator()

    @functools.wraps(Dataset.iterator)
    def iterator(self,mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):
        """
        Return an iterator for this dataset with the specified
        behaviour. Unspecified values are filled-in by the default.

        Parameters
        ----------
        mode : str or object, optional
            One of 'sequential', 'random_slice', or 'random_uniform',
            *or* a class that instantiates an iterator that returns
            slices or index sequences on every call to next().
        batch_size : int, optional
            The size of an individual batch. Optional if `mode` is
            'sequential' and `num_batches` is specified (batch size
            will be calculated based on full dataset size).
        num_batches : int, optional
            The total number of batches. Unnecessary if `mode` is
            'sequential' and `batch_size` is specified (number of
            batches will be calculated based on full dataset size).
        topo : boolean, optional
            Whether batches returned by the iterator should present
            examples in a topological view or not. Defaults to
            `False`.
        rng : int, object or array_like, optional
            Either an instance of `numpy.random.RandomState` (or
            something with a compatible interface), or a seed value
            to be passed to the constructor to create a `RandomState`.
            See the docstring for `numpy.random.RandomState` for
            details on the accepted seed formats. If unspecified,
            defaults to using the dataset's own internal random
            number generator, which persists across iterations
            through the dataset and may potentially be shared by
            multiple iterator objects simultaneously (see "Notes"
            below).
        targets: TODO WRITEME: DWF or LD should fill this in, but
            IG thinks it is just a bool saying whether to include
            the targets or not

        Returns
        -------
        iter_obj : object
            An iterator object implementing the standard Python
            iterator protocol (i.e. it has an `__iter__` method that
            return the object itself, and a `next()` method that
            returns results until it raises `StopIteration`).

        Notes
        -----
        Arguments are passed as instantiation parameters to classes
        that derive from `pylearn2.utils.iteration.SubsetIterator`.

        Iterating simultaneously with multiple iterator objects
        sharing the same random number generator could lead to
        difficult-to-reproduce behaviour during training. It is
        therefore *strongly recommended* that each iterator be given
        its own random number generator with the `rng` pabrameter
        in such situations.
        """

        assert self.provider is not None

        self.provider.reset()

        self._queue = None
        self._cache = None

        self._queue = Pylearn2CacheSimple.make_queue(15)
        self._cache = Pylearn2CacheSimple(queue=self._queue,
                                          provider=self.provider,
                                          batch_size=batch_size,
                                          preprocessor=self.preprocessor)
        self._cache.start()

        return QueuedDatasetIterator(queue=self._queue,
                                     dataset_size=self.provider.num_examples,
                                     batch_size=batch_size)

    def adjust_for_viewer(self, X):
        """
            X: a tensor in the same space as the data
            returns the same tensor shifted and scaled by a transformation
            that maps the data range to [-1, 1] so that it can be displayed
            with pylearn2.gui.patch_viewer tools

            for example, for MNIST X will lie in [0,1] and the return value
                should be X*2-1

            Default is to do nothing
        """

        return X

    def has_targets(self):
        """ Returns true if the dataset includes targets """
        return self.provider.has_targets()

    def get_topo_batch_axis(self):
        """
        Returns the index of the axis that corresponds to different examples
        in a batch when using topological_view.
        """

        # Subclasses that support topological view must implement this to
        # specify how their data is formatted.
        return 0

    def get_data_specs(self):
        return self.data_spec

    def cache(self):
        return self._cache

    def queue(self):
        return self._queue
