"""Functions for online preprocossing of minibatches sequentialy as they are returned from providers. In practice these are some feature-level transformers
i.e. splicing context frames or shuffling dimensions for frequency band convolutions"""

__authors__ = "Pawel Swietojanski"
__copyright__ = "Copyright 2013, University of Edinburgh"

import numpy
import theano


class OnlinePreprocessor(object):
    def __init__(self, act_on_sources='features'):
        assert act_on_sources is not None, (
            "You need to specify the source preprocessor should act on"
        )
        self.act_on_sources = act_on_sources if \
            isinstance(act_on_sources, tuple) else tuple(act_on_sources)

    def apply(self, xy):
        raise NotImplementedError('This is abstract class which does not implement preprocessing')

    def get_pre_data_spec(self):
        return NotImplementedError('This is abstract class which does not implement data_specs')

    def get_post_data_spec(self):
        return NotImplementedError('This is abstract class which does not implement data_specs')


class PipelineOnlinePreprocessor(OnlinePreprocessor):
    def __init__(self, items, act_on_sources='features'):
        super(PipelineOnlinePreprocessor, self).__init__(act_on_sources)
        self.items = items if items is not None else []
        
    def apply(self, xy):
        rval = xy
        for item in self.items:
            rval = item.apply(rval)
        return rval

    def get_pre_data_spec(self):
        return NotImplementedError('This is abstract class which did not implement data_specs')

    def get_post_data_spec(self):
        return NotImplementedError('This is abstract class which did not implement data_specs')


class SpliceFrames(OnlinePreprocessor):
    def __init__(self, left_context=4, right_context=4, add_paddings=True, act_on_sources='features'):
        """
        :param left_context: number of frames on the left from central one
        :param right_context: number of frames on the right from cetral one
        :param add_paddings: pad (replicate) frames at the beginning *_contex times
        :return: nparrray with spliced frames around central frame
        """
        super(SpliceFrames, self).__init__(act_on_sources)
        self._left_context = left_context
        self._right_context = right_context
        self._add_paddings = add_paddings

        assert self._left_context >= 0 and self._right_context >= 0

    def apply(self, xy):

        if (self._left_context + self._right_context)<1:
            return xy

        data = list(xy)
        x = data[0]

        if x.ndim == 2:
            rval_x = self.__splice_single_channel(x)
        elif x.ndim == 4: #Input from ReshapeAudioChannels, splice each of the channels separately

            #print 'SpliceFrames x shape', x.shape

            bs, channels, one, fs = x.shape

            tmprval = self.__splice_single_channel(x[:,0,0,:].reshape(bs,fs))
            feats_in_channel = tmprval.shape[1]

            rval = numpy.zeros((bs, channels, 1, feats_in_channel), dtype=theano.config.floatX)
            rval[:,0,0,:] = tmprval
            for c in xrange(1, channels):
                rval[:,c,0,:] = self.__splice_single_channel(x[:,c,0,:].reshape(bs,fs))

            #print 'SpliceFrames rval shape', rval.shape

            rval_x = rval
        else:
            raise Exception('SpliceFrames: expected tensor of length 2 or 4, got %i.'%len(x.shape))

        data[0] = rval_x

        return tuple(data)

    def __splice_single_channel(self, x):

        assert len(x.shape)==2, (
            "Frame splicing implemented only for 2D matrices."
        )

        num_examples, dim = x.shape
        ctx_win = numpy.arange((self._left_context+self._right_context+1) * dim)
        examples = numpy.arange(num_examples) * dim
        indexes  = examples[:, numpy.newaxis] + ctx_win[numpy.newaxis, :]

        inputs = x.flatten()
        if (self._add_paddings):
            padd_beg = numpy.tile(inputs[0:dim], self._left_context)
            padd_end = numpy.tile(inputs[-dim:], self._right_context)
            inputs = numpy.concatenate((padd_beg, inputs, padd_end))

        return numpy.asarray(inputs[indexes], dtype=numpy.float32)

    def get_pre_data_spec(self):
        return NotImplementedError('This is abstract class which did not implement data_specs')

    def get_post_data_spec(self):
        return NotImplementedError('This is abstract class which did not implement data_specs')


class CMVNNormaliser(OnlinePreprocessor):
    def __init__(self, mean,
                 stdev,
                 act_on_sources='features',
                 var_floor=0.01,
                 norm_vars=True):

        super(CMVNNormaliser, self).__init__(act_on_sources)
        self.mean = mean
        self.stdev = stdev
        self.var_floor = var_floor
        self.norm_vars = norm_vars

        assert var_floor > 0 and self.stdev > var_floor
        
    def apply(self, xy):
        x, y = xy
        rval=(x-self.mean)
        if self.norm_vars:
            rval /= self.stdev
        return (rval, y)

    def get_pre_data_spec(self):
        return NotImplementedError('This is abstract class which did not implement data_specs')

    def get_post_data_spec(self):
        return NotImplementedError('This is abstract class which did not implement data_specs')


class ReshapeAudioChannels(OnlinePreprocessor):
    """
    Should be called *before* SpliceFeats so splicing will be performed for each channel separately
    When the inputs are provided as multiple microphone array channels the feats are concatenated into a single large matrix -
    this is the most convenient way to do this and keep compatibility with Kaldi and also with pylearn2 MultiStreamKaldi... speech provider 
    This preprocessor reshapes the minibatches from (bs, feats_channels) to (bs, channels, 1, feats) which is compatible with desired output
    """
    def __init__(self, num_input_channels, act_on_sources='features'):
        super(ReshapeAudioChannels, self).__init__(act_on_sources)
        self._num_input_channels = num_input_channels
        assert self._num_input_channels >= 1
        
    def apply(self, xy):

        x, y = xy

        if self._num_input_channels == 1:
            return (x, y)

        #print 'ReshapeAudioChannels x shape', x.shape

        assert len(x.shape) == 2
        batch_size, tot_feats_dim = x.shape
        assert (tot_feats_dim % self._num_input_channels) == 0
        
        feats_in_channel = tot_feats_dim/self._num_input_channels
        rval = numpy.zeros((batch_size, self._num_input_channels, 1, feats_in_channel), dtype=theano.config.floatX)
        for i in xrange(self._num_input_channels):
            rval[:,i,0,:] = x[:,i*feats_in_channel:(i+1)*feats_in_channel]
        
        #print 'ReshapeAudioChannels rval shape', rval.shape
        
        return (rval, y)

    def get_pre_data_spec(self):
        return NotImplementedError('This is abstract class which did not implement data_specs')

    def get_post_data_spec(self):
        return NotImplementedError('This is abstract class which did not implement data_specs')


class FilterFrames(OnlinePreprocessor):
    """The preprocessor allows to remove or limit certain classes
    from training, i.e. limit the number of silence frames in adaptation
    """
    def __init__(self, exclude_list=[], ratio=1.0, act_on_sources=None):
        """
        exclude_list: id of classess one want to remove from minibatches
        ratio: remove 1/ratio datapoints describing certain classes from training
        """

        super(FilterFrames, self).__init__(act_on_sources)
        self.exclude_list = exclude_list
        self.ratio = ratio

        if self.ratio < 1.0:
            self.ratio = 1.0

    def apply(self, xy):
        x, y = xy
        if len(self.exclude_list) == 0:
            return xy
        return xy

    def get_pre_data_spec(self):
        return NotImplementedError('This is abstract class which did not implement data_specs')

    def get_post_data_spec(self):
        return NotImplementedError('This is abstract class which did not implement data_specs')


class ReorderByBands(OnlinePreprocessor):
    """
    WRITEME
    1) Apply this preprocessor *after* splicing
    """
    def __init__(self, num_bands,
                 context_window,
                 act_on_sources='features',
                 axes=('b','c',0,1),
                 tied_deltas=False,
                 tied_channels=True,
                 reshape_stride_for_1D=-1):

        super(ReorderByBands, self).__init__(act_on_sources)
        self.num_bands = num_bands
        self.context_window = context_window
        self.axes = axes
        self.tied_deltas = tied_deltas
        self.tied_channels = False #tied_channels - this has to be done at model level by dimshuffling
        self.reshape_stride_for_1D = reshape_stride_for_1D #test whether that will speed up efficiency
        
    def apply(self, xy):
        """
        1) x is assumed to be feature vector as read from kaldi or htk
          providers which is batch_size x (*s*tatics [+ *d*eltas + *d*elta *d*elta + ...]) : s+d+dd+...
        2) Features should be in (mel)-spectral domain (it's up to the user to check this!)
        3) Reordering shuffles dimensions so s,d and dd for the given band are together
           across all context frames (hence it is important do it after splicing)  
        4) If tied_deltas is true, s, d and dd will be in separate channels
        5) if x is multi-stream (i.e. one stream per microphone) are by defaults treated as separate streams
        6) Look at tests to get more intuition
        """

        data = list(xy)
        x = data[0]

        x_shape = x.shape
        xx = x
        if len(x_shape)==2:
            #Data is formatted as batches x dimensionality, reshape to (batches, channels, 1 /*since 1D conv*/, dimensionality)
            xx = numpy.reshape(x, (x_shape[0], 1, 1, x_shape[1]))
        
        batch_size, num_channels, _, dimension = xx.shape
        #print 'ReorderByBands xx shape ', xx.shape 
        
        if num_channels>1 and self.tied_deltas is True:
            raise NotImplementedError('Deltas and delta deltas can get separate filters only for a single channel inputs.')
        
        num_deltas = dimension/(self.num_bands * self.context_window) #this is actually the order of time derivatives + 1 for statics
        assert isinstance(num_deltas, (int, long)) and num_deltas >= 1
        
        #print 'Input shape %s '%(x.shape,)
        #print 'XX shape is %s '%(xx.shape,)
        
        #reorder s,d and dd according to bands across the whole con qtext window 
        #i.e. s0,d0,dd0_{t:T}s1,d1,dd1_{t:T},...,sb,db,ddb_{t:T} where b = 0..num_bands
        rval = numpy.zeros((batch_size, num_channels, 1, dimension), dtype=theano.config.floatX)
        feats_in_band = num_deltas*self.context_window
        for b in xrange(self.num_bands):
            rval[:, :, 0, b*feats_in_band:(b+1)*feats_in_band] = \
               xx[:, :, 0, b:dimension:self.num_bands]
        
        if self.tied_deltas and num_channels==1: 
            #rval has bandwise feats in order s1,d1,dd1,s2,d2,dd2 so we pick every num_deltas elems and put into separate channels
            tied_d_dimension = dimension/num_deltas
            rval2 = numpy.zeros((batch_size, num_deltas, 1, tied_d_dimension), dtype=theano.config.floatX)
            for d in xrange(num_deltas):
                rval2[:, d, :, :] = rval[:, 0, 0, d:dimension:num_deltas].reshape(batch_size, 1, tied_d_dimension)
            rval = rval2
            
        
        if self.reshape_stride_for_1D > 1:
            #instead of (mb, c, 1, feats) produces (mb, c, feats/stride, stride)
            assert isinstance(rval.shape[3]/self.reshape_stride_for_1D, (int, long))
            rval = rval.reshape(rval.shape[0]. rval.shape[1], rval.shape[3]/self.reshape_stride_for_1D, self.reshape_stride_for_1D)

        data[0] = rval
        #print 'ReorderByBands rval shape is ',rval.shape
        return tuple(data)

    def get_pre_data_spec(self):
        return NotImplementedError('This is abstract class which did not implement data_specs')

    def get_post_data_spec(self):
        return NotImplementedError('This is abstract class which did not implement data_specs')


class ReorderByBandsPermute(OnlinePreprocessor):
    """
    WRITEME
    1) Apply this preprocessor *after* splicing
    """
    def __init__(self, num_bands,
                 context_window,
                 act_on_sources='features',
                 axes=('b','c',0,1),
                 tied_deltas=True,
                 reshape_stride_for_1D=-1):

        super(ReorderByBandsPermute, self).__init__(act_on_sources)
        self.num_bands = num_bands
        self.context_window = context_window
        self.axes = axes
        self.tied_deltas = tied_deltas
        self.tied_channels = False #tied_channels - this has to be done at model level by dimshuffling
        self.reshape_stride_for_1D = reshape_stride_for_1D #test whether that will speed up efficiency
        self.permute = None
        
        if tied_deltas is True:
            raise NotImplementedError('ReorderByBandsPermute does not support reordering deltas into separate channels. Use ReorderByBands instead.')
        
    def apply(self, xy):
        """
        1) x is assumed to be feature vector as read from kaldi or htk
          providers which is batch_size x (*s*tatics [+ *d*eltas + *d*elta *d*elta + ...]) : s+d+dd+...
        2) Features should be in (mel)-spectral domain otherwise all of this does not make sense
        3) Reordering shuffles dimensions so s,d and dd for the given band are together 
           across all context frames (hence it is important do it after splicing)  
        4) If tied_deltas is true, s, d and dd will be reordered to separate channels
        5) if x is 4D tensor (i.e. one channel per microphone) are by default treated as separate streams
        6) Look at tests to get more intuition
        """

        x, y = xy

        x_shape = x.shape
        xx = x
        if len(x_shape)==2:
            #Data is formatted as batches x dimensionality, reshape to batches x channels x 1 (since 1d conv) x dimensionality
            xx = numpy.reshape(x, (x_shape[0], 1, 1, x_shape[1]))
        
        batch_size, num_channels, _, dimension = xx.shape
        #print 'ReorderByBands xx shape ', xx.shape 
        
        if num_channels>1 and self.tied_deltas is True:
            raise NotImplementedError('Deltas and delta deltas can get separate filters only for a single channel inputs.')
                
        num_deltas = dimension/(self.num_bands * self.context_window) #this is actually the number of differentials + statics
        assert isinstance(num_deltas, (int, long)) and num_deltas >= 1
        
        if self.permute is None:
            self.permute = self.__generate_permute_matrix(dimension, self.num_bands)
        
        #print 'Input shape %s '%(x.shape,)
        #print 'XX shape is %s '%(xx.shape,)
        
        rval = numpy.dot(x, self.permute)
        
        if self.reshape_stride_for_1D > 1:
            #instead of (mb, c, 1, feats) produces (mb, c, feats/stride, stride)
            rval = rval.reshape(rval.shape[0]. rval.shape[1], rval.shape[3]/self.reshape_stride_for_1D, self.reshape_stride_for_1D)
        
        #print 'ReorderByBandsPermute rval shape is ',rval.shape
        return (rval, y)
    
    def invert(self, x):
        if self.permute is None:
            raise NotImplementedError('ReorderByBandsPermute: implement generation of self.permute invert function')
        return numpy.dot(x, self.permute.transpose())
            
    
    def __generate_permute_matrix(self, dim, num_bands):
        permute = numpy.zeros((dim, dim), dtype=theano.config.floatX)
        # B-th band in Kaldi features (statics xN, d xN, dd xN, ...) is xx[:,:,:,B:dimension:num_bands]
        for k in xrange(num_bands):
            idx = numpy.arange(k, dim, num_bands)
            band_size = idx.shape[0]
            for i in xrange(band_size):
                permute[idx[i],i+k*band_size] = 1     
        return permute

    def get_pre_data_spec(self):
        return NotImplementedError('This is abstract class which did not implement data_specs')

    def get_post_data_spec(self):
        return NotImplementedError('This is abstract class which did not implement data_specs')


class SpliceChannels(OnlinePreprocessor):
    def __init__(self):
        pass
    
    def apply(self, xy):
        pass

