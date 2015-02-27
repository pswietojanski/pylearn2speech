__authors__ = "Pawel Swietojanski"
__copyright__ = "Copyright 2013, University od Edinburgh"
__credits__ = ["Pawel Swietojanski, (Thanks to Sander Dieleman for sugession on Theano-user group)"]
__license__ = "3-clause BSD"
__maintainer__ = "Pawel Swietojanski"
__email__ = "p.swietojanski@ed.ac.uk"

import functools
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d

from pylearn2.packaged_dependencies.theano_linear.conv2d import Conv2d as OrigConv2D
from pylearn2.linear.linear_transform import LinearTransform as P2LT
from pylearn2.utils import sharedX

class Conv1D(OrigConv2D):
    """ Use the theanoLinear Conv2d class to support conv1d operations

    Also extend it to handle different axis semantics."""

    def __init__(self,
            filters,
            batch_size,
            input_space,
            output_axes = ('b','c',0,1),
        subsample = (1, 1), border_mode = 'valid',
        filters_shape = None, message = '',
        tied_input_channels = False,
        channelwise_conv = False):

        self.input_space = input_space
        self.output_axes = output_axes
        self.tied_input_channels = tied_input_channels
        self.channelwise_conv = channelwise_conv 
        
        super(Conv1D,self).__init__(filters = filters,
                img_shape = (batch_size, input_space.num_channels,\
                    input_space.shape[0], input_space.shape[1]),
                subsample = subsample,
                border_mode = border_mode,
                filters_shape = filters_shape,
                message = message)

    @functools.wraps(P2LT.get_params)
    def get_params(self):
        return [ self._filters ]

    @functools.wraps(P2LT.get_weights_topo)
    def get_weights_topo(self,borrow):
        return np.transpose(self._filters.get_value(borrow = borrow),(0,2,3,1))

    def lmul(self, x):
        """
        dot(x, A)

        This method overrides the original Conv2D lmul to make it work
        with arbitrary axis orders """

        # x must be formatted as (batch index, channel, topo dim 0, topo dim 1)
        # for use with nnet.conv2d
        assert x.ndim == 4
        axes = self.input_space.axes
        assert len(axes) == 4
        
        #reshape only if stride is > 1, otherwise use con2d as it is
        reshape_for_strided = (self._subsample[1] > 1 and
                                (self._img_shape[3] % self._subsample[1])==0)
        
        if reshape_for_strided:
            r_x_shape = (self._img_shape[0], self._img_shape[1],
                         self._img_shape[3] / self._subsample[1],
                         self._subsample[1]) # (mb size, #out, length/stride, stride)
            r_x = x.reshape(r_x_shape)    
            r_filters_shape = (self._filters_shape[0], self._filters_shape[1],
                               self._filters_shape[3] / self._subsample[1],
                               self._subsample[1])
        else:
            r_x_shape = self._img_shape
            r_x = x
            r_filters_shape = self._filters_shape
       
        #tying channels make sense only if there is more than one         
        if self.tied_input_channels and r_filters_shape[1]>1: 
            #self._filters are (nb filt, 1, 1, filt cols) - concatenate to
            # get (nb filt, nb input channels, 1, filt cols)
            r_filters_reshape = (r_filters_shape[0], 1, r_filters_shape[2], r_filters_shape[3])
            r_filters_t = self._filters.reshape(r_filters_reshape)
            r_filters = T.concatenate([r_filters_t for i in xrange(r_filters_shape[1])], axis=1)
        else:      
            r_filters = self._filters.reshape(r_filters_shape)

        op_axes = ('b', 'c', 0, 1)

        if tuple(axes) != op_axes:
            x = x.dimshuffle(
                axes.index('b'),
                axes.index('c'),
                axes.index(0),
                axes.index(1))       
        
        if self.channelwise_conv:
            
            #shapes need to reflect the real (channelwise) convolved regions i.e. 4D tensor
            elem_x_shape = (r_x_shape[0], 1, r_x_shape[2], r_x_shape[3])
            elem_filters_shape = (r_filters_shape[0], 1, r_filters_shape[2], r_filters_shape[3]) 
            
            #below Op will convolve the filters with input channels sequentially and store the
            #output in a 4D tensor of shape (batch_size, num_filters*num_inp_channels, 1, cols)
            #this allows to later on, for example, maxpool the channel-specific featrue maps channels
            rval = T.concatenate([
                                    conv2d(r_x[:,i,:,:].dimshuffle(0,'x',1,2),
                                    r_filters[:,i,:,:].dimshuffle(0,'x',1,2),
                                    image_shape = elem_x_shape,
                                    filter_shape = elem_filters_shape,
                                    subsample = (1,1),
                                    border_mode = self._border_mode)
                                    for i in xrange(r_filters_shape[1])
                                 ], axis=1)
        else:
            rval =  conv2d(
                r_x, r_filters,
                image_shape=r_x_shape,
                filter_shape=r_filters_shape,
                subsample=(1,1),
                border_mode=self._border_mode,
                )
        
        if reshape_for_strided: #when reshaping for conv1D get rid of the last dimension, and reshape to get a desired conv2d 4D rval
            rval = rval[:,:,:,0]
            #rval = rval.reshape((self._img_shape[0], self._filters_shape[0], 1, -1)) #(mb, c, 1, convbands)
            rval = rval.dimshuffle(0,1,'x',2) #make it compatible with (b,c,0,1) where 0 topological axis has always 1 dimension

        # Format the output based on the output space
        axes = self.output_axes
        assert len(axes) == 4

        if tuple(axes) != op_axes:
            rval = rval.dimshuffle(
                    op_axes.index(axes[0]),
                    op_axes.index(axes[1]),
                    op_axes.index(axes[2]),
                    op_axes.index(axes[3]))

        return rval

    def lmul_T(self, x):
        """ override the original Conv2D lmul_T to make it work
        with pylearn format of topological data using dimshuffles """
        raise NotImplementedError('Conv1D: lmul_T not implemented')

    def lmul_sq_T(self, x):
        """ Kind of a stupid hacky method used to support convolutional score matching.
        Ought to find a way to make _filters symbolic rather than shared.
        """
        raise NotImplementedError('Conv1D: lmul_T not implemented')

    def set_batch_size(self, batch_size):
        self._img_shape = tuple([ batch_size ] + list(self._img_shape[1:]))


def default_rng():
    return np.random.RandomState([2012, 11, 6, 9])


def make_random_conv1D(irange, input_space, output_space,
        kernel_shape, batch_size, \
        subsample = (1,1), border_mode = 'valid', message = "", rng = None, 
        tied_input_channels=False, channelwise_conv=False):
    """ Creates a Conv1D with random kernels """
        
    if rng is None:
        rng = default_rng()
    
    output_channels = output_space.num_channels
    if channelwise_conv is True: #output space in this case is num inp channels times bigger, but we want to have the same number of weights
        output_channels /= input_space.num_channels
    
    filters_shape = (output_channels, input_space.num_channels, kernel_shape[0], kernel_shape[1])
    
    if tied_input_channels is True:
        W = sharedX( rng.uniform(-irange,irange, (output_channels, 1, kernel_shape[0], kernel_shape[1]))) #will concatenate across channels
        #W = T.concatenate([WW for i in xrange(filters_shape[1])], axis=1)
    else:
        W = sharedX( rng.uniform(-irange, irange, filters_shape))
    
    #print 'make_random_conv1D shape', W.get_value().shape
    
    return Conv1D(filters = W,
        batch_size = batch_size,
        input_space = input_space,
        output_axes = output_space.axes,
        subsample = subsample, border_mode = border_mode,
        filters_shape = filters_shape, message = message, 
        tied_input_channels = tied_input_channels,
        channelwise_conv = channelwise_conv)


def default_sparse_rng():
    return np.random.RandomState([2012, 11, 6])

def make_sparse_random_conv1D(num_nonzero, input_space, output_space,
        kernel_shape, batch_size, \
        subsample = (1,1), border_mode = 'valid', message = "", rng=None, tied_input_channels=False):
    """ Creates a Conv2D with random kernels, where the randomly initialized
    values are sparse"""

    if rng is None:
        rng = default_sparse_rng()

    W = np.zeros(( output_space.num_channels, input_space.num_channels, \
            kernel_shape[0], kernel_shape[1]))

    def random_coord():
        return [ rng.randint(dim) for dim in W.shape ]

    for i in xrange(num_nonzero):
        o, ch, r, c = random_coord()
        while W[o, ch, r, c] != 0:
            o, ch, r, c = random_coord()
        W[o, ch, r, c] = rng.randn()

    if tied_input_channels is True:
        WW = sharedX( rng.uniform(-irange,irange,( output_space.num_channels, 1, \
            kernel_shape[0], kernel_shape[1])))
        W = WW.dimshuffle(0,'x',2,3)
    else:
        W = sharedX( rng.uniform(-irange,irange,( output_space.num_channels, input_space.num_channels, \
            kernel_shape[0], kernel_shape[1])))

    return Conv1D(filters = W,
        batch_size = batch_size,
        input_space = input_space,
        output_axes = output_space.axes,
        subsample = subsample, border_mode = border_mode,
        filters_shape = W.get_value(borrow=True).shape, message = message)
