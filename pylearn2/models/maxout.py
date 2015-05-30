"""
MLP Layer objects related to the paper

Maxout Networks. Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron
Courville, and Yoshua Bengio. ICML 2013.

If you use this code in your research, please cite this paper.

The objects in this module are Layer objects for use with
pylearn2.models.mlp.MLP. You need to make an MLP object in
order for thse to do anything. For an example of how to build
an MLP with maxout hidden layers, see pylearn2/scripts/papers/maxout.

Note that maxout is designed for use with dropout, so you really should
use dropout in your MLP when using these layers.

Note to developers / maintainers: when making changes to this module,
ensure that the changes do not break the examples in
pylearn2/scripts/papers/maxout.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

import warnings

import numpy as np
from theano import config
from theano.compat.python2x import OrderedDict
from theano.gof.op import get_debug_values
from theano.sandbox import cuda
from theano import tensor as T

from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.linear import conv2d, conv1d
from pylearn2.models.mlp import Layer
from pylearn2.space import Conv2DSpace
from pylearn2.space import VectorSpace
from pylearn2.utils import py_integer_types
from pylearn2.utils import sharedX

from pylearn2.linear.conv2d_c01b import setup_detector_layer_c01b
if cuda.cuda_available:
    from pylearn2.sandbox.cuda_convnet.pool import max_pool_c01b
from pylearn2.linear import local_c01b
from pylearn2.sandbox.cuda_convnet import check_cuda


class Maxout(Layer):
    """
    A hidden layer that does max pooling over groups of linear
    units. If you use this code in a research project, please
    cite

    "Maxout Networks" Ian J. Goodfellow, David Warde-Farley,
    Mehdi Mirza, Aaron Courville, and Yoshua Bengio. ICML 2013
    """

    def __init__(self,
                 layer_name,
                 num_units,
                 num_pieces,
                 pool_stride = None,
                 randomize_pools = False,
                 irange = None,
                 sparse_init = None,
                 sparse_stdev = 1.,
                 include_prob = 1.0,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 max_col_norm = None,
                 max_row_norm = None,
                 mask_weights = None,
                 min_zero = False
        ):
        """
            layer_name: A name for this layer that will be prepended to
                        monitoring channels related to this layer.
            num_units: The number of maxout units to use in this layer.
            num_pieces: The number of linear pieces to use in each maxout
                        unit.
            pool_stride: The distance between the start of each max pooling
                        region. Defaults to num_pieces, which makes the
                        pooling regions disjoint. If set to a smaller number,
                        can do overlapping pools.
            randomize_pools: Does max pooling over randomized subsets of
                        the linear responses, rather than over sequential
                        subsets.
            irange: if specified, initializes each weight randomly in
                U(-irange, irange)
            sparse_init: if specified, irange must not be specified.
                        This is an integer specifying how many weights to make
                        non-zero. All non-zero weights will be initialized
                        randomly in N(0, sparse_stdev^2)
            include_prob: probability of including a weight element in the set
               of weights initialized to U(-irange, irange). If not included
               a weight is initialized to 0. This defaults to 1.
            init_bias: All biases are initialized to this number
            W_lr_scale: The learning rate on the weights for this layer is
                multiplied by this scaling factor
            b_lr_scale: The learning rate on the biases for this layer is
                multiplied by this scaling factor
            max_col_norm: The norm of each column of the weight matrix is
                constrained to have at most this norm. If unspecified, no
                constraint. Constraint is enforced by re-projection (if
                necessary) at the end of each update.
            max_row_norm: Like max_col_norm, but applied to the rows.
            mask_weights: A binary matrix multiplied by the weights after each
                         update, allowing you to restrict their connectivity.
            min_zero: If true, includes a zero in the set we take a max over
                    for each maxout unit. This is equivalent to pooling over
                    rectified linear units.
        """

        detector_layer_dim = num_units * num_pieces
        pool_size = num_pieces

        if pool_stride is None:
            pool_stride = pool_size

        self.__dict__.update(locals())
        del self.self

        self.b = sharedX( np.zeros((self.detector_layer_dim,)) + init_bias, name = layer_name + '_b')


        if max_row_norm is not None:
            raise NotImplementedError()

    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    def set_input_space(self, space):
        """ Note: this resets parameters! """

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)


        if not ((self.detector_layer_dim - self.pool_size) % self.pool_stride == 0):
            if self.pool_stride == self.pool_size:
                raise ValueError("detector_layer_dim = %d, pool_size = %d. Should be divisible but remainder is %d" %
                             (self.detector_layer_dim, self.pool_size, self.detector_layer_dim % self.pool_size))
            raise ValueError()

        self.h_space = VectorSpace(self.detector_layer_dim)
        self.pool_layer_dim = (self.detector_layer_dim - self.pool_size)/ self.pool_stride + 1
        self.output_space = VectorSpace(self.pool_layer_dim)

        rng = self.mlp.rng
        if self.irange is not None:
            assert self.sparse_init is None
            W = rng.uniform(-self.irange,
                            self.irange,
                            (self.input_dim, self.detector_layer_dim)) * \
                (rng.uniform(0.,1., (self.input_dim, self.detector_layer_dim))
                 < self.include_prob)
        else:
            assert self.sparse_init is not None
            W = np.zeros((self.input_dim, self.detector_layer_dim))
            def mask_rejects(idx, i):
                if self.mask_weights is None:
                    return False
                return self.mask_weights[idx, i] == 0.
            for i in xrange(self.detector_layer_dim):
                assert self.sparse_init <= self.input_dim
                for j in xrange(self.sparse_init):
                    idx = rng.randint(0, self.input_dim)
                    while W[idx, i] != 0 or mask_rejects(idx, i):
                        idx = rng.randint(0, self.input_dim)
                    W[idx, i] = rng.randn()
            W *= self.sparse_stdev

        W = sharedX(W)
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)

        W ,= self.transformer.get_params()
        assert W.name is not None

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if self.randomize_pools:
            permute = np.zeros((self.detector_layer_dim, self.detector_layer_dim))
            for j in xrange(self.detector_layer_dim):
                i = rng.randint(self.detector_layer_dim)
                permute[i,j] = 1
            self.permute = sharedX(permute)

        if self.mask_weights is not None:
            expected_shape =  (self.input_dim, self.detector_layer_dim)
            if expected_shape != self.mask_weights.shape:
                raise ValueError("Expected mask with shape "+str(expected_shape)+" but got "+str(self.mask_weights.shape))
            self.mask = sharedX(self.mask_weights)

    def censor_updates(self, updates):

        # Patch old pickle files
        if not hasattr(self, 'mask_weights'):
            self.mask_weights = None

        if self.mask_weights is not None:
            W ,= self.transformer.get_params()
            if W in updates:
                updates[W] = updates[W] * self.mask

        if self.max_col_norm is not None:
            assert self.max_row_norm is None
            W ,= self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))

    def get_params(self):
        assert self.b.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    def get_l1_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.abs(W).sum()

    def get_weights(self):
        if self.requires_reformat:
            # This is not really an unimplemented case.
            # We actually don't know how to format the weights
            # in design space. We got the data in topo space
            # and we don't have access to the dataset
            raise NotImplementedError()
        W ,= self.transformer.get_params()
        W = W.get_value()

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if self.randomize_pools:
            warnings.warn("randomize_pools makes get_weights multiply by the permutation matrix. "
                    "If you call set_weights(W) and then call get_weights(), the return value will "
                    "WP not W.")
            P = self.permute.get_value()
            return np.dot(W,P)

        return W

    def set_weights(self, weights):
        W, = self.transformer.get_params()
        W.set_value(weights)

    def set_biases(self, biases):
        self.b.set_value(biases)

    def get_biases(self):
        return self.b.get_value()

    def get_weights_format(self):
        return ('v', 'h')

    def get_weights_view_shape(self):
        total = self.detector_layer_dim
        cols = self.pool_size
        if cols == 1:
            # Let the PatchViewer decide how to arrange the units
            # when they're not pooled
            raise NotImplementedError()
        # When they are pooled, make each pooling unit have one row
        rows = total // cols
        if rows * cols < total:
            rows = rows + 1
        return rows, cols


    def get_weights_topo(self):

        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()

        # There was an implementation of this, but it was broken
        raise NotImplementedError()

    def get_monitoring_channels(self):

        W ,= self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return OrderedDict([
                            ('row_norms_min'  , row_norms.min()),
                            ('row_norms_mean' , row_norms.mean()),
                            ('row_norms_max'  , row_norms.max()),
                            ('col_norms_min'  , col_norms.min()),
                            ('col_norms_mean' , col_norms.mean()),
                            ('col_norms_max'  , col_norms.max()),
                            ])


    def get_monitoring_channels_from_state(self, state):

        P = state

        rval = OrderedDict()

        if self.pool_size == 1:
            vars_and_prefixes = [ (P,'') ]
        else:
            vars_and_prefixes = [ (P, 'p_') ]

        for var, prefix in vars_and_prefixes:
            v_max = var.max(axis=0)
            v_min = var.min(axis=0)
            v_mean = var.mean(axis=0)
            v_range = v_max - v_min

            # max_x.mean_u is "the mean over *u*nits of the max over e*x*amples"
            # The x and u are included in the name because otherwise its hard
            # to remember which axis is which when reading the monitor
            # I use inner.outer rather than outer_of_inner or something like that
            # because I want mean_x.* to appear next to each other in the alphabetical
            # list, as these are commonly plotted together
            for key, val in [
                             ('max_x.max_u', v_max.max()),
                             ('max_x.mean_u', v_max.mean()),
                             ('max_x.min_u', v_max.min()),
                             ('min_x.max_u', v_min.max()),
                             ('min_x.mean_u', v_min.mean()),
                             ('min_x.min_u', v_min.min()),
                             ('range_x.max_u', v_range.max()),
                             ('range_x.mean_u', v_range.mean()),
                             ('range_x.min_u', v_range.min()),
                             ('mean_x.max_u', v_mean.max()),
                             ('mean_x.mean_u', v_mean.mean()),
                             ('mean_x.min_u', v_mean.min())
                             ]:
                rval[prefix+key] = val

        return rval

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)

        z = self.transformer.lmul(state_below) + self.b

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if not hasattr(self, 'pool_stride'):
            self.pool_stride = self.pool_size

        if self.randomize_pools:
            z = T.dot(z, self.permute)

        if not hasattr(self, 'min_zero'):
            self.min_zero = False

        if self.min_zero:
            p = 0.
        else:
            p = None

        last_start = self.detector_layer_dim  - self.pool_size
        for i in xrange(self.pool_size):
            cur = z[:,i:last_start+i+1:self.pool_stride]
            if p is None:
                p = cur
            else:
                p = T.maximum(cur, p)

        p.name = self.layer_name + '_p_'

        return p

    def foo(self, state_below):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)

        z = self.transformer.lmul(state_below) + self.b

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if not hasattr(self, 'pool_stride'):
            self.pool_stride = self.pool_size

        if self.randomize_pools:
            z = T.dot(z, self.permute)

        if not hasattr(self, 'min_zero'):
            self.min_zero = False

        if self.min_zero:
            p = T.zeros_like(z)
        else:
            p = None

        last_start = self.detector_layer_dim  - self.pool_size

        pooling_stack = []
        for i in xrange(self.pool_size):
            cur = z[:,i:last_start+i+1:self.pool_stride]
            cur = cur.reshape((cur.shape[0], cur.shape[1], 1))
            assert cur.ndim == 3
            pooling_stack.append(cur)
        if self.min_zero:
            pooling_stack.append(T.zeros_like(cur))
        pooling_stack = T.concatenate(pooling_stack, axis=2)
        p = pooling_stack.max(axis=2)
        counts = (T.eq(pooling_stack, p.dimshuffle(0, 1, 'x'))).sum(axis=0)

        p.name = self.layer_name + '_p_'

        return p, counts

class MaxoutLp(Layer):
    """
    A hidden layer that does max pooling over groups of linear
    units. If you use this code in a research project, please
    cite

    "Maxout Networks" Ian J. Goodfellow, David Warde-Farley,
    Mehdi Mirza, Aaron Courville, and Yoshua Bengio. ICML 2013
    """

    def __init__(self,
                 layer_name,
                 num_units,
                 num_pieces,
                 pool_stride = None,
                 pool_order = 2.0,
                 randomize_pools = False,
                 irange = None,
                 sparse_init = None,
                 sparse_stdev = 1.,
                 include_prob = 1.0,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 max_col_norm = None,
                 max_row_norm = None,
                 mask_weights = None,
                 min_zero = False,
                 pool_normalisation = False,
                 post_pool_normalisation = False,
                 weighted_normalisation = False,
        ):
        """
            layer_name: A name for this layer that will be prepended to
                        monitoring channels related to this layer.
            num_units: The number of maxout units to use in this layer.
            num_pieces: The number of linear pieces to use in each maxout
                        unit.
            pool_stride: The distance between the start of each max pooling
                        region. Defaults to num_pieces, which makes the
                        pooling regions disjoint. If set to a smaller number,
                        can do overlapping pools.
            randomize_pools: Does max pooling over randomized subsets of
                        the linear responses, rather than over sequential
                        subsets.
            irange: if specified, initializes each weight randomly in
                U(-irange, irange)
            sparse_init: if specified, irange must not be specified.
                        This is an integer specifying how many weights to make
                        non-zero. All non-zero weights will be initialized
                        randomly in N(0, sparse_stdev^2)
            include_prob: probability of including a weight element in the set
               of weights initialized to U(-irange, irange). If not included
               a weight is initialized to 0. This defaults to 1.
            init_bias: All biases are initialized to this number
            W_lr_scale: The learning rate on the weights for this layer is
                multiplied by this scaling factor
            b_lr_scale: The learning rate on the biases for this layer is
                multiplied by this scaling factor
            max_col_norm: The norm of each column of the weight matrix is
                constrained to have at most this norm. If unspecified, no
                constraint. Constraint is enforced by re-projection (if
                necessary) at the end of each update.
            max_row_norm: Like max_col_norm, but applied to the rows.
            mask_weights: A binary matrix multiplied by the weights after each
                         update, allowing you to restrict their connectivity.
            min_zero: If true, includes a zero in the set we take a max over
                    for each maxout unit. This is equivalent to pooling over
                    rectified linear units.
        """

        detector_layer_dim = num_units * num_pieces
        pool_size = num_pieces

        if pool_stride is None:
            pool_stride = pool_size

        self.__dict__.update(locals())
        del self.self

        self.b = sharedX( np.zeros((self.detector_layer_dim,)) + init_bias, name = layer_name + '_b')

        if max_row_norm is not None:
            raise NotImplementedError()

    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    def set_input_space(self, space):
        """ Note: this resets parameters! """

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)


        if not ((self.detector_layer_dim - self.pool_size) % self.pool_stride == 0):
            if self.pool_stride == self.pool_size:
                raise ValueError("detector_layer_dim = %d, pool_size = %d. Should be divisible but remainder is %d" %
                             (self.detector_layer_dim, self.pool_size, self.detector_layer_dim % self.pool_size))
            raise ValueError()

        self.h_space = VectorSpace(self.detector_layer_dim)
        self.pool_layer_dim = (self.detector_layer_dim - self.pool_size)/ self.pool_stride + 1
        self.output_space = VectorSpace(self.pool_layer_dim)

        rng = self.mlp.rng
        if self.irange is not None:
            assert self.sparse_init is None
            W = rng.uniform(-self.irange,
                            self.irange,
                            (self.input_dim, self.detector_layer_dim)) * \
                (rng.uniform(0.,1., (self.input_dim, self.detector_layer_dim))
                 < self.include_prob)
        else:
            assert self.sparse_init is not None
            W = np.zeros((self.input_dim, self.detector_layer_dim))
            def mask_rejects(idx, i):
                if self.mask_weights is None:
                    return False
                return self.mask_weights[idx, i] == 0.
            for i in xrange(self.detector_layer_dim):
                assert self.sparse_init <= self.input_dim
                for j in xrange(self.sparse_init):
                    idx = rng.randint(0, self.input_dim)
                    while W[idx, i] != 0 or mask_rejects(idx, i):
                        idx = rng.randint(0, self.input_dim)
                    W[idx, i] = rng.randn()
            W *= self.sparse_stdev

        W = sharedX(W)
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)

        W ,= self.transformer.get_params()
        assert W.name is not None

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if self.randomize_pools:
            permute = np.zeros((self.detector_layer_dim, self.detector_layer_dim))
            for j in xrange(self.detector_layer_dim):
                i = rng.randint(self.detector_layer_dim)
                permute[i,j] = 1
            self.permute = sharedX(permute)

        if self.mask_weights is not None:
            expected_shape =  (self.input_dim, self.detector_layer_dim)
            if expected_shape != self.mask_weights.shape:
                raise ValueError("Expected mask with shape "+str(expected_shape)+" but got "+str(self.mask_weights.shape))
            self.mask = sharedX(self.mask_weights)

    def censor_updates(self, updates):

        # Patch old pickle files
        if not hasattr(self, 'mask_weights'):
            self.mask_weights = None

        if self.mask_weights is not None:
            W ,= self.transformer.get_params()
            if W in updates:
                updates[W] = updates[W] * self.mask

        if self.max_col_norm is not None:
            assert self.max_row_norm is None
            W ,= self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))

    def get_params(self):
        assert self.b.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    def get_l1_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.abs(W).sum()

    def get_weights(self):
        if self.requires_reformat:
            # This is not really an unimplemented case.
            # We actually don't know how to format the weights
            # in design space. We got the data in topo space
            # and we don't have access to the dataset
            raise NotImplementedError()
        W ,= self.transformer.get_params()
        W = W.get_value()

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if self.randomize_pools:
            warnings.warn("randomize_pools makes get_weights multiply by the permutation matrix. "
                    "If you call set_weights(W) and then call get_weights(), the return value will "
                    "WP not W.")
            P = self.permute.get_value()
            return np.dot(W,P)

        return W

    def set_weights(self, weights):
        W, = self.transformer.get_params()
        W.set_value(weights)

    def set_biases(self, biases):
        self.b.set_value(biases)

    def get_biases(self):
        return self.b.get_value()

    def get_weights_format(self):
        return ('v', 'h')

    def get_weights_view_shape(self):
        total = self.detector_layer_dim
        cols = self.pool_size
        if cols == 1:
            # Let the PatchViewer decide how to arrange the units
            # when they're not pooled
            raise NotImplementedError()
        # When they are pooled, make each pooling unit have one row
        rows = total // cols
        if rows * cols < total:
            rows = rows + 1
        return rows, cols


    def get_weights_topo(self):

        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()

        # There was an implementation of this, but it was broken
        raise NotImplementedError()

    def get_monitoring_channels(self):

        W ,= self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return OrderedDict([
                            ('row_norms_min'  , row_norms.min()),
                            ('row_norms_mean' , row_norms.mean()),
                            ('row_norms_max'  , row_norms.max()),
                            ('col_norms_min'  , col_norms.min()),
                            ('col_norms_mean' , col_norms.mean()),
                            ('col_norms_max'  , col_norms.max()),
                            ])


    def get_monitoring_channels_from_state(self, state):

        P = state

        rval = OrderedDict()

        if self.pool_size == 1:
            vars_and_prefixes = [ (P,'') ]
        else:
            vars_and_prefixes = [ (P, 'p_') ]

        for var, prefix in vars_and_prefixes:
            v_max = var.max(axis=0)
            v_min = var.min(axis=0)
            v_mean = var.mean(axis=0)
            v_range = v_max - v_min

            # max_x.mean_u is "the mean over *u*nits of the max over e*x*amples"
            # The x and u are included in the name because otherwise its hard
            # to remember which axis is which when reading the monitor
            # I use inner.outer rather than outer_of_inner or something like that
            # because I want mean_x.* to appear next to each other in the alphabetical
            # list, as these are commonly plotted together
            for key, val in [
                             ('max_x.max_u', v_max.max()),
                             ('max_x.mean_u', v_max.mean()),
                             ('max_x.min_u', v_max.min()),
                             ('min_x.max_u', v_min.max()),
                             ('min_x.mean_u', v_min.mean()),
                             ('min_x.min_u', v_min.min()),
                             ('range_x.max_u', v_range.max()),
                             ('range_x.mean_u', v_range.mean()),
                             ('range_x.min_u', v_range.min()),
                             ('mean_x.max_u', v_mean.max()),
                             ('mean_x.mean_u', v_mean.mean()),
                             ('mean_x.min_u', v_mean.min())
                             ]:
                rval[prefix+key] = val

        return rval

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)

        z = self.transformer.lmul(state_below) + self.b

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if not hasattr(self, 'pool_stride'):
            self.pool_stride = self.pool_size

        if self.randomize_pools:
            z = T.dot(z, self.permute)

        if not hasattr(self, 'min_zero'):
            self.min_zero = False

        if self.min_zero:
            z_act = T.maximum(z, 0)
        else:
            z_act = T.maximum(T.abs_(z), 1e-10)
        
        z_act = z_act**self.pool_order
        
        r = None
        if self.weighted_normalisation:
            last_start = self.detector_layer_dim  - self.pool_size
            for i in xrange(self.pool_size):
                cur = z_act[:,i:last_start+i+1:self.pool_stride]
                if r is None:
                    r = cur
                else:
                    r = r + cur
            r = 1./r
            r.name='_r_'
        else:
            r = 1.
            
        p = None
        last_start = self.detector_layer_dim  - self.pool_size
        for i in xrange(self.pool_size):
            cur = z_act[:,i:last_start+i+1:self.pool_stride]
            if p is None:
                p = r*cur
            else:
                p = p + r*cur
        
        # this temporarily fix some derivative stability issues w.r.t x in case Lp norm sum_i x_i**p == 0.0
        # i.e. in practice, in case the above sum is zero or close enough, we take derivative w.r.t constatnt which has desirable derivative 0
        if self.pool_normalisation:
            p = p/self.pool_size
            
        p = T.maximum(p, 1e-10) 
        p = p**(1.0/self.pool_order)
        
        if self.post_pool_normalisation:
            p = p/self.pool_size
        
        p.name = self.layer_name + '_p_'

        return p

    def foo(self, state_below):
        raise NotImplementedError()

class MaxoutLpLearnable(Layer):
    """
    A hidden layer that does max pooling over groups of linear
    units. If you use this code in a research project, please
    cite

    "Maxout Networks" Ian J. Goodfellow, David Warde-Farley,
    Mehdi Mirza, Aaron Courville, and Yoshua Bengio. ICML 2013
    """

    def __init__(self,
                 layer_name,
                 num_units,
                 num_pieces,
                 pool_stride = None,
                 pool_order_constraints= None,
                 randomize_pools = False,
                 irange = None,
                 sparse_init = None,
                 sparse_stdev = 1.,
                 include_prob = 1.0,
                 init_bias = 0.,
                 init_pool_order = 3.0,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 p_lr_scale = None,
                 max_col_norm = None,
                 max_row_norm = None,
                 mask_weights = None,
                 min_zero = False
        ):
        """
            layer_name: A name for this layer that will be prepended to
                        monitoring channels related to this layer.
            num_units: The number of maxout units to use in this layer.
            num_pieces: The number of linear pieces to use in each maxout
                        unit.
            pool_stride: The distance between the start of each max pooling
                        region. Defaults to num_pieces, which makes the
                        pooling regions disjoint. If set to a smaller number,
                        can do overlapping pools.
            randomize_pools: Does max pooling over randomized subsets of
                        the linear responses, rather than over sequential
                        subsets.
            irange: if specified, initializes each weight randomly in
                U(-irange, irange)
            sparse_init: if specified, irange must not be specified.
                        This is an integer specifying how many weights to make
                        non-zero. All non-zero weights will be initialized
                        randomly in N(0, sparse_stdev^2)
            include_prob: probability of including a weight element in the set
               of weights initialized to U(-irange, irange). If not included
               a weight is initialized to 0. This defaults to 1.
            init_bias: All biases are initialized to this number
            W_lr_scale: The learning rate on the weights for this layer is
                multiplied by this scaling factor
            b_lr_scale: The learning rate on the biases for this layer is
                multiplied by this scaling factor
            max_col_norm: The norm of each column of the weight matrix is
                constrained to have at most this norm. If unspecified, no
                constraint. Constraint is enforced by re-projection (if
                necessary) at the end of each update.
            max_row_norm: Like max_col_norm, but applied to the rows.
            mask_weights: A binary matrix multiplied by the weights after each
                         update, allowing you to restrict their connectivity.
            min_zero: If true, includes a zero in the set we take a max over
                    for each maxout unit. This is equivalent to pooling over
                    rectified linear units.
        """

        detector_layer_dim = num_units * num_pieces
        pool_size = num_pieces

        if pool_stride is None:
            pool_stride = pool_size

        if pool_order_constraints is None:
            pool_order_constraints = [1, 10]

        self.__dict__.update(locals())
        del self.self

        self.b = sharedX( np.zeros((self.detector_layer_dim,)) + init_bias, name = layer_name + '_b')
        if self.init_pool_order is not None:
            self.Lp = sharedX( np.zeros((self.num_units,)) + self.init_pool_order, name = layer_name + '_Lp')
        else:
            self.Lp = sharedX( np.random.uniform(pool_order_constraints[0], pool_order_constraints[1], (self.num_units,)), name = layer_name + '_Lp')

        if max_row_norm is not None:
            raise NotImplementedError()

    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        if not hasattr(self, 'p_lr_scale'):
            self.p_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale
            
        if self.p_lr_scale is not None:
            rval[self.Lp] = self.p_lr_scale

        return rval

    def set_input_space(self, space):
        """ Note: this resets parameters! """

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)


        if not ((self.detector_layer_dim - self.pool_size) % self.pool_stride == 0):
            if self.pool_stride == self.pool_size:
                raise ValueError("detector_layer_dim = %d, pool_size = %d. Should be divisible but remainder is %d" %
                             (self.detector_layer_dim, self.pool_size, self.detector_layer_dim % self.pool_size))
            raise ValueError()

        self.h_space = VectorSpace(self.detector_layer_dim)
        self.pool_layer_dim = (self.detector_layer_dim - self.pool_size)/ self.pool_stride + 1
        self.output_space = VectorSpace(self.pool_layer_dim)

        rng = self.mlp.rng
        if self.irange is not None:
            assert self.sparse_init is None
            W = rng.uniform(-self.irange,
                            self.irange,
                            (self.input_dim, self.detector_layer_dim)) * \
                (rng.uniform(0.,1., (self.input_dim, self.detector_layer_dim))
                 < self.include_prob)
        else:
            assert self.sparse_init is not None
            W = np.zeros((self.input_dim, self.detector_layer_dim))
            def mask_rejects(idx, i):
                if self.mask_weights is None:
                    return False
                return self.mask_weights[idx, i] == 0.
            for i in xrange(self.detector_layer_dim):
                assert self.sparse_init <= self.input_dim
                for j in xrange(self.sparse_init):
                    idx = rng.randint(0, self.input_dim)
                    while W[idx, i] != 0 or mask_rejects(idx, i):
                        idx = rng.randint(0, self.input_dim)
                    W[idx, i] = rng.randn()
            W *= self.sparse_stdev

        W = sharedX(W)
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)

        W ,= self.transformer.get_params()
        assert W.name is not None

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if self.randomize_pools:
            permute = np.zeros((self.detector_layer_dim, self.detector_layer_dim))
            for j in xrange(self.detector_layer_dim):
                i = rng.randint(self.detector_layer_dim)
                permute[i,j] = 1
            self.permute = sharedX(permute)

        if self.mask_weights is not None:
            expected_shape =  (self.input_dim, self.detector_layer_dim)
            if expected_shape != self.mask_weights.shape:
                raise ValueError("Expected mask with shape "+str(expected_shape)+" but got "+str(self.mask_weights.shape))
            self.mask = sharedX(self.mask_weights)

    def censor_updates(self, updates):

        # Patch old pickle files
        if not hasattr(self, 'mask_weights'):
            self.mask_weights = None

        if self.mask_weights is not None:
            W ,= self.transformer.get_params()
            if W in updates:
                updates[W] = updates[W] * self.mask

        if self.max_col_norm is not None:
            assert self.max_row_norm is None
            W ,= self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, self.max_col_norm)
            
#        if self.pool_order_constraints is not None:
#            assert len(self.pool_order_constraints)==2
#            assert self.Lp in updates
#            Lp_orders = updates[self.Lp]
#            updates[self.Lp] = T.clip(Lp_orders, self.pool_order_constraints[0], self.pool_order_constraints[1])
            

    def get_params(self):
        assert self.b.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        assert self.Lp not in rval
        rval.append(self.Lp)
        return rval

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    def get_l1_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.abs(W).sum()

    def get_weights(self):
        if self.requires_reformat:
            # This is not really an unimplemented case.
            # We actually don't know how to format the weights
            # in design space. We got the data in topo space
            # and we don't have access to the dataset
            raise NotImplementedError()
        W ,= self.transformer.get_params()
        W = W.get_value()

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if self.randomize_pools:
            warnings.warn("randomize_pools makes get_weights multiply by the permutation matrix. "
                    "If you call set_weights(W) and then call get_weights(), the return value will "
                    "WP not W.")
            P = self.permute.get_value()
            return np.dot(W,P)

        return W

    def set_weights(self, weights):
        W, = self.transformer.get_params()
        W.set_value(weights)

    def set_biases(self, biases):
        self.b.set_value(biases)

    def get_biases(self):
        return self.b.get_value()

    def get_weights_format(self):
        return ('v', 'h')

    def get_weights_view_shape(self):
        total = self.detector_layer_dim
        cols = self.pool_size
        if cols == 1:
            # Let the PatchViewer decide how to arrange the units
            # when they're not pooled
            raise NotImplementedError()
        # When they are pooled, make each pooling unit have one row
        rows = total // cols
        if rows * cols < total:
            rows = rows + 1
        return rows, cols


    def get_weights_topo(self):

        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()

        # There was an implementation of this, but it was broken
        raise NotImplementedError()

    def get_monitoring_channels(self):

        W ,= self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))
        
        return OrderedDict([
                            ('row_norms_min'  , row_norms.min()),
                            ('row_norms_mean' , row_norms.mean()),
                            ('row_norms_max'  , row_norms.max()),
                            ('col_norms_min'  , col_norms.min()),
                            ('col_norms_mean' , col_norms.mean()),
                            ('col_norms_max'  , col_norms.max()),
                            ('Lp_orders_min'  , self.Lp.min()),
                            ('Lp_orders_mean' , self.Lp.mean()),
                            ('Lp_orders_max'  , self.Lp.max()),
                            ])


    def get_monitoring_channels_from_state(self, state):

        P = state

        rval = OrderedDict()

        if self.pool_size == 1:
            vars_and_prefixes = [ (P,'') ]
        else:
            vars_and_prefixes = [ (P, 'p_') ]

        for var, prefix in vars_and_prefixes:
            v_max = var.max(axis=0)
            v_min = var.min(axis=0)
            v_mean = var.mean(axis=0)
            v_range = v_max - v_min

            # max_x.mean_u is "the mean over *u*nits of the max over e*x*amples"
            # The x and u are included in the name because otherwise its hard
            # to remember which axis is which when reading the monitor
            # I use inner.outer rather than outer_of_inner or something like that
            # because I want mean_x.* to appear next to each other in the alphabetical
            # list, as these are commonly plotted together
            for key, val in [
                             ('max_x.max_u', v_max.max()),
                             ('max_x.mean_u', v_max.mean()),
                             ('max_x.min_u', v_max.min()),
                             ('min_x.max_u', v_min.max()),
                             ('min_x.mean_u', v_min.mean()),
                             ('min_x.min_u', v_min.min()),
                             ('range_x.max_u', v_range.max()),
                             ('range_x.mean_u', v_range.mean()),
                             ('range_x.min_u', v_range.min()),
                             ('mean_x.max_u', v_mean.max()),
                             ('mean_x.mean_u', v_mean.mean()),
                             ('mean_x.min_u', v_mean.min())
                             ]:
                rval[prefix+key] = val

        return rval

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)

        z = self.transformer.lmul(state_below) + self.b

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if not hasattr(self, 'pool_stride'):
            self.pool_stride = self.pool_size

        if self.randomize_pools:
            z = T.dot(z, self.permute)

        if not hasattr(self, 'min_zero'):
            self.min_zero = False

        if self.min_zero:
            p = T.zeros_like(z)
        else:
            p = None
        
        eps = 1e-3
        lpr = (1+T.log(1+T.exp(self.Lp)))
        last_start = self.detector_layer_dim  - self.pool_size
        for i in xrange(self.pool_size):
            cur = z[:,i:last_start+i+1:self.pool_stride]
            cur = T.abs_(eps+cur)**lpr
            if p is None:
                p = cur
            else:
                p = p + cur
                
        p = p*(1.0/self.pool_size)
        p = p**(1.0/lpr)
        p.name = self.layer_name + '_p_'

        return p

    def foo(self, state_below):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)

        z = self.transformer.lmul(state_below) + self.b

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if not hasattr(self, 'pool_stride'):
            self.pool_stride = self.pool_size

        if self.randomize_pools:
            z = T.dot(z, self.permute)

        if not hasattr(self, 'min_zero'):
            self.min_zero = False

        if self.min_zero:
            p = T.zeros_like(z)
        else:
            p = None

        last_start = self.detector_layer_dim  - self.pool_size

        pooling_stack = []
        for i in xrange(self.pool_size):
            cur = z[:,i:last_start+i+1:self.pool_stride]
            cur = cur.reshape((cur.shape[0], cur.shape[1], 1))
            assert cur.ndim == 3
            pooling_stack.append(cur)
        if self.min_zero:
            pooling_stack.append(T.zeros_like(cur))
        pooling_stack = T.concatenate(pooling_stack, axis=2)
        p = pooling_stack.max(axis=2)
        counts = (T.eq(pooling_stack, p.dimshuffle(0, 1, 'x'))).sum(axis=0)

        p.name = self.layer_name + '_p_'

        return p, counts

class MaxoutConvC01B(Layer):
    """
    Maxout units arranged in a convolutional layer, with
    spatial max pooling on top of the maxout. If you use this
    code in a research project, please cite

    "Maxout Networks" Ian J. Goodfellow, David Warde-Farley,
    Mehdi Mirza, Aaron Courville, and Yoshua Bengio. ICML 2013


    This uses the C01B ("channels", topological axis 0,
    topological axis 1, "batch") format of tensors for input
    and output.

    The back-end is Alex Krizhevsky's cuda-convnet library,
    so it is extremely fast, but requires a GPU.
    """

    def __init__(self,
                 num_channels,
                 num_pieces,
                 kernel_shape,
                 pool_shape,
                 pool_stride,
                 layer_name,
                 irange = None,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 pad = 0,
                 fix_pool_shape = False,
                 fix_pool_stride = False,
                 fix_kernel_shape = False,
                 partial_sum = 1,
                 tied_b = False,
                 max_kernel_norm = None,
                 input_normalization = None,
                 detector_normalization = None,
                 min_zero = False,
                 output_normalization = None,
                 kernel_stride=(1, 1)):
        """
            num_channels: The number of output channels the layer should have.
                          Note that it must internally compute num_channels * num_pieces
                          convolution channels.
            num_pieces:   The number of linear pieces used to make each maxout unit.
            kernel_shape: The shape of the convolution kernel.
            pool_shape:   The shape of the spatial max pooling. A two-tuple of ints.
                          This is redundant as cuda-convnet requires the pool shape to
                          be square.
            pool_stride:  The stride of the spatial max pooling. Also must be square.
            layer_name: A name for this layer that will be prepended to
                        monitoring channels related to this layer.
            irange: if specified, initializes each weight randomly in
                U(-irange, irange)
            init_bias: All biases are initialized to this number
            W_lr_scale: The learning rate on the weights for this layer is
                multiplied by this scaling factor
            b_lr_scale: The learning rate on the biases for this layer is
                multiplied by this scaling factor
            pad: The amount of zero-padding to implicitly add to the boundary of the
                image when computing the convolution. Useful for making sure pixels
                at the edge still get to influence multiple hidden units.
            fix_pool_shape: If True, will modify self.pool_shape to avoid having
                pool shape bigger than the entire detector layer.
                If you have this on, you should probably also have
                fix_pool_stride on, since the pool shape might shrink
                smaller than the stride, even if the stride was initially
                valid.
                The "fix" parameters are useful for working with a hyperparameter
                optimization package, which might often propose sets of hyperparameters
                that are not feasible, but can easily be projected back into the feasible
                set.
            fix_kernel_shape: if True, will modify self.kernel_shape to avoid
            having the kernel shape bigger than the implicitly
            zero padded input layer

            partial_sum: a parameter that controls whether to prefer runtime savings
                        or memory savings when computing the gradient with respect to
                        the kernels. See pylearn2.sandbox.cuda_convnet.weight_acts.py
                        for details. The default is to prefer high speed.
                        Note that changing this setting may change the value of computed
                        results slightly due to different rounding error.
            tied_b: If true, all biases in the same channel are constrained to be the same
                    as each other. Otherwise, each bias at each location is learned independently.
            max_kernel_norm: If specifed, each kernel is constrained to have at most this norm.
            input_normalization, detector_normalization, output_normalization:
                if specified, should be a callable object. the state of the network is optionally
                replaced with normalization(state) at each of the 3 points in processing:
                    input: the input the layer receives can be normalized right away
                    detector: the maxout units can be normalized prior to the spatial pooling
                    output: the output of the layer, after sptial pooling, can be normalized as well
            kernel_stride: vertical and horizontal pixel stride between
                           each detector.
        """
        check_cuda(str(type(self)))

        detector_channels = num_channels * num_pieces

        self.__dict__.update(locals())
        del self.self

    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    def set_input_space(self, space):
        """ Note: this resets parameters! """

        setup_detector_layer_c01b(layer=self,
                input_space=space,
                rng=self.mlp.rng,
                irange=self.irange)

        rng = self.mlp.rng

        detector_shape = self.detector_space.shape

        def handle_pool_shape(idx):
            if self.pool_shape[idx] < 1:
                raise ValueError("bad pool shape: " + str(self.pool_shape))
            if self.pool_shape[idx] > detector_shape[idx]:
                if self.fix_pool_shape:
                    assert detector_shape[idx] > 0
                    self.pool_shape[idx] = detector_shape[idx]
                else:
                    raise ValueError("Pool shape exceeds detector layer shape on axis %d" % idx)

        map(handle_pool_shape, [0, 1])

        assert self.pool_shape[0] == self.pool_shape[1]
        assert self.pool_stride[0] == self.pool_stride[1]
        assert all(isinstance(elem, py_integer_types) for elem in self.pool_stride)
        if self.pool_stride[0] > self.pool_shape[0]:
            if self.fix_pool_stride:
                warnings.warn("Fixing the pool stride")
                ps = self.pool_shape[0]
                assert isinstance(ps, py_integer_types)
                self.pool_stride = [ps, ps]
            else:
                raise ValueError("Stride too big.")
        assert all(isinstance(elem, py_integer_types) for elem in self.pool_stride)

        dummy_detector = sharedX(self.detector_space.get_origin_batch(2)[0:16,:,:,:])

        dummy_p = max_pool_c01b(c01b=dummy_detector, pool_shape=self.pool_shape,
                                pool_stride=self.pool_stride,
                                image_shape=self.detector_space.shape)
        dummy_p = dummy_p.eval()
        self.output_space = Conv2DSpace(shape=[dummy_p.shape[1], dummy_p.shape[2]],
                                        num_channels = self.num_channels, axes = ('c', 0, 1, 'b') )

        print 'Output space: ', self.output_space.shape

    def censor_updates(self, updates):

        if self.max_kernel_norm is not None:
            W ,= self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=(0,1,2)))
                desired_norms = T.clip(row_norms, 0, self.max_kernel_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + row_norms)).dimshuffle('x', 'x', 'x', 0)

    def get_params(self):
        assert self.b.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    def set_weights(self, weights):
        W, = self.transformer.get_params()
        W.set_value(weights)

    def set_biases(self, biases):
        self.b.set_value(biases)

    def get_biases(self):
        return self.b.get_value()

    def get_weights_topo(self):
        return self.transformer.get_weights_topo()

    def get_monitoring_channels(self):

        W ,= self.transformer.get_params()

        assert W.ndim == 4

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=(0,1,2)))

        return OrderedDict([
                            ('kernel_norms_min'  , row_norms.min()),
                            ('kernel_norms_mean' , row_norms.mean()),
                            ('kernel_norms_max'  , row_norms.max()),
                            ])

    def fprop(self, state_below):
        check_cuda(str(type(self)))

        self.input_space.validate(state_below)

        if not hasattr(self, 'input_normalization'):
            self.input_normalization = None

        if self.input_normalization:
            state_below = self.input_normalization(state_below)

        # Alex's code requires # input channels to be <= 3 or a multiple of 4
        # so we add dummy channels if necessary
        if not hasattr(self, 'dummy_channels'):
            self.dummy_channels = 0
        if self.dummy_channels > 0:
            state_below = T.concatenate((state_below,
                                         T.zeros_like(state_below[0:self.dummy_channels, :, :, :])),
                                        axis=0)

        z = self.transformer.lmul(state_below)
        if not hasattr(self, 'tied_b'):
            self.tied_b = False
        if self.tied_b:
            b = self.b.dimshuffle(0, 'x', 'x', 'x')
        else:
            b = self.b.dimshuffle(0, 1, 2, 'x')


        z = z + b
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        self.detector_space.validate(z)

        assert self.detector_space.num_channels % 16 == 0

        if self.output_space.num_channels % 16 == 0:
            # alex's max pool op only works when the number of channels
            # is divisible by 16. we can only do the cross-channel pooling
            # first if the cross-channel pooling preserves that property
            if self.num_pieces != 1:
                s = None
                for i in xrange(self.num_pieces):
                    t = z[i::self.num_pieces,:,:,:]
                    if s is None:
                        s = t
                    else:
                        s = T.maximum(s, t)
                z = s

            if self.detector_normalization:
                z = self.detector_normalization(z)

            p = max_pool_c01b(c01b=z, pool_shape=self.pool_shape,
                              pool_stride=self.pool_stride,
                              image_shape=self.detector_space.shape)
        else:

            if self.detector_normalization is not None:
                raise NotImplementedError("We can't normalize the detector "
                        "layer because the detector layer never exists as a "
                        "stage of processing in this implementation.")
            z = max_pool_c01b(c01b=z, pool_shape=self.pool_shape,
                              pool_stride=self.pool_stride,
                              image_shape=self.detector_space.shape)
            if self.num_pieces != 1:
                s = None
                for i in xrange(self.num_pieces):
                    t = z[i::self.num_pieces,:,:,:]
                    if s is None:
                        s = t
                    else:
                        s = T.maximum(s, t)
                z = s
            p = z


        self.output_space.validate(p)

        if hasattr(self, 'min_zero') and self.min_zero:
            p = p * (p > 0.)

        if not hasattr(self, 'output_normalization'):
            self.output_normalization = None

        if self.output_normalization:
            p = self.output_normalization(p)

        return p

    def get_weights_view_shape(self):
        total = self.detector_channels
        cols = self.num_pieces
        if cols == 1:
            # Let the PatchViewer decide how to arrange the units
            # when they're not pooled
            raise NotImplementedError()
        # When they are pooled, make each pooling unit have one row
        rows = total // cols
        if rows * cols < total:
            rows = rows + 1
        return rows, cols

    def get_monitoring_channels_from_state(self, state):

        P = state

        rval = OrderedDict()

        vars_and_prefixes = [ (P,'') ]

        for var, prefix in vars_and_prefixes:
            assert var.ndim == 4
            v_max = var.max(axis=(1,2,3))
            v_min = var.min(axis=(1,2,3))
            v_mean = var.mean(axis=(1,2,3))
            v_range = v_max - v_min

            # max_x.mean_u is "the mean over *u*nits of the max over e*x*amples"
            # The x and u are included in the name because otherwise its hard
            # to remember which axis is which when reading the monitor
            # I use inner.outer rather than outer_of_inner or something like that
            # because I want mean_x.* to appear next to each other in the alphabetical
            # list, as these are commonly plotted together
            for key, val in [
                             ('max_x.max_u', v_max.max()),
                             ('max_x.mean_u', v_max.mean()),
                             ('max_x.min_u', v_max.min()),
                             ('min_x.max_u', v_min.max()),
                             ('min_x.mean_u', v_min.mean()),
                             ('min_x.min_u', v_min.min()),
                             ('range_x.max_u', v_range.max()),
                             ('range_x.mean_u', v_range.mean()),
                             ('range_x.min_u', v_range.min()),
                             ('mean_x.max_u', v_mean.max()),
                             ('mean_x.mean_u', v_mean.mean()),
                             ('mean_x.min_u', v_mean.min())
                             ]:
                rval[prefix+key] = val

        return rval

class MaxoutLocalC01B(Layer):
    """
    Maxout units arranged in a convolutional layer, with
    spatial max pooling on top of the maxout. If you use this
    code in a research project, please cite

    "Maxout Networks" Ian J. Goodfellow, David Warde-Farley,
    Mehdi Mirza, Aaron Courville, and Yoshua Bengio. ICML 2013

    This uses the C01B ("channels", topological axis 0,
    topological axis 1, "batch") format of tensors for input
    and output.

    Unlike MaxoutConvC01B, this class supports operation on CPU,
    thanks to James Bergstra's TheanoLinear library, which
    pylearn2 has forked. The GPU code is still based on Alex
    Krizvhevsky's cuda_convnet library.
    """

    def __init__(self,
                 num_channels,
                 num_pieces,
                 kernel_shape,
                 layer_name,
                 pool_shape=None,
                 pool_stride=None,
                 irange = None,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 pad = 0,
                 fix_pool_shape = False,
                 fix_pool_stride = False,
                 fix_kernel_shape = False,
                 partial_sum = 1,
                 tied_b = False,
                 max_filter_norm = None,
                 input_normalization = None,
                 detector_normalization = None,
                 min_zero = False,
                 output_normalization = None,
                 input_groups = 1,
                 kernel_stride=(1, 1)):
        """
            num_channels: The number of output channels the layer should have.
                          Note that it must internally compute num_channels * num_pieces
                          convolution channels.
            num_pieces:   The number of linear pieces used to make each maxout unit.
            kernel_shape: The shape of the convolution kernel.
            pool_shape:   The shape of the spatial max pooling. A two-tuple of ints.
                          This is redundant as cuda-convnet requires the pool shape to
                          be square.
                          Defaults to None, which means no spatial pooling
            pool_stride:  The stride of the spatial max pooling. Also must be square.
                          Defaults to None, which means no spatial pooling.
            layer_name: A name for this layer that will be prepended to
                        monitoring channels related to this layer.
            irange: if specified, initializes each weight randomly in
                U(-irange, irange)
            init_bias: All biases are initialized to this number
            W_lr_scale: The learning rate on the weights for this layer is
                multiplied by this scaling factor
            b_lr_scale: The learning rate on the biases for this layer is
                multiplied by this scaling factor
            pad: The amount of zero-padding to implicitly add to the boundary of the
                image when computing the convolution. Useful for making sure pixels
                at the edge still get to influence multiple hidden units.
            fix_pool_shape: If True, will modify self.pool_shape to avoid having
                pool shape bigger than the entire detector layer.
                If you have this on, you should probably also have
                fix_pool_stride on, since the pool shape might shrink
                smaller than the stride, even if the stride was initially
                valid.
                The "fix" parameters are useful for working with a hyperparameter
                optimization package, which might often propose sets of hyperparameters
                that are not feasible, but can easily be projected back into the feasible
                set.
            fix_kernel_shape: if True, will modify self.kernel_shape to avoid
            having the kernel shape bigger than the implicitly
            zero padded input layer

            partial_sum: a parameter that controls whether to prefer runtime savings
                        or memory savings when computing the gradient with respect to
                        the kernels. See pylearn2.sandbox.cuda_convnet.weight_acts.py
                        for details. The default is to prefer high speed.
                        Note that changing this setting may change the value of computed
                        results slightly due to different rounding error.
            tied_b: If true, all biases in the same channel are constrained to be the same
                    as each other. Otherwise, each bias at each location is learned independently.
            max_kernel_norm: If specifed, each kernel is constrained to have at most this norm.
            input_normalization, detector_normalization, output_normalization:
                if specified, should be a callable object. the state of the network is optionally
                replaced with normalization(state) at each of the 3 points in processing:
                    input: the input the layer receives can be normalized right away
                    detector: the maxout units can be normalized prior to the spatial pooling
                    output: the output of the layer, after sptial pooling, can be normalized as well
        """

        assert (pool_shape is None) == (pool_stride is None)

        detector_channels = num_channels * num_pieces

        self.__dict__.update(locals())
        del self.self

    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    def set_input_space(self, space):
        """ Note: this resets parameters! """

        self.input_space = space

        if not isinstance(self.input_space, Conv2DSpace):
            raise TypeError("The input to a convolutional layer should be a Conv2DSpace, "
                    " but layer " + self.layer_name + " got "+str(type(self.input_space)))
        # note: I think the desired space thing is actually redundant,
        # since LinearTransform will also dimshuffle the axes if needed
        # It's not hurting anything to have it here but we could reduce
        # code complexity by removing it
        self.desired_space = Conv2DSpace(shape=space.shape,
                                         channels=space.num_channels,
                                         axes=('c', 0, 1, 'b'))

        ch = self.desired_space.num_channels
        rem = ch % 4
        if ch > 3 and rem != 0:
            self.dummy_channels = 4 - rem
        else:
            self.dummy_channels = 0
        self.dummy_space = Conv2DSpace(shape=space.shape,
                                       channels=space.num_channels + self.dummy_channels,
                                       axes=('c', 0, 1, 'b'))

        rng = self.mlp.rng

        output_shape = [self.input_space.shape[0] + 2 * self.pad - self.kernel_shape[0] + 1,
                        self.input_space.shape[1] + 2 * self.pad - self.kernel_shape[1] + 1]

        def handle_kernel_shape(idx):
            if self.kernel_shape[idx] < 1:
                raise ValueError("kernel must have strictly positive size on all axes but has shape: "+str(self.kernel_shape))
            if output_shape[idx] <= 0:
                if self.fix_kernel_shape:
                    self.kernel_shape[idx] = self.input_space.shape[idx] + 2 * self.pad
                    assert self.kernel_shape[idx] != 0
                    output_shape[idx] = 1
                    warnings.warn("Had to change the kernel shape to make network feasible")
                else:
                    raise ValueError("kernel too big for input (even with zero padding)")

        map(handle_kernel_shape, [0, 1])

        self.detector_space = Conv2DSpace(shape=output_shape,
                                          num_channels = self.detector_channels,
                                          axes = ('c', 0, 1, 'b'))

        if self.pool_shape is not None:
            def handle_pool_shape(idx):
                if self.pool_shape[idx] < 1:
                    raise ValueError("bad pool shape: " + str(self.pool_shape))
                if self.pool_shape[idx] > output_shape[idx]:
                    if self.fix_pool_shape:
                        assert output_shape[idx] > 0
                        self.pool_shape[idx] = output_shape[idx]
                    else:
                        raise ValueError("Pool shape exceeds detector layer shape on axis %d" % idx)

            map(handle_pool_shape, [0, 1])

            assert self.pool_shape[0] == self.pool_shape[1]
            assert self.pool_stride[0] == self.pool_stride[1]
            assert all(isinstance(elem, py_integer_types) for elem in self.pool_stride)
            if self.pool_stride[0] > self.pool_shape[0]:
                if self.fix_pool_stride:
                    warnings.warn("Fixing the pool stride")
                    ps = self.pool_shape[0]
                    assert isinstance(ps, py_integer_types)
                    self.pool_stride = [ps, ps]
                else:
                    raise ValueError("Stride too big.")
            assert all(isinstance(elem, py_integer_types) for elem in self.pool_stride)

        if self.irange is not None:
            self.transformer = local_c01b.make_random_local(
                    input_groups = self.input_groups,
                    irange = self.irange,
                    input_axes = self.desired_space.axes,
                    image_shape = self.desired_space.shape,
                    output_axes = self.detector_space.axes,
                    input_channels = self.dummy_space.num_channels,
                    output_channels = self.detector_space.num_channels,
                    kernel_shape = self.kernel_shape,
                    kernel_stride=self.kernel_stride,
                    pad = self.pad,
                    partial_sum = self.partial_sum,
                    rng = rng)
        W, = self.transformer.get_params()
        W.name = 'W'

        if self.tied_b:
            self.b = sharedX(np.zeros((self.detector_space.num_channels)) + self.init_bias)
        else:
            self.b = sharedX(self.detector_space.get_origin() + self.init_bias)
        self.b.name = 'b'

        print 'Input shape: ', self.input_space.shape
        print 'Detector space: ', self.detector_space.shape

        assert self.detector_space.num_channels >= 16

        if self.pool_shape is None:
            self.output_space = Conv2DSpace(shape=self.detector_space.shape,
                    num_channels = self.num_channels,
                    axes = ('c', 0, 1, 'b'))
        else:
            dummy_detector = sharedX(self.detector_space.get_origin_batch(2)[0:16,:,:,:])

            dummy_p = max_pool_c01b(c01b=dummy_detector, pool_shape=self.pool_shape,
                                    pool_stride=self.pool_stride,
                                    image_shape=self.detector_space.shape)
            dummy_p = dummy_p.eval()
            self.output_space = Conv2DSpace(shape=[dummy_p.shape[1], dummy_p.shape[2]],
                                            num_channels = self.num_channels, axes = ('c', 0, 1, 'b') )

        print 'Output space: ', self.output_space.shape

    def censor_updates(self, updates):

        if self.max_filter_norm is not None:
            W ,= self.transformer.get_params()
            if W in updates:
                # TODO:    push some of this into the transformer itself
                updated_W = updates[W]
                updated_norms = self.get_filter_norms(updated_W)
                desired_norms = T.clip(updated_norms, 0, self.max_filter_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + updated_norms)
                        ).dimshuffle(0, 1, 'x', 'x', 'x', 2, 3)

    def get_params(self):
        assert self.b.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    def set_weights(self, weights):
        W, = self.transformer.get_params()
        W.set_value(weights)

    def set_biases(self, biases):
        self.b.set_value(biases)

    def get_biases(self):
        return self.b.get_value()

    def get_weights_topo(self):
        return self.transformer.get_weights_topo()

    def get_filter_norms(self, W = None):

        # TODO: push this into the transformer class itself

        if W is None:
            W ,= self.transformer.get_params()

        assert W.ndim == 7

        sq_W = T.sqr(W)

        norms = T.sqrt(sq_W.sum(axis=(2, 3, 4)))

        return norms

    def get_monitoring_channels(self):

        filter_norms = self.get_filter_norms()

        return OrderedDict([
                            ('filter_norms_min'  , filter_norms.min()),
                            ('filter_norms_mean' , filter_norms.mean()),
                            ('filter_norms_max'  , filter_norms.max()),
                            ])

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        state_below = self.input_space.format_as(state_below, self.desired_space)

        if not hasattr(self, 'input_normalization'):
            self.input_normalization = None

        if self.input_normalization:
            state_below = self.input_normalization(state_below)

        # Alex's code requires # input channels to be <= 3 or a multiple of 4
        # so we add dummy channels if necessary
        if not hasattr(self, 'dummy_channels'):
            self.dummy_channels = 0
        if self.dummy_channels > 0:
            state_below = T.concatenate((state_below,
                                         T.zeros_like(state_below[0:self.dummy_channels, :, :, :])),
                                        axis=0)

        z = self.transformer.lmul(state_below)
        if not hasattr(self, 'tied_b'):
            self.tied_b = False
        if self.tied_b:
            b = self.b.dimshuffle(0, 'x', 'x', 'x')
        else:
            b = self.b.dimshuffle(0, 1, 2, 'x')


        z = z + b
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        self.detector_space.validate(z)

        assert self.detector_space.num_channels % 16 == 0

        if self.output_space.num_channels % 16 == 0:
            # alex's max pool op only works when the number of channels
            # is divisible by 16. we can only do the cross-channel pooling
            # first if the cross-channel pooling preserves that property
            if self.num_pieces != 1:
                s = None
                for i in xrange(self.num_pieces):
                    t = z[i::self.num_pieces,:,:,:]
                    if s is None:
                        s = t
                    else:
                        s = T.maximum(s, t)
                z = s

            if self.detector_normalization:
                z = self.detector_normalization(z)

            if self.pool_shape is None:
                p = z
            else:
                p = max_pool_c01b(c01b=z, pool_shape=self.pool_shape,
                              pool_stride=self.pool_stride,
                              image_shape=self.detector_space.shape)
        else:

            if self.detector_normalization is not None:
                raise NotImplementedError("We can't normalize the detector "
                        "layer because the detector layer never exists as a "
                        "stage of processing in this implementation.")
            if self.pool_shape is not None:
                z = max_pool_c01b(c01b=z, pool_shape=self.pool_shape,
                              pool_stride=self.pool_stride,
                              image_shape=self.detector_space.shape)
            if self.num_pieces != 1:
                s = None
                for i in xrange(self.num_pieces):
                    t = z[i::self.num_pieces,:,:,:]
                    if s is None:
                        s = t
                    else:
                        s = T.maximum(s, t)
                z = s
            p = z


        self.output_space.validate(p)

        if hasattr(self, 'min_zero') and self.min_zero:
            p = p * (p > 0.)

        if not hasattr(self, 'output_normalization'):
            self.output_normalization = None

        if self.output_normalization:
            p = self.output_normalization(p)

        return p

    def get_weights_view_shape(self):
        total = self.detector_channels
        cols = self.num_pieces
        if cols == 1:
            # Let the PatchViewer decide how to arrange the units
            # when they're not pooled
            raise NotImplementedError()
        # When they are pooled, make each pooling unit have one row
        rows = total // cols
        if rows * cols < total:
            rows = rows + 1
        return rows, cols

    def get_monitoring_channels_from_state(self, state):

        P = state

        rval = OrderedDict()

        vars_and_prefixes = [ (P,'') ]

        for var, prefix in vars_and_prefixes:
            assert var.ndim == 4
            v_max = var.max(axis=(1,2,3))
            v_min = var.min(axis=(1,2,3))
            v_mean = var.mean(axis=(1,2,3))
            v_range = v_max - v_min

            # max_x.mean_u is "the mean over *u*nits of the max over e*x*amples"
            # The x and u are included in the name because otherwise its hard
            # to remember which axis is which when reading the monitor
            # I use inner.outer rather than outer_of_inner or something like that
            # because I want mean_x.* to appear next to each other in the alphabetical
            # list, as these are commonly plotted together
            for key, val in [
                             ('max_x.max_u', v_max.max()),
                             ('max_x.mean_u', v_max.mean()),
                             ('max_x.min_u', v_max.min()),
                             ('min_x.max_u', v_min.max()),
                             ('min_x.mean_u', v_min.mean()),
                             ('min_x.min_u', v_min.min()),
                             ('range_x.max_u', v_range.max()),
                             ('range_x.mean_u', v_range.mean()),
                             ('range_x.min_u', v_range.min()),
                             ('mean_x.max_u', v_mean.max()),
                             ('mean_x.mean_u', v_mean.mean()),
                             ('mean_x.min_u', v_mean.min())
                             ]:
                rval[prefix+key] = val

        return rval

class GaussianPooler(Layer):
    """
    
    """
    def __init__(self,
                 num_channels,
                 num_pieces,
                 kernel_shape,
                 layer_name,
                 pool_shape=None,
                 pool_stride=None,
                 irange = None,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 pad = 0,
                 fix_pool_shape = False,
                 fix_pool_stride = False,
                 fix_kernel_shape = False,
                 partial_sum = 1,
                 tied_b = False,
                 max_filter_norm = None,
                 input_normalization = None,
                 detector_normalization = None,
                 min_zero = False,
                 output_normalization = None,
                 input_groups = 1,
                 kernel_stride=(1, 1)):
        pass
        
        
class MaxoutBC01Strided(Layer):
    """
    Maxout units arranged in a convolutional layer, with
    spatial max pooling on top of the maxout. If you use this
    code in a research project, please cite

    "Maxout Networks" Ian J.q Goodfellow, David Warde-Farley,
    Mehdi Mirza, Aaron Courville, and Yoshua Bengio. ICML 2013

    This uses the C01B ("channels", topological axis 0,
    topological axis 1, "batch") format of tensors for input
    and output.

    Unlike MaxoutConvC01B, this class supports operation on CPU,
    thanks to James Bergstra's TheanoLinear library, which
    pylearn2 has forked. The GPU code is still based on Alex
    Krizvhevsky's cuda_convnet library.
    """

    def __init__(self,
                 num_channels,
                 num_pieces,
                 kernel_shape,
                 layer_name,
                 pool_shape=None,
                 pool_stride=None,
                 irange = None,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 pad = 0,
                 fix_pool_shape = False,
                 fix_pool_stride = False,
                 fix_kernel_shape = False,
                 partial_sum = 1,
                 tied_b = False,
                 max_filter_norm = None,
                 input_normalization = None,
                 detector_normalization = None,
                 min_zero = False,
                 output_normalization = None,
                 input_groups = 1,
                 kernel_stride=(1, 1)):
        """
            num_channels: The number of output channels the layer should have.
                          Note that it must internally compute num_channels * num_pieces
                          convolution channels.
            num_pieces:   The number of linear pieces used to make each maxout unit.
            kernel_shape: The shape of the convolution kernel.
            pool_shape:   The shape of the spatial max pooling. A two-tuple of ints.
                          This is redundant as cuda-convnet requires the pool shape to
                          be square.
                          Defaults to None, which means no spatial pooling
            pool_stride:  The stride of the spatial max pooling. Also must be square.
                          Defaults to None, which means no spatial pooling.
            layer_name: A name for this layer that will be prepended to
                        monitoring channels related to this layer.
            irange: if specified, initializes each weight randomly in
                U(-irange, irange)
            init_bias: All biases are initialized to this number
            W_lr_scale: The learning rate on the weights for this layer is
                multiplied by this scaling factor
            b_lr_scale: The learning rate on the biases for this layer is
                multiplied by this scaling factor
            pad: The amount of zero-padding to implicitly add to the boundary of the
                image when computing the convolution. Useful for making sure pixels
                at the edge still get to influence multiple hidden units.
            fix_pool_shape: If True, will modify self.pool_shape to avoid having
                pool shape bigger than the entire detector layer.
                If you have this on, you should probably also have
                fix_pool_stride on, since the pool shape might shrink
                smaller than the stride, even if the stride was initially
                valid.
                The "fix" parameters are useful for working with a hyperparameter
                optimization package, which might often propose sets of hyperparameters
                that are not feasible, but can easily be projected back into the feasible
                set.
            fix_kernel_shape: if True, will modify self.kernel_shape to avoid
            having the kernel shape bigger than the implicitly
            zero padded input layer

            partial_sum: a parameter that controls whether to prefer runtime savings
                        or memory savings when computing the gradient with respect to
                        the kernels. See pylearn2.sandbox.cuda_convnet.weight_acts.py
                        for details. The default is to prefer high speed.
                        Note that changing this setting may change the value of computed
                        results slightly due to different rounding error.
            tied_b: If true, all biases in the same channel are constrained to be the same
                    as each other. Otherwise, each bias at each location is learned independently.
            max_kernel_norm: If specifed, each kernel is constrained to have at most this norm.
            input_normalization, detector_normalization, output_normalization:
                if specified, should be a callable object. the state of the network is optionally
                replaced with normalization(state) at each of the 3 points in processing:
                    input: the input the layer receives can be normalized right away
                    detector: the maxout units can be normalized prior to the spatial pooling
                    output: the output of the layer, after sptial pooling, can be normalized as well
        """

        assert (pool_shape is None) == (pool_stride is None)

        detector_channels = num_channels * num_pieces

        self.__dict__.update(locals())
        del self.self

    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    def set_input_space(self, space):
        """ Note: this resets parameters! """

        self.input_space = space

        if not isinstance(self.input_space, Conv2DSpace):
            raise TypeError("The input to a convolutional layer should be a Conv2DSpace, "
                    " but layer " + self.layer_name + " got "+str(type(self.input_space)))
        # note: I think the desired space thing is actually redundant,
        # since LinearTransform will also dimshuffle the axes if needed
        # It's not hurting anything to have it here but we could reduce
        # code complexity by removing it
        self.desired_space = Conv2DSpace(shape=space.shape,
                                         channels=space.num_channels,
                                         axes=('c', 0, 1, 'b'))

        ch = self.desired_space.num_channels
        rem = ch % 4
        if ch > 3 and rem != 0:
            self.dummy_channels = 4 - rem
        else:
            self.dummy_channels = 0
        self.dummy_space = Conv2DSpace(shape=space.shape,
                                       channels=space.num_channels + self.dummy_channels,
                                       axes=('c', 0, 1, 'b'))

        rng = self.mlp.rng

        output_shape = [self.input_space.shape[0] + 2 * self.pad - self.kernel_shape[0] + 1,
                        self.input_space.shape[1] + 2 * self.pad - self.kernel_shape[1] + 1]

        def handle_kernel_shape(idx):
            if self.kernel_shape[idx] < 1:
                raise ValueError("kernel must have strictly positive size on all axes but has shape: "+str(self.kernel_shape))
            if output_shape[idx] <= 0:
                if self.fix_kernel_shape:
                    self.kernel_shape[idx] = self.input_space.shape[idx] + 2 * self.pad
                    assert self.kernel_shape[idx] != 0
                    output_shape[idx] = 1
                    warnings.warn("Had to change the kernel shape to make network feasible")
                else:
                    raise ValueError("kernel too big for input (even with zero padding)")

        map(handle_kernel_shape, [0, 1])

        self.detector_space = Conv2DSpace(shape=output_shape,
                                          num_channels = self.detector_channels,
                                          axes = ('c', 0, 1, 'b'))

        if self.pool_shape is not None:
            def handle_pool_shape(idx):
                if self.pool_shape[idx] < 1:
                    raise ValueError("bad pool shape: " + str(self.pool_shape))
                if self.pool_shape[idx] > output_shape[idx]:
                    if self.fix_pool_shape:
                        assert output_shape[idx] > 0
                        self.pool_shape[idx] = output_shape[idx]
                    else:
                        raise ValueError("Pool shape exceeds detector layer shape on axis %d" % idx)

            map(handle_pool_shape, [0, 1])

            assert self.pool_shape[0] == self.pool_shape[1]
            assert self.pool_stride[0] == self.pool_stride[1]
            assert all(isinstance(elem, py_integer_types) for elem in self.pool_stride)
            if self.pool_stride[0] > self.pool_shape[0]:
                if self.fix_pool_stride:
                    warnings.warn("Fixing the pool stride")
                    ps = self.pool_shape[0]
                    assert isinstance(ps, py_integer_types)
                    self.pool_stride = [ps, ps]
                else:
                    raise ValueError("Stride too big.")
            assert all(isinstance(elem, py_integer_types) for elem in self.pool_stride)

        if self.irange is not None:
            self.transformer = local_c01b.make_random_local(
                    input_groups = self.input_groups,
                    irange = self.irange,
                    input_axes = self.desired_space.axes,
                    image_shape = self.desired_space.shape,
                    output_axes = self.detector_space.axes,
                    input_channels = self.dummy_space.num_channels,
                    output_channels = self.detector_space.num_channels,
                    kernel_shape = self.kernel_shape,
                    kernel_stride=self.kernel_stride,
                    pad = self.pad,
                    partial_sum = self.partial_sum,
                    rng = rng)
        W, = self.transformer.get_params()
        W.name = 'W'

        if self.tied_b:
            self.b = sharedX(np.zeros((self.detector_space.num_channels)) + self.init_bias)
        else:
            self.b = sharedX(self.detector_space.get_origin() + self.init_bias)
        self.b.name = 'b'

        print 'Input shape: ', self.input_space.shape
        print 'Detector space: ', self.detector_space.shape

        assert self.detector_space.num_channels >= 16

        if self.pool_shape is None:
            self.output_space = Conv2DSpace(shape=self.detector_space.shape,
                    num_channels = self.num_channels,
                    axes = ('c', 0, 1, 'b'))
        else:
            dummy_detector = sharedX(self.detector_space.get_origin_batch(2)[0:16,:,:,:])

            dummy_p = max_pool_c01b(c01b=dummy_detector, pool_shape=self.pool_shape,
                                    pool_stride=self.pool_stride,
                                    image_shape=self.detector_space.shape)
            dummy_p = dummy_p.eval()
            self.output_space = Conv2DSpace(shape=[dummy_p.shape[1], dummy_p.shape[2]],
                                            num_channels = self.num_channels, axes = ('c', 0, 1, 'b') )

        print 'Output space: ', self.output_space.shape

    def censor_updates(self, updates):

        if self.max_filter_norm is not None:
            W ,= self.transformer.get_params()
            if W in updates:
                # TODO:    push some of this into the transformer itself
                updated_W = updates[W]
                updated_norms = self.get_filter_norms(updated_W)
                desired_norms = T.clip(updated_norms, 0, self.max_filter_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + updated_norms)
                        ).dimshuffle(0, 1, 'x', 'x', 'x', 2, 3)

    def get_params(self):
        assert self.b.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    def set_weights(self, weights):
        W, = self.transformer.get_params()
        W.set_value(weights)

    def set_biases(self, biases):
        self.b.set_value(biases)

    def get_biases(self):
        return self.b.get_value()

    def get_weights_topo(self):
        return self.transformer.get_weights_topo()

    def get_filter_norms(self, W = None):

        # TODO: push this into the transformer class itself

        if W is None:
            W ,= self.transformer.get_params()

        assert W.ndim == 7

        sq_W = T.sqr(W)

        norms = T.sqrt(sq_W.sum(axis=(2, 3, 4)))

        return norms

    def get_monitoring_channels(self):

        filter_norms = self.get_filter_norms()

        return OrderedDict([
                            ('filter_norms_min'  , filter_norms.min()),
                            ('filter_norms_mean' , filter_norms.mean()),
                            ('filter_norms_max'  , filter_norms.max()),
                            ])

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        state_below = self.input_space.format_as(state_below, self.desired_space)

        if not hasattr(self, 'input_normalization'):
            self.input_normalization = None

        if self.input_normalization:
            state_below = self.input_normalization(state_below)

        # Alex's code requires # input channels to be <= 3 or a multiple of 4
        # so we add dummy channels if necessary
        if not hasattr(self, 'dummy_channels'):
            self.dummy_channels = 0
        if self.dummy_channels > 0:
            state_below = T.concatenate((state_below,
                                         T.zeros_like(state_below[0:self.dummy_channels, :, :, :])),
                                        axis=0)

        z = self.transformer.lmul(state_below)
        if not hasattr(self, 'tied_b'):
            self.tied_b = False
        if self.tied_b:
            b = self.b.dimshuffle(0, 'x', 'x', 'x')
        else:
            b = self.b.dimshuffle(0, 1, 2, 'x')


        z = z + b
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        self.detector_space.validate(z)

        assert self.detector_space.num_channels % 16 == 0

        if self.output_space.num_channels % 16 == 0:
            # alex's max pool op only works when the number of channels
            # is divisible by 16. we can only do the cross-channel pooling
            # first if the cross-channel pooling preserves that property
            if self.num_pieces != 1:
                s = None
                for i in xrange(self.num_pieces):
                    t = z[i::self.num_pieces,:,:,:]
                    if s is None:
                        s = t
                    else:
                        s = T.maximum(s, t)
                z = s

            if self.detector_normalization:
                z = self.detector_normalization(z)

            if self.pool_shape is None:
                p = z
            else:
                p = max_pool_c01b(c01b=z, pool_shape=self.pool_shape,
                              pool_stride=self.pool_stride,
                              image_shape=self.detector_space.shape)
        else:

            if self.detector_normalization is not None:
                raise NotImplementedError("We can't normalize the detector "
                        "layer because the detector layer never exists as a "
                        "stage of processing in this implementation.")
            if self.pool_shape is not None:
                z = max_pool_c01b(c01b=z, pool_shape=self.pool_shape,
                              pool_stride=self.pool_stride,
                              image_shape=self.detector_space.shape)
            if self.num_pieces != 1:
                s = None
                for i in xrange(self.num_pieces):
                    t = z[i::self.num_pieces,:,:,:]
                    if s is None:
                        s = t
                    else:
                        s = T.maximum(s, t)
                z = s
            p = z


        self.output_space.validate(p)

        if hasattr(self, 'min_zero') and self.min_zero:
            p = p * (p > 0.)

        if not hasattr(self, 'output_normalization'):
            self.output_normalization = None

        if self.output_normalization:
            p = self.output_normalization(p)

        return p

    def get_weights_view_shape(self):
        total = self.detector_channels
        cols = self.num_pieces
        if cols == 1:
            # Let the PatchViewer decide how to arrange the units
            # when they're not pooled
            raise NotImplementedError()
        # When they are pooled, make each pooling unit have one row
        rows = total // cols
        if rows * cols < total:
            rows = rows + 1
        return rows, cols

    def get_monitoring_channels_from_state(self, state):

        P = state

        rval = OrderedDict()

        vars_and_prefixes = [ (P,'') ]

        for var, prefix in vars_and_prefixes:
            assert var.ndim == 4
            v_max = var.max(axis=(1,2,3))
            v_min = var.min(axis=(1,2,3))
            v_mean = var.mean(axis=(1,2,3))
            v_range = v_max - v_min

            # max_x.mean_u is "the mean over *u*nits of the max over e*x*amples"
            # The x and u are included in the name because otherwise its hard
            # to remember which axis is which when reading the monitor
            # I use inner.outer rather than outer_of_inner or something like that
            # because I want mean_x.* to appear next to each other in the alphabetical
            # list, as these are commonly plotted together
            for key, val in [
                             ('max_x.max_u', v_max.max()),
                             ('max_x.mean_u', v_max.mean()),
                             ('max_x.min_u', v_max.min()),
                             ('min_x.max_u', v_min.max()),
                             ('min_x.mean_u', v_min.mean()),
                             ('min_x.min_u', v_min.min()),
                             ('range_x.max_u', v_range.max()),
                             ('range_x.mean_u', v_range.mean()),
                             ('range_x.min_u', v_range.min()),
                             ('mean_x.max_u', v_mean.max()),
                             ('mean_x.mean_u', v_mean.mean()),
                             ('mean_x.min_u', v_mean.min())
                             ]:
                rval[prefix+key] = val

        return rval

class MaxoutMDM(Layer):
    """
    A hidden maxout layer supporiting multi-channel input with simples linear transformation.
    The layer additionally does two way max pooling 1) as usual maxout per each channel and then 
    2) over multiple input (acoustic) channels. The weights are alwayws tied across channels so the 
    max pooling between channels is related in terms of feature receptors. 
    
    In contrast to Maxout model, here we allow Conv2D input space and perform tensor dot product 
    separately for each of the channels which is then followe by max pooling in usual way.
    """

    def __init__(self,
                 layer_name,
                 num_units,
                 num_pieces,
                 pool_channels = True,
                 pool_stride = None,
                 randomize_pools = False,
                 irange = None,
                 sparse_init = None,
                 sparse_stdev = 1.,
                 include_prob = 1.0,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 max_col_norm = None,
                 max_row_norm = None,
                 mask_weights = None,
                 min_zero = False
        ):
        """
            layer_name: A name for this layer that will be prepended to
                        monitoring channels related to this layer.
            num_units: The number of maxout units to use in this layer.
            num_pieces: The number of linear pieces to use in each maxout
                        unit.
            pool_stride: The distance between the start of each max pooling
                        region. Defaults to num_pieces, which makes the
                        pooling regions disjoint. If set to a smaller number,
                        can do overlapping pools.
            randomize_pools: Does max pooling over randomized subsets of
                        the linear responses, rather than over sequential
                        subsets.
            irange: if specified, initializes each weight randomly in
                U(-irange, irange)
            sparse_init: if specified, irange must not be specified.
                        This is an integer specifying how many weights to make
                        non-zero. All non-zero weights will be initialized
                        randomly in N(0, sparse_stdev^2)
            include_prob: probability of including a weight element in the set
               of weights initialized to U(-irange, irange). If not included
               a weight is initialized to 0. This defaults to 1.
            init_bias: All biases are initialized to this number
            W_lr_scale: The learning rate on the weights for this layer is
                multiplied by this scaling factor
            b_lr_scale: The learning rate on the biases for this layer is
                multiplied by this scaling factor
            max_col_norm: The norm of each column of the weight matrix is
                constrained to have at most this norm. If unspecified, no
                constraint. Constraint is enforced by re-projection (if
                necessary) at the end of each update.
            max_row_norm: Like max_col_norm, but applied to the rows.
            mask_weights: A binary matrix multiplied by the weights after each
                         update, allowing you to restrict their connectivity.
            min_zero: If true, includes a zero in the set we take a max over
                    for each maxout unit. This is equivalent to pooling over
                    rectified linear units.
        """

        detector_layer_dim = num_units * num_pieces
        pool_size = num_pieces

        if pool_stride is None:
            pool_stride = pool_size

        self.__dict__.update(locals())
        del self.self

        self.b = sharedX( np.zeros((self.detector_layer_dim,)) + init_bias, name = layer_name + '_b')

        if max_row_norm is not None:
            raise NotImplementedError()

    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    def set_input_space(self, space):
        """ Note: this resets parameters! """

        self.input_space = space

        assert isinstance(space, Conv2DSpace)
            
        if not ((self.detector_layer_dim - self.pool_size) % self.pool_stride == 0):
            if self.pool_stride == self.pool_size:
                raise ValueError("detector_layer_dim = %d, pool_size = %d. Should be divisible but remainder is %d" %
                             (self.detector_layer_dim, self.pool_size, self.detector_layer_dim % self.pool_size))
            raise ValueError()

        h_shape = (1, self.detector_layer_dim)

        self.h_space = Conv2DSpace(shape=h_shape, num_channels=self.input_space.num_channels, axes=('b','c',0,1))
        self.pool_layer_dim = (self.detector_layer_dim - self.pool_size)/ self.pool_stride + 1
        
        o_shape = (1, self.pool_layer_dim)
        if self.pool_channels is True:
            self.output_space = Conv2DSpace(shape = o_shape, num_channels = 1,  axes = ('b','c',0,1))
        else:
            self.output_space = Conv2DSpace(shape = o_shape, num_channels = self.input_space.num_channels,  axes = ('b','c',0,1))

        self.input_dim = self.input_space.shape[1] #TODO: refactor

        rng = self.mlp.rng
        if self.irange is not None:
            assert self.sparse_init is None
            W = rng.uniform(-self.irange,
                            self.irange,
                            (self.input_dim, self.detector_layer_dim)) * \
                (rng.uniform(0.,1., (self.input_dim, self.detector_layer_dim))
                 < self.include_prob)
        else:
            assert self.sparse_init is not None
            W = np.zeros((self.input_dim, self.detector_layer_dim))
            def mask_rejects(idx, i):
                if self.mask_weights is None:
                    return False
                return self.mask_weights[idx, i] == 0.
            for i in xrange(self.detector_layer_dim):
                assert self.sparse_init <= self.input_dim
                for j in xrange(self.sparse_init):
                    idx = rng.randint(0, self.input_dim)
                    while W[idx, i] != 0 or mask_rejects(idx, i):
                        idx = rng.randint(0, self.input_dim)
                    W[idx, i] = rng.randn()
            W *= self.sparse_stdev

        W = sharedX(W)
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)

        W ,= self.transformer.get_params()
        assert W.name is not None

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if self.randomize_pools:
            permute = np.zeros((self.detector_layer_dim, self.detector_layer_dim))
            for j in xrange(self.detector_layer_dim):
                i = rng.randint(self.detector_layer_dim)
                permute[i,j] = 1
            self.permute = sharedX(permute)

        if self.mask_weights is not None:
            expected_shape =  (self.input_dim, self.detector_layer_dim)
            if expected_shape != self.mask_weights.shape:
                raise ValueError("Expected mask with shape "+str(expected_shape)+" but got "+str(self.mask_weights.shape))
            self.mask = sharedX(self.mask_weights)

    def censor_updates(self, updates):

        # Patch old pickle files
        if not hasattr(self, 'mask_weights'):
            self.mask_weights = None

        if self.mask_weights is not None:
            W ,= self.transformer.get_params()
            if W in updates:
                updates[W] = updates[W] * self.mask

        if self.max_col_norm is not None:
            assert self.max_row_norm is None
            W ,= self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))

    def get_params(self):
        assert self.b.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    def get_l1_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.abs(W).sum()

    def get_weights(self):
        if self.requires_reformat:
            # This is not really an unimplemented case.
            # We actually don't know how to format the weights
            # in design space. We got the data in topo space
            # and we don't have access to the dataset
            raise NotImplementedError()
        W ,= self.transformer.get_params()
        W = W.get_value()

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if self.randomize_pools:
            warnings.warn("randomize_pools makes get_weights multiply by the permutation matrix. "
                    "If you call set_weights(W) and then call get_weights(), the return value will "
                    "WP not W.")
            P = self.permute.get_value()
            return np.dot(W,P)

        return W

    def set_weights(self, weights):
        W, = self.transformer.get_params()
        W.set_value(weights)

    def set_biases(self, biases):
        self.b.set_value(biases)

    def get_biases(self):
        return self.b.get_value()

    def get_weights_format(self):
        return ('v', 'h')

    def get_weights_view_shape(self):
        total = self.detector_layer_dim
        cols = self.pool_size
        if cols == 1:
            # Let the PatchViewer decide how to arrange the units
            # when they're not pooled
            raise NotImplementedError()
        # When they are pooled, make each pooling unit have one row
        rows = total // cols
        if rows * cols < total:
            rows = rows + 1
        return rows, cols


    def get_weights_topo(self):

        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()

        # There was an implementation of this, but it was broken
        raise NotImplementedError()

    def get_monitoring_channels(self):

        W ,= self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return OrderedDict([
                            ('row_norms_min'  , row_norms.min()),
                            ('row_norms_mean' , row_norms.mean()),
                            ('row_norms_max'  , row_norms.max()),
                            ('col_norms_min'  , col_norms.min()),
                            ('col_norms_mean' , col_norms.mean()),
                            ('col_norms_max'  , col_norms.max()),
                            ])


    def get_monitoring_channels_from_state(self, state):

        P = state

        rval = OrderedDict()

        if self.pool_size == 1:
            vars_and_prefixes = [ (P,'') ]
        else:
            vars_and_prefixes = [ (P, 'p_') ]

        for var, prefix in vars_and_prefixes:
            v_max = var.max(axis=0)
            v_min = var.min(axis=0)
            v_mean = var.mean(axis=0)
            v_range = v_max - v_min

            # max_x.mean_u is "the mean over *u*nits of the max over e*x*amples"
            # The x and u are included in the name because otherwise its hard
            # to remember which axis is which when reading the monitor
            # I use inner.outer rather than outer_of_inner or something like that
            # because I want mean_x.* to appear next to each other in the alphabetical
            # list, as these are commonly plotted together
            for key, val in [
                             ('max_x.max_u', v_max.max()),
                             ('max_x.mean_u', v_max.mean()),
                             ('max_x.min_u', v_max.min()),
                             ('min_x.max_u', v_min.max()),
                             ('min_x.mean_u', v_min.mean()),
                             ('min_x.min_u', v_min.min()),
                             ('range_x.max_u', v_range.max()),
                             ('range_x.mean_u', v_range.mean()),
                             ('range_x.min_u', v_range.min()),
                             ('mean_x.max_u', v_mean.max()),
                             ('mean_x.mean_u', v_mean.mean()),
                             ('mean_x.min_u', v_mean.min())
                             ]:
                rval[prefix+key] = val

        return rval

    def fprop(self, state_below):

        self.input_space.validate(state_below)

#        if self.requires_reformat:
#            if not isinstance(state_below, tuple):
#                for sb in get_debug_values(state_below):
#                    if sb.shape[0] != self.dbm.batch_size:
#                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
#                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim
#
#            state_below = self.input_space.format_as(state_below, self.desired_space)

        z = self.transformer.lmul(state_below) + self.b.dimshuffle('x', 'x', 'x', 0)

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if not hasattr(self, 'pool_stride'):
            self.pool_stride = self.pool_size

        if self.randomize_pools:
            z = T.dot(z, self.permute)

        if not hasattr(self, 'min_zero'):
            self.min_zero = False

        if self.min_zero:
            p = T.zeros_like(z)
        else:
            p = None

        last_start = self.detector_layer_dim  - self.pool_size
        for i in xrange(self.pool_size):
            cur = z[:, :, :, i:last_start+i+1:self.pool_stride]
            if p is None:
                p = cur
            else:
                p = T.maximum(cur, p)

        if self.pool_channels is True:
            p = max_pool_channels_1D(p, self.input_space.num_channels, 1)
        
        p.name = self.layer_name + '_p_'

        return p

    def foo(self, state_below):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)

        z = self.transformer.lmul(state_below) + self.b

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if not hasattr(self, 'pool_stride'):
            self.pool_stride = self.pool_size

        if self.randomize_pools:
            z = T.dot(z, self.permute)

        if not hasattr(self, 'min_zero'):
            self.min_zero = False

        if self.min_zero:
            p = T.zeros_like(z)
        else:
            p = None

        last_start = self.detector_layer_dim  - self.pool_size

        pooling_stack = []
        for i in xrange(self.pool_size):
            cur = z[:,i:last_start+i+1:self.pool_stride]
            cur = cur.reshape((cur.shape[0], cur.shape[1], 1))
            assert cur.ndim == 3
            pooling_stack.append(cur)
        if self.min_zero:
            pooling_stack.append(T.zeros_like(cur))
        pooling_stack = T.concatenate(pooling_stack, axis=2)
        p = pooling_stack.max(axis=2)
        counts = (T.eq(pooling_stack, p.dimshuffle(0, 1, 'x'))).sum(axis=0)

        p.name = self.layer_name + '_p_'

        return p, counts

class MaxoutMDMLp(Layer):
    """
    A hidden maxout layer supporiting multi-channel input with simples linear transformation.
    The layer additionally does two way max pooling 1) as usual maxout per each channel and then 
    2) Lp pooling over multiple input (acoustic) channels. The weights are alwayws tied across channels so the 
    max pooling between channels is related in terms of feature receptors. 
    
    In contrast to Maxout model, here we allow Conv2D input space and perform tensor dot product 
    separately for each of the channels which is then followe by max pooling in usual way.
    """

    def __init__(self,
                 layer_name,
                 num_units,
                 num_pieces,
                 Lp_order = 2.0,
                 pool_stride = None,
                 randomize_pools = False,
                 irange = None,
                 sparse_init = None,
                 sparse_stdev = 1.,
                 include_prob = 1.0,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 max_col_norm = None,
                 max_row_norm = None,
                 mask_weights = None,
                 min_zero = False
        ):
        """
            layer_name: A name for this layer that will be prepended to
                        monitoring channels related to this layer.
            num_units: The number of maxout units to use in this layer.
            num_pieces: The number of linear pieces to use in each maxout
                        unit.
            pool_stride: The distance between the start of each max pooling
                        region. Defaults to num_pieces, which makes the
                        pooling regions disjoint. If set to a smaller number,
                        can do overlapping pools.
            randomize_pools: Does max pooling over randomized subsets of
                        the linear responses, rather than over sequential
                        subsets.
            irange: if specified, initializes each weight randomly in
                U(-irange, irange)
            sparse_init: if specified, irange must not be specified.
                        This is an integer specifying how many weights to make
                        non-zero. All non-zero weights will be initialized
                        randomly in N(0, sparse_stdev^2)
            include_prob: probability of including a weight element in the set
               of weights initialized to U(-irange, irange). If not included
               a weight is initialized to 0. This defaults to 1.
            init_bias: All biases are initialized to this number
            W_lr_scale: The learning rate on the weights for this layer is
                multiplied by this scaling factor
            b_lr_scale: The learning rate on the biases for this layer is
                multiplied by this scaling factor
            max_col_norm: The norm of each column of the weight matrix is
                constrained to have at most this norm. If unspecified, no
                constraint. Constraint is enforced by re-projection (if
                necessary) at the end of each update.
            max_row_norm: Like max_col_norm, but applied to the rows.
            mask_weights: A binary matrix multiplied by the weights after each
                         update, allowing you to restrict their connectivity.
            min_zero: If true, includes a zero in the set we take a max over
                    for each maxout unit. This is equivalent to pooling over
                    rectified linear units.
        """

        detector_layer_dim = num_units * num_pieces
        pool_size = num_pieces

        if pool_stride is None:
            pool_stride = pool_size

        self.__dict__.update(locals())
        del self.self

        self.b = sharedX( np.zeros((self.detector_layer_dim,)) + init_bias, name = layer_name + '_b')

        if max_row_norm is not None:
            raise NotImplementedError()

    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    def set_input_space(self, space):
        """ Note: this resets parameters! """

        self.input_space = space

        assert isinstance(space, Conv2DSpace)
            
        if not ((self.detector_layer_dim - self.pool_size) % self.pool_stride == 0):
            if self.pool_stride == self.pool_size:
                raise ValueError("detector_layer_dim = %d, pool_size = %d. Should be divisible but remainder is %d" %
                             (self.detector_layer_dim, self.pool_size, self.detector_layer_dim % self.pool_size))
            raise ValueError()

        h_shape = (1, self.detector_layer_dim)

        self.h_space = Conv2DSpace(shape=h_shape, num_channels=self.input_space.num_channels, axes=('b','c',0,1))
        self.pool_layer_dim = (self.detector_layer_dim - self.pool_size)/ self.pool_stride + 1
        
        o_shape = (1, self.pool_layer_dim)
        self.output_space = Conv2DSpace(shape = o_shape, num_channels = 1,  axes = ('b','c',0,1))

        self.input_dim = self.input_space.shape[1] #TODO: refactor

        rng = self.mlp.rng
        if self.irange is not None:
            assert self.sparse_init is None
            W = rng.uniform(-self.irange,
                            self.irange,
                            (self.input_dim, self.detector_layer_dim)) * \
                (rng.uniform(0.,1., (self.input_dim, self.detector_layer_dim))
                 < self.include_prob)
        else:
            assert self.sparse_init is not None
            W = np.zeros((self.input_dim, self.detector_layer_dim))
            def mask_rejects(idx, i):
                if self.mask_weights is None:
                    return False
                return self.mask_weights[idx, i] == 0.
            for i in xrange(self.detector_layer_dim):
                assert self.sparse_init <= self.input_dim
                for j in xrange(self.sparse_init):
                    idx = rng.randint(0, self.input_dim)
                    while W[idx, i] != 0 or mask_rejects(idx, i):
                        idx = rng.randint(0, self.input_dim)
                    W[idx, i] = rng.randn()
            W *= self.sparse_stdev

        W = sharedX(W)
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)

        W ,= self.transformer.get_params()
        assert W.name is not None

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if self.randomize_pools:
            permute = np.zeros((self.detector_layer_dim, self.detector_layer_dim))
            for j in xrange(self.detector_layer_dim):
                i = rng.randint(self.detector_layer_dim)
                permute[i,j] = 1
            self.permute = sharedX(permute)

        if self.mask_weights is not None:
            expected_shape =  (self.input_dim, self.detector_layer_dim)
            if expected_shape != self.mask_weights.shape:
                raise ValueError("Expected mask with shape "+str(expected_shape)+" but got "+str(self.mask_weights.shape))
            self.mask = sharedX(self.mask_weights)

    def censor_updates(self, updates):

        # Patch old pickle files
        if not hasattr(self, 'mask_weights'):
            self.mask_weights = None

        if self.mask_weights is not None:
            W ,= self.transformer.get_params()
            if W in updates:
                updates[W] = updates[W] * self.mask

        if self.max_col_norm is not None:
            assert self.max_row_norm is None
            W ,= self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))

    def get_params(self):
        assert self.b.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    def get_l1_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.abs(W).sum()

    def get_weights(self):
        if self.requires_reformat:
            # This is not really an unimplemented case.
            # We actually don't know how to format the weights
            # in design space. We got the data in topo space
            # and we don't have access to the dataset
            raise NotImplementedError()
        W ,= self.transformer.get_params()
        W = W.get_value()

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if self.randomize_pools:
            warnings.warn("randomize_pools makes get_weights multiply by the permutation matrix. "
                    "If you call set_weights(W) and then call get_weights(), the return value will "
                    "WP not W.")
            P = self.permute.get_value()
            return np.dot(W,P)

        return W

    def set_weights(self, weights):
        W, = self.transformer.get_params()
        W.set_value(weights)

    def set_biases(self, biases):
        self.b.set_value(biases)

    def get_biases(self):
        return self.b.get_value()

    def get_weights_format(self):
        return ('v', 'h')

    def get_weights_view_shape(self):
        total = self.detector_layer_dim
        cols = self.pool_size
        if cols == 1:
            # Let the PatchViewer decide how to arrange the units
            # when they're not pooled
            raise NotImplementedError()
        # When they are pooled, make each pooling unit have one row
        rows = total // cols
        if rows * cols < total:
            rows = rows + 1
        return rows, cols


    def get_weights_topo(self):

        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()

        # There was an implementation of this, but it was broken
        raise NotImplementedError()

    def get_monitoring_channels(self):

        W ,= self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return OrderedDict([
                            ('row_norms_min'  , row_norms.min()),
                            ('row_norms_mean' , row_norms.mean()),
                            ('row_norms_max'  , row_norms.max()),
                            ('col_norms_min'  , col_norms.min()),
                            ('col_norms_mean' , col_norms.mean()),
                            ('col_norms_max'  , col_norms.max()),
                            ])


    def get_monitoring_channels_from_state(self, state):

        P = state

        rval = OrderedDict()

        if self.pool_size == 1:
            vars_and_prefixes = [ (P,'') ]
        else:
            vars_and_prefixes = [ (P, 'p_') ]

        for var, prefix in vars_and_prefixes:
            v_max = var.max(axis=0)
            v_min = var.min(axis=0)
            v_mean = var.mean(axis=0)
            v_range = v_max - v_min

            # max_x.mean_u is "the mean over *u*nits of the max over e*x*amples"
            # The x and u are included in the name because otherwise its hard
            # to remember which axis is which when reading the monitor
            # I use inner.outer rather than outer_of_inner or something like that
            # because I want mean_x.* to appear next to each other in the alphabetical
            # list, as these are commonly plotted together
            for key, val in [
                             ('max_x.max_u', v_max.max()),
                             ('max_x.mean_u', v_max.mean()),
                             ('max_x.min_u', v_max.min()),
                             ('min_x.max_u', v_min.max()),
                             ('min_x.mean_u', v_min.mean()),
                             ('min_x.min_u', v_min.min()),
                             ('range_x.max_u', v_range.max()),
                             ('range_x.mean_u', v_range.mean()),
                             ('range_x.min_u', v_range.min()),
                             ('mean_x.max_u', v_mean.max()),
                             ('mean_x.mean_u', v_mean.mean()),
                             ('mean_x.min_u', v_mean.min())
                             ]:
                rval[prefix+key] = val

        return rval

    def fprop(self, state_below):

        self.input_space.validate(state_below)

#        if self.requires_reformat:
#            if not isinstance(state_below, tuple):
#                for sb in get_debug_values(state_below):
#                    if sb.shape[0] != self.dbm.batch_size:
#                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
#                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim
#
#            state_below = self.input_space.format_as(state_below, self.desired_space)

        z = self.transformer.lmul(state_below) + self.b.dimshuffle('x', 'x', 'x', 0)

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if not hasattr(self, 'pool_stride'):
            self.pool_stride = self.pool_size

        if self.randomize_pools:
            z = T.dot(z, self.permute)

        if not hasattr(self, 'min_zero'):
            self.min_zero = False

        if self.min_zero:
            p = T.zeros_like(z)
        else:
            p = None

        last_start = self.detector_layer_dim  - self.pool_size
        for i in xrange(self.pool_size):
            cur = z[:, :, :, i:last_start+i+1:self.pool_stride]
            if p is None:
                p = cur
            else:
                p = T.maximum(cur, p)

        if self.Lp_order >= 1:
            p = Lp_pool_channels_1D(p, self.input_space.num_channels, 1, self.Lp_order)
        
        p.name = self.layer_name + '_p_'

        return p

    def foo(self, state_below):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)

        z = self.transformer.lmul(state_below) + self.b

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if not hasattr(self, 'pool_stride'):
            self.pool_stride = self.pool_size

        if self.randomize_pools:
            z = T.dot(z, self.permute)

        if not hasattr(self, 'min_zero'):
            self.min_zero = False

        if self.min_zero:
            p = T.zeros_like(z)
        else:
            p = None

        last_start = self.detector_layer_dim  - self.pool_size

        pooling_stack = []
        for i in xrange(self.pool_size):
            cur = z[:,i:last_start+i+1:self.pool_stride]
            cur = cur.reshape((cur.shape[0], cur.shape[1], 1))
            assert cur.ndim == 3
            pooling_stack.append(cur)
        if self.min_zero:
            pooling_stack.append(T.zeros_like(cur))
        pooling_stack = T.concatenate(pooling_stack, axis=2)
        p = pooling_stack.max(axis=2)
        counts = (T.eq(pooling_stack, p.dimshuffle(0, 1, 'x'))).sum(axis=0)

        p.name = self.layer_name + '_p_'

        return p, counts

def max_pool_channels_1D(bc01, num_inp_channels, num_out_channels):
    pp = None
    for fc in xrange(num_inp_channels):
        t = bc01[:,fc*num_out_channels:(fc+1)*num_out_channels,:,:]
        if pp is None:
            pp = t
        else:
            pp = T.maximum(pp,t)
    return pp

def Lp_pool_channels_1D(bc01, num_inp_channels, num_out_channels, lp_order):
    pp = None
    for fc in xrange(num_inp_channels):
        t = bc01[:,fc*num_out_channels:(fc+1)*num_out_channels,:,:]
        t = T.abs_(t)**lp_order
        if pp is None:
            pp = t
        else:
            pp = pp + t
    return pp**(1.0/lp_order)

