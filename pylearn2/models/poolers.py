
__authors__ = "Pawel Swietojanski"
__copyright__ = "Copyright 2014, University of Edinburgh"

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


class LpFixedPooler(Layer):
    """
    This code implements the functionality proposed in:
    Learn p-norm units.

    Note: this is not the official code from the authors.
    """

    def __init__(self,
                 layer_name,
                 pool_size,
                 pool_stride = None,
                 lp_order = 2.0,
                 init_b = 0.0,
                 init_c = 0.0,
                 center_bias = False,
                 post_bias = False,
                 c_lr_scale = None,
                 b_lr_scale = None,
                 pool_normalisation = False,
                 mv_normalisation = False,
                 min_zero = False,
        ):
        """

        """

        if pool_stride is None:
            pool_stride = pool_size

        self.__dict__.update(locals())
        del self.self

    def get_lr_scalers(self):

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None
        if not hasattr(self, 'c_lr_scale'):
            self.c_lr_scale = None

        rval = OrderedDict()

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        if self.c_lr_scale is not None:
            rval[self.c] = self.c_lr_scale

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

        self.detector_layer_dim = self.input_dim
        if not ((self.detector_layer_dim - self.pool_size) % self.pool_stride == 0):
            if self.pool_stride == self.pool_size:
                raise ValueError("detector_layer_dim = %d, pool_size = %d. Should be divisible but remainder is %d" %
                             (self.detector_layer_dim, self.pool_size, self.detector_layer_dim % self.pool_size))
            raise ValueError()

        self.h_space = VectorSpace(self.detector_layer_dim)
        self.pool_layer_dim = (self.detector_layer_dim - self.pool_size)/ self.pool_stride + 1
        self.output_space = VectorSpace(self.pool_layer_dim)

        if self.center_bias:
            self.c = sharedX( np.zeros((self.detector_layer_dim,)) + self.init_c, name = self.layer_name + '_c')
        else:
            assert self.c_lr_scale is None

        if self.post_bias:
            self.b = sharedX( np.zeros((self.pool_layer_dim,)) + self.init_b, name = self.layer_name + '_b')
        else:
            self.b = sharedX( np.zeros((self.detector_layer_dim,)) + self.init_b, name = self.layer_name + '_b')

    def censor_updates(self, updates):

        # Patch old pickle files
        if not hasattr(self, 'mask_weights'):
            self.mask_weights = None


    def get_params(self):
        rval = []

        assert self.b.name is not None
        assert self.b not in rval
        rval.append(self.b)

        if self.center_bias:
            assert self.c.name is not None
            assert self.c not in rval
            rval.append(self.c)

        return rval

    def get_weight_decay(self, coeff):
        raise NotImplementedError()

    def get_l1_weight_decay(self, coeff):
        raise NotImplementedError()

    def get_monitoring_channels(self):


        rval =  OrderedDict([
                            ('b_min', self.b.min()),
                            ('b_mean',  self.b.mean()),
                            ('b_max',  self.b.max()),
                            ('b_std',  self.b.std())
                            ])

        if self.center_bias:
            rval['c_min'] = self.c.min()
            rval['c_mean'] = self.c.mean()
            rval['c_max'] = self.c.max()
            rval['c_std'] = self.c.std()

        return rval


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

        z = state_below

        if self.post_bias is False:
            z = z + self.b

        if not hasattr(self, 'pool_stride'):
            self.pool_stride = self.pool_size

        if not hasattr(self, 'min_zero'):
            self.min_zero = False

        epsilon = 1e-10

        z_act = z
        if self.center_bias:
            z_act = z_act - self.c.dimshuffle('x',0)

        if self.min_zero:
            z_act = T.maximum(z_act, 0)
        else:
            z_act = T.maximum(T.abs_(z_act), epsilon) # fix to gradients w.r.t p in case sum_i x_i**p is close to or 0

        p, s = None, None
        last_start = self.detector_layer_dim  - self.pool_size
        for i in xrange(self.pool_size):
            cur = z_act[:,i:last_start+i+1:self.pool_stride]
            cur = cur**self.lp_order
            if p is None:
                p = cur
            else:
                p = p + cur

        if self.pool_normalisation:
            p = p/self.pool_size

        p = T.maximum(p, epsilon) # fix to gradients w.r.t x in case sum_i x_i**p is close to or 0
        p = p**(1.0/self.lp_order)

        self.post_pool_normalisation = False
        if self.post_pool_normalisation:
            p = p/self.pool_size

        if self.post_bias:
            p = p + self.b

        if self.mv_normalisation:
            stddev = T.sqrt(T.mean(T.sqr(p), axis=1)).dimshuffle(0,'x')
            p = p*(1.0/stddev)*(stddev>1) + p*(stddev<=1)

        p.name = self.layer_name + '_p_'

        return p

    def foo(self, state_below):
        raise NotImplementedError();

class LpPooler(Layer):
    """
    This code implements the functionality proposed in:
    Learn p-norm units.

    Note: this is not the official code from the authors.
    """

    def __init__(self,
                 layer_name,
                 pool_size,
                 pool_stride = None,
                 init_p = 2.0,
                 init_b = 0.0,
                 init_c = 0.0,
                 center_bias = False,
                 post_bias = False,
                 p_order_constraints = [1.5,10.],
                 p_lr_scale = None,
                 c_lr_scale = None,
                 b_lr_scale = None,
                 pool_normalisation = False,
                 mv_normalisation = False,
                 min_zero = False,
                 reparam_p = 'softplus'
        ):
        """

        """

        assert reparam_p in ['softplus', 'rect']

        if pool_stride is None:
            pool_stride = pool_size

        self.__dict__.update(locals())
        del self.self
        
    def get_lr_scalers(self):

        if not hasattr(self, 'p_lr_scale'):
            self.p_lr_scale = None
        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None
        if not hasattr(self, 'c_lr_scale'):
            self.c_lr_scale = None

        rval = OrderedDict()

        if self.p_lr_scale is not None:
            rval[self.p] = self.p_lr_scale
            
        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale
            
        if self.c_lr_scale is not None:
            rval[self.c] = self.c_lr_scale

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

        self.detector_layer_dim = self.input_dim
        if not ((self.detector_layer_dim - self.pool_size) % self.pool_stride == 0):
            if self.pool_stride == self.pool_size:
                raise ValueError("detector_layer_dim = %d, pool_size = %d. Should be divisible but remainder is %d" %
                             (self.detector_layer_dim, self.pool_size, self.detector_layer_dim % self.pool_size))
            raise ValueError()

        self.h_space = VectorSpace(self.detector_layer_dim)
        self.pool_layer_dim = (self.detector_layer_dim - self.pool_size)/ self.pool_stride + 1
        self.output_space = VectorSpace(self.pool_layer_dim)

        rng = self.mlp.rng
        if self.init_p > 1.0:
            if self.reparam_p == 'softplus':
                init_rho_order = np.log(np.exp(self.init_p-1)-1)
            elif self.reparam_p == 'rect':
                init_rho_order = self.init_p
            self.p = sharedX( np.zeros((self.pool_layer_dim,)) + init_rho_order, name = self.layer_name + '_p')
        else:
            p0 = np.log(np.exp(self.p_order_constraints[0]-1)-1)
            pk = np.log(np.exp(self.p_order_constraints[1]-1)-1)
            self.p = sharedX( np.random.uniform(p0, pk, (self.pool_layer_dim,)), name = self.layer_name + '_p')
        
        if self.center_bias:
            self.c = sharedX( np.zeros((self.detector_layer_dim,)) + self.init_c, name = self.layer_name + '_c')
        else:
            assert self.c_lr_scale is None
        
        if self.post_bias:
            self.b = sharedX( np.zeros((self.pool_layer_dim,)) + self.init_b, name = self.layer_name + '_b')
        else:
            self.b = sharedX( np.zeros((self.detector_layer_dim,)) + self.init_b, name = self.layer_name + '_b')

    def censor_updates(self, updates):

        # Patch old pickle files
        if not hasattr(self, 'mask_weights'):
            self.mask_weights = None
            

    def get_params(self):
        rval = []
        
        assert self.p.name is not None
        assert self.p not in rval
        rval.append(self.p)
        
        assert self.b.name is not None
        assert self.b not in rval
        rval.append(self.b)
        
        if self.center_bias:
            assert self.c.name is not None
            assert self.c not in rval
            rval.append(self.c)
            
        return rval

    def get_weight_decay(self, coeff):
        raise NotImplementedError()

    def get_l1_weight_decay(self, coeff):
        raise NotImplementedError()

    def get_monitoring_channels(self):
        
        if self.reparam_p == 'softplus':
            toLp  = lambda x: 1+T.log(T.exp(x)+1) 
        elif self.reparam_p == 'rect':
            toLp = lambda x: T.maximum(1., x)
        
        rval =  OrderedDict([
                            ('p_orders_min'  , toLp(self.p.min())),
                            ('p_orders_mean' , toLp(self.p.mean())),
                            ('p_orders_max'  , toLp(self.p.max())),
                            ('p_orders_std'  , toLp(self.p.std())),
                            ('b_min', self.b.min()),
                            ('b_mean',  self.b.mean()),
                            ('b_max',  self.b.max()),
                            ('b_std',  self.b.std())
                            ])
        
        if self.center_bias:
            rval['c_min'] = self.c.min()
            rval['c_mean'] = self.c.mean()
            rval['c_max'] = self.c.max()
            rval['c_std'] = self.c.std()
        
        return rval


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

        z = state_below

        if self.post_bias is False:
            z = z + self.b

        if not hasattr(self, 'pool_stride'):
            self.pool_stride = self.pool_size
            
        if not hasattr(self, 'min_zero'):
            self.min_zero = False
        
        if self.reparam_p == 'softplus':
            pr  = 1+T.log(1+T.exp(self.p))
        elif self.reparam_p == 'rect':
            pr = T.maximum(1., self.p)
        else:
            raise Exception('WTF?')
        
        epsilon = 1e-10
        
        z_act = z  
        if self.center_bias:
            z_act = z_act - self.c.dimshuffle('x',0)
            
        if self.min_zero:
            z_act = T.maximum(z_act, 0)
        else:
            z_act = T.maximum(T.abs_(z_act), epsilon) # fix to gradients w.r.t p in case sum_i x_i**p is close to or 0       
        
        p, s = None, None
        last_start = self.detector_layer_dim  - self.pool_size
        for i in xrange(self.pool_size):
            cur = z_act[:,i:last_start+i+1:self.pool_stride]
            cur = cur**pr
            if p is None:
                p = cur
            else:
                p = p + cur
        
        if self.pool_normalisation:   
            p = p*(1.0/self.pool_size)
        
        p = T.maximum(p, epsilon) # fix to gradients w.r.t x in case sum_i x_i**p is close to or 0         
        p = p**(1.0/pr)
        
        self.post_pool_normalisation = False   
        if self.post_pool_normalisation:   
            p = p/self.pool_size
            
        if self.post_bias:
            p = p + self.b
        
        if self.mv_normalisation:
            stddev = T.sqrt(T.mean(T.sqr(p), axis=1)).dimshuffle(0,'x')
            p = p*(1.0/stddev)*(stddev>1) + p*(stddev<=1)
        
        p.name = self.layer_name + '_p_'

        return p

    def foo(self, state_below):
        raise NotImplementedError();


class LpPoolerR(Layer):
    """
    """

    def __init__(self,
                 layer_name,
                 pool_size,
                 pool_stride = None,
                 init_p = 2.0,
                 init_b = 0.0,
                 init_c = 0.0,
                 center_bias = False,
                 post_bias = False,
                 p_order_constraints = [1.5,10.],
                 p_lr_scale = None,
                 c_lr_scale = None,
                 b_lr_scale = None,
                 pool_normalisation = False,
                 mv_normalisation = False,
                 min_zero = False,
                 reparam_p = 'softplus'
        ):
        """

        """

        assert reparam_p in ['softplus', 'rect']

        if pool_stride is None:
            pool_stride = pool_size

        self.__dict__.update(locals())
        del self.self
        
    def get_lr_scalers(self):

        if not hasattr(self, 'p_lr_scale'):
            self.p_lr_scale = None
        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None
        if not hasattr(self, 'c_lr_scale'):
            self.c_lr_scale = None

        rval = OrderedDict()

        if self.p_lr_scale is not None:
            rval[self.p] = self.p_lr_scale
            
        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale
            
        if self.c_lr_scale is not None:
            rval[self.c] = self.c_lr_scale

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

        self.detector_layer_dim = self.input_dim
        if not ((self.detector_layer_dim - self.pool_size) % self.pool_stride == 0):
            if self.pool_stride == self.pool_size:
                raise ValueError("detector_layer_dim = %d, pool_size = %d. Should be divisible but remainder is %d" %
                             (self.detector_layer_dim, self.pool_size, self.detector_layer_dim % self.pool_size))
            raise ValueError()

        self.h_space = VectorSpace(self.detector_layer_dim)
        self.pool_layer_dim = (self.detector_layer_dim - self.pool_size)/ self.pool_stride + 1
        self.output_space = VectorSpace(self.pool_layer_dim)

        rng = self.mlp.rng
        if self.init_p > 1.0:
            if self.reparam_p == 'softplus':
                init_rho_order = np.log(np.exp(self.init_p-1)-1)
            elif self.reparam_p == 'rect':
                init_rho_order = self.init_p
            self.p = sharedX( np.zeros((self.pool_layer_dim,)) + init_rho_order, name = self.layer_name + '_p')
        else:
            p0 = np.log(np.exp(self.p_order_constraints[0]-1)-1)
            pk = np.log(np.exp(self.p_order_constraints[1]-1)-1)
            self.p = sharedX( np.random.uniform(p0, pk, (self.pool_layer_dim,)), name = self.layer_name + '_p')
        
        if self.center_bias:
            self.c = sharedX( np.zeros((self.detector_layer_dim,)) + self.init_c, name = self.layer_name + '_c')
        else:
            assert self.c_lr_scale is None
        
        if self.post_bias:
            self.b = sharedX( np.zeros((self.pool_layer_dim,)) + self.init_b, name = self.layer_name + '_b')
        else:
            self.b = sharedX( np.zeros((self.detector_layer_dim,)) + self.init_b, name = self.layer_name + '_b')

    def censor_updates(self, updates):

        # Patch old pickle files
        if not hasattr(self, 'mask_weights'):
            self.mask_weights = None
            

    def get_params(self):
        rval = []
        
        assert self.p.name is not None
        assert self.p not in rval
        rval.append(self.p)
        
        assert self.b.name is not None
        assert self.b not in rval
        rval.append(self.b)
        
        if self.center_bias:
            assert self.c.name is not None
            assert self.c not in rval
            rval.append(self.c)
            
        return rval

    def get_weight_decay(self, coeff):
        raise NotImplementedError()

    def get_l1_weight_decay(self, coeff):
        raise NotImplementedError()

    def get_monitoring_channels(self):
        
        if self.reparam_p == 'softplus':
            toLp  = lambda x: 1+T.log(T.exp(x)+1) 
        elif self.reparam_p == 'rect':
            toLp = lambda x: T.maximum(1., x)
        
        rval =  OrderedDict([
                            ('p_orders_min'  , toLp(self.p.min())),
                            ('p_orders_mean' , toLp(self.p.mean())),
                            ('p_orders_max'  , toLp(self.p.max())),
                            ('p_orders_std'  , toLp(self.p.std())),
                            ('b_min', self.b.min()),
                            ('b_mean',  self.b.mean()),
                            ('b_max',  self.b.max()),
                            ('b_std',  self.b.std())
                            ])
        
        if self.center_bias:
            rval['c_min'] = self.c.min()
            rval['c_mean'] = self.c.mean()
            rval['c_max'] = self.c.max()
            rval['c_std'] = self.c.std()
        
        return rval


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

        z = state_below

        if self.post_bias is False:
            z = z + self.b

        if not hasattr(self, 'pool_stride'):
            self.pool_stride = self.pool_size
            
        if not hasattr(self, 'min_zero'):
            self.min_zero = False
        
        if self.reparam_p == 'softplus':
            pr  = 1+T.log(1+T.exp(self.p))
        elif self.reparam_p == 'rect':
            pr = T.maximum(1., self.p)
        else:
            raise Exception('WTF?')
        
        epsilon = 1e-10
        
        z_act = z
        
        r = None
        last_start = self.detector_layer_dim  - self.pool_size
        for i in xrange(self.pool_size):
            cur = T.abs_(z[:,i:last_start+i+1:self.pool_stride])
            if r is None:
                r = cur
            else:
                r = r + cur
         
        z_norm = T.zeros_like(z)
        for i in xrange(self.pool_size):
            z_norm = T.set_subtensor(z_norm[:,i:last_start+i+1:self.pool_stride], r)
                 
        z_act = z/(z_norm+1e-7)  #apply weighted scaling
        
        if self.center_bias:
            z_act = z_act - self.c.dimshuffle('x',0)
            
        if self.min_zero:
            z_act = T.maximum(z_act, 0)
        else:
            z_act = T.maximum(T.abs_(z_act), epsilon) # fix to gradients w.r.t p in case sum_i x_i**p is close to or 0       
        
        z_act=z_act**pr
        
        p = None
        last_start = self.detector_layer_dim  - self.pool_size
        for i in xrange(self.pool_size):
            cur = z_act[:,i:last_start+i+1:self.pool_stride]
            if p is None:
                p = cur
            else:
                p = p + cur
        
        if self.pool_normalisation:   
            p = p/self.pool_size
        
        p = T.maximum(p, epsilon) # fix to gradients w.r.t x in case sum_i x_i**p is close to or 0         
        p = p**(1.0/pr)
        
        self.post_pool_normalisation = False   
        if self.post_pool_normalisation:   
            p = p/self.pool_size
            
        if self.post_bias:
            p = p + self.b
        
        if self.mv_normalisation:
            stddev = T.sqrt(T.mean(T.sqr(p), axis=1)).dimshuffle(0,'x')
            p = p/stddev*(stddev>1) + p*(stddev<=1)
            #mean = T.mean(p, axis=0).dimshuffle('x',0)
            #stddev = T.sqrt(T.mean(T.sqr(p), axis=0)).dimshuffle('x',0)
            #p = (p-mean)/(stddev+0.001)
        
        p.name = self.layer_name + '_p_'

        return p

    def foo(self, state_below):
        raise NotImplementedError();

class Normaliser(Layer):
    """
    """

    def __init__(self,
                 layer_name,
                 m_normalisation = True, #mean normalisation
                 v_normalisation = True, #variance normalisation
                 normalisation_stats = 'cmb', 
                 update_stats = False, #update states based on new data, for example, at testing stage in c mode
                 m_lr_scale = None,
                 v_lr_scale = None,
        ):
        """

        """

        #cmb - cumulative mb stats over training set, learn - simply tune these by backprop
        assert normalisation_stats in ['cmb','learn'] 
        #stats update only makes sense in unsupervised mode
        if update_stats is True: assert normalisation_stats in ['cmb']
        #in this case we do not want to update means and variances by backprop!
        if normalisation_stats in ['cmb']:
            m_lr_scale, v_lr_scale = 0., 0.
        
        self.__dict__.update(locals())
        del self.self

    def get_lr_scalers(self):
        
        if not hasattr(self, 'm_lr_scale'):
            self.m_lr_scale = None
        if not hasattr(self, 'v_lr_scale'):
            self.v_lr_scale = None
        
        rval = OrderedDict()
        
        if self.m_lr_scale is not None:
            rval[self.mean] = self.m_lr_scale
            
        if self.v_lr_scale is not None:
            rval[self.stdev] = self.v_lr_scale
        
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

        self.output_space = VectorSpace(self.input_dim)
      
        #mean is gonna to be needed regardless 'm_normalisation' option        
        self.mean = sharedX( np.zeros((self.input_dim,)),  name=self.layer_name +"_mean")
        if self.v_normalisation:
            self.stdev = sharedX( np.zeros((self.input_dim,)) + 1., name=self.layer_name +"_stdev")

        #sum of squares accumulator to compute variance
        self.m_sqr = sharedX( np.zeros((self.input_dim,)),  name=self.layer_name +"_m_sqr")
            
        self.datapoints = 0

    def censor_updates(self, updates):
        pass
            
    def get_params(self):
        rval = []
        
        if self.m_normalisation:
            assert self.mean.name is not None
            assert self.mean not in rval
            rval.append(self.mean)
            
        if self.v_normalisation:
            assert self.stdev.name is not None
            assert self.stdev not in rval
            rval.append(self.stdev)
        
        return rval

    def get_weight_decay(self, coeff):
        raise NotImplementedError()

    def get_l1_weight_decay(self, coeff):
        raise NotImplementedError()

    def get_monitoring_channels(self):
        
        rval = OrderedDict([])
        
        if self.m_normalisation:
            rval['m_min'] = self.mean.min();
            rval['m_mean'] = self.mean.mean();
            rval['m_max'] = self.mean.max();
            rval['m_std'] = self.mean.std();
            
        if self.v_normalisation:
            rval['std_min'] = self.stdev.min();
            rval['std_mean'] = self.stdev.mean();
            rval['std_max'] = self.stdev.max();
            rval['std_std'] = self.stdev.std();
        
        return rval


    def get_monitoring_channels_from_state(self, state):

        rval = OrderedDict()

        mx = state.max(axis=0)
        mean = state.mean(axis=0)
        mn = state.min(axis=0)
        rg = mx - mn

        rval['range_x_max_u'] = rg.max()
        rval['range_x_mean_u'] = rg.mean()
        rval['range_x_min_u'] = rg.min()

        rval['max_x_max_u'] = mx.max()
        rval['max_x_mean_u'] = mx.mean()
        rval['max_x_min_u'] = mx.min()

        rval['mean_x_max_u'] = mean.max()
        rval['mean_x_mean_u'] = mean.mean()
        rval['mean_x_min_u'] = mean.min()

        rval['min_x_max_u'] = mn.max()
        rval['min_x_mean_u'] = mn.mean()
        rval['min_x_min_u'] = mn.min()
        
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

        z = state_below
        
        if self.normalisation_stats == 'cmb' and self.update_stats is True:
    
            batch_size = self.mlp.batch_size        
            self.datapoints += batch_size
            
            z_m = z - T.mean(z, axis=0).dimshuffle('x',0)
            #update means and variances
            self.mean = self.mean + T.sum(z_m, axis=0)/self.datapoints
            if self.v_normalisation:
                z_m_new = z - self.mean.dimshuffle('x',0)
                self.m_sqr = self.m_sqr + T.mean(z_m*z_m_new, axis=0)
                self.stdev = self.m_sqr/(self.datapoints-1)
        
        if self.m_normalisation:
            z = z - self.mean
            
        if self.v_normalisation:
            if self.normalisation_stats == 'learn':
                vr = 0.1+T.log(1+T.exp(self.stdev)) #reparametrize variance so the optimised value is in R
                z = z/vr
            else:
                z = z/self.stdev
        
        z.name = self.layer_name + '_z'

        return z

    def foo(self, state_below):
        raise NotImplementedError();

class GaussianPooler(Layer):
    """
    """

    def __init__(self,
                 layer_name,
                 pool_size,
                 pool_stride = None,
                 init_u = 0.,
                 init_beta = 1.0,
                 u_lr_scale = None,
                 beta_lr_scale= None,
                 mv_normalisation = False,
        ):
        """

        """
        
        if pool_stride is None:
            pool_stride = pool_size

        self.__dict__.update(locals())
        del self.self

    def get_lr_scalers(self):

        if not hasattr(self, 'u_lr_scale'):
            self.u_lr_scale = None

        if not hasattr(self, 'beta_lr_scale'):
            self.beta_lr_scale = None

        rval = OrderedDict()

        if self.u_lr_scale is not None:
            rval[self.u] = self.u_lr_scale

        if self.beta_lr_scale is not None:
            rval[self.beta] = self.beta_lr_scale

        return rval

    def set_input_space(self, space):
        """ Note: this resets parameters! 
            TODO: work on any space type.
        """

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        self.detector_layer_dim = self.input_dim
        if not ((self.input_dim - self.pool_size) % self.pool_stride == 0):
            if self.pool_stride == self.pool_size:
                raise ValueError("detector_layer_dim = %d, pool_size = %d. Should be divisible but remainder is %d" %
                             (self.detector_layer_dim, self.pool_size, self.detector_layer_dim % self.pool_size))
            raise ValueError()

        self.pool_layer_dim = (self.detector_layer_dim - self.pool_size) / self.pool_stride + 1
        self.output_space = VectorSpace(self.pool_layer_dim)
       
        self.u = sharedX( np.zeros((self.pool_layer_dim,)) + self.init_u, name = self.layer_name + '_u')
        self.beta = sharedX( np.zeros((self.pool_layer_dim,)) + self.init_beta, name = self.layer_name + '_beta')
        
#        #mean is gonna to be needed regardless 'm_normalisation' option        
#        self.mean = sharedX( np.zeros((self.input_dim,)),  name=self.layer_name +"_mean")
#        self.stdev = sharedX( np.zeros((self.input_dim,)) + 1., name=self.layer_name +"_stdev")
#        #sum of squares accumulator to compute variance
#        self.mean_sqr = sharedX(np.zeros((self.input_dim,)),  name=self.layer_name +"_mean_sqr")
#        self.datapoints=0

    def censor_updates(self, updates):
        pass
            
    def get_params(self):
        assert self.u.name is not None
        assert self.beta.name is not None
        rval = []
        assert self.u not in rval
        rval.append(self.u)
        assert self.beta not in rval
        rval.append(self.beta)
        return rval

    def get_monitoring_channels(self):
        
        return OrderedDict([
                            ('u_min'  , self.u.min()),
                            ('u_mean' , self.u.mean()),
                            ('u_max'  , self.u.max()),
                            ('beta_min'  , self.beta.min()),
                            ('beta_mean' , self.beta.mean()),
                            ('beta_max'  , self.beta.max()),
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
            
        z = state_below
        
        if self.mv_normalisation:
            
#            batch_size = self.mlp.batch_size        
#            self.datapoints += batch_size
#            
#            z_m = z - T.mean(z, axis=0).dimshuffle('x',0)
#            #update means and variances
#            self.mean = self.mean + T.sum(z_m, axis=0)/self.datapoints
#            z_m_new = z - self.mean.dimshuffle('x',0)
#            self.m_sqr = self.mean_sqr + T.mean(z_m*z_m_new, axis=0)
#            self.stdev = self.mean_sqr/(self.datapoints-1)
#            
#            z=(z-self.mean)/T.maximum(self.stdev, 0.1)
            
            #stddev = T.sqrt(T.mean(T.sqr(z), axis=1)).dimshuffle(0,'x')
            #z = z/stddev*(stddev>1) + z*(stddev<=1)
            #mean = T.mean(z, axis=0).dimshuffle('x',0)
            #stddev = T.sqrt(T.mean(T.sqr(z), axis=0)).dimshuffle('x',0)
            #z = (z-mean)/(stddev+0.01)
            act_norm = T.sqrt(T.sum(T.sqr(z), axis=1))
            des_norm = T.clip(act_norm, 0., 10.)
            mult = des_norm/(1e-7+act_norm)
            z = z*mult.dimshuffle(0,'x')
        
        p, gi, gj = None, None, None
        #xi = 0.01 + T.log(1+T.exp(self.beta))
        
        last_start = self.detector_layer_dim  - self.pool_size
        for i in xrange(self.pool_size):
            cur = z[:,i:last_start+i+1:self.pool_stride]
            gi = T.exp(-0.5*self.beta*T.sqr(cur-self.u))
            if gj is None:
                p = cur*T.sqrt(gi)
                gj = gi
            else:
                p = p + cur*T.sqrt(gi)
                gj = gj + gi
        
        p = p/(1e-7+T.sqrt(gj))
        
        p.name = self.layer_name + '_p_'

        return p

    def foo(self, state_below):
        pass

class GaussianPoolerW(Layer):
    """
    """

    def __init__(self,
                 layer_name,
                 pool_size,
                 pool_stride = None,
                 init_u = 0.,
                 init_beta = 1.0,
                 u_lr_scale = None,
                 beta_lr_scale= None,
                 min_zero = False,
        ):
        """

        """
        
        if pool_stride is None:
            pool_stride = pool_size

        self.__dict__.update(locals())
        del self.self

    def get_lr_scalers(self):

        if not hasattr(self, 'u_lr_scale'):
            self.u_lr_scale = None

        if not hasattr(self, 'beta_lr_scale'):
            self.beta_lr_scale = None

        rval = OrderedDict()

        if self.u_lr_scale is not None:
            rval[self.u] = self.u_lr_scale

        if self.beta_lr_scale is not None:
            rval[self.beta] = self.beta_lr_scale

        return rval

    def set_input_space(self, space):
        """ Note: this resets parameters! 
            TODO: work on any space type.
        """

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        self.detector_layer_dim = self.input_dim
        if not ((self.input_dim - self.pool_size) % self.pool_stride == 0):
            if self.pool_stride == self.pool_size:
                raise ValueError("detector_layer_dim = %d, pool_size = %d. Should be divisible but remainder is %d" %
                             (self.detector_layer_dim, self.pool_size, self.detector_layer_dim % self.pool_size))
            raise ValueError()

        self.pool_layer_dim = (self.detector_layer_dim - self.pool_size) / self.pool_stride + 1
        self.output_space = VectorSpace(self.pool_layer_dim)
       
        self.u = sharedX( np.zeros((self.pool_layer_dim,)) + self.init_u, name = self.layer_name + '_u')
        self.beta = sharedX( np.zeros((self.pool_layer_dim,)) + self.init_beta, name = self.layer_name + '_beta')
        
#        #mean is gonna to be needed regardless 'm_normalisation' option        
#        self.mean = sharedX( np.zeros((self.input_dim,)),  name=self.layer_name +"_mean")
#        self.stdev = sharedX( np.zeros((self.input_dim,)) + 1., name=self.layer_name +"_stdev")
#        #sum of squares accumulator to compute variance
#        self.mean_sqr = sharedX(np.zeros((self.input_dim,)),  name=self.layer_name +"_mean_sqr")
#        self.datapoints=0

    def censor_updates(self, updates):
        pass
            
    def get_params(self):
        assert self.u.name is not None
        assert self.beta.name is not None
        rval = []
        assert self.u not in rval
        rval.append(self.u)
        assert self.beta not in rval
        rval.append(self.beta)
        return rval

    def get_monitoring_channels(self):
        
        return OrderedDict([
                            ('u_min'  , self.u.min()),
                            ('u_mean' , self.u.mean()),
                            ('u_max'  , self.u.max()),
                            ('beta_min'  , self.beta.min()),
                            ('beta_mean' , self.beta.mean()),
                            ('beta_max'  , self.beta.max()),
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
            
        z = state_below
        
        if self.min_zero:
            z = T.maximum(z, 0)
        else:
            #z = T.abs_(z)
            z
        
        r = None
        last_start = self.detector_layer_dim  - self.pool_size
        for i in xrange(self.pool_size):
            cur = z[:,i:last_start+i+1:self.pool_stride]
            cur = T.abs_(cur)
            if r is None:
                r = cur
            else:
                r = r + cur
        
        p, gi, gj = None, None, None
        #xi = 0.01 + T.log(1+T.exp(self.beta))
        
        last_start = self.detector_layer_dim  - self.pool_size
        for i in xrange(self.pool_size):
            cur = z[:,i:last_start+i+1:self.pool_stride]
            cur = cur/(r+1e-7)
            gi = T.exp(-0.5*self.beta*T.sqr(cur-self.u))
            if gj is None:
                p = cur*T.sqrt(gi)
                gj = gi
            else:
                p = p + cur*T.sqrt(gi)
                gj = gj + gi
        
        p = p/(T.sqrt(gj)+1e-7)
        
        p.name = self.layer_name + '_p_'

        return p

    def foo(self, state_below):
        pass


class GaussianPoolerT(Layer):
    """
    This class was used in:

    Differentiable Pooling for Unsupervised Speaker Adaptation,
    Swietojanski and Renals, In Proc. IEEE ICASSP, 2014


    """

    def __init__(self,
                 layer_name,
                 pool_size,
                 pool_stride = None,
                 init_u = 0.,
                 init_beta = 1.0,
                 u_lr_scale = None,
                 beta_lr_scale= None,
                 amp_lr_scale= None,
                 arg_lr_scale= None,
                 learn_amp = True,
                 learn_slope = True,
                 learn_per_pool = False,
                 nonneg_beta = False,
                 activation = 'tanh',
                 l2_reg_pools = False,
        ):
        """

        """
        
        if pool_stride is None:
            pool_stride = pool_size
        
        assert activation in ['tanh', 'sigmoid', 'relu']
        
        if activation == 'tanh':
            self.f = T.tanh
        elif activation == 'sigmoid':
            self.f = T.nnet.sigmoid
        
        if learn_per_pool:
            assert learn_amp is True or learn_slope is True

        self.__dict__.update(locals())
        del self.self

    def get_lr_scalers(self):

        if not hasattr(self, 'u_lr_scale'):
            self.u_lr_scale = None

        if not hasattr(self, 'beta_lr_scale'):
            self.beta_lr_scale = None

        if not hasattr(self, 'amp_lr_scale'):
            self.amp_lr_scale = None
        
        if not hasattr(self, 'arg_lr_scale'):
            self.arg_lr_scale = None

        rval = OrderedDict()

        if self.u_lr_scale is not None:
            rval[self.u] = self.u_lr_scale

        if self.beta_lr_scale is not None:
            rval[self.beta] = self.beta_lr_scale

        if self.amp_lr_scale is not None:
            rval[self.amp] = self.amp_lr_scale
        
        if self.arg_lr_scale is not None:
            rval[self.arg] = self.arg_lr_scale

        return rval

    def set_input_space(self, space):
        """ Note: this resets parameters! 
            TODO: work on any space type.
        """

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        self.detector_layer_dim = self.input_dim
        if not ((self.input_dim - self.pool_size) % self.pool_stride == 0):
            if self.pool_stride == self.pool_size:
                raise ValueError("detector_layer_dim = %d, pool_size = %d. Should be divisible but remainder is %d" %
                             (self.detector_layer_dim, self.pool_size, self.detector_layer_dim % self.pool_size))
            raise ValueError()

        self.pool_layer_dim = (self.detector_layer_dim - self.pool_size) / self.pool_stride + 1
        self.output_space = VectorSpace(self.pool_layer_dim)
       
        self.u = sharedX( np.zeros((self.pool_layer_dim,)) + self.init_u, name = self.layer_name + '_u')
        self.beta = sharedX( np.zeros((self.pool_layer_dim,)) + self.init_beta, name = self.layer_name + '_beta')    
        
        param_dim = self.detector_layer_dim
        if self.learn_per_pool:
            param_dim = self.pool_layer_dim
        
        if self.learn_amp:
            self.amp = sharedX( np.ones((param_dim,)),  name=self.layer_name +"_amp")
        if self.learn_slope:
            self.arg = sharedX( np.ones((param_dim,)),  name=self.layer_name +"_arg")

    def censor_updates(self, updates):
        pass
            
    def get_params(self):
        assert self.u.name is not None
        assert self.beta.name is not None
        
        if self.learn_amp:
            assert self.amp.name is not None
        if self.learn_slope:
            assert self.arg.name is not None

        rval = []
        assert self.u not in rval
        rval.append(self.u)
        assert self.beta not in rval
        rval.append(self.beta)
        
        if self.learn_amp:
            assert self.amp not in rval
            rval.append(self.amp)
        if self.learn_slope:
            assert self.arg not in rval
            rval.append(self.arg)
        
        return rval

    def get_monitoring_channels(self):
        
        rval = OrderedDict([
                            ('u_min'  , self.u.min()),
                            ('u_mean' , self.u.mean()),
                            ('u_max'  , self.u.max()),
                            ('beta_min'  , self.beta.min()),
                            ('beta_mean' , self.beta.mean()),
                            ('beta_max'  , self.beta.max()),
                            ])
        
        if self.learn_amp:
            rval['amp_min'] = self.amp.min()
            rval['amp_mean'] = self.amp.mean()
            rval['amp_max']  = self.amp.max()
            
        if self.learn_slope:
            rval['arg_min']  = self.arg.min()
            rval['arg_mean'] = self.arg.mean()
            rval['arg_max']  = self.arg.max()
        
        return rval

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
                
        z = state_below
        
        amp = 1.7159
        slope = 0.66667
        if self.activation=='sigmoid':
            amp, slope = 1., 1.
        
        if self.learn_per_pool:

            assert self.pool_stride == self.pool_size
            
            z_pools = z.reshape((z.shape[0], self.pool_layer_dim, self.pool_size))
            
            if self.learn_amp:
                amp = self.amp.dimshuffle('x',0,'x')
            if self.learn_slope:
                slope = self.arg.dimshuffle('x',0,'x')
            
            z_pooled_act = amp*self.f(slope*z_pools)
            z = z_pooled_act.reshape((z.shape[0], self.pool_layer_dim*self.pool_size))

        else:
            if self.learn_amp:
                amp = self.amp
            if self.learn_slope:
                slope = self.arg
            z = amp*self.f(slope*z)
        
        p, gi, gj = None, None, None
        #xi = 0.01 + T.log(1+T.exp(self.beta))
        
        last_start = self.detector_layer_dim  - self.pool_size
        for i in xrange(self.pool_size):
            cur = z[:,i:last_start+i+1:self.pool_stride]
            gi = T.exp(-0.5*self.beta*T.sqr(cur-self.u))
            if gj is None:
                if self.l2_reg_pools is True:
                    p = cur*T.sqrt(gi)
                else:
                    p = cur*gi
                gj = gi
            else:
                if self.l2_reg_pools is True:
                    p = p + cur*T.sqrt(gi)
                else:
                    p = p + cur*gi
                gj = gj + gi

        if self.l2_reg_pools is True:
            p = p/(T.sqrt(gj)+1e-7)
        else:
            p = p/(gj+1e-7)

        p.name = self.layer_name + '_p_'

        return p

    def foo(self, state_below):
        pass


class GaussianPoolerLin(Layer):
    """
    """

    def __init__(self,
                 layer_name,
                 pool_size,
                 pool_stride = None,
                 init_u = 0.,
                 init_beta = 0.54,
                 u_lr_scale = None,
                 beta_lr_scale= None,
                 learn_amp = False,
                 relu = False,
        ):
        """

        """
        
        if pool_stride is None:
            pool_stride = pool_size

        self.__dict__.update(locals())
        del self.self

    def get_lr_scalers(self):

        if not hasattr(self, 'u_lr_scale'):
            self.u_lr_scale = None

        if not hasattr(self, 'beta_lr_scale'):
            self.beta_lr_scale = None

        rval = OrderedDict()

        if self.u_lr_scale is not None:
            rval[self.u] = self.u_lr_scale

        if self.beta_lr_scale is not None:
            rval[self.beta] = self.beta_lr_scale

        return rval

    def set_input_space(self, space):
        """ Note: this resets parameters! 
            TODO: work on any space type.
        """

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        self.detector_layer_dim = self.input_dim
        if not ((self.input_dim - self.pool_size) % self.pool_stride == 0):
            if self.pool_stride == self.pool_size:
                raise ValueError("detector_layer_dim = %d, pool_size = %d. Should be divisible but remainder is %d" %
                             (self.detector_layer_dim, self.pool_size, self.detector_layer_dim % self.pool_size))
            raise ValueError()

        self.pool_layer_dim = (self.detector_layer_dim - self.pool_size) / self.pool_stride + 1
        self.output_space = VectorSpace(self.pool_layer_dim)

        self.u = sharedX( np.zeros((self.pool_layer_dim,)) + self.init_u, name = self.layer_name + '_u')
        self.beta = sharedX( np.zeros((self.pool_layer_dim,)) + self.init_beta, name = self.layer_name + '_beta')
        #self.u = sharedX( self.init_beta*np.random.standard_normal((self.pool_layer_dim,)) + self.init_u, name = self.layer_name + '_u')
        #self.beta = sharedX( self.init_beta*np.random.standard_normal((self.pool_layer_dim,)) + 1.0, name = self.layer_name + '_beta')

    def censor_updates(self, updates):
        pass
        #betas = updates[self.beta]
        #updates[self.beta] = T.clip(betas, 0, 999)
            
    def get_params(self):
        assert self.u.name is not None
        assert self.beta.name is not None

        rval = []
        assert self.u not in rval
        rval.append(self.u)
        assert self.beta not in rval
        rval.append(self.beta)

        return rval

    def get_monitoring_channels(self):
        
        rval = OrderedDict([
                            ('u_min'  , self.u.min()),
                            ('u_mean' , self.u.mean()),
                            ('u_max'  , self.u.max()),
                            ('beta_min'  , self.beta.min()),
                            ('beta_mean' , self.beta.mean()),
                            ('beta_max'  , self.beta.max()),
                            ])
        
        return rval

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
            state_below = self.input_space.format_as(state_below, self.desired_space)
        
        z = state_below
        
        #if self.relu is True:
        #    z = T.maximum(0.0, z)
        
        p, gi, gj = None, None, None
        
        betar = T.log(1.0 + T.exp(self.beta))
        
        last_start = self.detector_layer_dim - self.pool_size
        for i in xrange(self.pool_size):
            cur = z[:,i:last_start+i+1:self.pool_stride]
            gi = T.exp(-0.5*self.beta*T.sqr(cur-self.u))
            if gj is None:
                #p = cur*T.sqrt(gi)
                p = cur*gi
                gj = gi
            else:
                #p = p + cur*T.sqrt(gi)
                p = p + cur*gi
                gj = gj + gi
        
        #p = p/(T.sqrt(gj)+1e-7)
        p = p/(gj+1e-7)
        
        p.name = self.layer_name + '_p_'

        return p

    def foo(self, state_below):
        pass


class MultiplicativeAdapter(Layer):
    """
    This code was used for speaker adaptation experiments
    for large vocabulary speech recognition in the following paper:

    Learning Hidden Unit Contributions for Unsupervised
    Adaptation of Neural Network Acoustic Models,
    Swietojanski and Renals, IEEE SLT, 2014

    This isn't really a pooling operator, but was used in a similar
    context (for adaptation) hence the code it's here.
    """
    def __init__(self, layer_name,
                       irange = None,
                       u_lr_scale = None,
                       constraint = 'sigmoid',
                       decoding = False):
        """
            layer_name: a layer name that will be
            irange: if specified, the initial multipliers will be randomly
                    selected from U(-irange, irange)
            u_lr_scale: learning rate scaling for u w.r.t the main learning
                    rate
            decoding: once the model is trained, u is fixed as such there is
                    no need to evaluate simgoid(u) for each example
        """

        assert constraint in ['sigmoid', 'exp', 'relu', 'none']

        self.__dict__.update(locals())
        del self.self

    def get_lr_scalers(self):

        rval = OrderedDict()

        if not hasattr(self, 'u_lr_scale'):
            self.u_lr_scale = None

        if self.u_lr_scale is not None:
            rval[self.u] = self.u_lr_scale

        return rval

    def get_monitoring_channels(self):

        return OrderedDict([
                            ('u_min'  , self.u.min()),
                            ('u_mean' , self.u.mean()),
                            ('u_max'  , self.u.max()),
                            ])

    def set_input_space(self, space):

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        self.output_space = VectorSpace(self.input_dim)

        if self.irange is not None:
            u = np.random.uniform(-self.irange, +self.iragne,
                                    (self.input_dim,), dtype=config.floatX)
        else:
            if self.constraint in ['sigmoid', 'exp']:
                offset = 0.0
            else:
                offset = 1.0
            u = np.zeros((self.input_dim,), dtype=config.floatX) + offset

        self.u = sharedX(u, name=self.layer_name+'_u')

    def get_weights_topo(self):
        raise NotImplementedError()

    def get_weights(self):
        raise NotImplementedError()

    def censor_updates(self, updates):
        return OrderedDict()

    def get_params(self):
        assert self.u.name is not None

        rval = []
        rval.append(self.u)

        return rval

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            state_below = self.input_space.format_as(state_below, self.desired_space)

        self.output_space.validate(state_below)

        z = state_below
        if self.constraint == "sigmoid":
            amp = 2*T.nnet.sigmoid(self.u)
        elif self.constraint == "exp":
            amp = T.exp(self.u)
        elif self.constraint == "relu":
            amp = T.maximum(0.0, self.relu)
        else:
            amp = self.u

        a = z*amp

        a.name = "_a_"

        return a


class RecurrentMultiplicativeAdapter(Layer):
    """
    This code was used for speaker adaptation experiments
    for large vocabulary speech recognition in the following paper:

    Learning Hidden Unit Contributions for Unsupervised
    Adaptation of Neural Network Acoustic Models,
    Swietojanski and Renals, IEEE SLT, 2014

    This isn't really a pooling operator, but was used in a similar
    context (for adaptation) hence the code it's here.
    """
    def __init__(self, layer_name,
                       irange = None,
                       u_lr_scale = None,
                       constraint = 'sigmoid',
                       decoding = False):
        """
            layer_name: a layer name that will be
            irange: if specified, the initial multipliers will be randomly
                    selected from U(-irange, irange)
            u_lr_scale: learning rate scaling for u w.r.t the main learning
                    rate
            decoding: once the model is trained, u is fixed as such there is
                    no need to evaluate simgoid(u) for each example
        """

        assert constraint in ['sigmoid', 'exp', 'relu', 'none']

        self.__dict__.update(locals())
        del self.self

    def get_lr_scalers(self):

        rval = OrderedDict()

        if not hasattr(self, 'u_lr_scale'):
            self.u_lr_scale = None

        if self.u_lr_scale is not None:
            rval[self.u] = self.u_lr_scale

        return rval

    def get_monitoring_channels(self):

        return OrderedDict([
                            ('u_min'  , self.u.min()),
                            ('u_mean' , self.u.mean()),
                            ('u_max'  , self.u.max()),
                            ])

    def set_input_space(self, space):

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        self.output_space = VectorSpace(self.input_dim)

        if self.irange is not None:
            u = np.random.uniform(-self.irange, +self.iragne,
                                    (self.input_dim,), dtype=config.floatX)
        else:
            if self.constraint in ['sigmoid', 'exp']:
                offset = 0.0
            else:
                offset = 1.0
            u = np.zeros((self.input_dim,), dtype=config.floatX) + offset

        self.u = sharedX(u, name=self.layer_name+'_u')

    def get_weights_topo(self):
        raise NotImplementedError()

    def get_weights(self):
        raise NotImplementedError()

    def censor_updates(self, updates):
        return OrderedDict()

    def get_params(self):
        assert self.u.name is not None

        rval = []
        rval.append(self.u)

        return rval

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            state_below = self.input_space.format_as(state_below, self.desired_space)

        self.output_space.validate(state_below)

        z = state_below
        if self.constraint == "sigmoid":
            amp = 2*T.nnet.sigmoid(self.u)
        elif self.constraint == "exp":
            amp = T.exp(self.u)
        elif self.constraint == "relu":
            amp = T.maximum(0.0, self.relu)
        else:
            amp = self.u

        a = z*amp

        a.name = "_a_"

        return a


class FactoredMultiplicativeAdapter(Layer):
    """
    This code was used for speaker adaptation experiments
    for large vocabulary speech recognition in the following paper:

    Learning Hidden Unit Contributions for Unsupervised
    Adaptation of Neural Network Acoustic Models,
    Swietojanski and Renals, IEEE SLT, 2014

    This isn't really a pooling operator, but was used in a similar
    context (for adaptation) hence the code it's here.
    """
    def __init__(self, layer_name,
                       irange = None,
                       u_lr_scale = None,
                       decoding = False):
        """
            layer_name: a layer name that will be
            irange: if specified, the initial multipliers will be randomly
                    selected from U(-irange, irange)
            u_lr_scale: learning rate scaling for u w.r.t the main learning
                    rate
            decoding: once the model is trained, u is fixed as such there is
                    no need to evaluate simgoid(u) for each example
        """

        self.__dict__.update(locals())
        del self.self

    def get_lr_scalers(self):

        rval = OrderedDict()

        if not hasattr(self, 'u_lr_scale'):
            self.u_lr_scale = None


        if self.u_lr_scale is not None:
            rval[self.u] = self.u_lr_scale

        return rval

    def get_monitoring_channels(self):

        return OrderedDict([
                            ('u_min'  , self.u.min()),
                            ('u_mean' , self.u.mean()),
                            ('u_max'  , self.u.max()),
                            ])

    def set_input_space(self, space):

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        self.output_space = VectorSpace(self.input_dim)

        if self.irange is not None:
            u = np.random.uniform(-self.irange, +self.iragne,
                                    (self.input_dim,), dtype=config.floatX)
        else:
            u = np.zeros((self.input_dim,), dtype=config.floatX)

        self.u = sharedX(u, name=self.layer_name+'_u')

    def get_weights_topo(self):
        raise NotImplementedError()

    def get_weights(self):
        raise NotImplementedError()

    def censor_updates(self, updates):
        return OrderedDict()

    def get_params(self):
        assert self.u.name is not None

        rval = []
        rval.append(self.u)

        return rval

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            state_below = self.input_space.format_as(state_below, self.desired_space)

        self.output_space.validate(state_below)

        z = state_below
        amp = 2*T.nnet.sigmoid(self.u)

        a = z*amp
        a.name = "_a_"

        return a