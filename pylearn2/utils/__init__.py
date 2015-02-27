import os

from .general import is_iterable
import theano
# Delay import of pylearn2.config.yaml_parse and pylearn2.datasets.control
# to avoid circular imports
yaml_parse = None
control = None
from itertools import izip
cuda = None

import numpy
np = numpy

def make_name(variable, anon = "anonymous_variable"):
    """
    If variable has a name, returns that name.
    Otherwise, returns anon
    """

    if hasattr(variable,'name') and variable.name is not None:
        return variable.name

    return anon


def sharedX(value, name=None, borrow=False):
    """Transform value into a shared variable of type floatX"""
    return theano.shared(theano._asarray(value, dtype=theano.config.floatX),
         name=name,
         borrow=borrow)

def as_floatX(variable):
    """Casts a given variable into dtype config.floatX
    numpy ndarrays will remain numpy ndarrays
    python floats will become 0-D ndarrays
    all other types will be treated as theano tensors"""

    if isinstance(variable, float):
        return numpy.cast[theano.config.floatX](variable)

    if isinstance(variable, numpy.ndarray):
        return numpy.cast[theano.config.floatX](variable)

    return theano.tensor.cast(variable, theano.config.floatX)

def constantX(value):
    """
        Returns a constant of value `value` with floatX dtype
    """
    return theano.tensor.constant(numpy.asarray(value,
                                     dtype=theano.config.floatX))
def subdict(d, keys):
    """ Create a subdictionary of d with the keys in keys """
    result = {}
    for key in keys:
        if key in d: result[key] = d[key]
    return result

def safe_update(dict_to, dict_from):
    """
    Like dict_to.update(dict_from), except don't overwrite any keys.
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to


class CallbackOp(theano.gof.Op):
    """A Theano Op that implements the identity transform but
    also does an arbitrary (user-specified) side effect. """


    view_map = { 0: [0] }

    def __init__(self, callback):
        self.callback = callback

    def make_node(self, xin):
        xout = xin.type.make_variable()
        return theano.gof.Apply(op=self, inputs=[xin], outputs=[xout])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin
        self.callback(xin)

    def grad(self, inputs, output_gradients):
        return output_gradients

    def R_op(self, inputs, eval_points):
        return [x for x in eval_points]

    def __eq__(self, other):
        return type(self) == type(other) and self.callback == other.callback

    def hash(self):
        return hash(self.callback)


def get_dataless_dataset(model):
    """
    Loads the dataset that model was trained on, without loading data.
    This is useful if you just need the dataset's metadata, like for
    formatting views of the model's weights.
    """

    global yaml_parse
    global control

    if yaml_parse is None:
        from pylearn2.config import yaml_parse

    if control is None:
        from pylearn2.datasets import control

    control.push_load_data(False)
    try:
        rval = yaml_parse.load(model.dataset_yaml_src)
    finally:
        control.pop_load_data()
    return rval

def safe_zip(*args):
    """Like zip, but ensures arguments are of same length"""
    base = len(args[0])
    for i, arg in enumerate(args[1:]):
        if len(arg) != base:
            raise ValueError("Argument 0 has length "+str(base)+\
                " but argument "+str(i+1)+" has length "+str(len(arg)))
    return zip(*args)

def safe_izip(*args):
    """Like izip, but ensures arguments are of same length"""
    assert all([len(arg) == len(args[0]) for arg in args])
    return izip(*args)

def gpu_mem_free():
    global cuda
    if cuda is None:
        from theano.sandbox import cuda
    return cuda.mem_info()[0]/1024./1024

class _ElemwiseNoGradient(theano.tensor.Elemwise):

    def connection_pattern(self, node):

        return [ [ False ] ]

    def grad(self, inputs, output_gradients):

        return [ theano.gradient.DisconnectedType()() ]

# Call this on a theano variable to make a copy of that variable
# No gradient passes through the copying operation
# This is equivalent to making my_copy = var.copy() and passing
# my_copy in as part of consider_constant to tensor.grad
# However, this version doesn't require as much long range
# communication between parts of the code
block_gradient = _ElemwiseNoGradient(theano.scalar.identity)


def safe_union(a, b):
    """
    Does the logic of a union operation without the non-deterministic
    ordering of python sets
    """
    if not isinstance(a, list):
        raise TypeError("Expected first argument to be a list, but got "+str(type(a)))
    assert isinstance(b, list)
    c = []
    for x in a + b:
        if x not in c:
            c.append(x)
    return c

# This was moved to theano, but I include a link to avoid breaking
# old imports
from theano.printing import hex_digest

def function(*args, **kwargs):
    """
    A wrapper around theano.function that:

        -Disables the on_unused_input error. Almost no part of
         pylearn2 can assume that an unused input is an error, so
         the default from theano is inappropriate for this project.
    """

    return theano.function(*args, on_unused_input='ignore', **kwargs)

def grad(*args, **kwargs):
    """
    A wrapper around theano.gradient.grad that disable the disconnected_inputs
    error. Almost no part of pylearn2 can assume that a disconnected input
    is an error.
    """
    return theano.gradient.grad(*args, disconnected_inputs='ignore', **kwargs)

# Groups of Python types that are often used together in `isinstance`
py_integer_types = (int, long, np.integer)
py_float_types = (float, np.floating)
py_complex_types = (complex, np.complex)
py_number_types = (int, long, float, complex, np.number)


def get_choice(choice_to_explanation ):
    """
    choice_to_explanation: a dictionary mapping possible user responses
    to strings describing what that response will
    cause the script to do
    """
    d = choice_to_explanation

    for key in d:
        print '\t'+key + ': '+d[key]
    prompt = '/'.join(d.keys())+'? '

    first = True
    choice = ''
    while first or choice not in d.keys():
        if not first:
            print 'unrecognized choice'
        first = False
        choice = raw_input(prompt)
    return choice


def float32_floatX(f):
    """
    This function change floatX to float32 for the call to f.

    This is usefull in GPU tests.
    """
    def new_f(*args, **kwargs):
        old_floatX = theano.config.floatX
        theano.config.floatX = 'float32'
        try:
            f(*args, **kwargs)
        finally:
            theano.config.floatX = old_floatX

    # If we don't do that, tests function won't be run.
    new_f.func_name = f.func_name
    return new_f
