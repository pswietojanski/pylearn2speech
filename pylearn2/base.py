"""
A mostly outdated module that isn't used much anymore.
See tutorials/*ipynb or scripts/train_example to get
a quick introduction to the library.
"""
# Standard library imports
import warnings

# Third-party imports
import theano
from theano import tensor
try:
    from theano.sparse import SparseType
except ImportError:
    warnings.warn("Could not import theano.sparse.SparseType")
from theano.compile.mode import get_default_mode

theano.config.warn.sum_div_dimshuffle_bug = False

if 0:
    print 'WARNING: using SLOW rng'
    RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
    import theano.sandbox.rng_mrg
    RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams


class Block(object):
    """
    Basic building block that represents a simple transformation.
    By chaining Blocks together we can represent complex
    feed-forward transformations.

    TODO: give this input and output spaces to make it different from
          a theano Op
          supporting CompositeSpace would allow more complicated structures
          than just chains
    """
    def __init__(self):
        self.fn = None
        self.cpu_only = False

    def __call__(self, inputs):
        """
        WRITEME: what does this function do?
        WRITEME: how should inputs be formatted? is it a single tensor, a list
            of tensors, a tuple of tensors?
        """
        raise NotImplementedError(str(type(self)) + 'does not implement Block.__call__')

    def function(self, name=None):
        """
        Returns a compiled theano function to compute a representation
        """
        inputs = tensor.matrix()
        if self.cpu_only:
            return theano.function([inputs], self(inputs), name=name,
                                   mode=get_default_mode().excluding('gpu'))
        else:
            return theano.function([inputs], self(inputs), name=name)

    def perform(self, X):
        if self.fn is None:
            self.fn = self.function("perform")
        return self.fn(X)

    def inverse(self):
        raise NotImplementedError()

    def set_input_space(self, space):
        raise NotImplementedError(
                "%s does not implement set_input_space yet" % str(type(self)))

    def get_input_space(self):
        raise NotImplementedError(
                "%s does not implement get_input_space yet" % str(type(self)))

    def get_output_space(self):
        raise NotImplementedError(
                "%s does not implement get_output_space yet" % str(type(self)))


class StackedBlocks(Block):
    """
    A stack of Blocks, where the output of a block is the input of the next.
    """
    def __init__(self, layers):
        """
        Build a stack of layers.

        Parameters
        ----------
        layers: list of Blocks
            The layers to be stacked, ordered
            from bottom (input) to top (output)
        """

        super(StackedBlocks, self).__init__()

        self._layers = layers
        # Do not duplicate the parameters if some are shared between layers
        self._params = set([p for l in self._layers for p in l._params])

    def layers(self):
        return list(self._layers)

    def __len__(self):
        return len(self._layers)

    def __call__(self, inputs):
        """
        Return the output representation of all layers, including the inputs.

        Parameters
        ----------
        inputs : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) to be encoded. Assumed to be 2-tensors, with the
            first dimension indexing training examples and the second indexing
            data dimensions.

        Returns
        -------
        reconstructed : tensor_like or list of tensor_like
            A list of theano symbolic (or list thereof), each containing
            the representation at one level. The first element is the input.
        """
        # Build the hidden representation at each layer
        repr = [inputs]

        for layer in self._layers:
            outputs = layer(repr[-1])
            repr.append(outputs)

        return repr

    def function(self, name=None, repr_index=-1, sparse_input=False):
        """
        Compile a function computing representations on given layers.

        Parameters
        ----------
        name: string
            name of the function
        repr_index: int
            Index of the hidden representation to return.
            0 means the input, -1 the last output.
        """

        if sparse_input:
            inputs = SparseType('csr', dtype=theano.config.floatX)()
        else:
            inputs = tensor.matrix()

        return theano.function(
                [inputs],
                outputs=self(inputs)[repr_index],
                name=name)

    def concat(self, name=None, start_index=-1, end_index=None):
        """
        Compile a function concatenating representations on given layers.

        Parameters
        ----------
        name: string
            name of the function
        start_index: int
            Index of the hidden representation to start the concatenation.
            0 means the input, -1 the last output.
        end_index: int
            Index of the hidden representation from which to stop
            the concatenation. We must have start_index < end_index.
        """
        inputs = tensor.matrix()
        return theano.function([inputs],
            outputs=tensor.concatenate(self(inputs)[start_index:end_index]),
            name=name)

    def append(self, layer):
        """
        Add a new layer on top of the last one
        """
        self._layers.append(layer)
        self._params.update(layer._params)

    def get_input_space(self):
        return self._layers[0].get_input_space()

    def get_output_space(self):
        return self._layers[-1].get_output_space()

    def set_input_space(self, space):
        for layer in self._layers:
            layer.set_input_space(space)
            space = layer.get_output_space()
