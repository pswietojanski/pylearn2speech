"""
Functionality for preprocessing Datasets.
"""

__authors__ = "Ian Goodfellow, David Warde-Farley, Guillaume Desjardins, and Mehdi Mirza"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow", "David Warde-Farley", "Guillaume Desjardins",
               "Mehdi Mirza"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"


import copy
import logging
import warnings
import numpy as np
try:
    from scipy import linalg
except ImportError:
    warnings.warn("Could not import scipy.linalg")
from theano import function
import theano.tensor as T

from pylearn2.base import Block
from pylearn2.linear.conv2d import Conv2D
from pylearn2.space import Conv2DSpace, VectorSpace
from pylearn2.expr.preprocessing import global_contrast_normalize
from pylearn2.utils.insert_along_axis import insert_columns
from pylearn2.utils import sharedX


log = logging.getLogger(__name__)

convert_axes = Conv2DSpace.convert_numpy

class Preprocessor(object):
    """
        Abstract class.

        An object that can preprocess a dataset.

        Preprocessing a dataset implies changing the data that
        a dataset actually stores. This can be useful to save
        memory--if you know you are always going to access only
        the same processed version of the dataset, it is better
        to process it once and discard the original.

        Preprocessors are capable of modifying many aspects of
        a dataset. For example, they can change the way that it
        converts between different formats of data. They can
        change the number of examples that a dataset stores.
        In other words, preprocessors can do a lot more than
        just example-wise transformations of the examples stored
        in the dataset.
    """

    def apply(self, dataset, can_fit=False):
        """
            dataset: The dataset to act on.
            can_fit: If True, the Preprocessor can adapt internal parameters
                     based on the contents of dataset. Otherwise it must not
                     fit any parameters, or must re-use old ones.
                     Subclasses should still have this default to False, so
                     that the behavior of the preprocessors is uniform.

            Typical usage:
                # Learn PCA preprocessing and apply it to the training set
                my_pca_preprocessor.apply(training_set, can_fit = True)
                # Now apply the same transformation to the test set
                my_pca_preprocessor.apply(test_set, can_fit = False)

            Note: this method must take a dataset, rather than a numpy ndarray,
                  for a variety of reasons:
                      1) Preprocessors should work on any dataset, and not all
                         datasets will store their data as ndarrays.
                      2) Preprocessors often need to change a dataset's metadata.
                         For example, suppose you have a DenseDesignMatrix dataset
                         of images. If you implement a fovea Preprocessor that
                         reduces the dimensionality of images by sampling them finely
                         near the center and coarsely with blurring at the edges,
                         then your preprocessor will need to change the way that the
                         dataset converts example vectors to images for visualization.
        """

        raise NotImplementedError(str(type(self))+" does not implement an apply method.")

    def invert(self):
        """
        Do any necessary prep work to be able to support the "inverse" method
        later. Default implementation is no-op.
        """

class ExamplewisePreprocessor(Preprocessor):
    """
        Abstract class.

        A Preprocessor that restricts the actions it can do in its
        apply method so that it could be implemented as a Block's
        perform method.

        In other words, this Preprocessor can't modify the Dataset's
        metadata, etc.

        TODO: can these things fit themselves in their apply method?
        That seems like a difference from Block.
    """

    def as_block(self):
        raise NotImplementedError(str(type(self))+" does not implement as_block.")

class BlockPreprocessor(ExamplewisePreprocessor):
    """
        An ExamplewisePreprocessor implemented by a Block.
    """

    def __init__(self, block):
        self.block = block

    def apply(self, dataset, can_fit = False):
        assert not can_fit
        dataset.X = self.block.perform(dataset.X)



class Pipeline(Preprocessor):
    """
        A Preprocessor that sequentially applies a list
        of other Preprocessors.
    """
    def __init__(self, items=None):
        self.items = items if items is not None else []

    def apply(self, dataset, can_fit=False):
        for item in self.items:
            item.apply(dataset, can_fit)


class ExtractGridPatches(Preprocessor):
    """
    Converts a dataset of images into a dataset of patches extracted along a
    regular grid from each image.  The order of the images is
    preserved.
    """
    def __init__(self, patch_shape, patch_stride):
        self.patch_shape = patch_shape
        self.patch_stride = patch_stride

    def apply(self, dataset, can_fit=False):
        X = dataset.get_topological_view()
        num_topological_dimensions = len(X.shape) - 2
        if num_topological_dimensions != len(self.patch_shape):
            raise ValueError("ExtractGridPatches with "
                             + str(len(self.patch_shape))
                             + " topological dimensions called on"
                             + " dataset with " +
                             str(num_topological_dimensions) + ".")
        num_patches = X.shape[0]
        max_strides = [X.shape[0] - 1]
        for i in xrange(num_topological_dimensions):
            patch_width = self.patch_shape[i]
            data_width = X.shape[i + 1]
            last_valid_coord = data_width - patch_width
            if last_valid_coord < 0:
                raise ValueError('On topological dimension ' + str(i) +
                                 ', the data has width ' + str(data_width) +
                                 ' but the requested patch width is ' +
                                 str(patch_width))
            stride = self.patch_stride[i]
            if stride == 0:
                max_stride_this_axis = 0
            else:
                max_stride_this_axis = last_valid_coord / stride
            num_strides_this_axis = max_stride_this_axis + 1
            max_strides.append(max_stride_this_axis)
            num_patches *= num_strides_this_axis
        # batch size
        output_shape = [num_patches]
        # topological dimensions
        for dim in self.patch_shape:
            output_shape.append(dim)
        # number of channels
        output_shape.append(X.shape[-1])
        output = np.zeros(output_shape, dtype=X.dtype)
        channel_slice = slice(0, X.shape[-1])
        coords = [0] * (num_topological_dimensions + 1)
        keep_going = True
        i = 0
        while keep_going:
            args = [coords[0]]
            for j in xrange(num_topological_dimensions):
                coord = coords[j + 1] * self.patch_stride[j]
                args.append(slice(coord, coord + self.patch_shape[j]))
            args.append(channel_slice)
            patch = X[args]
            output[i, :] = patch
            i += 1
            # increment coordinates
            j = 0
            keep_going = False
            while not keep_going:
                if coords[-(j + 1)] < max_strides[-(j + 1)]:
                    coords[-(j + 1)] += 1
                    keep_going = True
                else:
                    coords[-(j + 1)] = 0
                    if j == num_topological_dimensions:
                        break
                    j = j + 1
        dataset.set_topological_view(output)


class ReassembleGridPatches(Preprocessor):
    """ Converts a dataset of patches into a dataset of full examples
        This is the inverse of ExtractGridPatches for patch_stride=patch_shape
    """
    def __init__(self, orig_shape, patch_shape):
        self.patch_shape = patch_shape
        self.orig_shape = orig_shape

    def apply(self, dataset, can_fit=False):

        patches = dataset.get_topological_view()

        num_topological_dimensions = len(patches.shape) - 2

        if num_topological_dimensions != len(self.patch_shape):
            raise ValueError("ReassembleGridPatches with " +
                             str(len(self.patch_shape)) +
                             " topological dimensions called on dataset " +
                             " with " +
                             str(num_topological_dimensions) + ".")
        num_patches = patches.shape[0]
        num_examples = num_patches
        for im_dim, patch_dim in zip(self.orig_shape, self.patch_shape):
            if im_dim % patch_dim != 0:
                raise Exception('Trying to assemble patches of shape ' +
                                str(self.patch_shape) + ' into images of ' +
                                'shape ' + str(self.orig_shape))
            patches_this_dim = im_dim / patch_dim
            if num_examples % patches_this_dim != 0:
                raise Exception('Trying to re-assemble ' + str(num_patches) +
                                ' patches of shape ' + str(self.patch_shape) +
                                ' into images of shape ' + str(self.orig_shape)
                               )
            num_examples /= patches_this_dim

        # batch size
        reassembled_shape = [num_examples]
        # topological dimensions
        for dim in self.orig_shape:
            reassembled_shape.append(dim)
        # number of channels
        reassembled_shape.append(patches.shape[-1])
        reassembled = np.zeros(reassembled_shape, dtype=patches.dtype)
        channel_slice = slice(0, patches.shape[-1])
        coords = [0] * (num_topological_dimensions + 1)
        max_strides = [num_examples - 1]
        for dim, pd in zip(self.orig_shape, self.patch_shape):
            assert dim % pd == 0
            max_strides.append(dim / pd - 1)
        keep_going = True
        i = 0
        while keep_going:
            args = [coords[0]]
            for j in xrange(num_topological_dimensions):
                coord = coords[j + 1]
                args.append(slice(coord * self.patch_shape[j],
                                  (coord + 1) * self.patch_shape[j]))
                next_shape_coord = reassembled.shape[j + 1]
                assert (coord + 1) * self.patch_shape[j] <= next_shape_coord

            args.append(channel_slice)

            try:
                patch = patches[i, :]
            except IndexError:
                raise IndexError('Gave index of ' + str(i) +
                                 ', : into thing of shape ' +
                                 str(patches.shape))
            reassembled[args] = patch
            i += 1
            j = 0
            keep_going = False
            while not keep_going:
                if coords[-(j + 1)] < max_strides[-(j + 1)]:
                    coords[-(j + 1)] += 1
                    keep_going = True
                else:
                    coords[-(j + 1)] = 0
                    if j == num_topological_dimensions:
                        break
                    j = j + 1

        dataset.set_topological_view(reassembled)


class ExtractPatches(Preprocessor):
    """ Converts an image dataset into a dataset of patches
        extracted at random from the original dataset. """
    def __init__(self, patch_shape, num_patches, rng=None):
        self.patch_shape = patch_shape
        self.num_patches = num_patches

        if rng != None:
            self.start_rng = copy.copy(rng)
        else:
            self.start_rng = np.random.RandomState([1, 2, 3])

    def apply(self, dataset, can_fit=False):
        rng = copy.copy(self.start_rng)

        X = dataset.get_topological_view()

        num_topological_dimensions = len(X.shape) - 2

        if num_topological_dimensions != len(self.patch_shape):
            raise ValueError("ExtractPatches with "
                             + str(len(self.patch_shape))
                             + " topological dimensions called on "
                             + "dataset with "
                             + str(num_topological_dimensions) + ".")

        # batch size
        output_shape = [self.num_patches]
        # topological dimensions
        for dim in self.patch_shape:
            output_shape.append(dim)
        # number of channels
        output_shape.append(X.shape[-1])
        output = np.zeros(output_shape, dtype=X.dtype)
        channel_slice = slice(0, X.shape[-1])
        for i in xrange(self.num_patches):
            args = []
            args.append(rng.randint(X.shape[0]))

            for j in xrange(num_topological_dimensions):
                max_coord = X.shape[j + 1] - self.patch_shape[j]
                coord = rng.randint(max_coord + 1)
                args.append(slice(coord, coord + self.patch_shape[j]))
            args.append(channel_slice)
            output[i, :] = X[args]
        dataset.set_topological_view(output)
        dataset.y = None


class ExamplewiseUnitNormBlock(Block):
    """
    A block that takes n-tensors, with training examples indexed along
    the first axis, and normalizes each example to lie on the unit
    sphere.
    """

    def __init__(self, input_space=None):
        super(ExamplewiseUnitNormBlock, self).__init__()
        self.input_space = input_space

    def __call__(self, batch):
        if self.input_space:
            self.input_space.validate(batch)
        squared_batch = batch ** 2
        squared_norm = squared_batch.sum(axis=1)
        norm = T.sqrt(squared_norm)
        return batch / norm

    def set_input_space(self, space):
        self.input_space = space

    def get_input_space(self):
        if self.input_space is not None:
            return self.input_space
        raise ValueError("No input space was specified for this Block (%s). "
                "You can call set_input_space to correct that." % str(self))

    def get_output_space(self):
        return self.get_input_space()


class MakeUnitNorm(ExamplewisePreprocessor):
    def apply(self, dataset, can_fit=False):
        X = dataset.get_design_matrix()
        X_norm = np.sqrt(np.sum(X ** 2, axis=1))
        X /= X_norm[:, None]
        dataset.set_design_matrix(X)

    def as_block(self):
        return ExamplewiseUnitNormBlock()


class ExamplewiseAddScaleTransform(Block):
    """
    A block that encodes an per-feature addition/scaling transform.
    The addition/scaling can be done in either order.
    """
    def __init__(self, add=None, multiply=None, multiply_first=False,
                 input_space=None):
        """
        Initialize an ExamplewiseAddScaleTransform instance.

        Parameters
        ----------
        add : array_like or scalar, optional
            Array or array-like object or scalar, to be added to each
            training example by this Block.

        multiply : array_like, optional
            Array or array-like object or scalar, to be element-wise
            multiplied with each training example by this Block.

        multiply_first : boolean, optional
            Whether to perform the multiplication before the addition.
            (default is False).

        input_space: Space, optional
            The input space describing the data
        """
        self._add = np.asarray(add)
        self._multiply = np.asarray(multiply)
        # TODO: put the constant somewhere sensible.
        if multiply is not None:
            self._has_zeros = np.any(abs(multiply) < 1e-14)
        else:
            self._has_zeros = False
        self._multiply_first = multiply_first
        self.input_space = input_space

    def _multiply(self, batch):
        if self._multiply is not None:
            batch *= self._multiply
        return batch

    def _add(self, batch):
        if self._add is not None:
            batch += self._add
        return batch

    def __call__(self, batch):
        if self.input_space:
            self.input_space.validate(batch)
        cur = batch
        if self._multiply_first:
            batch = self._add(self._multiply(batch))
        else:
            batch = self._multiply(self._add(batch))
        return batch

    def inverse(self):
        if self._multiply is not None and self._has_zeros:
            raise ZeroDivisionError("%s transformation not invertible "
                                    "due to (near-) zeros in multiplicand" %
                                    self.__class__.__name__)
        else:
            mult_inverse = self._multiply ** -1.
            return self.__class__(add=-self._add, multiply=mult_inverse,
                                  multiply_first=not self._multiply_first)

    def set_input_space(self, space):
        self.input_space = space

    def get_input_space(self):
        if self.input_space is not None:
            return self.input_space
        raise ValueError("No input space was specified for this Block (%s). "
                "You can call set_input_space to correct that." % str(self))

    def get_output_space(self):
        return self.get_input_space()


class RemoveMean(ExamplewisePreprocessor):
    """
    Subtracts the mean along a given axis, or from every element
    if `axis=None`.
    """
    def __init__(self, axis=0):
        """
        Initialize a RemoveMean preprocessor.

        Parameters
        ----------
        axis : int or None
            Axis over which to take the mean, with the exact same
            semantics as the `axis` parameter of `numpy.mean`.
        """
        self._axis = axis
        self._mean = None

    def apply(self, dataset, can_fit=True):
        X = dataset.get_design_matrix()
        if can_fit:
            self._mean = X.mean(axis=self._axis)
        else:
            if self._mean is None:
                raise ValueError("can_fit is False, but RemoveMean object "
                                 "has no stored mean or standard deviation")
        X -= self._mean
        dataset.set_design_matrix(X)

    def as_block(self):
        if self._mean is None:
            raise  ValueError("can't convert %s to block without fitting"
                              % self.__class__.__name__)
        return ExamplewiseAddScaleTransform(add=-self._mean)


class Standardize(ExamplewisePreprocessor):
    """Subtracts the mean and divides by the standard deviation."""
    def __init__(self, global_mean=False, global_std=False, std_eps=1e-4):
        """
        Initialize a Standardize preprocessor.

        Parameters
        ----------
        global_mean : bool
            If `True`, subtract the (scalar) mean over every element
            in the design matrix. If `False`, subtract the mean from
            each column (feature) separately. Default is `False`.
        global_std : bool
            If `True`, after centering, divide by the (scalar) standard
            deviation of every element in the design matrix. If `False`,
            divide by the column-wise (per-feature) standard deviation.
            Default is `False`.
        std_eps : float
            Stabilization factor added to the standard deviations before
            dividing, to prevent standard deviations very close to zero
            from causing the feature values to blow up too much.
            Default is `1e-4`.
        """
        self._global_mean = global_mean
        self._global_std = global_std
        self._std_eps = std_eps
        self._mean = None
        self._std = None

    def apply(self, dataset, can_fit=False):
        X = dataset.get_design_matrix()
        if can_fit:
            self._mean = X.mean() if self._global_mean else X.mean(axis=0)
            self._std = X.std() if self._global_std else X.std(axis=0)
        else:
            if self._mean is None or self._std is None:
                raise ValueError("can_fit is False, but Standardize object "
                                 "has no stored mean or standard deviation")
        new = (X - self._mean) / (self._std_eps + self._std)
        dataset.set_design_matrix(new)

    def as_block(self):
        if self._mean is None or self._std is None:
            raise  ValueError("can't convert %s to block without fitting"
                              % self.__class__.__name__)
        return ExamplewiseAddScaleTransform(add=-self._mean,
                                            multiply=self._std ** -1)


class ColumnSubsetBlock(Block):
    def __init__(self, columns, total):
        self._columns = columns
        self._total = total

    def __call__(self, batch):
        if batch.ndim != 2:
            raise ValueError("Only two-dimensional tensors are supported")
        return batch.dimshuffle(1, 0)[self._columns].dimshuffle(1, 0)

    def inverse(self):
        return ZeroColumnInsertBlock(self._columns, self._total)

    def get_input_space(self):
        return VectorSpace(dim=self._total)

    def get_output_space(self):
        return VectorSpace(dim=self._columns)


class ZeroColumnInsertBlock(Block):
    def __init__(self, columns, total):
        self._columns = columns
        self._total = total

    def __call__(self, batch):
        if batch.ndim != 2:
            raise ValueError("Only two-dimensional tensors are supported")
        return insert_columns(batch, self._total, self._columns)

    def inverse(self):
        return ColumnSubsetBlock(self._columns, self._total)

    def get_input_space(self):
        return VectorSpace(dim=self._columns)

    def get_output_space(self):
        return VectorSpace(dim=self._total)


class RemoveZeroColumns(ExamplewisePreprocessor):
    _eps = 1e-8

    def __init__(self):
        self._block = None

    def apply(self, dataset, can_fit=False):
        design_matrix = dataset.get_design_matrix()
        mean = design_matrix.mean(axis=0)
        var = design_matrix.var(axis=0)
        columns, = np.where((var < self._eps) & (mean < self._eps))
        self._block = ColumnSubsetBlock

    def as_block(self):
        if self._block is None:
            raise  ValueError("can't convert %s to block without fitting"
                              % self.__class__.__name__)
        return self._block


class RemapInterval(ExamplewisePreprocessor):
    # TODO: Implement as_block
    def __init__(self, map_from, map_to):
        assert map_from[0] < map_from[1] and len(map_from) == 2
        assert map_to[0] < map_to[1] and len(map_to) == 2
        self.map_from = [np.float(x) for x in map_from]
        self.map_to = [np.float(x) for x in map_to]

    def apply(self, dataset, can_fit=False):
        X = dataset.get_design_matrix()
        X = (X - self.map_from[0]) / np.diff(self.map_from)
        X = X * np.diff(self.map_to) + self.map_to[0]
        dataset.set_design_matrix(X)


class PCA_ViewConverter(object):
    def __init__(self, to_pca, to_input, to_weights, orig_view_converter):
        self.to_pca = to_pca
        self.to_input = to_input
        self.to_weights = to_weights
        if orig_view_converter is None:
            raise ValueError("It doesn't make any sense to make a PCA view "
                             "converter when there's no original view "
                             "converter to define a topology in the first "
                             "place.")
        self.orig_view_converter = orig_view_converter

    def view_shape(self):
        return self.orig_view_converter.shape

    def design_mat_to_topo_view(self, X):
        to_input = self.to_input(X)
        return self.orig_view_converter.design_mat_to_topo_view(to_input)

    def design_mat_to_weights_view(self, X):
        to_weights = self.to_weights(X)
        return self.orig_view_converter.design_mat_to_weights_view(to_weights)

    def topo_view_to_design_mat(self, V):
        return self.to_pca(self.orig_view_converter.topo_view_to_design_mat(V))


class PCA(object):
    def __init__(self, num_components):
        self._num_components = num_components
        self._pca = None
        # TODO: Is storing these really necessary? This computation
        # can't really be merged since we're basically creating the
        # functions in apply(); I see no reason to keep these around.
        self._input = T.matrix()
        self._output = T.matrix()

    def apply(self, dataset, can_fit=False):
        if self._pca is None:
            if not can_fit:
                raise ValueError("can_fit is False, but PCA preprocessor "
                                 "object has no fitted model stored")
            from pylearn2 import pca
            self._pca = pca.CovEigPCA(self._num_components)
            self._pca.train(dataset.get_design_matrix())
            self._transform_func = function([self._input],
                                            self._pca(self._input))
            self._invert_func = function([self._output],
                                         self._pca.reconstruct(self._output))
            self._convert_weights_func = function(
                [self._output],
                self._pca.reconstruct(self._output, add_mean=False)
            )

        orig_data = dataset.get_design_matrix()
        dataset.set_design_matrix(
            self._transform_func(dataset.get_design_matrix())
        )
        proc_data = dataset.get_design_matrix()
        orig_var = orig_data.var(axis=0)
        proc_var = proc_data.var(axis=0)
        assert proc_var[0] > orig_var.max()
        # TODO: logging
        print 'original variance: ' + str(orig_var.sum())
        print 'processed variance: ' + str(proc_var.sum())
        if dataset.view_converter is not None:
            new_converter = PCA_ViewConverter(self._transform_func,
                                              self._invert_func,
                                              self._convert_weights_func,
                                              dataset.view_converter)
            dataset.view_converter = new_converter


class Downsample(object):
    def __init__(self, sampling_factor):
        """
            downsamples the topological view

            parameters
            ----------
            sampling_factor: a list or array with one element for
                            each topological dimension of the data
        """

        self.sampling_factor = sampling_factor

    def apply(self, dataset, can_fit=False):
        X = dataset.get_topological_view()
        d = len(X.shape) - 2
        assert d in [2, 3]
        assert X.dtype == 'float32' or X.dtype == 'float64'
        if d == 2:
            X = X.reshape([X.shape[0], X.shape[1], X.shape[2], 1, X.shape[3]])
        kernel_size = 1
        kernel_shape = [X.shape[-1]]
        for factor in self.sampling_factor:
            kernel_size *= factor
            kernel_shape.append(factor)
        if d == 2:
            kernel_shape.append(1)
        kernel_shape.append(X.shape[-1])
        kernel_value = 1. / float(kernel_size)
        kernel = np.zeros(kernel_shape, dtype=X.dtype)
        for i in xrange(X.shape[-1]):
            kernel[i, :, :, :, i] = kernel_value
        from theano.tensor.nnet.Conv3D import conv3D
        X_var = T.TensorType(broadcastable=[s == 1 for s in X.shape],
                             dtype=X.dtype)()
        downsampled = conv3D(X_var, kernel, np.zeros(X.shape[-1], X.dtype),
                             kernel_shape[1:-1])
        f = function([X_var], downsampled)
        X = f(X)
        if d == 2:
            X = X.reshape([X.shape[0], X.shape[1], X.shape[2], X.shape[4]])
        dataset.set_topological_view(X)


class GlobalContrastNormalization(Preprocessor):
    def __init__(self, subtract_mean=True,
                 scale=1., sqrt_bias=None, use_std=None, min_divisor=1e-8,
                 std_bias=None, use_norm=None,
                 batch_size=None):
        """
        See the docstring for `global_contrast_normalize` in
        `pylearn2.expr.preprocessing`.

        Parameters
        ----------
        batch_size : int or None, optional
            If specified, read, apply and write the transformed
            data in batches no larger than `batch_size`.

        std_bias is a deprecated alias for sqrt_bias.
        use_norm is a deprecated argument that controls the same thing as use_std,
            except that use_norm=True means use_std=False.

        use_std defaults to True and sqrt_bias defaults to 10 if nothing is specified.
        Both of these defaults will change for consistency with pylearn2.expr.preprocessing
        sometime after October 12, 2013.
        The defaults aren't specified as part of the method signature so that we can tell
        whether the client is using each name for each option.
        """

        if std_bias is not None:
            warnings.warn("std_bias is deprecated, and may be removed after October 12, 2013. Switch to sqrt_bias.", stacklevel=2)
            if sqrt_bias is not None:
                if std_bias == sqrt_bias:
                    warnings.warn("You're specifying both std_bias and sqrt_bias, which are actually aliases for the same parameter. You're setting them both to the same thing so it's OK, but you probably want to change your script to just specify sqrt_bias.",stacklevel=2)
                else:
                    raise ValueError("You specified sqrt_bias and std_bias to different values, but they are aliases of each other. Specify only sqrt_bias. std_bias is a deprecated alias.", stacklevel=2)
            sqrt_bias = std_bias

        if sqrt_bias is None:
            warnings.warn("You are not specifying a value for sqrt_bias. Note that the default value will change on or after October 12, 2013, to be consistent with pylearn2.expr.preprocessing.")
            sqrt_bias = 10.

        if use_norm is not None:
            warnings.warn("use_norm is deprecated, and may be removed after October 12, 2013. Pass the opposite value to use_std.", stacklevel=2)
            if use_std is not None:
                if use_std == (not use_norm):
                    warnings.warn("You're specifying both use_std and use_norm. You have them set to the opposite of each other, i.e. both are requesting the same behavior, so you're OK, but you probably want to change your script to only specify one.", stacklevel=2)
                else:
                    raise ValueError("use_std conflicts with use_norm.")
            use_std = not use_norm

        if use_std is None:
            warnings.warn("You are not specifying a value for use_std. The default of use_std will change on or after October 12, 2013 to be consistent with pylearn2.expr.preprocessing.")

            use_std = True

        self._subtract_mean = subtract_mean
        self._use_std = use_std
        self._sqrt_bias = sqrt_bias
        # These were not parameters of the old preprocessor.
        self._scale = scale
        self._min_divisor = min_divisor
        if batch_size is not None:
            batch_size = int(batch_size)
            assert batch_size > 0, "batch_size must be positive"
        self._batch_size = batch_size

    def apply(self, dataset, can_fit=False):
        if self._batch_size is None:
            X = global_contrast_normalize(dataset.get_design_matrix(),
                                          scale=self._scale,
                                          subtract_mean=self._subtract_mean,
                                          use_std=self._use_std,
                                          sqrt_bias=self._sqrt_bias,
                                          min_divisor=self._min_divisor)
            dataset.set_design_matrix(X)
        else:
            X = dataset.get_design_matrix()
            data_size = X.shape[0]
            last = (np.floor(data_size / float(self._batch_size)) *
                    self._batch_size)
            for i in xrange(0, data_size, self._batch_size):
                if i >= last:
                    stop = i + np.mod(data_size, self._batch_size)
                else:
                    stop = i + self._batch_size
                log.info("GCN processing data from %d to %d" % (i, stop))
                data = self.transform(X[i:stop])
                dataset.set_design_matrix(data, start = i)


class GlobalContrastNormalizationPyTables(object):
    def __init__(self, subtract_mean=True, std_bias=10.0, use_norm=False, batch_size = 5000):
        """

        Optionally subtracts the mean of each example
        Then divides each example either by the standard deviation of the
        pixels contained in that example or by the norm of that example

        Parameters:

            subtract_mean: boolean, if True subtract the mean of each example
            std_bias: Add this amount inside the square root when computing
                      the standard deviation or the norm
            use_norm: If True uses the norm instead of the standard deviation


            The default parameters of subtract_mean = True, std_bias = 10.0,
            use_norm = False are used in replicating one step of the
            preprocessing used by Coates, Lee and Ng on CIFAR10 in their paper
            "An Analysis of Single Layer Networks in Unsupervised Feature
            Learning"
        """

        self.subtract_mean = subtract_mean
        self.std_bias = std_bias
        self.use_norm = use_norm
        self._batch_size = batch_size
        warnings.warn("GlobalContrastNormalizationPyTables has been rolled "
                      "into GlobalContrastNormalization. This class will "
                      "disappear after October 12, 2013.")

    def transform(self, X):
        assert X.dtype == 'float32' or X.dtype == 'float64'

        if self.subtract_mean:
            X -= X[:].mean(axis=1)[:, None]

        if self.use_norm:
            scale = np.sqrt(np.square(X).sum(axis=1) + self.std_bias)
        else:
            # use standard deviation
            scale = np.sqrt(np.square(X).mean(axis=1) + self.std_bias)
        eps = 1e-8
        scale[scale < eps] = 1.
        X /= scale[:, None]
        return X

    def apply(self, dataset, can_fit=False):
        X = dataset.get_design_matrix()
        data_size = X.shape[0]
        last = np.floor(data_size / float(self._batch_size)) * self._batch_size
        for i in xrange(0, data_size, self._batch_size):
            if i >= last:
                stop = i + np.mod(data_size, self._batch_size)
            else:
                stop = i + self._batch_size
            print "GCN processing data from %d to %d" % (i, stop)
            data = self.transform(X[i:stop])
            dataset.set_design_matrix(data, start = i)

class ZCA(Preprocessor):
    """
    Performs ZCA whitening.
    TODO: add reference
    """
    def __init__(self, n_components=None, n_drop_components=None,
                 filter_bias=0.1):
        """
        n_components: TODO: WRITEME
        n_drop_components: TODO: WRITEME
        filter_bias: Filters are scaled by 1/sqrt(filter_bias + variance)
                    TODO: verify that default of 0.1 is what was used in the
                          Coates and Ng paper, add reference
        """
        warnings.warn("This ZCA preprocessor class is known to yield very "
                      "different results on different platforms. If you plan "
                      "to conduct experiments with this preprocessing on "
                      "multiple machines, it is probably a good idea to do "
                      "the preprocessing on a single machine and copy the "
                      "preprocessed datasets to the others, rather than "
                      "preprocessing the data independently in each "
                      "location.")
        # TODO: test to see if differences across platforms
        # e.g., preprocessing STL-10 patches in LISA lab versus on
        # Ian's Ubuntu 11.04 machine
        # are due to the problem having a bad condition number or due to
        # different version numbers of scipy or something
        self.n_components = n_components
        self.n_drop_components = n_drop_components
        self.copy = True
        self.filter_bias = filter_bias
        self.has_fit_ = False

    def fit(self, X):
        assert X.dtype in ['float32', 'float64']
        assert not np.any(np.isnan(X))
        assert len(X.shape) == 2
        n_samples = X.shape[0]
        if self.copy:
            X = X.copy()
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        # TODO: logging
        print 'computing zca'
        eigs, eigv = linalg.eigh(np.dot(X.T, X) / X.shape[0])
        print 'done with eigh'
        assert not np.any(np.isnan(eigs))
        assert not np.any(np.isnan(eigv))
        if self.n_components:
            eigs = eigs[:self.n_components]
            eigv = eigv[:, :self.n_components]
        if self.n_drop_components:
            eigs = eigs[self.n_drop_components:]
            eigv = eigv[:, self.n_drop_components:]
        self.P_ = np.dot(eigv * np.sqrt(1.0 / (eigs + self.filter_bias)),
                         eigv.T)
        # print 'zca components'
        # print np.square(self.P_).sum(axis=0)
        assert not np.any(np.isnan(self.P_))
        self.has_fit_ = True

    def apply(self, dataset, can_fit=False):
        X = dataset.get_design_matrix()
        assert X.dtype in ['float32', 'float64']
        if not self.has_fit_:
            assert can_fit
            self.fit(X)
        new_X = np.dot(X - self.mean_, self.P_)
        dataset.set_design_matrix(new_X)

    def invert(self):
        """
        Do any necessary prep work to be able to support the "inverse" method
        later.
        """
        self.inv_P_ = np.linalg.inv(self.P_)

    def inverse(self, X):
        assert X.ndim == 2
        return np.dot(X, self.inv_P_) + self.mean_

class LeCunLCN(ExamplewisePreprocessor):
    """ Yann LeCun local contrast normalization
    """

    def __init__(self, img_shape, kernel_size = 7, batch_size = 5000,
                threshold = 1e-4, channels = None):
        """
        img_shape: image shape
        kernel_size: local contrast kernel size
        batch_size: batch size. If dataset is based on PyTables use a
                    batch size smaller than 10000. Otherwise any
                    batch size diffrent than datasize is not supported yet.
        threshold: threshold for denominator
        channels: List of channels to normalize.
                    If none will apply it on all channels
        """
        self._img_shape = img_shape
        self._kernel_size = kernel_size
        self._batch_size = batch_size
        self._threshold = threshold
        if channels is None:
            self._channels = range(3)
        else:
            if isinstance(channels, list) or isinstance(channels, tuple):
                self._channels = channels
            elif isinstance(channels, int):
                self._channels = [channels]
            else:
                raise ValueError("channesl should be either a list or int")

    def transform(self, x):
        """
        X: data with axis [b, 0, 1, c]
        """
        for i in self._channels:
            assert isinstance(i, int)
            assert i >= 0 and i <= x.shape[3]

            x[:, :, :, i] = lecun_lcn(x[:, :, :, i], self._img_shape,
                                                    self._kernel_size,
                                                    self._threshold)
        return x

    def apply(self, dataset, can_fit=False):
        axes = ['b', 0, 1, 'c']
        data_size = dataset.X.shape[0]

        if self._channels is None:
            self._channels

        last = np.floor(data_size / float(self._batch_size)) * self._batch_size
        for i in xrange(0, data_size, self._batch_size):
            stop = i + np.mod(data_size, self._batch_size) if i>= last else i + self._batch_size
            print "LCN processing data from %d to %d" % (i, stop)
            transformed = self.transform(convert_axes(
                                dataset.get_topological_view(
                                dataset.X[i:stop, :]),
                                dataset.axes, axes))
            transformed = convert_axes(transformed, axes, dataset.axes)
            if self._batch_size != data_size:
                if isinstance(dataset.X, np.ndarray):
                    # TODO have a separate class for non pytables datasets
                    transformed = convert_axes(transformed, dataset.axes, ['b', 0, 1, 'c'])
                    transformed = transformed.reshape(transformed.shape[0],
                                        transformed.shape[1] * transformed.shape[2] * transformed.shape[3])
                    dataset.X[i:stop] = transformed
                else:
                    dataset.set_topological_view(transformed, dataset.axes,
                                            start = i)

        if self._batch_size == data_size:
            dataset.set_topological_view(transformed, dataset.axes)

class RGB_YUV(ExamplewisePreprocessor):

    def __init__(self, rgb_yuv = True, batch_size = 5000):
        """
        Converts image color channels from rgb to yuv and vice versa

        Parameters:

            rgb_yuv: If true converts from rgb to yuv, if false
            converts from yuv to rgb
            batch_size: batch_size to make conversions in batches
        """

        self._batch_size = batch_size
        self._rgb_yuv = rgb_yuv

    def yuv_rgb(self, x):
        y = x[:,:,:,0]
        u = x[:,:,:,1]
        v = x[:,:,:,2]

        r = y + 1.13983 * v
        g = y - 0.39465 * u - 0.58060 * v
        b = y + 2.03211 * u

        x[:,:,:,0] = r
        x[:,:,:,1] = g
        x[:,:,:,2] = b

        return x

    def rgb_yuv(self, x):
        r = x[:,:,:,0]
        g = x[:,:,:,1]
        b = x[:,:,:,2]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v = 0.615 * r -0.51499 * g  -0.10001 * b

        x[:,:,:,0] = y
        x[:,:,:,1] = u
        x[:,:,:,2] = v

        return x

    def transform(self, x, dataset_axes):

        axes = ['b', 0, 1, 'c']
        x = convert_axes(x, dataset_axes, axes)
        if self._rgb_yuv:
            x = self.rgb_yuv(x)
        else:
            x = self.yuv_rgb(x)
        x = convert_axes(x, axes, dataset_axes)
        return x

    def apply(self, dataset, can_fit=False):

        X = dataset.X
        data_size = X.shape[0]
        last = np.floor(data_size / float(self._batch_size)) * self._batch_size
        for i in xrange(0, data_size, self._batch_size):
            stop = i + np.mod(data_size, self._batch_size) if i>= last else i + self._batch_size
            print "RGB_YUV processing data from %d to %d" % (i, stop)
            data = dataset.get_topological_view(X[i:stop])
            transformed = self.transform(data, dataset.axes)

            # TODO have a separate class for non pytables datasets
            # or add start option to dense_design_matrix
            if isinstance(dataset.X, np.ndarray):
                transformed = convert_axes(transformed, dataset.axes, ['b', 0, 1, 'c'])
                transformed = transformed.reshape(transformed.shape[0],
                                    transformed.shape[1] * transformed.shape[2] * transformed.shape[3])
                dataset.X[i:stop] = transformed
            else:
                dataset.set_topological_view(transformed, dataset.axes, start = i)

class CentralWindow(Preprocessor):
    """
    Preprocesses an image dataset to contain only the central window.
    """

    def __init__(self, window_shape):

        self.__dict__.update(locals())
        del self.self

    def apply(self, dataset, can_fit=False):

        w_rows, w_cols = self.window_shape

        arr = dataset.get_topological_view()

        try:
            axes = dataset.view_converter.axes
        except AttributeError:
            raise NotImplementedError("I don't know how to tell what the axes of this kind of dataset are.")

        needs_transpose = not axes[1:3] == (0, 1)

        if needs_transpose:
            arr = np.transpose(arr, (axes.index('c'), axes.index(0), axes.index(1), axes.index('b')))

        r_off = (arr.shape[1] - w_rows) // 2
        c_off = (arr.shape[2] - w_cols) // 2
        new_arr = arr[:, r_off:r_off + w_rows, c_off:c_off + w_cols, :]

        if needs_transpose:
            new_arr = np.transpose(new_arr, tuple(('c', 0, 1, 'b').index(axis) for axis in axes))

        dataset.set_topological_view(new_arr, axes=axes)

def lecun_lcn(input, img_shape, kernel_shape, threshold = 1e-4):
    """
    Yann LeCun's local contrast normalization
    Orginal code in Theano by: Guillaume Desjardins
    """
    input = input.reshape(input.shape[0], input.shape[1], input.shape[2], 1)
    X = T.matrix(dtype=input.dtype)
    X = X.reshape((len(input), img_shape[0], img_shape[1], 1))

    filter_shape = (1, 1, kernel_shape, kernel_shape)
    filters = sharedX(gaussian_filter(kernel_shape).reshape(filter_shape))

    input_space = Conv2DSpace(shape = img_shape, num_channels = 1)
    transformer = Conv2D(filters = filters, batch_size = len(input),
                        input_space = input_space,
                        border_mode = 'full')
    convout = transformer.lmul(X)

    # For each pixel, remove mean of 9x9 neighborhood
    mid = int(np.floor(kernel_shape/ 2.))
    centered_X = X - convout[:,mid:-mid,mid:-mid,:]

    # Scale down norm of 9x9 patch if norm is bigger than 1
    transformer = Conv2D(filters = filters, batch_size = len(input),
                        input_space = input_space,
                        border_mode = 'full')
    sum_sqr_XX = transformer.lmul(X**2)

    denom = T.sqrt(sum_sqr_XX[:,mid:-mid,mid:-mid,:])
    per_img_mean = denom.mean(axis = [1,2])
    divisor = T.largest(per_img_mean.dimshuffle(0,'x', 'x', 1), denom)
    divisor = T.maximum(divisor, threshold)

    new_X = centered_X / divisor
    new_X = T.flatten(new_X, outdim=3)

    f = function([X], new_X)
    return f(input)

def gaussian_filter(kernel_shape):

    x = np.zeros((kernel_shape, kernel_shape), dtype='float32')

    def gauss(x, y, sigma=2.0):
        Z = 2 * np.pi * sigma**2
        return  1./Z * np.exp(-(x**2 + y**2) / (2. * sigma**2))

    mid = np.floor(kernel_shape/ 2.)
    for i in xrange(0,kernel_shape):
        for j in xrange(0,kernel_shape):
            x[i,j] = gauss(i-mid, j-mid)

    return x / np.sum(x)

class CentralWindow(Preprocessor):
    """
    Preprocesses an image dataset to contain only the central window.
    """

    def __init__(self, window_shape):

        self.__dict__.update(locals())
        del self.self

    def apply(self, dataset, can_fit=False):

        w_rows, w_cols = self.window_shape

        arr = dataset.get_topological_view()

        try:
            axes = dataset.view_converter.axes
        except AttributeError:
            raise NotImplementedError("I don't know how to tell what the axes of this kind of dataset are.")

        needs_transpose = not axes[1:3] == (0, 1)

        if needs_transpose:
            arr = np.transpose(arr, (axes.index('c'), axes.index(0), axes.index(1), axes.index('b')))

        r_off = (arr.shape[1] - w_rows) // 2
        c_off = (arr.shape[2] - w_cols) // 2
        new_arr = arr[:, r_off:r_off + w_rows, c_off:c_off + w_cols, :]

        if needs_transpose:
            new_arr = np.transpose(new_arr, tuple(('c', 0, 1, 'b').index(axis) for axis in axes))

        dataset.set_topological_view(new_arr, axes=axes)

class ShuffleAndSplit(Preprocessor):

    def __init__(self, seed, start, stop):
        """
        Allocates a numpy rng with the specified seed.
        Note: this must be a seed, not a RandomState. A new RandomState is
        re-created with the same seed every time the preprocessor is called.
        This way if you save the preprocessor and re-use it later it will give
        the same dataset regardless of whether you save the preprocessor before
        or after applying it.
        Shuffles the data, then takes examples in range (start, stop)
        """

        self.__dict__.update(locals())
        del self.self

    def apply(self, dataset, can_fit=False):
        start = self.start
        stop = self.stop
        rng = np.random.RandomState(self.seed)
        X = dataset.X
        y = dataset.y

        if y is not None:
            assert X.shape[0] == y.shape[0]

        for i in xrange(X.shape[0]):
            j = rng.randint(X.shape[0])
            tmp = X[i, :].copy()
            X[i,:] = X[j, :].copy()
            X[j,:] = tmp

            if y is not None:
                tmp = y[i, :].copy()
                y[i, :] = y[j,:].copy()
                y[j, :] = tmp
        assert start >= 0
        assert stop > start
        assert stop <= X.shape[0]

        dataset.X = X[start:stop, :]
        if y is not None:
            dataset.y = y[start:stop, :]




