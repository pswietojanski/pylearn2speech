__authors__ = 'Ian Goodfellow'
__copyright__ = "Copyright 2013, Universite de Montreal"

from pylearn2.costs.cost import Cost
from pylearn2.space import CompositeSpace
from pylearn2.utils import sharedX
from collections import OrderedDict

class Dropout(Cost):
    """
    Implements the dropout training technique described in
    "Improving neural networks by preventing co-adaptation of feature
    detectors"
    Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever,
    Ruslan R. Salakhutdinov
    arXiv 2012

    This paper suggests including each unit with probability p during training,
    then multiplying the outgoing weights by p at the end of training.
    We instead include each unit with probability p and divide its
    state by p during training. Note that this means the initial weights should
    be multiplied by p relative to Hinton's.
    The SGD learning rate on the weights should also be scaled by p^2 (use
    W_lr_scale rather than adjusting the global learning rate, because the
    learning rate on the biases should not be adjusted).
    """

    supervised = True

    def __init__(self, default_input_include_prob=.5, input_include_probs=None,
            default_input_scale=2., input_scales=None, per_example=True):
        """
        During training, each input to each layer is randomly included or excluded
        for each example. The probability of inclusion is independent for each input
        and each example. Each layer uses "default_input_include_prob" unless that
        layer's name appears as a key in input_include_probs, in which case the input
        inclusion probability is given by the corresponding value.

        Each feature is also multiplied by a scale factor. The scale factor for each
        layer's input scale is determined by the same scheme as the input probabilities.
        """

        if input_include_probs is None:
            input_include_probs = {}

        if input_scales is None:
            input_scales = {}

        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        (X, Y) = data
        Y_hat = model.dropout_fprop(
            X,
            default_input_include_prob=self.default_input_include_prob,
            input_include_probs=self.input_include_probs,
            default_input_scale=self.default_input_scale,
            input_scales=self.input_scales,
            per_example=self.per_example
        )
        return model.cost(Y, Y_hat)

    def get_data_specs(self, model):
        space = CompositeSpace([model.get_input_space(), model.get_output_space()])
        sources = (model.get_input_source(), model.get_target_source())
        return (space, sources)


class AnnealedDropout(Cost):
    """
    Implements the annealed dropout training technique described in:

    Annealed Dropout for Maxout Model Training for Large Vocabulary Speech Recognition
    Rennie S, Thomas S, IEEE SLT, 2014
    """

    supervised = True

    def __init__(self, default_input_include_prob=1.0, input_include_probs=None,
            default_input_scale=1.0, input_scales=None, per_example=True):
        """
        During training, each input to each layer is randomly included or excluded
        for each example. The probability of inclusion is independent for each input
        and each example. Each layer uses "default_input_include_prob" unless that
        layer's name appears as a key in input_include_probs, in which case the input
        inclusion probability is given by the corresponding value.

        Each feature is also multiplied by a scale factor. The scale factor for each
        layer's input scale is determined by the same scheme as the input probabilities.
        """

        if input_include_probs is None:
            input_include_probs = {}

        if input_scales is None:
            input_scales = {}

        self.__dict__.update(locals())

        self.input_probs = {}
        self.input_scales = {}

        for key, prob in input_include_probs.iteritems():
            self.input_probs[key] = sharedX(prob, '%s_input_include_prob'%key)
            if key in input_scales:
                self.input_scales[key] = sharedX(input_scales[key], '%s_input_include_prob'%key)
            else:
                self.input_scales[key] = sharedX(1.0/prob, '%s_input_include_prob'%key)

        del self.self

    def expr(self, model, data, ** kwargs):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        (X, Y) = data
        Y_hat = model.annealed_dropout_fprop(
            X,
            default_input_include_prob=self.default_input_include_prob,
            input_include_probs=self.input_probs,
            default_input_scale=self.default_input_scale,
            input_scales=self.input_scales,
            per_example=self.per_example
        )
        return model.cost(Y, Y_hat)

    def get_data_specs(self, model):
        space = CompositeSpace([model.get_input_space(), model.get_output_space()])
        sources = (model.get_input_source(), model.get_target_source())
        return (space, sources)
    
    def get_monitoring_channels(self, model, data, **kwargs):
        """
        Returns a dictionary mapping channel names to expressions for
        channel values.

        WRITEME: how do you do prereqs in this setup? (there is a way,
            but I forget how right now)

        Parameters
        ----------
        model: the model to use to compute the monitoring channels
        data: symbolic expressions for the monitoring data

        kwargs: used so that custom algorithms can use extra variables
                for monitoring.

        """
        self.get_data_specs(model)[0].validate(data)
        rval = OrderedDict()

        for key in self.input_probs.keys():
            rval['%s_prob'%key] = self.input_probs[key]
            rval['%s_scale'%key] = self.input_scales[key]

        return rval
