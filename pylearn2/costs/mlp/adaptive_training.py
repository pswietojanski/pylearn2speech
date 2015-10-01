__authors__ = 'Pawel Swietojanski'
__copyright__ = "Copyright 2015, University of Edinburgh"

from pylearn2.costs.cost import Cost
from pylearn2.space import CompositeSpace, VectorSpace

class ClusterAdaptiveCost(Cost):
    """
    Implements a form of speaker adaptive traing scheme
    where some subset of parameters in the model are
    trained separaterly depending on the cluster it came
    from. The cluster could be a given speaker's data and/or any other
    meaningful set of data that can benefit from adaptive training.
    """

    supervised = True

    def __init__(self):
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        ((X, SPK_IDX), Y) = data
        Y_hat = model.fprop_sat((X, SPK_IDX), SPK_IDX)
        return model.cost(Y, Y_hat)

    def get_data_specs(self, model):
        space = CompositeSpace([model.get_input_space(), model.get_output_space()])
        sources = (model.get_input_source(), model.get_target_source())
        return (space, sources)


class ClusterConcAdaptiveCost(Cost):
    """
    Implements a form of speaker adaptive traing scheme
    where some subset of parameters in the model are
    trained separaterly depending on the cluster it came
    from. The cluster could be a given speaker's data and/or any other
    meaningful set of data that can benefit from adaptive training.
    """

    supervised = True

    def __init__(self):
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        (X, Y) = data

        if X.ndim == 2:
            SPK_IDX = X[:,-1]
        elif X.ndim == 4:
            SPK_IDX = X[:,:,:,-1].reshape((-1,))
        else:
            raise ValueError('Expected to get 2D or 4D ndarray')

        Y_hat = model.fprop_sat(X, SPK_IDX)
        return model.cost(Y, Y_hat)

    def get_data_specs(self, model):
        space = CompositeSpace([model.get_input_space(), model.get_output_space()])
        sources = (model.get_input_source(), model.get_target_source())
        return (space, sources)

