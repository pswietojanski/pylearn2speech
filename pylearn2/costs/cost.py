"""
Classes representing loss functions.
Currently, these are primarily used to specify
the objective function for the SGD and BGD
training algorithms.
"""
from itertools import izip
import warnings

import theano.tensor as T
from theano.compat.python2x import OrderedDict

from pylearn2.utils import safe_zip
from pylearn2.utils import safe_union
from pylearn2.space import CompositeSpace, NullSpace
from pylearn2.utils.data_specs import DataSpecsMapping



class Cost(object):
    """
    Represents a cost that can be called either as a supervised cost or an
    unsupervised cost.
    """

    # If True, the data argument to expr and get_gradients must be a
    # (X, Y) pair, and Y cannot be None.
    supervised = False

    def expr(self, model, data, ** kwargs):
        """
        Parameters
        ----------
        model: a pylearn2 Model instance
        data : a batch in cost.get_data_specs() form

        Returns a symbolic expression for a cost function applied to the
        minibatch of data.
        Optionally, may return None. This represents that the cost function
        is intractable but may be optimized via the get_gradients method.

        """

        raise NotImplementedError(str(type(self))+" does not implement expr.")

    def get_gradients(self, model, data, ** kwargs):
        """
        Parameters
        ----------
        model: a pylearn2 Model instance
        data : a batch in cost.get_data_specs() form

        returns: gradients, updates
            gradients:
                a dictionary mapping from the model's parameters
                         to their gradients
                The default implementation is to compute the gradients
                using T.grad applied to the value returned by expr.
                However, subclasses may return other values for the gradient.
                For example, an intractable cost may return a sampling-based
                approximation to its gradient.
            updates:
                a dictionary mapping shared variables to updates that must
                be applied to them each time these gradients are computed.
                This is to facilitate computation of sampling-based approximate
                gradients.
                The parameters should never appear in the updates dictionary.
                This would imply that computing their gradient changes
                their value, thus making the gradient value outdated.
        """

        try:
            cost = self.expr(model=model, data=data, **kwargs)
        except TypeError,e:
            # If anybody knows how to add type(seslf) to the exception message
            # but still preserve the stack trace, please do so
            # The current code does neither
            e.message += " while calling "+str(type(self))+".expr"
            print str(type(self))
            print e.message
            raise e

        if cost is None:
            raise NotImplementedError(str(type(self))+" represents an intractable "
                    " cost and does not provide a gradient approximation scheme.")

        params = list(model.get_params())

        grads = T.grad(cost, params, disconnected_inputs = 'ignore')

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()

        return gradients, updates

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
        return OrderedDict()

    def get_fixed_var_descr(self, model, data):
        """
        Subclasses should override this if they need variables held
        constant across multiple updates to a minibatch.

        TrainingAlgorithms that do multiple updates to a minibatch should
        respect this. See FixedVarDescr below for details.
        """
        self.get_data_specs(model)[0].validate(data)
        return FixedVarDescr()

    def get_data_specs(self, model):
        """
        Returns a composite space, describing the format of the data
        which the cost (and the model) expects.
        """
        raise NotImplementedError(str(type(self))+" does not implement " +
                                  "get_data_specs.")


class CostCDCI(Cost):
    """
    The customised cost for softmax layer with auxliliary task
    """
    def __init__(self, coeff_cd, coeff_ci, zero_ci_grad_for_cd=True):
        self.coeff_cd = coeff_cd
        self.coeff_ci = coeff_ci
        self.zero_ci_grad_for_cd = zero_ci_grad_for_cd

    def expr(self, model, data, ** kwargs):
        """
        Returns the sum of the costs the SumOfCosts instance was given at
        initialization.

        Parameters
        ----------
        model : pylearn2.models.model.Model
            the model for which we want to calculate the sum of costs
        data : flat tuple of tensor_like variables.
            data has to follow the format defined by self.get_data_specs(),
            but this format will always be a flat tuple.
        """
        cost_cd, cost_ci = model.cost_from_X(data)
        sum_of_costs = self.coeff_cd*cost_cd + self.coeff_ci*cost_ci

        return sum_of_costs

    def get_composite_data_specs(self, model):
        """
        Build and return a composite data_specs of all costs.

        The returned space is a CompositeSpace, where the components are
        the spaces of each of self.costs, in the same order. The returned
        source is a tuple of the corresponding sources.
        """
        raise NotImplementedError(str(type(self))+" does not implement " +
                                  "get_composite_data_specs.")

    def get_composite_specs_and_mapping(self, model):
        """
        Build the composite data_specs and a mapping to flatten it, return both

        Build the composite data_specs described in `get_composite_specs`,
        and build a DataSpecsMapping that can convert between it and a flat
        equivalent version. In particular, it helps building a flat data_specs
        to request data, and nesting this data back to the composite data_specs,
        so it can be dispatched among the different sub-costs.

        This is a helper function used by `get_data_specs` and `get_gradients`,
        and possibly other methods.
        """
        raise NotImplementedError(str(type(self))+" does not implement " +
                                  "get_composite_specs_and_mapping.")

    def get_data_specs(self, model):
        """
        Get a flat data_specs containing all information for all sub-costs.

        This data_specs should be non-redundant. It is built by flattening
        the composite data_specs returned by `get_composite_specs`.

        This is the format that SumOfCosts will request its data in. Then,
        this flat data tuple will be nested into the composite data_specs,
        in order to dispatch it among the different sub-costs.
        """
        space = CompositeSpace([model.get_input_space(), model.get_output_space()])
        sources = (model.get_input_source(), model.get_target_source())
        return (space, sources)

    def get_gradients(self, model, data, ** kwargs):

        cost_cd, cost_ci = model.cost_from_X(data)
        params_dict = model.get_params()
        params = list(params_dict)

        zero_grads = []
        if self.zero_ci_grad_for_cd:
            #how to get this in less explicit way, i.e. using only dict?
            print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
            assert model.layers[-1].M in params_dict
            assert model.layers[-1].m in params_dict
            zero_grads = [model.layers[-1].M, model.layers[-1].m]

        grads_cd = T.grad(cost_cd, params, disconnected_inputs = 'ignore', consider_constant=zero_grads)
        grads_ci = T.grad(cost_ci, params, disconnected_inputs = 'ignore')

        gradients_cd = OrderedDict(izip(params, grads_cd))
        gradients_ci = OrderedDict(izip(params, grads_ci))

        indiv_results = []
        indiv_results.append((gradients_cd, OrderedDict()))
        indiv_results.append((gradients_ci, OrderedDict()))

        grads = OrderedDict()
        updates = OrderedDict()
        params = model.get_params()

        for coeff, packed in zip([self.coeff_cd, self.coeff_ci], indiv_results):
            g, u = packed
            for param in g:
                if param not in params:
                    raise ValueError("A shared variable ("+str(param)+") that is not a parameter appeared in a cost gradient dictionary.")
            for param in g:
                assert param.ndim == g[param].ndim
                v = coeff * g[param]
                if param not in grads:
                    grads[param] = v
                else:
                    grads[param] = grads[param] + v
                assert grads[param].ndim == param.ndim
            assert not any([state in updates for state in u])
            assert not any([state in params for state in u])
            updates.update(u)

        return grads, updates

    def get_monitoring_channels(self, model, data, ** kwargs):
        self.get_data_specs(model)[0].validate(data)
        rval = OrderedDict()
        value = self.expr(model, data, ** kwargs)
        if value is not None:
            name = ''
            if hasattr(value, 'name') and value.name is not None:
                name = '_' + value.name
            rval['sum_of_costs_'+name] = value
        return rval

    def get_fixed_var_descr(self, model, data):
        self.get_data_specs(model)[0].validate(data)
        return FixedVarDescr()


class SumOfCosts(Cost):
    """
    Combines multiple costs by summing them.
    """
    def __init__(self, costs):
        """
        Initialize the SumOfCosts object and make sure that the list of costs
        contains only Cost instances.

        Parameters
        ----------
        costs: list
            List of Cost objects or (coeff, Cost) pairs
        """
        assert isinstance(costs, list)
        assert len(costs) > 0

        self.costs = []
        self.coeffs = []

        for cost in costs:
            if isinstance(cost, (list, tuple)):
                coeff, cost = cost
            else:
                coeff = 1.
            self.coeffs.append(coeff)
            self.costs.append(cost)

            if not isinstance(cost, Cost):
                raise ValueError("one of the costs is not " + \
                                 "Cost instance")

        # TODO: remove this when it is no longer necessary
        self.supervised = any([cost.supervised for cost in self.costs])

    def expr(self, model, data, ** kwargs):
        """
        Returns the sum of the costs the SumOfCosts instance was given at
        initialization.

        Parameters
        ----------
        model : pylearn2.models.model.Model
            the model for which we want to calculate the sum of costs
        data : flat tuple of tensor_like variables.
            data has to follow the format defined by self.get_data_specs(),
            but this format will always be a flat tuple.
        """
        self.get_data_specs(model)[0].validate(data)
        composite_specs, mapping = self.get_composite_specs_and_mapping(model)
        nested_data = mapping.nest(data)
        costs = []
        for cost, cost_data in safe_zip(self.costs, nested_data):
            print cost_data
            costs.append(cost.expr(model, cost_data, **kwargs))
        assert len(costs) > 0

        if any([cost is None for cost in costs]):
            sum_of_costs = None
        else:
            costs = [coeff * cost for coeff, cost in safe_zip(self.coeffs, costs)]
            assert len(costs) > 0
            sum_of_costs = reduce(lambda x, y: x + y, costs)

        return sum_of_costs

    def get_composite_data_specs(self, model):
        """
        Build and return a composite data_specs of all costs.

        The returned space is a CompositeSpace, where the components are
        the spaces of each of self.costs, in the same order. The returned
        source is a tuple of the corresponding sources.
        """
        spaces = []
        sources = []
        for cost in self.costs:
            space, source = cost.get_data_specs(model)
            spaces.append(space)
            sources.append(source)

        # Build composite space representing all inputs
        composite_space = CompositeSpace(spaces)
        sources = tuple(sources)
        return (composite_space, sources)

    def get_composite_specs_and_mapping(self, model):
        """
        Build the composite data_specs and a mapping to flatten it, return both

        Build the composite data_specs described in `get_composite_specs`,
        and build a DataSpecsMapping that can convert between it and a flat
        equivalent version. In particular, it helps building a flat data_specs
        to request data, and nesting this data back to the composite data_specs,
        so it can be dispatched among the different sub-costs.

        This is a helper function used by `get_data_specs` and `get_gradients`,
        and possibly other methods.
        """
        composite_space, sources = self.get_composite_data_specs(model)
        mapping = DataSpecsMapping((composite_space, sources))
        return (composite_space, sources), mapping

    def get_data_specs(self, model):
        """
        Get a flat data_specs containing all information for all sub-costs.

        This data_specs should be non-redundant. It is built by flattening
        the composite data_specs returned by `get_composite_specs`.

        This is the format that SumOfCosts will request its data in. Then,
        this flat data tuple will be nested into the composite data_specs,
        in order to dispatch it among the different sub-costs.
        """
        composite_specs, mapping = self.get_composite_specs_and_mapping(model)
        composite_space, sources = composite_specs
        flat_composite_space = mapping.flatten(composite_space)
        flat_sources = mapping.flatten(sources)
        data_specs = (flat_composite_space, flat_sources)
        return data_specs

    def get_gradients(self, model, data, ** kwargs):
        indiv_results = []
        composite_specs, mapping = self.get_composite_specs_and_mapping(model)
        nested_data = mapping.nest(data)
        for cost, cost_data in safe_zip(self.costs, nested_data):
            result = cost.get_gradients(model, cost_data, ** kwargs)
            indiv_results.append(result)

        grads = OrderedDict()
        updates = OrderedDict()
        params = model.get_params()

        for coeff, packed in zip(self.coeffs, indiv_results):
            g, u = packed
            for param in g:
                if param not in params:
                    raise ValueError("A shared variable ("+str(param)+") that is not a parameter appeared in a cost gradient dictionary.")
            for param in g:
                assert param.ndim == g[param].ndim
                v = coeff * g[param]
                if param not in grads:
                    grads[param] = v
                else:
                    grads[param] = grads[param] + v
                assert grads[param].ndim == param.ndim
            assert not any([state in updates for state in u])
            assert not any([state in params for state in u])
            updates.update(u)

        return grads, updates

    def get_monitoring_channels(self, model, data, ** kwargs):
        self.get_data_specs(model)[0].validate(data)
        rval = OrderedDict()
        composite_specs, mapping = self.get_composite_specs_and_mapping(model)
        nested_data = mapping.nest(data)

        for i, cost in enumerate(self.costs):
            cost_data = nested_data[i]
            try:
                rval.update(cost.get_monitoring_channels(model, cost_data, **kwargs))
            except TypeError:
                print 'SumOfCosts.get_monitoring_channels encountered TypeError while calling ' \
                        + str(type(cost))+'.get_monitoring_channels'
                raise

            value = cost.expr(model, cost_data, ** kwargs)
            if value is not None:
                name = ''
                if hasattr(value, 'name') and value.name is not None:
                    name = '_' + value.name
                rval['term_'+str(i)+name] = value

        return rval

    def get_fixed_var_descr(self, model, data):
        data_specs = self.get_data_specs(model)
        data_specs[0].validate(data)
        composite_specs, mapping = self.get_composite_specs_and_mapping(model)
        nested_data = mapping.nest(data)

        descrs = [cost.get_fixed_var_descr(model, cost_data)
                  for cost, cost_data in safe_zip(self.costs, nested_data)]

        rval = FixedVarDescr()
        rval.data_specs = data_specs
        rval.on_load_batch = []
        # To avoid calling the same function more than once
        on_load_batch_seen = []

        for i, descr in enumerate(descrs):
            # We assume aliasing is a bug
            assert descr.fixed_vars is not rval.fixed_vars
            assert descr.on_load_batch is not rval.on_load_batch

            for key in descr.fixed_vars:
                if key in rval.fixed_vars:
                    raise ValueError("Cannot combine these FixedVarDescrs, "
                            "two different ones contain %s" % key)
            rval.fixed_vars.update(descr.fixed_vars)

            for on_load in descr.on_load_batch:
                if on_load in on_load_batch_seen:
                    continue
                # Using default argument binds the variables used in the lambda
                # function to the value they have when the lambda is defined.
                new_on_load = (lambda batch, mapping=mapping, i=i,
                                      on_load=on_load:
                        on_load(mapping.nest(batch)[i]))
                rval.on_load_batch.append(new_on_load)

        return rval


class ScaledCost(Cost):
    """
    Represents a given cost scaled by a constant factor.
    TODO: why would you want to use this? SumOfCosts allows you to scale individual
        terms, and if this is the only cost, why not just change the learning rate?
        If there's an obvious use case or rationale we should document it, if not,
        we should remove it.
    """
    def __init__(self, cost, scaling):
        """
        Parameters
        ----------
        cost: Cost
            cost to be scaled
        scaling : float
            scaling of the cost
        """
        self.cost = cost
        self.supervised = cost.supervised
        self.scaling = scaling

    def expr(self, model, data):
        """
        Returns cost scaled by its scaling factor.

        Parameters
        ----------
        model : pylearn2.models.model.Model
            the model for which we want to calculate the scaled cost
        X : tensor_like
            input to the model
        Y : tensor_like
            the target, if necessary
        """
        self.get_data_specs(model)[0].validate(data)
        return self.scaling * self.cost(model, data)

    def get_data_specs(self, model):
        return self.cost.get_data_specs(model)


class LxReg(Cost):
    """
    L-x regularization term for the list of tensor variables provided.
    """
    def __init__(self, variables, x):
        """
        Initialize LxReg with the variables and scaling provided.

        Parameters:
        -----------
        variables: list
            list of tensor variables to be regularized
        x: int
            the x in "L-x regularization""
        """
        self.variables = variables
        self.x = x

    def expr(self, model=None, data=None):
        """
        Return the scaled L-x regularization term. The optional parameters are
        never used, they're there only to provide an interface consistent with
        both SupervisedCost and UnsupervisedCost.
        """
        # This Cost does not depend on any data, and get_data_specs does not
        # ask for any data, so we should not be provided with some.
        self.get_data_specs(model)[0].validate(data)

        Lx = 0
        for var in self.variables:
            Lx = Lx + abs(var ** self.x).sum()
        return Lx

    def get_data_specs(self, model):
        # This cost does not use any data
        return (NullSpace(), '')


class CrossEntropy(Cost):
    """WRITEME"""
    def __init__(self):
        self.supervised = True

    def expr(self, model, data, ** kwargs):
        """WRITEME"""
        self.get_data_specs(model)[0].validate(data)

        # unpack data
        (X, Y) = data
        return (-Y * T.log(model(X)) - \
                (1 - Y) * T.log(1 - model(X))).sum(axis=1).mean()

    def get_data_specs(self, model):
        data = CompositeSpace([model.get_input_space(),
                               model.get_output_space()])
        sources = (model.get_input_source(), model.get_target_source())
        return (data, sources)
  
  
class MethodCost(Cost):
    """
    A cost specified via the string name of a method of the model.
    """

    def __init__(self, method, data_specs=None, supervised=None):
        """
            method: a string specifying the name of the method of the model
                    that should be called to generate the objective function.
            supervised: deprecated argument, ignored
            data_specs: a string specifying the name of a method/property of
                    the model that describe the data specs required by
                    method
        """
        if supervised != None:
            if data_specs is not None:
                raise TypeError("Deprecated argument 'supervised' and new "
                                "argument 'data_specs' were both specified.")
            warnings.warn("Usage of 'supervised' argument of MethodCost "
                          "is deprecated. Use 'data_specs' to provide the "
                          "name of a method or property of the model "
                          "that describes the data specs required by method "
                          "%s. %s will be used by default."
                          % (method, method + '_data_specs'),
                          stacklevel=2)
        self.method = method
        self.data_specs = data_specs

    def expr(self, model, data, *args, **kwargs):
            """ Patches calls through to a user-specified method of the model """
            self.get_data_specs(model)[0].validate(data)
            fn = getattr(model, self.method)
            return fn(data, *args, **kwargs)

    def get_data_specs(self, model):
        if self.data_specs is not None:
            fn = getattr(model, self.data_specs)
        else:
            # To be compatible with earlier scripts,
            # try (self.method)_data_specs
            fn = getattr(model, '%s_data_specs' % self.method)

        if callable(fn):
            return fn()
        else:
            return fn

def _no_op(data):
    """
    An on_load_batch callback that does nothing.
    """

class FixedVarDescr(object):
    """
    An object used to describe variables that influence the cost but that should
    be held fixed for each minibatch, even if the learning algorithm makes multiple
    changes to the parameters on this minibatch, ie, for a line search, etc.
    """

    def __init__(self):
        """
        fixed_vars: maps string names to shared variables or some sort of data structure
                    surrounding shared variables.
                    Any learning algorithm that does multiple updates on the same minibatch
                    should pass fixed_vars to the cost's expr and get_gradient methods
                    as keyword arguments.
        """
        self.fixed_vars = {}

        """
        A list of callable objects that the learning algorithm should
        call with input data (formatted as self.data_specs) as appropriate
        whenever a new batch of data is loaded.
        This will update the shared variables mapped to by fixed_vars.

        TODO: figure out why on_load_batch uses _no_op instead of an
            empty list--either there is a reason and it should be
            documented, or there is not reason and it should just be
            an empty list.
        """
        self.on_load_batch = [_no_op]

        """
        A (space, source) pair describing the inputs of every function
        in self.on_load_batch.
        """
        self.data_specs = (NullSpace(), '')


def merge(left, right):
    """
    Combine two FixedVarDescrs
    """

    assert left is not right
    # We assume aliasing is a bug
    assert left.fixed_vars is not right.fixed_vars
    assert left.on_load_batch is not right.on_load_batch

    rval = FixedVarDescr()
    for key in left.fixed_vars:
        if key in right.fixed_vars:
            raise ValueError("Can't merge these FixedVarDescrs, both contain "+key)
    assert not any([key in left.fixed_vars for key in right.fixed_vars])
    rval.fixed_vars.update(left.fixed_vars)
    rval.fixed_vars.update(right.fixed_vars)

    if left.data_specs == right.data_specs:
        # Combining the on_load_batch functions is easy, as they take
        # the same input arguments
        rval.data_specs = left.fixed_vars
        rval.on_load_batch = safe_union(left.on_load_batch, right.on_load_batch)
    else:
        # We would have to build a composite data_specs
        raise NotImplementedError()

    return rval
