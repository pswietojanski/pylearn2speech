"""
This module contains cost functions to use with deep Boltzmann machines
(pylearn2.models.dbm).
"""
__authors__ = ["Ian Goodfellow", "Vincent Dumoulin"]
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

import warnings

from theano.compat.python2x import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T

from pylearn2.costs.cost import Cost
from pylearn2.models import dbm
from pylearn2.models.dbm import flatten
from pylearn2.space import CompositeSpace, NullSpace
from pylearn2.utils import safe_izip
from pylearn2.utils import safe_zip


class BaseCD(Cost):
    def __init__(self, num_chains, num_gibbs_steps, supervised=False,
                 toronto_neg=False):
        """
            toronto_neg: If True, use a bit of mean field in the negative phase
                        Ruslan Salakhutdinov's matlab code does this.
        """
        self.__dict__.update(locals())
        del self.self
        self.theano_rng = MRG_RandomStreams(2012 + 10 + 14)
        assert supervised in [True, False]

    def expr(self, model, data):
        """
        The partition function makes this intractable.
        """
        self.get_data_specs(model)[0].validate(data)

        return None

    def get_monitoring_channels(self, model, data):
        self.get_data_specs(model)[0].validate(data)
        rval = OrderedDict()

        if self.supervised:
            X, Y = data
        else:
            X = data
            Y = None

        history = model.mf(X, return_history = True)
        q = history[-1]

        if self.supervised:
            assert len(data) == 2
            Y_hat = q[-1]
            true = T.argmax(Y, axis=1)
            pred = T.argmax(Y_hat, axis=1)

            #true = Print('true')(true)
            #pred = Print('pred')(pred)

            wrong = T.neq(true, pred)
            err = T.cast(wrong.mean(), X.dtype)
            rval['misclass'] = err

            if len(model.hidden_layers) > 1:
                q = model.mf(X, Y=Y)
                pen = model.hidden_layers[-2].upward_state(q[-2])
                Y_recons = model.hidden_layers[-1].mf_update(state_below=pen)
                pred = T.argmax(Y_recons, axis=1)
                wrong = T.neq(true, pred)

                rval['recons_misclass'] = T.cast(wrong.mean(), X.dtype)

        return rval

    def get_gradients(self, model, data):
        self.get_data_specs(model)[0].validate(data)
        if self.supervised:
            X, Y = data
            assert Y is not None
        else:
            X = data
            Y = None

        pos_phase_grads, pos_updates = self._get_positive_phase(model, X, Y)

        neg_phase_grads, neg_updates = self._get_negative_phase(model, X, Y)

        updates = OrderedDict()
        for key, val in pos_updates.items():
            updates[key] = val
        for key, val in neg_updates.items():
            updates[key] = val

        gradients = OrderedDict()
        for param in list(pos_phase_grads.keys()):
            gradients[param] = neg_phase_grads[param] + pos_phase_grads[param]

        return gradients, updates

    def _get_toronto_neg(self, model, layer_to_chains):
        # Ruslan Salakhutdinov's undocumented negative phase from
        # http://www.mit.edu/~rsalakhu/code_DBM/dbm_mf.m
        # IG copied it here without fully understanding it, so it
        # only applies to exactly the same model structure as
        # in that code.

        assert isinstance(model.visible_layer, dbm.BinaryVector)
        assert isinstance(model.hidden_layers[0], dbm.BinaryVectorMaxPool)
        assert model.hidden_layers[0].pool_size == 1
        assert isinstance(model.hidden_layers[1], dbm.BinaryVectorMaxPool)
        assert model.hidden_layers[1].pool_size == 1
        assert isinstance(model.hidden_layers[2], dbm.Softmax)
        assert len(model.hidden_layers) == 3

        params = list(model.get_params())

        V_samples = layer_to_chains[model.visible_layer]
        H1_samples, H2_samples, Y_samples = [layer_to_chains[layer] for
                                             layer in model.hidden_layers]

        H1_mf = model.hidden_layers[0].mf_update(
            state_below=model.visible_layer.upward_state(V_samples),
            state_above=model.hidden_layers[1].downward_state(H2_samples),
            layer_above=model.hidden_layers[1])
        Y_mf = model.hidden_layers[2].mf_update(
            state_below=model.hidden_layers[1].upward_state(H2_samples))
        H2_mf = model.hidden_layers[1].mf_update(
            state_below=model.hidden_layers[0].upward_state(H1_mf),
            state_above=model.hidden_layers[2].downward_state(Y_mf),
            layer_above=model.hidden_layers[2])

        expected_energy_p = model.energy(
            V_samples, [H1_mf, H2_mf, Y_samples]
        ).mean()

        constants = flatten([V_samples, H1_mf, H2_mf, Y_samples])

        neg_phase_grads = OrderedDict(
            safe_zip(params, T.grad(-expected_energy_p, params,
                                    consider_constant=constants)))
        return neg_phase_grads

    def _get_standard_neg(self, model, layer_to_chains):
        params = list(model.get_params())

        warnings.warn("""TODO: reduce variance of negative phase by
                         integrating out the even-numbered layers. The
                         Rao-Blackwellize method can do this for you when
                         expected gradient = gradient of expectation, but
                         doing this in general is trickier.""")
        #layer_to_chains = model.rao_blackwellize(layer_to_chains)
        expected_energy_p = model.energy(
            layer_to_chains[model.visible_layer],
            [layer_to_chains[layer] for layer in model.hidden_layers]
        ).mean()

        samples = flatten(layer_to_chains.values())
        for i, sample in enumerate(samples):
            if sample.name is None:
                sample.name = 'sample_'+str(i)

        neg_phase_grads = OrderedDict(
            safe_zip(params, T.grad(-expected_energy_p, params,
                                    consider_constant=samples,
                                    disconnected_inputs='ignore'))
        )
        return neg_phase_grads

    def _get_variational_pos(self, model, X, Y):
        if self.supervised:
            assert Y is not None
            # note: if the Y layer changes to something without linear energy,
            # we'll need to make the expected energy clamp Y in the positive
            # phase
            assert isinstance(model.hidden_layers[-1], dbm.Softmax)

        q = model.mf(X, Y)

        """
            Use the non-negativity of the KL divergence to construct a lower
            bound on the log likelihood. We can drop all terms that are
            constant with repsect to the model parameters:

            log P(v) = L(v, q) + KL(q || P(h|v))
            L(v, q) = log P(v) - KL(q || P(h|v))
            L(v, q) = log P(v) - sum_h q(h) log q(h) + q(h) log P(h | v)
            L(v, q) = log P(v) + sum_h q(h) log P(h | v) + const
            L(v, q) = log P(v) + sum_h q(h) log P(h, v)
                               - sum_h q(h) log P(v) + const
            L(v, q) = sum_h q(h) log P(h, v) + const
            L(v, q) = sum_h q(h) -E(h, v) - log Z + const

            so the cost we want to minimize is
            expected_energy + log Z + const


            Note: for the RBM, this bound is exact, since the KL divergence
                  goes to 0.
        """

        variational_params = flatten(q)

        # The gradients of the expected energy under q are easy, we can just
        # do that in theano
        expected_energy_q = model.expected_energy(X, q).mean()
        params = list(model.get_params())
        gradients = OrderedDict(
            safe_zip(params, T.grad(expected_energy_q,
                                    params,
                                    consider_constant=variational_params,
                                    disconnected_inputs='ignore'))
        )
        return gradients

    def _get_sampling_pos(self, model, X, Y):
        layer_to_clamp = OrderedDict([(model.visible_layer, True)])
        layer_to_pos_samples = OrderedDict([(model.visible_layer, X)])
        if self.supervised:
            # note: if the Y layer changes to something without linear energy,
            #       we'll need to make the expected energy clamp Y in the
            #       positive phase
            assert isinstance(model.hidden_layers[-1], dbm.Softmax)
            layer_to_clamp[model.hidden_layers[-1]] = True
            layer_to_pos_samples[model.hidden_layers[-1]] = Y
            hid = model.hidden_layers[:-1]
        else:
            assert Y is None
            hid = model.hidden_layers

        for layer in hid:
            mf_state = layer.init_mf_state()

            def recurse_zeros(x):
                if isinstance(x, tuple):
                    return tuple([recurse_zeros(e) for e in x])
                return x.zeros_like()
            layer_to_pos_samples[layer] = recurse_zeros(mf_state)

        layer_to_pos_samples = model.mcmc_steps(
            layer_to_state=layer_to_pos_samples,
            layer_to_clamp=layer_to_clamp,
            num_steps=self.num_gibbs_steps,
            theano_rng=self.theano_rng)

        q = [layer_to_pos_samples[layer] for layer in model.hidden_layers]

        pos_samples = flatten(q)

        # The gradients of the expected energy under q are easy, we can just
        # do that in theano
        expected_energy_q = model.energy(X, q).mean()
        params = list(model.get_params())
        gradients = OrderedDict(
            safe_zip(params, T.grad(expected_energy_q, params,
                                    consider_constant=pos_samples,
                                    disconnected_inputs='ignore'))
        )
        return gradients


class PCD(BaseCD):
    """
    An intractable cost representing the negative log likelihood of a DBM.
    The gradient of this bound is computed using a persistent
    markov chain.

    TODO add citation to Tieleman paper, Younes paper
    """
    def _get_positive_phase(self, model, X, Y=None):
        return self._get_sampling_pos(model, X, Y), OrderedDict()

    def _get_negative_phase(self, model, X, Y=None):
        layer_to_chains = model.make_layer_to_state(self.num_chains)

        def recurse_check(l):
            if isinstance(l, (list, tuple)):
                for elem in l:
                    recurse_check(elem)
            else:
                assert l.get_value().shape[0] == self.num_chains

        recurse_check(layer_to_chains.values())

        model.layer_to_chains = layer_to_chains

        # Note that we replace layer_to_chains with a dict mapping to the new
        # state of the chains
        updates, layer_to_chains = model.get_sampling_updates(
            layer_to_chains, self.theano_rng, num_steps=self.num_gibbs_steps,
            return_layer_to_updated=True)

        if self.toronto_neg:
            neg_phase_grads = self._get_toronto_neg(model, layer_to_chains)
        else:
            neg_phase_grads = self._get_standard_neg(model, layer_to_chains)

        return neg_phase_grads, updates

    def get_data_specs(self, model):
        if self.supervised:
            space = CompositeSpace([model.get_input_space(),
                                    model.get_output_space()])
            sources = (model.get_input_source(), model.get_target_source())
            return (space, sources)
        else:
            return (model.get_input_space(), model.get_input_source())


class VariationalPCD(BaseCD):
    """
    An intractable cost representing the variational upper bound
    on the negative log likelihood of a DBM.
    The gradient of this bound is computed using a persistent
    markov chain.

    TODO add citation to Tieleman paper, Younes paper
    """

    def expr(self, model, data):
        """
        The partition function makes this intractable.
        """
        self.get_data_specs(model)[0].validate(data)
        return None

    def _get_positive_phase(self, model, X, Y=None):
        return self._get_variational_pos(model, X, Y), OrderedDict()

    def _get_negative_phase(self, model, X, Y=None):
        """
        d/d theta log Z = (d/d theta Z) / Z
                        = (d/d theta sum_h sum_v exp(-E(v,h)) ) / Z
                        = (sum_h sum_v - exp(-E(v,h)) d/d theta E(v,h) ) / Z
                        = - sum_h sum_v P(v,h)  d/d theta E(v,h)
        """
        layer_to_chains = model.make_layer_to_state(self.num_chains)

        def recurse_check(l):
            if isinstance(l, (list, tuple)):
                for elem in l:
                    recurse_check(elem)
            else:
                assert l.get_value().shape[0] == self.num_chains

        recurse_check(layer_to_chains.values())

        model.layer_to_chains = layer_to_chains

        # Note that we replace layer_to_chains with a dict mapping to the new
        # state of the chains
        updates, layer_to_chains = model.get_sampling_updates(
            layer_to_chains,
            self.theano_rng, num_steps=self.num_gibbs_steps,
            return_layer_to_updated=True)

        if self.toronto_neg:
            neg_phase_grads = self._get_toronto_neg(model, layer_to_chains)
        else:
            neg_phase_grads = self._get_standard_neg(model, layer_to_chains)

        return neg_phase_grads, updates

    def get_data_specs(self, model):
        if self.supervised:
            space = CompositeSpace([model.get_input_space(),
                                    model.get_output_space()])
            sources = (model.get_input_source(), model.get_target_source())
            return (space, sources)
        else:
            return (model.get_input_space(), model.get_input_source())


class VariationalCD(BaseCD):
    """
    An intractable cost representing the negative log likelihood of a DBM.
    The gradient of this bound is computed using a markov chain initialized
    with the training example.

    Source: Hinton, G. Training Products of Experts by Minimizing
            Contrastive Divergence
    """

    def _get_positive_phase(self, model, X, Y=None):
        return self._get_variational_pos(model, X, Y), OrderedDict()

    def _get_negative_phase(self, model, X, Y=None):
        """
        d/d theta log Z = (d/d theta Z) / Z
                        = (d/d theta sum_h sum_v exp(-E(v,h)) ) / Z
                        = (sum_h sum_v - exp(-E(v,h)) d/d theta E(v,h) ) / Z
                        = - sum_h sum_v P(v,h)  d/d theta E(v,h)
        """
        layer_to_clamp = OrderedDict([(model.visible_layer, True)])

        layer_to_chains = model.make_layer_to_symbolic_state(self.num_chains,
                                                             self.theano_rng)
        # The examples are used to initialize the visible layer's chains
        layer_to_chains[model.visible_layer] = X
        # If we use supervised training, we need to make sure the targets are
        # also clamped.
        if self.supervised:
            assert Y is not None
            # note: if the Y layer changes to something without linear energy,
            # we'll need to make the expected energy clamp Y in the positive
            # phase
            assert isinstance(model.hidden_layers[-1], dbm.Softmax)
            layer_to_clamp[model.hidden_layers[-1]] = True
            layer_to_chains[model.hidden_layers[-1]] = Y

        model.layer_to_chains = layer_to_chains

        # Note that we replace layer_to_chains with a dict mapping to the new
        # state of the chains
        # We first initialize the chain by clamping the visible layer and the
        # target layer (if it exists)
        layer_to_chains = model.mcmc_steps(layer_to_chains,
                                           self.theano_rng,
                                           layer_to_clamp=layer_to_clamp,
                                           num_steps=1)
        # We then do the required mcmc steps
        layer_to_chains = model.mcmc_steps(layer_to_chains,
                                           self.theano_rng,
                                           num_steps=self.num_gibbs_steps)

        if self.toronto_neg:
            neg_phase_grads = self._get_toronto_neg(model, layer_to_chains)
        else:
            neg_phase_grads = self._get_standard_neg(model, layer_to_chains)

        return neg_phase_grads, OrderedDict()

    def get_data_specs(self, model):
        if self.supervised:
            space = CompositeSpace([model.get_input_space(),
                                    model.get_output_space()])
            sources = (model.get_input_source(), model.get_target_source())
            return (space, sources)
        else:
            return (model.get_input_space(), model.get_input_source())


class MF_L2_ActCost(Cost):
    """
        An L2 penalty on the amount that the hidden unit mean field parameters
        deviate from desired target values.
    """

    def __init__(self, targets, coeffs, supervised=False):

        targets = fix(targets)
        coeffs = fix(coeffs)

        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, return_locals=False, **kwargs):
        """
        If returns locals is True, returns (objective, locals())
        Note that this means adding / removing / changing the value of
        local variables is an interface change.
        In particular, TorontoSparsity depends on "terms" and "H_hat"
        """
        self.get_data_specs(model)[0].validate(data)
        if self.supervised:
            (X, Y) = data
        else:
            X = data
            Y = None

        H_hat = model.mf(X, Y=Y)

        terms = []

        hidden_layers = model.hidden_layers
        #if self.supervised:
        #    hidden_layers = hidden_layers[:-1]

        for layer, mf_state, targets, coeffs in \
                safe_zip(hidden_layers, H_hat, self.targets, self.coeffs):
            try:
                cost = layer.get_l2_act_cost(mf_state, targets, coeffs)
            except NotImplementedError:
                if isinstance(coeffs, float) and coeffs == 0.:
                    cost = 0.
                else:
                    raise
            terms.append(cost)


        objective = sum(terms)

        if return_locals:
            return objective, locals()
        return objective

    def get_data_specs(self, model):
        if self.supervised:
            space = CompositeSpace([model.get_input_space(), model.get_output_space()])
            sources = (model.get_input_source(), model.get_target_source())
            return (space, sources)
        else:
            return (model.get_input_space(), model.get_input_source())

def fix(l):
    if isinstance(l, list):
        return [fix(elem) for elem in l]
    if isinstance(l, str):
        return float(l)
    return l

class TorontoSparsity(Cost):
    """
    TODO: add link to Ruslan Salakhutdinov's paper that this is based on
    """

    def __init__(self, targets, coeffs, supervised=False):
        self.__dict__.update(locals())
        del self.self

        self.base_cost = MF_L2_ActCost(targets=targets,
                coeffs=coeffs, supervised=supervised)

    def expr(self, model, data, return_locals=False, **kwargs):
        self.get_data_specs(model)[0].validate(data)
        return self.base_cost.expr(model, data, return_locals=return_locals,
                **kwargs)

    def get_gradients(self, model, data, **kwargs):
        self.get_data_specs(model)[0].validate(data)
        obj, scratch = self.base_cost.expr(model, data, return_locals=True,
                                           **kwargs)
        if self.supervised:
            assert isinstance(data, (list, tuple))
            assert len(data) == 2
            (X, Y) = data
        else:
            X = data

        H_hat = scratch['H_hat']
        terms = scratch['terms']
        hidden_layers = scratch['hidden_layers']

        grads = OrderedDict()

        assert len(H_hat) == len(terms)
        assert len(terms) == len(hidden_layers)
        num_layers = len(hidden_layers)
        for i in xrange(num_layers):
            state = H_hat[i]
            layer = model.hidden_layers[i]
            term = terms[i]

            if term == 0.:
                continue
            else:
                print 'term is ',term

            if i == 0:
                state_below = X
                layer_below = model.visible_layer
            else:
                layer_below = model.hidden_layers[i-1]
                state_below = H_hat[i-1]
            state_below = layer_below.upward_state(state_below)

            components = flatten(state)

            real_grads = T.grad(term, components)

            fake_state = layer.linear_feed_forward_approximation(state_below)

            fake_components = flatten(fake_state)
            real_grads = OrderedDict(safe_zip(fake_components, real_grads))

            params = list(layer.get_params())
            fake_grads = T.grad(cost=None, consider_constant=flatten(state_below),
                    wrt=params, known_grads = real_grads)

            for param, grad in safe_zip(params, fake_grads):
                if param in grads:
                    grads[param] = grads[param] + grad
                else:
                    grads[param] = grad

        return grads, OrderedDict()

    def get_data_specs(self, model):
        return self.base_cost.get_data_specs(model)


class WeightDecay(Cost):
    """
    coeff * sum(sqr(weights))

    for each set of weights.

    """

    def __init__(self, coeffs):
        """
        coeffs: a list, one element per layer, specifying the coefficient
                to put on the L1 activation cost for each layer.
                Each element may in turn be a list, ie, for CompositeLayers.
        """
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):
        self.get_data_specs(model)[0].validate(data)
        layer_costs = [ layer.get_weight_decay(coeff)
            for layer, coeff in safe_izip(model.hidden_layers, self.coeffs) ]

        assert T.scalar() != 0. # make sure theano semantics do what I want
        layer_costs = [ cost for cost in layer_costs if cost != 0.]

        if len(layer_costs) == 0:
            rval =  T.as_tensor_variable(0.)
            rval.name = '0_weight_decay'
            return rval
        else:
            total_cost = reduce(lambda x, y: x + y, layer_costs)
        total_cost.name = 'DBM_WeightDecay'

        assert total_cost.ndim == 0

        total_cost.name = 'weight_decay'

        return total_cost

    def get_data_specs(self, model):
        # This cost does not use or require data
        return (NullSpace(), '')
