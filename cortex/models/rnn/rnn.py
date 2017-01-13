'''
Module for RNN layers.
'''

import copy
from collections import OrderedDict
import numpy as np
import pprint
import theano
from theano import tensor as T

from .. import Cell, dropout, norm_weight, ortho_weight
from ..extra_layers import Averager
from .. import mlp as mlp_module
from ...costs import squared_error
from ...utils import concatenate, floatX, pi, scan
from ...utils.tools import _p

def unpack(rnn_args, **model_args):
    '''Unpacks a saved RNN.

    See `utils.tools.load_model` for details.

    Args:
        rnn_args (dict): dictionary of model arguments for forming object.
        **model_args: keyword arguments of saved parameters.

    Returns:
        list: list of models.
        dict: dictionary of saved parameters.

    '''

    model = factory(**rnn_args)
    models = [model] + model.get_components()
    return models, model_args, None


class RNNInitializer(Cell):
    '''Initializer for RNNs.

    Currently supports MLP intialization and averager.

    Attributes:
        initialization (str): intialization type.
        dim_in (int): input dimension. For MLP.
        dim_outs (list): hidden dimensions.

    '''
    _required = ['dim_in', 'dim_out']
    _dim_map = {
        'input': 'dim_in',
        'output': 'dim_out'
    }

    def __init__(self, dim_in, dim_out, initialization='MLP',
                 name='rnn_initializer', **kwargs):
        '''Initialization function for RNN_Initializer.

        Args:
            dim_in (int): input dimension. For MLP.
            dim_outs (list): hidden dimensions.
            initialization (str): intialization type.
            **kwargs: keyword arguments for initialization.

        '''
        self.initialization = initialization
        self.dim_in = dim_in
        self.dim_out = dim_out

        super(RNNInitializer, self).__init__(name=name)

    def set_components(self, **kwargs):
        if self.initialization is None:
            init_args = None
        elif self.initialization == 'MLP':
            init_args = {
                'cell_type': 'MLP',
                'dim_in': self.dim_in,
                'dim_out': self.dim_out,
                'dim_hs': [self.dim_in],
                'dropout': 0.1,
                'batch_normalization': False,
                '_passed': ['dim_h', 'dim_hs', 'n_layers', 'h_act', 'dropout'],
                '_required': {'out_act': 'tanh'}
            }
        elif self.initialization == 'Averager':
            init_args = {'cell_type': 'Averager',
                         'shape': (self.dim_out,)}
        else:
            raise TypeError()

        components = {'initializer': init_args}

        return super(RNNInitializer, self).set_components(
            components=components, **kwargs)

    def _cost(self, X=None, H=None):
        '''Call function for RNN_Initializer.

        Updates the initializer or returns cost.

        Args:
            X (T.tensor): input
            hs (list): hidden states

        Returns:
            OrderedDict: cost, mlp output, hs
            theano.OrderedUpdates: updates
            list: constants

        '''
        H = H.copy()
        constants = [H]

        if self.initialization == 'MLP':
            Y = self.initializer(X)['Y']
            cost = squared_error(H[:-1], Y[1:])
            return OrderedDict(cost=cost, constants=constants)
        elif self.initialization == 'Averager':
            updates = self.initializer(H)
            return OrderedDict(updates=updates, constants=constants)

    def _feed(self, X, *params):
        '''Initialize the hidden states.

        Args:
            X (T.tensor): input.

        Returns:
            list: initial states.

        '''
        if self.initialization == 'MLP':
            outs = self.initializer._feed(X, *params)
            outs['output'] = outs['Y']
        elif self.initialization == 'Averager':
            if X.ndim == 1:
                y = params[0]
            elif X.ndim == 2:
                y = T.zeros((X.shape[0], params[0].shape[0])).astype(floatX) + params[0][None, :]
            outs = {'output': y}
        return outs


class RecurrentUnit(Cell):
    _required = ['dim_h']
    _options = {'weight_noise': False}
    _args = ['dim_h']
    _weights = ['W']
    _dim_map = {
        'input': 'dim_h',
        'output': 'dim_h'
    }

    def __init__(self, dim_h, name='RU', **kwargs):
        self.dim_h = dim_h
        super(RecurrentUnit, self).__init__(name=name, **kwargs)

    def init_params(self):
        '''Initialize RNN parameters.

        '''
        self.params = OrderedDict()
        W = ortho_weight(self.dim_h)
        self.params['W'] = W

    def _recurrence(self, m, y, h_, W):
        '''Recurrence function.

        Args:
            m (T.tensor): masks.
            y (T.tensor): inputs.
            h_ (T.tensor): recurrent state.
            W (theano.shared): recurrent weights.

        Returns:
            T.tensor: next recurrent state.

        '''
        preact = T.dot(h_, W) + y
        h      = T.tanh(preact)
        h      = m * h + (1 - m) * h_
        return preact, h

    def _feed(self, X, M, H0, *params):
        n_steps = X.shape[0]
        seqs         = [M[:, :, None], X]
        outputs_info = [None, H0]
        non_seqs     = params

        (preact, h), updates = theano.scan(
            self._recurrence,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=self.name + '_recurrent_steps',
            n_steps=n_steps)

        return OrderedDict(H=h, H_preact=preact, updates=updates)


class RNN(Cell):
    _options = {'dropout': False}
    _components = {
        'initializer': {
            'cell_type': 'RNNInitializer',
            '_passed': ['initialization']
        },
        'RU': {
            'cell_type': '&recurrence_type',
            '_passed': ['weight_noise', 'dim_h']
        },
        'input_net': {
            'cell_type': 'MLP',
            '_required': {'out_act': 'identity'},
            '_passed': ['dim_in']
        }
    }
    _links = [
        ('input_net.output', 'RU.input'),
        ('initializer.output', 'RU.input'),
        ('initializer.input', 'input_net.input')]
    _dim_map = {
        'input': 'dim_in',
        'output': 'dim_h'
    }
    _test_order = ['input_net.Y', 'input_Y', 'H0', 'H', 'output']

    def __init__(self, name='RNN', recurrence_type='RecurrentUnit', **kwargs):
        '''Init function for RNN.

        Args:
            **kwargs: additional keyword arguments.

        '''
        self.recurrence_type = recurrence_type
        super(RNN, self).__init__(name=name, **kwargs)

    def init_args(self, X, M=None):
        if M is None: M = T.ones((X.shape[0], X.shape[1])).astype(floatX)
        return (X, M)
    
    def set_components(self, components=None, **kwargs):
        if components is None:
            components = OrderedDict((k, v) for k, v in self._components.items())
        if self.recurrence_type in ['GRU']:
            components['aux_net'] = dict((k, v)
                for k, v in components['input_net'].iteritems())
            components['aux_net']['dim_out'] = 2 * kwargs['dim_h']
            components['aux_net']['dim_in'] = kwargs['dim_in']
        else:
            components['aux_net'] = None

        return super(RNN, self).set_components(
            components=components, **kwargs)
    
    def feed_in(self, X, *params):
        input_params = self.select_params('input_net', *params)
        outs = self.input_net._feed(X, *input_params)
        outs = OrderedDict((_p('input_net', k), v)
            for k, v in outs.iteritems())
        if self.aux_net is not None:
            aux_params = self.select_params('aux_net', *params)
            outs_a = self.aux_net._feed(X, *aux_params)
            outs_a = OrderedDict((_p('aux_net', k), v)
            for k, v in outs_a.iteritems())
            outs.update(**outs_a)
            outs['input_Y'] = concatenate(
                [outs[_p('aux_net', 'Y')],
                 outs[_p('input_net', 'Y')]],
                axis=X.ndim-1)
        else:
            outs['input_Y'] = outs[_p('input_net', 'Y')]
        return outs

    def _feed(self, X, M, *params):
        ru_params = self.select_params('RU', *params)
        initializer_params = self.select_params('initializer', *params)
        
        outs = self.feed_in(X, *params)
        outs_init = self.initializer._feed(X[0], *initializer_params)
        outs.update(**dict((_p('initializer', k), v)
            for k, v in outs_init.iteritems()))
        outs['H0'] = outs_init['output']

        outs.update(**self.RU(outs['input_Y'], M, outs['H0']))
        H = outs['H'][-1]
        if self.dropout and self.noise_switch():
            H = dropout(H, T.tanh, self.dropout, self.trng)
        outs['output'] = H

        return outs

class GenRNN(RNN):
    _required = ['distribution_type']
    _components = copy.deepcopy(RNN._components)
    _components.update(**
        {
            'output_net': {
                'cell_type': 'DistributionMLP',
                'distribution_type': '&distribution_type'
            }
        })
    _links = RNN._links[:]
    _links += [
        ('output_net.samples', 'input_net.input'),
        ('output_net.input', 'RU.output')]
    _dim_map = copy.deepcopy(RNN._dim_map)
    _dim_map.update(**{
        'output': 'dim_in',
        'P': 'dim_in',
        'samples': 'dim_in'
    })
    _dist_map = {'P': 'distribution_type'}
    _costs = {
        'nll': '_cost',
        'negative_log_likelihood': '_cost'}

    def __init__(self, distribution_type, name='GenRNN', **kwargs):
        self.distribution_type = distribution_type
        super(GenRNN, self).__init__(name=name, **kwargs)

    def _feed(self, X, M, *params):
        outs = super(GenRNN, self)._feed(X, M, *params)
        H = outs['H']
        output_params = self.select_params('output_net', *params)
        outs_out_net = self.output_net._feed(H, *output_params)
        outs.update(
            **dict((_p('output_net', k), v)
                for k, v in outs_out_net.iteritems()))
        outs['P'] = outs[_p('output_net', 'P')]
        return outs

    def _cost(self, X=None, P=None):
        if P is None:
            session = self.manager.get_session()
            P = session.tensors[_p(self.name, 'P')]
        nll = self.output_net.distribution.neg_log_prob(X[1:], P=P[:-1])
        return nll.sum(axis=0).mean()

    # Step functions -----------------------------------------------------------
    def _step_sample(self, X, H, epsilon, *params):
        '''Returns preact for sampling step.

        '''
        ru_params = self.select_params('RU', *params)
        input_params = self.select_params('input_net', *params)
        initializer_params = self.select_params('initializer', *params)
        output_params = self.select_params('output_net', *params)

        Y = self.input_net._feed(X, *input_params)['output']
        H = self.RU._recurrence(1, Y, H, *ru_params)
        P_ = self.output_net._feed(H, *output_params)['output']
        X_ = self.output_net._sample(epsilon, P=P_)

        return H, X_, P_

    def generate_random_variables(self, shape, P=None):
        if P is None:
            P = T.zeros((self.output_net.mlp.dim_out,)).astype(floatX)
        return self.output_net.generate_random_variables(shape, P=P)

    def _sample(self, epsilon, X0=None, H0=None, P=None):
        '''Samples from an initial state.

        Args:
            x0 (Optional[T.tensor]): initial input state.
            h0 (Optional[T.tensor]): initial recurrent state.
            n_samples (Optional[int]): if no x0 or h0, used to initial batch.
                Number of chains.
            n_steps (int): number of sampling steps.

        Returns:
            OrderedDict: dictionary of results. hiddens, probabilities, and preacts.
            theano.OrderedUpdates: updates.

        '''
        if X0 is None: X0 = self.output_net.simple_sample(epsilon.shape[1], P=0.5)
        if H0 is None: H0 = self.initializer(X0)['output']

        seqs = [epsilon]
        outputs_info = [X0, H0, None]
        non_seqs = self.get_params()

        (H, X, P), updates = scan(
            self._step_sample, seqs, outputs_info, non_seqs, epsilon.shape[0],
            name=self.name+'_sampling')

        return OrderedDict(samples=X, P=P, H=H, updates=updates)

    def get_center(self, P):
        return self.output_net.distribution.get_center(P)
    
    
class GenRNNwithContext(RNN):
    _required = ['distribution_type']
    _components = copy.deepcopy(RNN._components)
    _components.update(**
        {
            'output_net': {
                'cell_type': 'DistributionMLP',
                'distribution_type': '&distribution_type'
            },
            'context_net': {
                'cell_type': 'MLP',
                'dim_in': '&c_dim',
                '_required': {'out_act': 'identity'},
            }
        })
    _links = RNN._links[:]
    _links += [
        ('output_net.samples', 'input_net.input'),
        ('output_net.input', 'RU.output'),
        ('context_net.output', 'RU.input')]
    _dim_map = copy.deepcopy(RNN._dim_map)
    _dim_map.update(**{
        'output': 'dim_in',
        'P': 'dim_in',
        'samples': 'dim_in'
    })
    _dist_map = {'P': 'distribution_type'}
    _costs = {
        'nll': '_cost',
        'negative_log_likelihood': '_cost'}

    def __init__(self, distribution_type, c_dim=None, name='GenRNN', **kwargs):
        self.distribution_type = distribution_type
        self.c_dim = c_dim
        super(GenRNNwithContext, self).__init__(name=name, **kwargs)
        
    def init_args(self, X, C, M=None):
        if M is None: M = T.ones((X.shape[0], X.shape[1])).astype(floatX)
        return (X, C, M)

    def _feed(self, X, C, M, *params):
        ru_params = self.select_params('RU', *params)
        initializer_params = self.select_params('initializer', *params)
        context_params = self.select_params('context_net', *params)
        
        outs = self.feed_in(X, *params)
        outs_init = self.initializer._feed(X[0], *initializer_params)
        outs.update(**dict((_p('initializer', k), v)
            for k, v in outs_init.iteritems()))
        outs['H0'] = outs_init['output']
        input_Y = outs['input_Y']
        input_Y += self.context_net._feed(C, *context_params)['output']

        outs.update(**self.RU(input_Y, M, outs['H0']))
        H = outs['H'][-1]
        if self.dropout and self.noise_switch():
            H = dropout(H, T.tanh, self.dropout, self.trng)
        outs['output'] = H
        
        output_params = self.select_params('output_net', *params)
        outs_out_net = self.output_net._feed(H, *output_params)
        outs.update(
            **dict((_p('output_net', k), v)
                for k, v in outs_out_net.iteritems()))
        outs['P'] = outs[_p('output_net', 'P')]
        return outs

    def _cost(self, X=None, P=None):
        if P is None:
            session = self.manager.get_session()
            P = session.tensors[_p(self.name, 'P')]
        nll = self.output_net.distribution.neg_log_prob(X[1:], P=P[:-1])
        return nll.sum(axis=0).mean()

    # Step functions -----------------------------------------------------------
    def _step_sample(self, epsilon, H, X, C, *params):
        '''Returns preact for sampling step.

        '''
        ru_params = self.select_params('RU', *params)
        input_params = self.select_params('input_net', *params)
        initializer_params = self.select_params('initializer', *params)
        output_params = self.select_params('output_net', *params)
        context_params = self.select_params('context_net', *params)

        Y = self.input_net._feed(X, *input_params)['output'] + C
        H = self.RU._recurrence(1, Y, H, *ru_params)[1]
        P_ = self.output_net._feed(H, *output_params)['output']
        X_ = self.output_net._sample(epsilon, P=P_)

        return H, X_, P_

    def generate_random_variables(self, shape, P=None):
        if P is None:
            P = T.zeros((self.output_net.mlp.dim_out,)).astype(floatX)
        return self.output_net.generate_random_variables(shape, P=P)

    def _sample(self, epsilon, Z, X0=None, H0=None):
        '''Samples from an initial state.

        Args:
            x0 (Optional[T.tensor]): initial input state.
            h0 (Optional[T.tensor]): initial recurrent state.
            n_samples (Optional[int]): if no x0 or h0, used to initial batch.
                Number of chains.
            n_steps (int): number of sampling steps.

        Returns:
            OrderedDict: dictionary of results. hiddens, probabilities, and preacts.
            theano.OrderedUpdates: updates.

        '''
        if X0 is None: X0 = self.output_net.simple_sample(epsilon.shape[1], P=0.5)
        if H0 is None: H0 = self.initializer(X0)['output']
        C = self.context_net(Z)['output']

        seqs = [epsilon]
        outputs_info = [H0, X0, None]
        non_seqs = [C] + list(self.get_params())

        (H, X, P), updates = scan(
            self._step_sample, seqs, outputs_info, non_seqs, epsilon.shape[0],
            name=self.name+'_sampling')

        return OrderedDict(samples=X, P=P, H=H, C=C, updates=updates)

    def get_center(self, P):
        return self.output_net.distribution.get_center(P)
    

_classes = {'RNNInitializer': RNNInitializer,
            'RecurrentUnit': RecurrentUnit,
            'GenRNNwithContext': GenRNNwithContext,
            'RNN': RNN,
            'GenRNN': GenRNN}