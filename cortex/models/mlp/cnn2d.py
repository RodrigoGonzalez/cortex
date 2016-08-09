'''Convolutional NN

'''

import numpy as np
from theano import tensor as T

from . import resolve_nonlinearity
from .. import batch_normalization, Cell, dropout, norm_weight


class CNN2D(Cell):
    _required = ['input_shape', 'n_filters', 'filter_shapes', 'pool_sizes']
    _options = {'dropout': False, 'weight_noise': 0,
                'batch_normalization': False}
    _args = ['input_shape', 'n_filters', 'filter_shapes', 'pool_sizes']

    def __init__(self, input_shape, n_filters, filter_shapes, pool_sizes,
                 h_act='sigmoid', out_act=None, name='CNN2D', **kwargs):
        if not(len(n_filters) == len(filter_shapes)):
            raise TypeError(
            '`filter_shapes` and `n_filters` must have the same length')

        self.input_shape = input_shape
        self.filter_shapes = filter_shapes
        self.n_filters = n_filters
        self.n_layers = len(self.n_filters)
        self.pool_sizes = pool_sizes
        self.h_act = resolve_nonlinearity(h_act)

        super(CNN2D, self).__init__(name=name, **kwargs)

    @classmethod
    def set_link_value(C, key, input_shape=None, filter_shapes=None,
                       n_filters=None, pool_sizes=None):

        if key not in ['output']:
            return super(CNN2D, C).set_link_value(link, key)
        if link.value is None:
            raise ValueError
        if n_filters is None: raise ValueError('n_filters')
        if input_shape is None: raise ValueError('input_shape')
        if filter_shapes is None: raise ValueError('filter_shapes')
        if pool_sizes is None: raise ValueError('pool_sizes')

        for filter_shape, pool_size in zip(filter_shapes, pool_sizes):
            dim_x = (input_shape[0] - shape[0] + 1) // pool_size
            dim_y = (input_shape[1] - shape[1] + 1) // pool_size
            input_shape = (dim_x, dim_y)

        return dim_x * dim_y * n_filters[-1]

    def init_params(self, weight_scale=1e-3):
        dim_ins = [self.input_shape[0]] + self.n_filters[:-1]
        dim_outs = self.n_filters

        weights = []
        biases = []

        for dim_in, dim_out, (dim_x, dim_y) in zip(
            dim_ins, dim_outs, self.filter_shapes):
            W = norm_weight(dim_in, dim_out, dim_x, dim_y)
            b = np.zeros((dim_out,))
            weights.append(W)
            biases.append(b)

        self.params = dict(weights=weights, biases=biases)

    def get_params(self):
        params = zip(self.weights, self.biases)
        params = [i for sl in params for i in sl]
        return super(CNN2D, self).get_params(params=params)

    def _feed(self, X, *params):
        session = self.manager._current_session
        params = list(params)
        outs = OrderedDict(X=X)
        outs['input'] = X

        X = X.reshape((X.shape[0], self.input_shape[0], self.input_shape[1]))
        input_shape = X.shape

        for l in xrange(self.n_filters):
            if self.batch_normalization:
                self.logger.debug('Batch normalization on layer %d' % l)
                X = batch_normalization(X, session=session)

            W = params.pop(0)
            b = params.pop(0)
            shape = self.filter_shapes[l]
            pool_size = self.pool_sizes[l]

            conv_out = T.nnet.conv2d(input=X, filters=W, filter_shape=shape,
                                     input_shape=input_shape)
            dim_x = input_shape[0] - shape[0] + 1
            dim_y = input_shape[1] - shape[1] + 1
            pool_out = T.signal.pool(input=conv_out, ds=pool_size,
                                     ignore_borders=True)
            dim_x = dim_x // pool_size[0]
            dim_y = dim_y // pool_size[1]

            preact = pooled_out + self.b[None, :, None, None]
            X = self.h_act(preact)

            if self.dropout and self.noise_switch():
                self.logger.debug('Adding dropout to layer {layer} for CNN2D '
                              '`{name}`'.format(layer=l, name=self.name))
                X = dropout(X, self.h_act, self.dropout, self.trng)

            outs.update(**{
                ('G_%d' % l): preact,
                ('H_%d' % l): X,
                ('C_%d' % l): conv_out,
                ('P_%d' % l): pool_out})

            input_shape = (dim_x, dim_y)

        X = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
        outs['output'] = X

        assert len(params) == 0

_classes = {'CNN2D': CNN2D}