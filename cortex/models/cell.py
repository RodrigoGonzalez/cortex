'''Base Cell class.

'''

from collections import OrderedDict
import copy
import logging
import numpy as np
from pprint import pprint
import random
import re
import theano
from theano import tensor as T

from ..utils import floatX, _rng
from ..utils.logger import get_class_logger
from ..utils.tools import warn_kwargs, get_trng, _p


logger = logging.getLogger(__name__)


def init_rngs(cell):
    '''Initialization function for RNGs.

    Args:
        cell (Cell).

    '''
    cell.rng = _rng
    cell.trng = get_trng()

def ortho_weight(ndim, rng=None):
    '''Make ortho weight tensor.

    '''
    if not rng: rng = _rng
    W = rng.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(floatX)

def norm_weight(nin, nout=None, scale=0.01, ortho=True, rng=None):
    '''Make normal weight tensor.

    '''
    if not rng:
        rng = _rng
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin, rng=rng)
    else:
        W = scale * rng.randn(nin, nout)
    return W.astype(floatX)

def dropout(x, act, rate, trng, epsilon=None):
        if epsilon is None:
            epsilon = trng.binomial(x.shape, p=1-rate, n=1, dtype=x.dtype)
        if act == T.tanh:
            x = 2. * (epsilon * (x + 1.) / 2) / (1 - rate) - 1
        elif act in [T.nnet.sigmoid, T.nnet.softplus, T.nnet.relu]:
            x = x * epsilon / (1 - rate)
        else:
            raise NotImplementedError('No dropout for %s yet' % activ)
        return x

def batch_normalization(x, gamma, beta, epsilon=1e-5, session=None):
    if x.ndim == 1:
        mu = 0.
        sigma = 1.
    elif x.ndim == 2:
        mu = x.mean(axis=0, keepdims=True)
        sigma = x.std(axis=0, keepdims=True)
    elif x.ndim == 3:
        mu = x.mean(axis=(0, 1), keepdims=True)
        sigma = x.std(axis=(0, 1), keepdims=True)
    else:
        raise TypeError(x.ndim)
    y = gamma * (x - mu) / T.sqrt(sigma ** 2 + epsilon) + beta
    return y


class NoiseSwitch(object):
    '''Object to control noise of model.

    '''
    _instance = None
    def __init__(self):
        if NoiseSwitch._instance is None:
            NoiseSwitch._instance = self
        self.noise = True

    def switch(self, switch_to):
        ons = ['on', 'On', 'ON']
        offs = ['off', 'Off', 'OFF']
        if switch_to in ons:
            self.noise = True
        elif switch_to in offs:
            self.noise = False

    def __call__(self):
        return self.noise

def get_noise_switch():
    if NoiseSwitch._instance is None:
        return NoiseSwitch()
    else:
        return NoiseSwitch._instance
    
    
def step_method(inputs=None, options=None, outputs=None):
    inputs = inputs or []
    options = options or []
    outputs = outputs or [] # Never default to lists or dicts, or changing them in function will change the defaults.
    
    def decorate(step):
        def do_step(*args, **kwargs):
            
            if len(args) != len(inputs):
                raise ValueError('Wrong number of inputs')
            
            for arg, input_ in zip(args, inputs):
                pass
                #<check dims or do some matching, make sure input_ actually exists>
            for k in kwargs.keys():
                if k not in options: raise ValueError('Unknown option')
            
            cortex.add_step(step, args=args, kwargs=kwargs) # registers the step and its arguments
            for output in outputs:
                cortex.add_tensor(output) # registers the dummy tensors so that other functions can use them, adds them to the namespace.
            
            return # doesn't actually return anything
        
        return do_step
    return decorate


class Cell(object):
    '''Base class for all models.

    Attributes:
        name (str): name identifier of cell.
        params (dict): dictionary of numpy.arrays
        learn (bool): if False, do not change params.
        n_params (int): number of parameters

    '''
    _components = {}    # Cells that this cell controls.
    _options = {'weight_noise': 0}      # Dictionary of optional arguments and default values
    _required = []      # Required arguments for __init__
    _args = ['name']    # Arguments necessary to uniquely id the cell. Used for
                        #   save.
    _dim_map = {}       #
    _links = []
    _dist_map = {}
    _call_args = ['input']
    _costs = {}
    _evals = {}
    _weights = []
    _test_order = None
    _sample_tensors = []

    noise_switch = get_noise_switch()

    def __init__(self, name='cell_proto', inits=None, excludes=None, **kwargs):
        '''Init function for Cell.

        Args:
            name (str): name identifier of cell.

        '''
        from .. import _manager as manager

        self.passed = {}
        self.name = name
        self.manager = manager
        self.inits = inits or dict()
        self.excludes = excludes or []
        kwargs = self.set_options(**kwargs)
        init_rngs(self)

        self.logger = get_class_logger(self)
        self.logger.debug(
            'Forming model cell %r with name %s' % (self.__class__, name))
        self.logger.debug('Formation parameters: %s' % self.get_args())

        kwargs = self.set_components(**kwargs)
        self.init_params(**kwargs)
        self.logger.debug('Parameters and shapes: %s' % self.profile_params())
        self.register()

        self.n_params = 0
        for param in self.params.values():
            if isinstance(param, list):
                self.n_params += len(param)
            else:
                self.n_params += 1
        self.n_component_params = 0
        for key in self.component_keys:
            component = self.__dict__[key]
            if component is None: continue
            if isinstance(component, list):
                for c_ in component:
                    self.n_component_params += c_.total_params
            else:
                self.n_component_params += component.total_params
        self.total_params = self.n_params + self.n_component_params
        self.set_tparams()

    def set_options(self, **kwargs):
        for k, v in self._options.iteritems():
            self.__dict__[k] = kwargs.pop(k, v)
        return kwargs

    def profile_params(self):
        d = OrderedDict()
        for k, p in self.params.iteritems():
            if isinstance(p, list):
                for i, pp in enumerate(p):
                    d['%s[%d]' % (k, i)] = pp.shape
            else:
                d[k] = p.shape
        return d

    @classmethod
    def set_link_value(C, key, **kwargs):
        from .. import manager
        logger.debug('Setting link value for class _dim_map %s with key `%s` and'
                    ' kwargs %s' % (C._dim_map, key, kwargs))
        if key in C._dim_map.keys():
            value = kwargs.get(C._dim_map[key], None)
            if not isinstance(value, manager.link.Link) and value is not None:
                return value
            elif isinstance(value, manager.link.Link) and value.value is not None:
                return value.value
            else:
                raise ValueError
        else:
            raise KeyError

    @classmethod
    def get_link_value(C, link, key):
        logger.debug('Attempting to get link value for cell class %s from `%s` '
                     'with key `%s`' % (C, link, key))
        if key in C._dim_map.keys():
            if link.value is None:
                raise ValueError
            else:
                return (C._dim_map[key], link.value)
                logger.debug('Resulting kwargs: %s')
        else:
            raise KeyError

    @classmethod
    def set_link_distribution(C, key, **kwargs):
        from .. import manager

        logger.debug('Setting link distribution for class _dist_map %s with key '
                    '`%s` and kwargs %s' % (C._dist_map, key, kwargs))
        if key in C._dist_map.keys():
            value = kwargs.get(C._dist_map[key], None)
            if not isinstance(value, manager.link.Link) and value is not None:
                return value
            else:
                raise ValueError
        else:
            raise KeyError

    @classmethod
    def factory(C, cell_type=None, **kwargs):
        '''Cell factory.

        Convenience function for building Cells.

        Args:
            **kwargs: construction keyword arguments.

        Returns:
            C

        '''
        reqs = OrderedDict(
            (k, kwargs[k]) for k in C._required if k in kwargs.keys())
        logger.debug('Required args for %s found: %s' % (C, reqs))
        options = dict((k, v) for k, v in kwargs.iteritems() if not k in C._required)

        for req in C._required:
            if req not in reqs.keys() or reqs[req] is None:
                raise TypeError('Required argument %s not provided for '
                                'constructor of %s or is `None`. Got %s'
                                % (req, C, kwargs))

        return C(*reqs.values(), **options)

    def register(self):
        self.manager[self.name] = self

    def set_components(self, components=None, **kwargs):
        from ..utils.tools import _p
        if components is None: components = self._components
        self.component_keys = components.keys()

        for k, v in components.iteritems():
            if v is None: continue
            new_args = kwargs.pop(k, {})
            args = dict((k_, v_) for k_, v_ in v.iteritems())
            args.update(**new_args)

            passed = args.pop('_passed', [])
            if 'cell_type' in args.keys():
                cell_type = args['cell_type']
                if cell_type.startswith('&'):
                    cell_type = self.__dict__[cell_type[1:]]
                C = self.manager.resolve_class(cell_type)
                passed += C._args

            for p in passed: self.passed[p] = k

            # Required arguments
            required = args.pop('_required', dict())
            args.update(**required)

            # Arguments passed as arguments to owner.
            passed_args = dict((kk, kwargs[kk])
                for kk in passed
                if kk in kwargs.keys())
            kwargs = dict((kk, kwargs[kk]) for kk in kwargs.keys()
                if kk not in passed)
            args.update(**passed_args)

            # Leading `&` indicates reference to owner attribute
            final_args = {}
            for kk, vv in args.iteritems():
                if isinstance(vv, str) and vv.startswith('&'):
                    final_args[kk] = self.__dict__[vv[1:]]
                else:
                    final_args[kk] = vv
                    
            self.manager.prepare_cell(name=k, requestor=self, **final_args)

        for f, t in self._links:
            f = _p(self.name, f)
            t = _p(self.name, t)
            self.manager.match_dims(f, t)

        for k, v in components.iteritems():
            if v is None:
                comp = None
            else:
                name = _p(self.name, k)
                self.manager.build_cell(name)
                comp = self.manager[name]
            self.__dict__[k] = comp

        return kwargs

    def copy(self):
        '''Copy the cell.

        '''
        return copy.deepcopy(self)

    def init_params(self, **init_kwargs):
        '''Initialize the parameters.

        '''
        self.params = OrderedDict()

    def set_tparams(self):
        '''Sets the tensor parameters.

        '''
        self.param_keys = []
        if self.params is None: raise ValueError('Params not set yet')
        tparams = OrderedDict()

        for k, p in self.params.iteritems():
            if isinstance(p, list):
                self.__dict__[k] = []
                for i, pp in enumerate(p):
                    kk = '%s[%d]' % (k, i)
                    name = _p(self.name, kk)
                    if name not in self.manager.tparams.keys():
                        tp = theano.shared(pp.astype(floatX), name=name)
                        self.manager.tparams[name] = tp
                    else:
                        tp = self.manager.tparams[name]
                    self.__dict__[k].append(tp)
                    self.param_keys.append(kk)
            else:
                name = _p(self.name, k)
                if name not in self.manager.tparams.keys():
                    tp = theano.shared(p.astype(floatX), name=name)
                    self.manager.tparams[name] = tp
                else:
                    tp = self.manager.tparams[name]
                self.__dict__[k] = tp
                self.param_keys.append(k)

    def get_params(self, params=None):
        if params is None: params = [self.__dict__[k] for k in self.param_keys]
        if self.noise_switch() and self.weight_noise:
            for i in range(len(params)):
                param = params[i]
                for w in self._weights:
                    if w in param.name:
                        self.logger.debug(
                            'Adding weight noise (%.2e) to %s'
                            % (self.weight_noise, param.name))
                        param += self.trng.normal(
                            avg=0.,
                            std=self.weight_noise,
                            size=param.shape).astype(floatX)
                        break

                self.params[i] = param

        for key in self.component_keys:
            component = self.__dict__[key]
            if component is None: continue
            if isinstance(component, (list, tuple)):
                for c_ in component:
                    c_params = c_.get_params()
                    params += c_params
            else:
                c_params = component.get_params()
                params += c_params
        return params

    def select_params(self, key, *params):
        params = list(params)
        start = 0
        end = 0
        if key is None:
            end = self.n_params
        else:
            start = self.n_params
            found = False
            for k in self.component_keys:
                component = self.__dict__[k]
                if component is None: continue
                if isinstance(component, (list, tuple)):
                    for i in xrange(len(component)):
                        l = component[i].total_params
                        k_ = '{comp}_{index}'.format(comp=k, index=i)
                        if k_ == key:
                            end = start + l
                            found = True
                            break
                        else:
                            start = start + l
                else:
                    l = component.total_params
                    if k == key:
                        end = start + l
                        found = True
                        break
                    else:
                        start = start + l
            if not found:
                raise KeyError('Component `%s` not found' % key)

        return params[start:end]

    def get_n_params(self):
        return self.n_params

    def get_args(self):
        d = dict((k, self.__dict__[k]) for k in self._args)
        try:
            c = next(c for c, v in self.manager.classes.iteritems()
                     if v == self.__class__)
        except StopIteration:
            raise

        d['cell_type'] = c
        return d

    def _feed(self, *args, **kwargs):
        '''Basic feed method.

        This is the identity graph. Generally it is `scan` safe.

        Args:
            args (list): list of tensor inputs.

        Returns:
            OrderedDict: theano tensor variables.

        '''
        return OrderedDict(('X_%d' % i, args[i]) for i in range(len(args)))

    def init_args(self, *args, **kwargs):
        return args

    def __call__(self, *args, **kwargs):
        '''Call function.

        Args:
            args (list): list of tensor inputs.

        Returns:
            OrderedDict: theano tensor variables.

        '''
        params = tuple(self.get_params())
        args = self.init_args(*args, **kwargs)
        return self._feed(*(args + params))

    def get_components(self):
        '''Gets cell components.

        '''
        components = []
        for k in self._components:
            component = getattr(self, k)
            if component is None: continue
            if isinstance(component, list):
                components += [c for c in component if c is not None]
            elif component is not None:
                components.append(component)

        c_components = []
        for component in components:
            if component is None: continue
            c_components += component.get_components()
        components += c_components
        return components


    def help(self):
        pprint(self._help)

    @classmethod
    def get_arg_reference(C, key, kwargs):
        try:
            k = C._arg_map[key]
        except KeyError:
            raise KeyError('cell %s has no argument %s. Available arguments: '
                           '%s' % (C, key, C._arg_map))
        return kwargs[k]

    def __getattr__(self, key):
        if key in ['passed', 'components']:
            raise AttributeError(key)

        if not hasattr(self, 'passed') or key not in self.passed.keys():
            raise AttributeError('Cell of type %s has no attribute %s'
                                 % (type(self), key))
        component = self.__dict__[self.passed[key]]
        return object.__getattribute__(component, key)

    def __str__(self):
        attributes = {}
        attributes.update(**self.__dict__)
        for k in ['trng', 'manager', 'passed', 'component_keys', 'rng',
                  'logger']:
            attributes.pop(k, None)

        params = {}
        params.update(**attributes['params'])

        for k, v in params.iteritems():
            if isinstance(v, np.ndarray):
                params[k] = '<numpy.ndarray: {shape: %s}>' % (v.shape,)
            elif isinstance(v, list):
                params[k] = [
                    '<numpy.ndarray: {shape: %s}>' % (a.shape,) for a in v]
        attributes.update(params=params)
        attr_str = ''
        
        for k, a in attributes.iteritems():
            if k in self._components and a is not None:
                if isinstance(a, (list, tuple)):
                    c_str = ': '
                    for a_ in a:
                        c_str += '<' + a_.name + '>'
                else:
                    c_str = ': <' + a.name + '>'
                new_str = '\n\t' + k + c_str
            else:
                new_str = '\n\t%s: %s' % (k, a)
            attr_str += new_str

        s = ('<Cell %s: %s>' % (self.__class__.__name__, attr_str))
        return s

_classes = {'Cell': Cell}
