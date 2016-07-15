'''Setup scripts for Cortex.

'''
from collections import OrderedDict
import logging
import readline, glob
from os import path
import urllib2

from .utils.tools import get_paths, _p
from .utils.extra import (
    complete_path, query_yes_no, write_default_theanorc, write_path_conf)


logger = logging.getLogger(__name__)

def get_manager():
    if Manager._instance is None:
        return Manager()
    else:
        return Manager._instance

def resolve_class(cell_type, classes=None):
    from models import _classes
    if classes is None:
        classes = _classes
    try:
        C = classes[cell_type]
    except KeyError:
        raise KeyError('Unexpected cell subclass `%s`, '
                       'available classes: %s' % (cell_type, classes.keys()))
    return C


class Link(object):
    class Node(object):
        def __init__(self, C, key):
            if key not in C._dim_map.keys():
                raise TypeError('Class %s has no key `%s`' % (C, key))
            self.C = C
            self.link_key = key
            self.dim_key = C._dim_map[key]
            self.dist_key = C._distribution

    def __init__(self, cm, f, t):
        self.value = None
        self.distribution = None
        self.nodes = {}
        self.cm = cm
        self.name = f + '->' + t

        def split_arg(arg, idx=-1):
            s = arg.split('.')
            name = '.'.join(s[:idx])
            arg = '.'.join(s[idx:])
            return name, arg

        def get_args(name):
            if name.split('.')[0] in cm.datasets.keys():
                name_, key = split_arg(name)
                C = None
            elif '.'.join(name.split('.')[:-1]) in cm.cell_args.keys():
                name_, key = split_arg(name)
                C = self.cm.resolve_class(cm.cell_args[name_]['cell_type'])
            else:
                raise KeyError('Cell or data %s not found' % name)
            return name_, key, C

        f_name, f_key, f_class = get_args(f)
        t_name, t_key, t_class = get_args(t)

        dataset_name = None
        dataset_key = None

        if f_name in cm.datasets.keys():
            if t_name in cm.datasets.keys():
                raise ValueError('Cannot link 2 datasets')
            dataset_name = f_name
            dataset_key = f_key
        elif t_name in cm.datasets.keys():
            dataset_name = t_name
            dataset_key = t_key

        if dataset_name is not None:
            self.value = cm.datasets[dataset_name]['dims'][dataset_key]
            self.distribution = cm.datasets[dataset_name]['distributions'][dataset_key]
        else:
            t_args = cm.cell_args[t_name]
            f_args = cm.cell_args[f_name]
            try:
                self.value = t_class.set_link_value(t_key, **t_args)
            except ValueError:
                pass
            try:
                self.value = f_class.set_link_value(f_key, **f_args)
            except ValueError:
                pass
            try:
                self.distribution = t_class.set_link_distribution(**t_args)
            except ValueError:
                pass
            try:
                self.distribution = f_class.set_link_distribution(**f_args)
            except ValueError:
                pass

        if self.value is None:
            raise TypeError('Link between %s and %s requires a resolvable '
                            'dimension' % (f, t))

        if f_name != dataset_name:
            self.nodes[f_name] = self.Node(f_class, f_key)
        if t_name != dataset_name:
            self.nodes[t_name] = self.Node(t_class, t_key)
        cm.links.append(self)

    def query(self, name, key):
        if not name in self.nodes.keys():
            raise KeyError('Link does not have node `%s`' % name)

        node = self.nodes[name]
        if key == node.dim_key:
            (vk_, value) = node.C.get_link_value(self, node.link_key)
            if value is None:
                raise ValueError
            return value
        elif key == 'cell_type' or key == node.dist_key[1:]:
            return self.distribution
        else:
            raise KeyError('Link with node `%s` does not support key `%s`'
                           % (name, key))

    def __repr__(self):
        return ('<link>(%s)' % self.name)


class Manager(object):
    '''cortex manager.

    Ensures that connected objects have the right dimensionality as well as
        manages passing the correct tensors as input and cost.

    '''
    _instance = None

    def __init__(self):
        from models import _classes
        from datasets import _classes as _dataset_classes

        self.logger = logging.getLogger(
            '.'.join([self.__module__, self.__class__.__name__]))
        if Manager._instance is not None:
            logger.warn('New `Manager` instance. Old one will be lost.')
        Manager._instance = self
        self.cells = OrderedDict()
        self.cell_args = OrderedDict()
        self.links = []
        self.classes = _classes
        self.dataset_classes = _dataset_classes
        self.tparams = {}
        self.datasets = {}

    def add_cell_class(name, C):
        self.classes[name] = C

    def add_dataset_class(name, C):
        self.dataset_classes[name] = C

    @staticmethod
    def split_ref(ref):
        l = ref.split('.')
        cell_id = '.'.join(l[:-1])
        arg = l[-1]
        return cell_id, arg

    def reset(self):
        self.links = []
        self.tparams = {}
        self.cells = OrderedDict()
        self.cell_args = OrderedDict()
        self.datasets = {}

    def resolve_class(self, cell_type):
        return resolve_class(cell_type, self.classes)

    def match_args(self, cell_name, **kwargs):
        fail_on_mismatch = bool(cell_name in self.cells.keys())

        if fail_on_mismatch and (cell_name not in self.cell_args):
            raise KeyError('Cell args of %s not found but cell already set'
                           % cell_name)
        else:
            self.cell_args[cell_name] = {}

        args = self.cell_args[cell_name]
        for k, v in kwargs.iteritems():
            if k not in args.keys():
                if fail_on_mismatch:
                    raise KeyError('Requested key %s not found in %s and cell '
                                    'already exists.' % (k, cell_name))
                else:
                    args[k] = v
            if args[k] is not None and args[k] != v:
                raise ValueError('Key %s already set and differs from '
                                 'requested value (% vs %s)' % (k, args[k], v))

    def make_data(self, dataset, **kwargs):
        C = resolve_class(dataset, self.dataset_classes)
        C(**kwargs)

    def build(self, name=None):
        if name is not None and name not in self.cells:
            self.build_cell(name)
        else:
            for k, kwargs in self.cell_args.iteritems():
                if k not in self.cells:
                    self.build_cell(k)

    def build_cell(self, name):
        kwargs = self.cell_args[name]
        C = self.resolve_class(kwargs['cell_type'])

        for k in kwargs.keys():
            if isinstance(kwargs[k], Link):
                link = kwargs[k]
                value = link.query(name, k)
                kwargs[k] = value

        C.factory(name=name, **kwargs)

    def prepare_cell(self, cell_type, requestor=None, name=None, **kwargs):
        C = self.resolve_class(cell_type)

        if name is None and requestor is None:
            name = cell_type + '_cell'
        elif name is None:
            name = _p(requestor.name, cell_type)
        elif name is not None and requestor is not None:
            name = _p(requestor.name, name)

        self.match_args(name, cell_type=cell_type, **kwargs)

    def register_cell(self, name=None, cell_type=None, **layer_args):
        if name is None:
            name = cell_type
        if name in self.cells.keys():
            self.logger.warn(
                'Cell with name `%s` already found: overwriting. '
                'Use `cortex.manager.remove_cell` to avoid this warning' % key)
        try:
            self.cell_classes = self.classes[cell_type]
        except KeyError:
            raise TypeError('`cell_type` must be provided. Got %s. Available: '
                            '%s' % (cell_type, self.classes))

        self.cell_args[name] = cell_args

    def remove_cell(self, key):
        del self.cells[key]
        del self.cell_args[key]

    def resolve_links(self, name):
        self.logger.debug('Resolving %s' % name)
        for link in self.links:
            if name in link.members:
                self.logger.debug('Resolving link %s' % link)
                link.resolve()

    def add_link(self, f, t):
        link = Link(self, f, t)
        for name, node in link.nodes.iteritems():
            if name in self.datasets.keys():
                pass
            else:
                if self.cell_args[name].get(node.dim_key, None) is None:
                    self.cell_args[name][node.dim_key] = link
                dk = node.dist_key
                if dk is None:
                    pass
                elif dk.startswith('&'):
                    dk = dk[1:]
                else:
                    dk = 'cell_type'

                if (self.cell_args[name].get(dk, None) is None
                    and node.C._distribution is not None):
                    self.cell_args[name][dk] = link

    def __getitem__(self, key):
        return self.cells[key]

    def __setitem__(self, key, cell):
        from models import Cell

        if key in self.cells.keys():
            self.logger.warn(
                'Cell with name `%s` already found: overwriting. '
                'Use `cortex.manager.remove_cell` to avoid this warning' % key)
        if isinstance(cell, Cell):
            self.cells[key] = cell
            self.cell_args[key] = cell.get_args()
        else:
            raise TypeError('`cell` must be of type %s, got %s'
                            % (Cell, type(cell)))


manager = get_manager()

def main():
    from cortex.datasets import fetch_basic_data
    from cortex.datasets.neuroimaging import fetch_neuroimaging_data

    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind('tab: complete')
    readline.set_completer(complete_path)
    print ('Welcome to Cortex: a deep learning toolbox for '
            'neuroimaging')
    print ('Cortex requires that you enter some paths for '
            'default dataset and output directories. These '
            'can be changed at any time and are customizable '
            'via the ~/.cortexrc file.')

    try:
        path_dict = get_paths()
    except ValueError:
        path_dict = dict()

    if '$data' in path_dict:
        data_path = raw_input(
            'Default data path: [%s] ' % path_dict['$data']) or path_dict['$data']
    else:
        data_path = raw_input('Default data path: ')
    data_path = path.expanduser(data_path)
    if not path.isdir(data_path):
        raise ValueError('path %s does not exist. Please create it.' % data_path)

    if '$outs' in path_dict:
        out_path = raw_input(
            'Default output path: [%s] ' % path_dict['$outs']) or path_dict['$outs']
    else:
        out_path = raw_input('Default output path: ')
    out_path = path.expanduser(out_path)
    if not path.isdir(out_path):
        raise ValueError('path %s does not exist. Please create it.' % out_path)
    write_path_conf(data_path, out_path)

    print ('Cortex demos require additional data that is not necessary for '
           'general use of the Cortex as a package.'
           'This includes MNIST, Caltech Silhoettes, and some UCI dataset '
           'samples.')

    answer = query_yes_no('Download basic dataset? ')

    if answer:
        try:
            fetch_basic_data()
        except urllib2.HTTPError:
            print 'Error: basic dataset not found.'

    print ('Cortex also requires neuroimaging data for the neuroimaging data '
           'for the neuroimaging demos. These are large and can be skipped.')

    answer = query_yes_no('Download neuroimaging dataset? ')

    if answer:
        try:
            fetch_neuroimaging_data()
        except urllib2.HTTPError:
            print 'Error: neuroimaging dataset not found.'

    home = path.expanduser('~')
    trc = path.join(home, '.theanorc')
    if not path.isfile(trc):
        print 'No %s found, adding' % trc
        write_default_theanorc()