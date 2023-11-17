'''Tests the demos.
'''


from os import path

from cortex.demos.demos_basic import classifier
from cortex.demos.demos_basic import rbm_mnist
from cortex.demos.demos_basic import vae
from cortex.utils.tools import load_experiment

d = path.abspath(f'{path.dirname(path.realpath(__file__))}/..')

def test_classifier(epochs=5):
    yaml = path.join(d, 'classifier_mnist.yaml')
    exp_dict = load_experiment(yaml)
    exp_dict['learning_args']['epochs'] = epochs
    exp_dict['dataset_args']['stop'] = 100
    classifier.train(**exp_dict)

def test_rbm(epochs=5, yaml='rbm_mnist.yaml'):
    yaml = path.join(d, 'rbm_mnist.yaml')
    exp_dict = load_experiment(yaml)
    exp_dict['learning_args']['epochs'] = epochs
    exp_dict['dataset_args']['stop'] = 100
    rbm_mnist.train(**exp_dict)

def test_rbm_cifar(epochs=5):
    test_rbm(epochs=epochs, yaml='rbm_cifar.yaml')

def test_vae(epochs=5):
    yaml = path.join(d, 'vae_mnist.yaml')
    exp_dict = load_experiment(yaml)
    exp_dict['learning_args']['epochs'] = epochs
    exp_dict['dataset_args']['stop'] = 100
    vae.train(**exp_dict)