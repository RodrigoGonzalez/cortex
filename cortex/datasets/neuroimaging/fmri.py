'''
Module for fMRI data
'''

import cPickle
from collections import OrderedDict
from glob import glob
import nipy
from nipy.core.api import Image
import numpy as np
import os
from os import path
import pprint
import random
import theano
import yaml

from ...analysis.mri import rois
from .. import Dataset, make_one_hot
from .mri import MRI
from . import nifti_viewer
from ...utils import floatX
from ...utils.tools import resolve_path


class FMRI_IID(MRI):
    '''fMRI data treated as IID.

    Use this dataset if you plan to use a model that needs identical and
    independently sampled data.

    Attributes:
        extras (dict): dictionary of additional arrays for analysis.

    '''
    def __init__(self, name='fmri_iid', **kwargs):
        super(FMRI_IID, self).__init__(name=name, **kwargs)

    def get_data(self, source):
        '''Fetch the fMRI dataset.

        fMRI dataset source is a yaml file.
        An example format of said yaml is::

            name: 'aod',
            data: [
                '/Users/devon/Data/AOD/AOD_0.npy',
                '/Users/devon/Data/AOD/AOD_1.npy'
            ],
            mask: '/Users/devon/Data/AOD/AOD_mask.npy',
            nifti: '//Users/devon/Data/VBM/H000A.nii',
            tmp_path: '/Users/devon/Data/mri_tmp/',
            anat_file: '/Users/devon/Data/ch2better_whitebg_aligned2EPI_V4.nii',

        '''
        print('Loading file locations from %s' % source)
        source_dict = yaml.load(open(source))
        print('Source locations: %s' % pprint.pformat(source_dict))

        def unpack_source(name=None, nifti=None, mask=None, anat_file=None,
                          tmp_path=None, pca=None, data=None, **kwargs):
            return (name, nifti, mask, anat_file, tmp_path, pca, data, kwargs)

        (name, nifti_file, mask_file, self.anat_file,
         self.tmp_path, self.pca_file, data_files, extras) = unpack_source(
            **source_dict)

        self.base_nifti_file = nifti_file
        if not path.isdir(self.tmp_path):
            os.mkdir(self.tmp_path)

        mask = np.load(mask_file)
        if not np.all(np.bitwise_or(mask == 0, mask == 1)):
            raise ValueError("Mask has incorrect values.")
        self.mask = mask

        if self.pca_file is not None:
            try:
                with open(self.pca_file, 'rb') as f:
                    self.pca = cPickle.load(f)
            except (IOError, EOFError):
                self.pca = None
        else:
            self.pca = None

        self.extras = dict((k, np.load(v)) for k, v in extras.iteritems())

        if isinstance(data_files, str):
            data_files = [data_files]
        X = []
        Y = []
        for i, data_file in enumerate(data_files):
            print 'Loading %s' % data_file
            X_ = np.load(data_file)
            X.append(X_.astype(floatX))
            Y.append((np.zeros((X_.shape[0] * X_.shape[1],)) + i).astype(floatX))

        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)

        self.n_subjects, self.n_scans, _, _, _ = X.shape
        X = X.reshape((X.shape[0] * X.shape[1],) + X.shape[2:])

        return X, Y


class FMRI(FMRI_IID):
    '''fMRI dataset class.

    Treats fMRI as sequences, instead as IID as with FMRI_IID.

    Attributes:
        window (int): window size of fMRI batches.
        stride (int): stride of fMRI batches.
        n (int): number of subjects.
        idx (list): indices of subject, scan-window pairs.

    '''

    def __init__(self, name='fmri', window=10, stride=1, idx=None, **kwargs):
        '''Init function for fMRI.

        Args:
            name (str): name of dataset.
            window (int): window size of fMRI batches.
            stride (int): stride of fMRI batches.
            idx (list): indices of dataset (subjects).
            **kwargs: keyword arguments for initializaiton.

        '''
        super(FMRI, self).__init__(name=name, **kwargs)

        self.window = window
        self.stride = stride

        self.X = self.X.reshape((self.n_subjects, self.n_scans, self.X.shape[1]))
        self.Y = self.Y.reshape((self.n_subjects, self.n_scans, self.Y.shape[1]))

        if idx is not None:
            self.X = self.X[idx]
            self.Y = self.Y[idx]
            self.n_subjects = len(idx)

        scan_idx = range(0, self.n_scans - window + 1, stride)
        scan_idx_e = scan_idx * self.n_subjects
        subject_idx = range(self.n_subjects)
        # Similar to np.repeat, but using list comprehension.
        subject_idx_e = [i for j in [[s] * len(scan_idx) for s in subject_idx]
                         for i in j]
        # idx is list of (subject, scan)
        self.idx = zip(subject_idx_e, scan_idx_e)
        self.n = len(self.idx)

        if self.shuffle:
            self.randomize()

    def randomize(self):
        '''Randomize the fMRI dataset.

        Shuffles the idx.

        '''
        rnd_idx = np.random.permutation(np.arange(0, self.n, 1))
        self.idx = [self.idx[i] for i in rnd_idx]

    def next(self, batch_size=None):
        '''Draws the next batch of windowed fMRI.

        Args:
            batch_size (Optional[int]): number of windows in batch.

        Returns:
            dict: dictionary of batched data

        '''
        if batch_size is None:
            batch_size = self.batch_size

        if self.pos == -1:
            self.reset()
            raise StopIteration

        idxs = [self.idx[i] for i in range(self.pos, self.pos+batch_size)]
        x = np.array([self.X[i][j:j+self.window] for i, j in idxs]).astype(floatX).transpose(1, 0, 2)
        y = np.array([self.Y[i][j:j+self.window] for i, j in idxs]).astype(floatX).transpose(1, 0, 2)

        self.pos += batch_size

        if self.pos + batch_size > self.n:
            self.pos = -1

        return {self.name: x, 'group': y}
