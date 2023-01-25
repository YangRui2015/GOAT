"""
Adapted from https://github.com/spitis/mrl/blob/master/mrl/modules/density.pyat
Density modules for estimating density of items in the replay buffer (e.g., states / achieved goals).
"""

import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.special import entr
import os
import pickle
import multiprocessing
from wgcsl.common import logger


class KernelDensityModule:
    """
    A KDE-based density model for raw items in the replay buffer (e.g., states/goals).
    """
    def __init__(self, kernel='gaussian', bandwidth=0.2, normalize=True, log_entropy=False):
        self.kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.normalize = normalize
        self.kde_sample_mean = 0.
        self.kde_sample_std = 1.
        self.fitted_kde = None
        self.ready = False
        self.log_entropy = log_entropy 
        self.prop_names = ['kde', 'kde_sample_mean', 'kde_sample_std', 'fitted_kde', 'ready']

    def fit(self, kde_samples):
        if len(kde_samples):
            self.ready = True
            if self.normalize:
                self.kde_sample_mean = np.mean(kde_samples, axis=0, keepdims=True)
                self.kde_sample_std  = np.std(kde_samples, axis=0, keepdims=True) + 1e-4
                kde_samples = (kde_samples - self.kde_sample_mean) / self.kde_sample_std

            self.fitted_kde = self.kde.fit(kde_samples)

            # Now log the entropy
            if self.log_entropy:
                # Scoring samples is a bit expensive, so just use 500 points
                num_samples = 500
                s = self.fitted_kde.sample(num_samples)
                entropy = -self.fitted_kde.score(s)/num_samples + np.log(self.kde_sample_std).sum()
                logger.log('{}_entropy'.format('KDE'), entropy, log_every=500)

    def evaluate_log_density(self, samples):
        assert self.ready, "ENSURE READY BEFORE EVALUATING LOG DENSITY"
        return self.fitted_kde.score_samples( (samples  - self.kde_sample_mean) / self.kde_sample_std )

    def evaluate_elementwise_entropy(self, samples, beta=0.):
        """ Given an array of samples, compute elementwise function of entropy of the form:

            elem_entropy = - (p(samples) + beta)*log(p(samples) + beta)
        Args:
        samples: 1-D array of size N
        beta: float, offset entropy calculation
        Returns:
        elem_entropy: 1-D array of size N, elementwise entropy with beta offset
        """
        assert self.ready, "ENSURE READY BEFORE EVALUATING ELEMENT-WISE ENTROPY"
        log_px = self.fitted_kde.score_samples( (samples  - self.kde_sample_mean) / self.kde_sample_std )
        px = np.exp(log_px)
        elem_entropy = entr(px + beta)
        return elem_entropy

    def save(self, save_folder):
        """Convenience method for saving module attributes"""
        prop_dict = {prop: self.__dict__[prop] for prop in self.prop_names}
        with open(os.path.join(save_folder, "{}.pkl".format('KDE')), 'wb') as f:
            pickle.dump(prop_dict, f)

    def load(self, path):
        with open(path, 'rb') as f:
            prop_dict = pickle.load(f)
        for k, v in prop_dict.items():
            self.__dict__[k] = v

    def save_log_density(self, log_density, folder):
        with open(os.path.join(folder, 'log_density.pkl'), 'wb') as f:
            pickle.dump(log_density, f)

    def load_log_density(self, folder):
        with open(os.path.join(folder, 'log_density.pkl'), 'rb') as f:
            data = pickle.load(f)
        return data

    def parrallel_score_samples(self, samples, thread_count=int(1 * multiprocessing.cpu_count())):
        # print('parrallel cpu cores: {}'.format(thread_count))
        assert self.ready, "ENSURE READY BEFORE EVALUATING LOG DENSITY"
        samples = (samples  - self.kde_sample_mean) / self.kde_sample_std 
        fun = self.fitted_kde.score_samples
        with multiprocessing.Pool(thread_count) as p:
            res = np.concatenate(p.map(fun, np.array_split(samples, thread_count)))
        return res