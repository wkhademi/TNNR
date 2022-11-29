import os
import numpy as np


def add_noise(X, observed, sigma):
    '''
    Added Gaussian white noise to observed entries.

    Args:
        X: data matrix
        observed: indicator matrix indicating which entries of matrix X are observed
        sigma: standard deviation of added Gaussian white noise
    Returns:
        X_noisy: noisy version of data matrix X
    '''
    Z = np.random.randn(*X.shape)
    X_noisy = X + sigma*Z*observed

    return X_noisy


def corrupt(data, corruption, rate=0.25, width=):
    '''
    '''
    if corruption == 'noise':
        pass
    elif corruption == 'text':
        pass
    elif corruption == 'single_block':
        pass
    elif corruption == 'multi_block':
        pass


def load_data(data_root, dataset, img):
    '''
    Load ground truth image and generate a corrupted version of it.
    '''
    pass


def generate_synthetic_data(m, n, r, sigma):
    '''
    Generate synthetic matrices of rank r.

    Args:
        m: number of rows in image
        n: number of columns in image
        r: rank of generated synthetic matrix
        sigma: standard deviation of added Gaussian white noise
        p: percentage of observed entries between [0, 1]
    Returns:
        M: ground truth matrix
        M_obs: observed matrix
        observed
    '''
    # generate synthetic matrix with rank r
    M_L = np.random.randn(m, r)
    M_R = np.random.randn(r, n)
    M = M_L @ M_R

    # randomly select p% of entries in M to be the observed entries
    keep = np.random.rand(m, n)
    observed = np.where(keep <= p, 1., 0.)

    # add Gaussian white noise to observed entries
    M_obs = add_noise(M, observed, sigma)

    return M, M_obs, observed
