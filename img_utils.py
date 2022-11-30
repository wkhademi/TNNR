import os
import numpy as np
import matplotlib.pyplot as plt


def save(X, path):
    '''
    Save image to specified directory.
    '''
    plt.axis('off')
    plt.imshow(X)
    plt.savefig(path, bbox_inches='tight')


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
    X_noisy = X*observed + sigma*Z*observed

    return X_noisy


def corrupt(X, corruption, **kwargs):
    '''
    '''
    if corruption == 'drop':
        pass
    elif corruption == 'text':
        pass
    elif corruption == 'block':
        pass


def load_data(data_root, dataset, img, corruption, **kwargs):
    '''
    Load ground truth image and generate a corrupted version of it.
    '''
    pass


def generate_synthetic_data(m, n, r, p, sigma):
    '''
    Generate synthetic matrices of rank r.

    Args:
        m: number of rows in image
        n: number of columns in image
        r: rank of generated synthetic matrix
        p: percentage of observed entries between [0, 1]
        sigma: standard deviation of added Gaussian white noise
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

    return M[..., None], M_obs[..., None], observed[..., None]


def plot_error_vs_num_itrs(error, log=False):
    '''
    Plot the error (or log error) vs. number of out iterations of algorithm.
    '''
    if log:
        error = np.log10(error)

    iters = list(arange(len(error)))

    fig, ax = plt.subplots()
    ax.set_xlabel('number of outer iterations')
    if log:
        ax.set_ylabel('log10(error)')
    else:
        ax.set_ylabel('error')
    ax.plot(iters, error, c='k')
    plt.savefig('error_vs_num_itrs.png', bbox_inches='tight')


def plot_error_vs_r(error, r):
    '''
    Plot the error vs. parameter r (i.e., guess of rank)
    '''
    fig, ax = plt.subplots()
    ax.set_xlabel('r')
    ax.set_ylabel('error')
    ax.plot(r, error, c='k')
    plt.savefig('error_vs_r.png', bbox_inches='tight')
