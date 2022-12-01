import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


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


def load_mask(data_root, mask, height, width):
    '''
    Load an already created mask from directory.
    '''
    mask_path = os.path.join(data_root, 'masks', 'block_%s.bmp'%mask)
    mask = Image.open(mask_path)
    mask = mask.resize((width, height))
    observed = np.asarray(mask) / 255

    return observed.astype(np.int32)


def corrupt(X, corruption, data_root, config):
    '''
    Corrupt image using a specified masking type.
    '''
    if corruption == 'drop':
        drop_rate = config.rate
        p = np.random.rand(*X.shape[:2])
        observed = np.where(p[..., None] >= drop_rate, 1 , 0)
        observed = np.broadcast_to(observed, X.shape)
    elif corruption == 'text':
        observed = load_mask(data_root, 'text', *X.shape[:2])
    elif corruption == 'block':
        observed = load_mask(data_root, config.block_type, *X.shape[:2])

    X_obs = X*observed

    return X_obs, observed


def load_data(data_root, dataset, img_num, corruption, config):
    '''
    Load ground truth image and generate a corrupted version of it.
    '''
    # load image
    img_path = os.path.join(data_root, dataset, '%d.jpg'%img_num)
    img = Image.open(img_path)
    X_gt = np.asarray(img)

    # generate corruption of image
    X_obs, observed = corrupt(X_gt, corruption, data_root, config)

    # add Gaussian white noise to observed entries
    X_obs = add_noise(X_obs, observed, config.sigma)

    return X_gt, X_obs, observed


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
        observed: indicator matrix indicating observed entries in M_obs
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
