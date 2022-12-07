import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pypfm import PFMLoader


def save(X, root, dataset, optimizer, img_num):
    '''
    Save image to specified directory.
    '''
    save_path = os.path.join(root, dataset, '%s_%d.jpg'%(optimizer, img_num))
    plt.axis('off')
    plt.imshow(X)
    plt.savefig(save_path, bbox_inches='tight')


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
    mask = np.asarray(mask) / 255.

    observed = np.ones_like(mask[..., 0])
    observed[mask[...,0] < 1] = 0.
    observed[mask[...,1] < 1] = 0.
    observed[mask[...,2] < 1] = 0.
    observed = np.repeat(observed[..., None], 3, axis=-1)

    return observed


def load_image(data_root, dataset, img_num):
    '''
    Load JPG image from specified path.
    '''
    img_path = os.path.join(data_root, dataset, '%d.jpg'%img_num)
    img = Image.open(img_path)

    # normalize image
    min_val = 0
    max_val = 255
    X_gt = (np.asarray(img) - min_val) / (max_val - min_val)

    # plot singular values
    s0 = np.linalg.svd(X_gt[..., 0], compute_uv=False)
    s1 = np.linalg.svd(X_gt[..., 1], compute_uv=False)
    s2 = np.linalg.svd(X_gt[..., 2], compute_uv=False)
    fig, ax = plt.subplots()
    ax.plot(list(range(len(s0))), s0, 'r', label='red channel')
    ax.plot(list(range(len(s1))), s1, 'g', label='green channel')
    ax.plot(list(range(len(s2))), s2, 'b', label='blue channel')
    ax.set_xlabel('i-th singular value')
    ax.set_ylabel('magnitude')
    ax.legend()
    plt.show()

    return X_gt, min_val, max_val


def load_depth_map(data_root, dataset, img_num):
    '''
    Load depth map from specified path.
    '''
    calib_path = os.path.join(data_root, dataset, 'calib%d.txt'%img_num)

    # load calibration file containing relevant parameters
    calib_dict = {}
    with open(calib_path) as calib_file:
        for line in calib_file:
            name, val = line.partition('=')[::2]
            calib_dict[name] = val

    width = int(calib_dict['width'])
    height = int(calib_dict['height'])
    min_val = int(calib_dict['vmin'])
    max_val = int(calib_dict['vmax'])

    # load depth map
    loader = PFMLoader((width, height), color=False, compress=False)
    depth_path = os.path.join(data_root, dataset, 'disp%d.pfm'%img_num)
    depth = np.flip(loader.load_pfm(depth_path), axis=0)
    depth[depth == np.inf] = 0.

    # normalize depth map
    X_gt = (depth - min_val) / (max_val - min_val)
    X_gt[X_gt < 0] = 0.

    # resize depth map
    img = Image.fromarray(X_gt)
    X_gt = np.array(img.resize((300, 300)))[..., None]

    # plot singular values
    s = np.linalg.svd(X_gt[...,0], compute_uv=False)
    fig, ax = plt.subplots()
    ax.plot(list(range(len(s))), s)
    ax.set_xlabel('i-th singular value')
    ax.set_ylabel('magnitude')
    plt.show()

    return X_gt, min_val, max_val


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

    if X.shape != observed.shape:
        observed = observed[..., :1]

    X_obs = X*observed

    return X_obs, observed


def load_data(data_root, dataset, img_num, corruption, config):
    '''
    Load ground truth image and generate a corrupted version of it.
    '''
    if dataset == 'real':  # load image
        X_gt, min_val, max_val = load_image(data_root, dataset, img_num)
    elif dataset == 'depth':  # load depth map
        X_gt, min_val, max_val = load_depth_map(data_root, dataset, img_num)

    # generate corruption of image
    if corruption != 'none':
        X_obs, observed = corrupt(X_gt, corruption, data_root, config)

    # add Gaussian white noise to observed entries
    X_obs = add_noise(X_obs, observed, config.sigma)

    return X_gt, X_obs, observed, min_val, max_val


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
