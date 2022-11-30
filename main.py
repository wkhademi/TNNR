import os
import argparse
import numpy as np

import img_utils
import alg_utils
import metric_utils
import optimizers


OPT = {'admm': optimizers.ADMM,
       'apgl': optimizers.APGL}


def solve(X_gt, M_obs, observed, r, config):
    '''
    Iterative scheme described in Algorithm 1 of "Matrix Completion by Truncated
    Nuclear Norm Regularization" by Zhang et al.

    Args:
        M_obs: observed portion of image
        observed: indicator matrix indicating whether index in image is observed or not
        r: number for truncated nuclear norm (i.e., truncated nuclear norm is
         defined as sum of min(m,n)-r minimum singular values)
        config:
    Returns:
        X_sol: completed/inpainted image
    '''
    # intialize optimizer used at each iteration
    opt = OPT[config.optimizer]
    optimizer = opt(config.opt_max_itrs, config.opt_tol, config)

    X_sol = M_obs

    # solve for completed matrix (complete matrix per channel)
    num_channels = M_obs.shape[-1]
    for c in range(num_channels):
        X = M_obs[..., c]  # initialize X_1 to M_obs

        for iter in range(config.alg_max_itrs):
            # STEP 1: compute SVD of current iterate and get the first r
            #         columns of U and V
            A, S, B = alg_utils.truncated_svd(X, r)

            # STEP 2: update iterate by solving (17) or (26)
            X_new = optimizer.minimize(X, A, B, M_obs[..., c], observed[..., c])

            # check stopping criteria
            if np.linalg.norm(X_new - X) <= config.alg_tol:
                break

            X = X_new

            error = metric_utils.error(X[...,None], X_gt, observed)
            print('r: %d, error: %f'%(r, error))

        X_sol[..., c] = X_new

    return X_sol


def runner(config):
    # load data
    if config.dataset == 'synthetic':  # generate synthetic image
        m, n = config.img_size
        r = config.r
        p = config.p
        sigma = config.sigma
        X_gt, X_obs, observed = img_utils.generate_synthetic_data(m, n, r, p, sigma)
    #else:
    #    X_gt, X_obs, observed = img_utils.load_data()

    # solve for complete image (solve for all r \in [min_rank, max_rank] and select best)
    best_error = 1e10
    best_psnr = 0
    best_X_sol = X_obs
    for r in range(config.min_rank, config.max_rank+1):
        X_sol = solve(X_gt, X_obs, observed, r, config)

    # save best image inpainting result



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset arguments
    parser.add_argument('--data_root', type=str, default='data/',
        help='Path to directory containing the data')
    parser.add_argument('--dataset', type=str, default='real',
        help='Dataset used for solving matrix completion problem. Options are: ' + \
        '[real | synthetic | depth | microscopy]')
    parser.add_argument('--img_num', type=int, default=1,
        help='Image number to load in from dataset.')
    parser.add_argument('--corruption', type=str, default='text',
        help='Corruption method for corrupting image. Options are: [text | drop | block]')
    parser.add_argument('--rate', type=float, default=0.25,
        help='Amount of image to corrupt. Only used for drop style corruption.')
    parser.add_argument('--block_type', type=str, default='square_small',
        help='Type of block corruption used. Options are: [circle | circle_small | column ' + \
        '| diamond | diamond_small | increase | row | square | square_small | star ' + \
        '| star_small | triangle]')
    parser.add_argument('--img_size', type=int, nargs=2, default=[100, 200],
        help='Number of rows and columns in matrix. Used only for synthetic data.')
    parser.add_argument('--r', type=int, default=15,
        help='Rank of matrix. Used only for synthetic data.')
    parser.add_argument('--p', type=float, default=0.7,
        help='Percentage of observed pixels in image. Used only for synthetic data.')
    parser.add_argument('--sigma', type=float, default=0.1,
        help='Standard deviation of Gaussian white noise that is added to image.')

    # optimizer arguments
    parser.add_argument('--optimizer', type=str, default='admm',
        help='Optimizer used to solve the matrix completion problem. Options are: [admm | apgl]')
    parser.add_argument('--min_rank', type=int, default=1,
        help='Minimum assumed rank of matrix.')
    parser.add_argument('--max_rank', type=int, default=20,
        help='Maximum assumed rank of matrix.')
    parser.add_argument('--alg_max_itrs', type=int, default=100,
        help='max number of iterations to run algorithm for.')
    parser.add_argument('--opt_max_itrs', type=int, default=100,
        help='max number of iterations to run optimizer for at each iteration of algorithm.')
    parser.add_argument('--alg_tol', type=float, default=1e-4,
        help='Tolerance for stopping criteria of algorithm.')
    parser.add_argument('--opt_tol', type=float, default=1e-4,
        help='Tolerance for stopping criteria of optimizer used at each iteration.')
    parser.add_argument('--rho', type=float, default=1.,
        help='weighting parameter for augmented Lagrangian used by ADMM.')
    parser.add_argument('--lam', type=float, default=0.06,
        help='weighting parameter used by APGL for soft-constraint formulation.')
    config = parser.parse_args()

    print(config)

    runner(config)
