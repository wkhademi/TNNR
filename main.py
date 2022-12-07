import os
import time
import argparse
import numpy as np

import img_utils
import alg_utils
import metric_utils
import optimizers
import matplotlib.pyplot as plt


OPT = {'admm': optimizers.ADMM,
       'apgl': optimizers.APGL}


def solve(X_gt, M_obs, observed, r, config):
    '''
    Iterative scheme described in Algorithm 1 of "Matrix Completion by Truncated
    Nuclear Norm Regularization" by Zhang et al.

    Args:
        M_obs: observed portion of image X_gt
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

    X_sol = np.copy(M_obs)

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

            metric = metric_utils.error(X, X_gt[...,c], observed[...,c])
            print('Error (r = %d): %f'%(r, metric))

        X_sol[..., c] = X_new

    return X_sol


def runner(config):
    # load data
    if config.dataset == 'synthetic':  # generate synthetic image
        m, n = config.img_size
        X_gt, X_obs, observed = img_utils.generate_synthetic_data(m, n, config.r,
                                                                  config.p, config.sigma)
    else:  # load image and generate corruption of it
        X_gt, X_obs, observed, min_val, max_val = img_utils.load_data(config.data_root, config.dataset,
                                                                      config.img_num, config.corruption,
                                                                      config)

    num_channels = X_obs.shape[-1]
    best_metric = 1e10 if num_channels == 1 else -1e10
    best_r = 0
    best_X_sol = X_obs

    # solve for complete image (solve for all r \in [min_rank, max_rank] and select best)
    total_time = 0
    for r in range(config.min_rank, config.max_rank+1):
        start = time.time()
        X_sol = solve(X_gt, X_obs, observed, r, config)
        end = time.time()
        print('Runtime: %f'%(end-start))
        total_time += (end-start)

        # evaluate metric
        if num_channels == 1:
            metric = metric_utils.error(X_sol, X_gt, observed)
            print('Error (r = %d): %f'%(r, metric))
            better = metric < best_metric
        elif num_channels == 3:
            X_sol = np.minimum(np.maximum(0., X_sol), 1.)
            metric = metric_utils.PSNR(X_sol, X_gt, observed, min_val, max_val)
            print('PSNR (r = %d): %f'%(r, metric))
            better = metric > best_metric

        if better:
            best_metric = metric
            best_r = r
            best_X_sol = X_sol

    print('Total runtime: %f'%total_time)

    # print metric for best image completion (i.e., best choice of r)
    metric_name = 'Error' if num_channels == 1 else 'PSNR'
    print('Best %s (r = %d): %f'%(metric_name, r, metric))

    # save best image inpainting result
    if config.dataset != 'synthetic':
        if config.dataset == 'depth':
            img_utils.save(X_obs, 'real/', config.dataset, config.optimizer, config.img_num)
        img_utils.save(X_obs, 'corrupt/', config.dataset, config.optimizer, config.img_num)
        img_utils.save(X_sol, 'results/', config.dataset, config.optimizer, config.img_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset arguments
    parser.add_argument('--data_root', type=str, default='data/',
        help='Path to directory containing the data')
    parser.add_argument('--dataset', type=str, default='real',
        help='Dataset used for solving matrix completion problem. Options are: ' + \
        '[real | synthetic | depth]')
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
    parser.add_argument('--alg_max_itrs', type=int, default=10,
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
