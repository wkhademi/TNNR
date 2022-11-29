import os
import argparse

import img_utils
import alg_utils
import vis_utils
import optimizers


OPT = {'admm': optimizers.ADMM,
       'apgl': optimizers.APGL}


def solve(M, r, config):
    '''
    Iterative scheme described in Algorithm 1 of "Matrix Completion via Truncated
    Nuclear Norm Regularization" by Zhang et al.

    Args:
        M: matrix containing observed data
        r: number for truncated nuclear norm (i.e., truncated nuclear norm is
         defined as sum of min(m,n)-r minimum singular values)

    Returns:
        X: completed/inpainted matrix
    '''
    # intialize optimizer used at each iteration
    opt = OPT[config.optimizer]
    optimizer = opt(config.opt_max_itrs, config.opt_tol, config)

    X = M  # initialize X_0 to observed matrix

    # solve for completed matrix
    for iter in range(config.alg_max_itrs):
        # STEP 1: compute SVD of current iterate and get truncated
        #         columns of U and V
        A, S, B = utils.truncated_svd(X)

        # STEP 2: update iterate by solving (17) or (26)
        X_new = optimizer.minimize(X, A, B)

        # check stopping criteria
        if np.linalg.norm(X_new - X) <= config.tol:
            break

        X = X_new

    X = X_new

    return X


def runner(config):
    # load data


    # solve for complete matrix


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type='str', required=True,
        help='Dataset used for solving matrix completion problem')
    parser.add_argument('--corruption', type='str', default='text',
        help='Corruption method for corrupting image. Options are: [text | noise | block]')
    parser.add_argument('--rate', type=float, default=0.25,
        help='Amount of image to corrupt. Only used for noise or block corruption.')
    parser.add_argument('--optimizer', type='str', required=True,
        help='Optimizer used to solve the matrix completion problem. Options are: [admm | apgl]')
    parser.add_argument('--alg_max_itrs', type=int, default=100,
        help='max number of iterations to run algorithm for.')
    parser.add_argument('--opt_max_itrs', type=int, default=100,
        help='max number of iterations to run optimizer for at each iteration of algorithm.')
    parser.add_argument('--alg_tol', type=float, default=1e-4,
        help='Tolerance for stopping criteria of algorithm.')
    parser.add_argument('--opt_tol', type=float, default=1e-4,
        help='Tolerance for stopping criteria of optimizer used at each iteration.')
    parser.add_argument('--lambda', type=float, default=0.6,
        help='weighting parameter used by APGL for soft-constraint formulation')
    config = parser.parse_args()

    runner(config)
