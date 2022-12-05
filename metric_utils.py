import numpy as np
import matplotlib.pyplot as plt


def error(X_sol, X_gt, observed):
    '''
    Compute the error on the missing pixels that were predicted:

            error = ||(X_sol - X_gt)_miss||_F
    '''
    missing = np.ones_like(observed) - observed
    error = np.linalg.norm((X_sol - X_gt)*missing)

    return error


def MSE(X_sol, X_gt, observed):
    '''
    Compute the mean squared error on the missing pixels that were predicted.
    Let N be the number of missing pixels, then:

            MSE = (error_r**2 + error_g**2 + error_b**2) / 3N
    '''
    # compute error for each channel
    error_r = error(X_sol[..., 0], X_gt[..., 0], observed[..., 0])
    error_g = error(X_sol[..., 1], X_gt[..., 1], observed[..., 1])
    error_b = error(X_sol[..., 2], X_gt[..., 2], observed[..., 2])

    squared_error = error_r**2 + error_g**2 + error_b**2

    N = np.sum(np.ones_like(observed) - observed)  # compute 3*number of missing pixels
    mse = squared_error / N

    return mse


def PSNR(X_sol, X_gt, observed, min_val, max_val):
    '''
    Compute the Peak Signal-to-Noise Ration (PSNR) on the missing pixels that
    were predicted:

            PSNR = 10 * log_10(255**2 / MSE)
    '''
    # unnormalize images
    X_sol = (max_val - min_val)*X_sol + min_val
    X_gt = (max_val - min_val)*X_gt + min_val

    mse = MSE(X_sol, X_gt, observed)
    psnr = 10. * np.log10(255**2 / mse)

    return psnr
