import numpy as np


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
    error_r = error(X_sol[..., 0], X_gt[..., 0], observed)
    error_g = error(X_sol[..., 1], X_gt[..., 1], observed)
    error_b = error(X_sol[..., 2], X_gt[..., 2], observed)

    squared_error = error_r**2 + error_g**2 + error_b**2

    N = np.sum(np.ones_like(observed) - observed)  # compute number of missing pixels
    mse = squared_error / (3*N)

    return mse


def PSNR(X_sol, X_gt, observed):
    '''
    Compute the Peak Signal-to-Noise Ration (PSNR) on the missing pixels that
    were predicted:

            PSNR = 10 * log_10(255**2 / MSE)
    '''
    mse = MSE(X_sol, X_gt, observed)
    psnr = 10. * np.log10(255**2 / mse)

    return psnr
