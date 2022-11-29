import numpy as np

from abc import ABC

import alg_utils


class Optimizer(ABC):
    def __init__(self, max_itrs=100, tol=1e-4, **kwargs):
        self.max_itrs = max_itrs
        self.tol = tol

    @abstractmethod
    def minimize(self, X, A, B, M_obs, observed):
        pass


class ADMM(Optimizer):
    '''
    ADMM algorithm for solving (17) in "Matrix Completion by Truncated
    Nuclear Norm Regularization" by Zhang et al.
    '''
    name = 'ADMM'

    def __init__(self, max_itrs=100, tol=1e-4, **kwargs):
        super(ADMM, self).__init__(max_itrs, tol, kwargs)

        self.rho = kwargs['rho']

    def minimize(self, X, A, B, M_obs, observed):
        # indiciator matrix indicating missing portions in image
        missing = np.ones_like(observed) - observed

        X = M_obs  # initialize X_1 to observed matrix
        W = X  # from constraint X = W in (17)
        Y = X  # initialize dual variable to X_1

        for itr in range(self.max_itrs):
            # STEP 1: update X iterate using shrinkage operator
            X_new = alg_utils.shrinkage_operator(W - (1/self.rho)*Y, tau=(1/self.rho))

            # STEP 2: update W iterate
            W_new = X_new + (1/self.rho)*(A.T @ B + Y)

            # fix values at observed entries (for hard constraint W_obs = M_obs in (17))
            W_new = W_new*missing + M_obs

            # STEP 3: update dual variable
            Y_new = Y + self.rho * (X_new - W_new)

            # check stopping criteria
            if np.linalg.norm(X_new - X) <= self.tol:
                break

            X = X_new
            W = W_new
            Y = Y_new

        X = X_new

        return X


class APGL(Optimizer):
    '''
    APGL algorithm for solving (26) in "Matrix Completion by Truncated
    Nuclear Norm Regularization" by Zhang et al.
    '''
    name = 'APGL'

    def __init__(self, **kwargs):
        super(APGL, self).__init__(max_itrs, tol, kwargs)

        self.lambda = kwargs['lambda']

    def minimize(self, X, A, B, M_obs, observed):
        t = 1
        X = M_obs  # initialize X_1 to observed matrix
        X_prev = X
        Y = X  # initialize extrapolation point to X_1

        for itr in range(self.max_itrs):
            # STEP 1: update X iterate using shrinkage operator
            grad_f = A.T @ B - self.lambda*(Y*observed - M_obs)
            X_new = alg_utils.shrinkage_operator(Y + t*grad_f, t)

            # STEP 2: update t iterate (same as nesterov's acceleration update)
            t_new = (1 + np.sqrt(1 + 4*(t**2))) / 2.

            # STEP 3: update Y iterate (extrapolate point from current and previous X iterate)
            Y_new = X + ((t - 1) / t_new)*(X - X_prev)

            # check stopping criteria
            if np.linalg.norm(X_new - X) <= self.tol:
                break

            X_prev = X
            X = X_new
            t = t_new
            Y = Y_new

        X = X_new

        return X
