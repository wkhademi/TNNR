import numpy as np

from abc import ABC

import alg_utils


class Optimizer(ABC):
    def __init__(self, max_itrs=100, tol=1e-4, **kwargs):
        self.max_itrs = max_itrs
        self.tol = tol

    @abstractmethod
    def minimize(self, M, A, B):
        pass


class ADMM(Optimizer):
    '''
    ADMM algorithm for solving (17) in "Matrix Completion via Truncated
    Nuclear Norm Regularization" by Zhang et al.
    '''
    name = 'ADMM'

    def __init__(self, max_itrs=100, tol=1e-4, **kwargs):
        super(ADMM, self).__init__(max_itrs, tol, kwargs)

        self.rho = kwargs['rho']

    def minimize(self, M, A, B):
        X = M  # initialize X_1 to observed matrix
        W = X  # from constraint X = W in (17)
        Y = X  # initialize dual variable to X_1

        for itr in range(self.max_itrs):
            # STEP 1: update X iterate using shrinkage operator
            X_new = alg_utils.shrinkage_operator()

            # STEP 2: update W iterate
            W_new = W

            # fix values at observed entries (i.e., hard constraint W_obs = M_obs in (17))
            W_new = W_new

            # STEP 3: update dual variable
            Y_new = Y + self.rho * (X_new - W_new)

            # check stopping criteria
            if np.linalg.norm(X_new - X) <= self.tol:
                break

            X = X_new
            W = W_new
            Y = Y_new

        X = X_new
        W = W_new
        Y = Y_new

        return X


class APGL(Optimizer):
    '''
    APGL algorithm for solving (26) in "Matrix Completion via Truncated
    Nuclear Norm Regularization" by Zhang et al.
    '''
    name = 'APGL'

    def __init__(self, **kwargs):
        super(APGL, self).__init__(max_itrs, tol, kwargs)

    def minimize(self, M, A, B):
        pass
