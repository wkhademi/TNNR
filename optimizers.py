import numpy as np

from abc import ABC, abstractmethod

import alg_utils


class Optimizer(ABC):
    def __init__(self, max_itrs=100, tol=1e-4):
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

    def __init__(self, max_itrs, tol, config):
        super(ADMM, self).__init__(max_itrs, tol)

        self.rho = config.rho

    def minimize(self, X, A, B, M_obs, observed):
        obj_val = 1e10

        # indiciator matrix indicating missing portions in image
        missing = np.ones_like(observed) - observed

        AB = A.T @ B  # precompute A^TB
        X = M_obs  # intialize X to M_obs
        W = X  # from constraint X = W in (17)
        Y = X  # initialize dual variable to X_1

        for itr in range(self.max_itrs):
            # STEP 1: update X iterate using shrinkage operator
            X_new = alg_utils.shrinkage_operator(W - (1/self.rho)*Y, tau=(1/self.rho))

            # STEP 2: update W iterate
            W_new = X_new + (1/self.rho)*(AB + Y)

            # fix values at observed entries (for hard constraint W_obs = M_obs in (17))
            W_new = W_new*missing + M_obs

            # STEP 3: update dual variable
            Y_new = Y + self.rho * (X_new - W_new)

            obj_val_new = np.linalg.norm(X_new, 'nuc') - np.trace(A@W_new@B.T) + \
                            (self.rho/2)*(np.linalg.norm(X_new - W_new)**2) + np.trace(Y_new.T @ (X_new - W_new))

            # check stopping criteria
            if np.linalg.norm(X_new - X) <= self.tol:
                break

            X = X_new
            W = W_new
            Y = Y_new
            obj_val = obj_val_new

        X = X_new

        return X


class APGL(Optimizer):
    '''
    APGL algorithm for solving (26) in "Matrix Completion by Truncated
    Nuclear Norm Regularization" by Zhang et al.
    '''
    name = 'APGL'

    def __init__(self, max_itrs, tol, config):
        super(APGL, self).__init__(max_itrs, tol)

        self.lam = config.lam

    def p(self, Y, AB, M_obs, observed, L):
        '''
        Solve p_{L}(Y) = argmin_X Q(X, Y)
        '''
        grad = AB - self.lam*(Y - M_obs)*observed
        X = alg_utils.shrinkage_operator(Y + (1./L)*grad, (1./L))

        return X

    def F(self, X, A, B, M_obs, observed):
        '''
        Evaluate F(X) = -Tr(AXB^T) + (lam/2)(||X_obs - M_obs||_F)^2
        '''
        f = -np.trace(A@X@B.T) + (self.lam/2)*np.linalg.norm((X - M_obs)*observed)**2

        return f

    def Q(self, X, Y, AB, A, B, M_obs, observed, L):
        '''
        Evaluate Q(X, Y) = F(Y) + <X-Y, grad_F(Y)> + (L/2)(||X-Y||_F)^2 + ||X||_*
        '''
        F_Y = self.F(Y, A, B, M_obs, observed)
        XY = X - Y
        grad_F_Y = AB - self.lam*(Y - M_obs)*observed
        dot = np.trace(grad_F_Y.T @ XY)  # <X-Y, grad_F(Y)> = Tr(grad_F(Y)^T(X-Y))
        frob_norm = np.linalg.norm(XY)
        nuc_norm = np.linalg.norm(X, 'nuc')

        q = F_Y + dot + (L/2.)*(frob_norm**2) + nuc_norm

        return q

    def minimize(self, X, A, B, M_obs, observed):
        M_norm = np.linalg.norm(M_obs)
        obj_val = 1e10

        AB = A.T @ B  # precompute A^TB
        t = 1
        X = M_obs
        Y = X  # initialize extrapolation point to X_1

        # backtracking vars
        #eta = 1.1
        #L = 1e-5

        for itr in range(self.max_itrs):
            # find appropriate step size with backtracking
            #P = self.p(Y, AB, M_obs, observed, L)
            #f = self.F(P, A, B, M_obs, observed)
            #q = self.Q(P, Y, AB, A, B, M_obs, observed, L)
            #while f > q:
                #L *= eta
                #P = self.p(Y, AB, M_obs, observed, L)
                #f = self.F(P, A, B, M_obs, observed)
                #q = self.Q(P, Y, AB, A, B, M_obs, observed, L)
            #print(L)

            # STEP 1: update X iterate using shrinkage operator
            grad_f = AB - self.lam*(Y - M_obs)*observed
            X_new = alg_utils.shrinkage_operator(Y + (1./self.lam)*grad_f, (1./self.lam))

            # STEP 2: update t iterate (same as nesterov's acceleration update)
            t_new = (1 + np.sqrt(1 + 4*(t**2))) / 2.

            # STEP 3: update Y iterate (extrapolate point from current and previous X iterate)
            Y_new = X_new + ((t - 1) / t_new)*(X_new - X)

            # check if objective value has gotten worse
            obj_val_new = np.linalg.norm(X_new, 'nuc') - np.trace(A@X_new@B.T) + \
                            (self.lam/2)*np.linalg.norm((X_new - M_obs)*observed)**2

            # check stopping criteria
            if np.linalg.norm(X_new - X) <= self.tol:
                break

            X = X_new
            t = t_new
            Y = Y_new
            obj_val = obj_val_new

        #print(X_new[observed == 0])
        X = X_new

        return X
