import numpy as np


def truncated_svd(X, r):
    '''
    Compute singular value decomposition of matrix X. For othogonal matrices
    U and V, return their truncated versions with columns corresponding to
    the r largest singular values. (i.e., U[:,:r] and V[:, :r])

    Args:
        X: matrix to compute singular value decomposition for
        r: truncation number for columns of orthogonal matrices U and V
    Returns:
        A: matrix with rows composed of first r columns of U, A=(u_1, ..., u_r)^T
        S: diagonal matrix containing singular values of X
        B: matrix with rows composed of first r columns of V, B=(v_1, ..., v_r)^T
    '''
    U, s, Vh = np.linalg.svd(X)
    S = np.diag(s)
    V = Vh.T

    # truncate U and V
    A = U[:, :r].T
    B = V[:, :r].T

    return A, S, B


def tnn(S, r):
    '''
    Compute the truncated nuclear norm (i.e., sum of smallest min(m,n)-r
    singular values)

    Args:
        S: diagonal matrix containing the singular values of a matrix
        r: # of singular values to not include in truncated nuclear norm
    Returns:
        norm: truncated nuclear norm of a matrix
    '''
    s = S.diagonal()  # get singular values as vector
    idx = len(s) - r
    norm = np.sum(s[idx:])  # sum of min(m,n)-r minimum singular values

    return norm


def shrinkage_operator(X, tau):
    '''
    Singular value shrinkage operator defined in equation (15) of "Matrix
    Completion by Truncated Nuclear Norm Regularization" by Zhang et al.

    D_tau(X) = U*D_tau(S)*V^T, where D_tau(S) = diag{max(sigma_i - tau, 0)}
    '''
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    s = np.maximum(s - tau, 0)
    X = U @ np.diag(s) @ Vh

    return X
