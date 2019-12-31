import numpy as np
from ..coefficient_array import CoefficientArray, spdiag, threaded


@threaded
def gram_schmidt(X):
    """
    Arguments:
    X -- column vectors
    """
    X = np.matrix(X, copy=False)
    m = X.shape[1]
    Q = np.zeros_like(X)
    Q[:, 0] = X[:, 0] / np.linalg.norm(X[:, 0])
    for i in range(1, m):
        Xi = X[:, i].copy()
        for j in range(i):
            Xi -= np.tensordot(Xi, np.conj(Q[:, j]), axes=2) * Q[:, j]
        Q[:, i] = Xi / np.linalg.norm(Xi)
    return Q


@threaded
def modified_gram_schmidt(X):
    X = np.matrix(X, copy=False)
    m = X.shape[1]
    Q = np.zeros_like(X)
    for k in range(m):
        Q[:, k] = X[:, k]
        for i in range(k):
            Q[:, k] = Q[:, k] - np.tensordot(Q[:, k], np.conj(Q[:, i]), axes=2) * Q[:, i]
        Q[:, k] = Q[:, k] / np.linalg.norm(Q[:, k])
    return Q


def loewdin(X):
    S = X.H @ X
    w, U = S.eigh()
    Sm2 = U @ spdiag(1/np.sqrt(w)) @ U.H
    return X @ Sm2
