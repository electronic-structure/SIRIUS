import numpy as np
from ..coefficient_array import CoefficientArray, spdiag


def _gram_schmidt(X):
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


def gram_schmidt(X):
    if isinstance(X, CoefficientArray):
        out = type(X)(dtype=X.dtype, ctype=X.ctype)
        for key, val in X._data.items():
            out[key] = _gram_schmidt(val)
        return out
    else:
        return _gram_schmidt(X)


def loewdin(X):
    S = X.H @ X
    w, U = S.eigh()
    Sm2 = U @ spdiag(1/np.sqrt(w)) @ U.H
    return X @ Sm2
