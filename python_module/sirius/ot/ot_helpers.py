import numpy as np


def _matfun(x, f):
    """f(U) for U = sqrt(x.H @ x)"""
    x = np.asarray(x)
    w, R = np.linalg.eigh(np.conj(x).T @ x)
    w = np.sqrt(w)
    return R @ np.diag(f(w)) @ np.conj(R).T


def sinU(x):
    """sin(U), U = sqrt(x.H @ x)"""
    return _matfun(x, np.sin)


def cosU(x):
    """cos(U), U = sqrt(x.H @ x)"""
    return _matfun(x, np.cos)
