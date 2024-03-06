import numpy as np
from ..coefficient_array import spdiag, threaded, allthreaded
from numpy.typing import NDArray
from typing import TypeAlias

__all__ = ["gram_schmidt", "modified_gram_schmidt", "loewdin"]

complex_array_t: TypeAlias = NDArray[np.complex128]


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
            Q[:, k] = (
                Q[:, k] - np.tensordot(Q[:, k], np.conj(Q[:, i]), axes=2) * Q[:, i]
            )
        Q[:, k] = Q[:, k] / np.linalg.norm(Q[:, k])
    return Q


def loewdin_nc(X: complex_array_t) -> complex_array_t:
    """Loewdin for the norm-consvering case."""
    M = np.conj(X).T @ X
    w, U = np.linalg.eigh(M)
    R = U @ spdiag(1 / np.sqrt(w)) @ np.conj(U).T
    return X @ R


def loewdin_overlap(X: complex_array_t, S: complex_array_t) -> complex_array_t:
    """Loewdin with overlap matrix S."""
    M = np.conj(X).T @ (S @ X)
    w, U = np.linalg.eigh(M)
    R = U @ spdiag(1 / np.sqrt(w)) @ np.conj(U).T
    return X @ R


def loewdin(X, S=None):
    """Loewdin orthogonalization."""
    if S is None:
        return threaded(loewdin_nc)(X)
    return allthreaded(loewdin_overlap)(X, S)
