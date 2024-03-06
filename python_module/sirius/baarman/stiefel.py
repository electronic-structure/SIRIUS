from numpy.typing import ArrayLike, NDArray
from sirius.coefficient_array import threaded, allthreaded, CoefficientArray
from typing import Tuple, Any, TypeAlias
import numpy as np

complex_array_t: TypeAlias = NDArray[np.complex128]


def _stiefel_project_tangent(V: complex_array_t, X: complex_array_t) -> complex_array_t:
    """
    Keyword Arguments:
    V -- m times n
    X -- m times n
    """
    import numpy as np

    n, m = X.shape

    return V - 0.5 * X @ (np.conj(X).T @ V) - 0.5 * X @ (np.conj(V).T @ X)


def stiefel_project_tangent(V, X):
    """ """
    import numpy as np

    if isinstance(X, CoefficientArray):
        return allthreaded(_stiefel_project_tangent)(V, X)
    else:
        return _stiefel_project_tangent(V, X)


def _stiefel_decompose_tangent(
    Y: complex_array_t, X: complex_array_t
) -> Tuple[complex_array_t, complex_array_t, complex_array_t]:
    """ """
    import numpy as np

    B = np.conj(X).T @ Y

    Z = Y - X @ B
    Q, R = np.linalg.qr(Z)

    return B, Q, R


def stiefel_decompose_tangent(Y, X):
    """ """
    from ..coefficient_array import CoefficientArray

    if isinstance(X, CoefficientArray):
        B = type(X)()
        Q = type(X)()
        R = type(X)()
        for key, X_loc in X._data.items():
            Y_loc = Y[key]
            B[key], Q[key], R[key] = _stiefel_decompose_tangent(Y_loc, X_loc)
        return B, Q, R
    else:
        return _stiefel_decompose_tangent(Y, X)


class Geodesic:
    def __init__(self, M, N, Q):
        self.M = M
        self.N = N
        self.Q = Q
        m = Q.shape[0]
        self.shape = (m, m)

    def __matmul__(self, X):
        return X @ self.M + self.Q @ self.N


class ParallelTransport:
    def __init__(self, X, expm, Q, R, B):
        import numpy as np

        self.Q = Q
        self.R = R
        self.B = B
        n = expm.shape[0] // 2
        G00 = expm[:n, :n] - np.eye(n)
        G01 = expm[:n, n:]
        G10 = expm[n:, :n]
        G11 = expm[n:, n:] - np.eye(n)

        self.RX = G00 @ B + G01 @ R
        self.RQ = G10 @ B + G11 @ R
        self.X = X

    def __matmul__(self, Y):
        """ """
        return Y + self.X @ self.RX + self.Q @ self.RQ


def _stiefel_transport_operators(
    Y: complex_array_t, X: complex_array_t, tau: float
) -> Tuple[Geodesic, ParallelTransport]:
    """ """
    import numpy as np

    B, Q, R = _stiefel_decompose_tangent(Y, X)
    m, n = Q.shape

    exp_mat = np.vstack(
        [np.hstack([B, -np.conj(R).T]), np.hstack([R, np.zeros_like(R)])]
    )

    # compute eigenvalues of exp_mat, which is skew-Hermitian
    w, V = np.linalg.eigh(1j * exp_mat)
    w = -1j * np.real(w)
    # w must be purely imaginary
    D = np.diag(np.exp(tau * w))
    # assert(np.isclose(V.H@V, np.eye(2*n, 2*n), atol=1e-8).all())
    expm = V @ D @ np.conj(V).T

    # U = XQ @ expm @ XQ.H
    MN = expm @ np.eye(2 * n, n)
    M = MN[:n, :]
    N = MN[n:, :]
    U = Geodesic(M, N, Q)
    # W = np.eye(m) + XQ @ (expm - np.eye(2 * n)) @ XQ.H
    W = ParallelTransport(X, expm, Q, R, B)

    return U, W


def stiefel_transport_operators(Y, X, tau):
    """
    Returns:
    U(τ), W(τ)
    """

    if isinstance(X, CoefficientArray):
        def _op(y, x) -> Tuple[Geodesic, ParallelTransport]:
            return _stiefel_transport_operators(y, x, tau)

        return allthreaded(_op)(Y, X)
    else:
        return _stiefel_transport_operators(Y, X, tau)
