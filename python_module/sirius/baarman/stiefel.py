def matview(x):
    import numpy as np
    return np.matrix(x, copy=False)


def _stiefel_project_tangent(V, X):
    """
    Keyword Arguments:
    V -- m times n
    X -- m times n
    """
    import numpy as np

    # X = matview(X)
    # V = matview(V)

    n, m = X.shape

    return V - 0.5 * X @ (X.H @ V) - 0.5 * X @ (V.H @ X)


def stiefel_project_tangent(V, X):
    """

    """
    import numpy as np
    from ..coefficient_array import CoefficientArray

    if isinstance(X, CoefficientArray):
        Y = type(X)(dtype=X.dtype, ctype=np.matrix)
        for key, X_loc in X._data.items():
            V_loc = V[key]
            Y[key] = _stiefel_project_tangent(V_loc, X_loc)
        return Y
    else:
        return _stiefel_project_tangent(V, X)


def _stiefel_decompose_tangent(Y, X):
    """

    """
    import numpy as np

    X = matview(X)
    Y = matview(Y)

    B = X.H @ Y

    Z = Y - X @ B
    Q, R = np.linalg.qr(Z)

    return B, Q, R


def stiefel_decompose_tangent(Y, X):
    """

    """
    import numpy as np
    from ..coefficient_array import CoefficientArray

    if isinstance(X, CoefficientArray):
        B = type(X)(dtype=X.dtype, ctype=np.matrix)
        Q = type(X)(dtype=X.dtype, ctype=np.matrix)
        R = type(X)(dtype=X.dtype, ctype=np.matrix)
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
        """
        """
        return Y + self.X @ self.RX + self.Q @ self.RQ


def _stiefel_transport_operators(Y, X, tau):
    """

    """
    import numpy as np

    B, Q, R = stiefel_decompose_tangent(Y, X)
    m, n = Q.shape

    exp_mat = np.vstack(
        [np.hstack([B, -R.H]),
         np.hstack([R, np.zeros_like(R)])])

    # compute eigenvalues of exp_mat, which is skew-Hermitian
    w, V = np.linalg.eigh(1j * exp_mat)
    w = -1j * np.real(w)
    # w must be purely imaginary
    D = np.diag(np.exp(tau * w))
    # assert(np.isclose(V.H@V, np.eye(2*n, 2*n), atol=1e-8).all())
    expm = V @ D @ V.H

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
    import numpy as np
    from ..coefficient_array import CoefficientArray

    if isinstance(X, CoefficientArray):
        U = type(X)(dtype=X.dtype, ctype=np.matrix)
        W = type(X)(dtype=X.dtype, ctype=np.matrix)
        for key, X_loc in X._data.items():
            Y_loc = Y[key]
            U[key], W[key] = _stiefel_transport_operators(Y_loc, X_loc, tau)
        return U, W
    else:
        return _stiefel_transport_operators(Y, X, tau)
