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

    X = matview(X)
    V = matview(V)

    n, m = X.shape

    return (np.eye(n, n) - 0.5 * X @ X.H) @ V - 0.5 * X @ V.H @ X


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


def _stiefel_transport_operators(Y, X, tau):
    """

    """
    import numpy as np

    B, Q, R = stiefel_decompose_tangent(Y, X)
    m, n = Q.shape

    exp_mat = np.vstack([
        np.hstack([B, -R.H]),
        np.hstack([R, np.zeros_like(R)])
    ])

    # compute eigenvalues of exp_mat, which is skew-Hermitian
    w, V = np.linalg.eig(exp_mat)
    # w must be purely imaginary
    D = np.diag(np.exp(tau*w))
    expm = V @ D @ V.H

    XQ = np.hstack((X, Q))

    U = XQ @ expm @ XQ.H
    W = np.eye(m) + XQ @ (expm - np.eye(2*n)) @ XQ.H

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
