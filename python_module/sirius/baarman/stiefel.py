import numpy as np


def matview(x):
    import numpy as np
    return np.matrix(x, copy=False)


def stiefel_project_tangent(V, X):
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


def stiefel_decompose_tangent(Y, X):
    """

    """
    import numpy as np

    X = matview(X)
    Y = matview(Y)

    B = X.H @ Y
    assert (np.isclose(B.H + B, 0).all())

    Z = Y - X @ B
    Q, R = np.linalg.qr(Z)

    return B, Q, R


def stiefel_transport_operators(Y, X, tau):
    """
    Returns:
    U(τ), W(τ)
    """
    import numpy as np

    B, Q, R = stiefel_decompose_tangent(Y, X)
    m, n = Q.shape

    exp_mat = np.vstack(
        np.hstack((B, -R.H)),
        np.hstack((R, np.zeros(n, n)))
    )

    # compute eigenvalues of exp_mat, which is skew-Hermitian
    w, V = np.linalg.eig(exp_mat)
    # w must be purely imagin
    D = np.diag(np.exp(tau*w))
    expm = V @ D @ V.H

    XQ = np.hstack((X, Q))

    U = XQ @ expm @ XQ.H
    W = np.eye(m) + XQ @ (expm - np.eye(2*n)) @ XQ.H

    return U, W
