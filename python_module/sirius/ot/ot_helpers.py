from .ot_transformations import matview


def sinU(x):
    """
    """
    import numpy as np
    x = matview(x)
    XX = np.dot(x.H, x)
    w, R = np.linalg.eigh(XX)
    w = np.sqrt(w)
    R = matview(R)
    Wsin = np.diag(np.sin(w))
    sinU = np.dot(np.dot(R, Wsin), R.H)
    return sinU


def cosU(x):
    """
    """
    import numpy as np
    x = matview(x)
    XX = np.dot(x.H, x)
    w, R = np.linalg.eigh(XX)
    w = np.sqrt(w)
    R = matview(R)
    Wcos = np.diag(np.cos(w))
    cosU = np.dot(np.dot(R, Wcos), R.H)
    return cosU
