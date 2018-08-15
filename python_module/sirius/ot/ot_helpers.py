from .ot_transformations import matview


def store_pw_coeffs(kpointset, cn, ki=None, ispn=None):
    """
    kpoint -- K_point
    cn     -- numpy array
    ispn   -- spin component
    """
    from .coefficient_array import PwCoeffs
    from ..py_sirius import DeviceEnum

    on_device = kpointset.ctx().processing_unit() == DeviceEnum.GPU

    if isinstance(cn, PwCoeffs):
        assert(ki is None)
        assert(ispn is None)
        for key, v in cn.items():
            k, ispn = key
            kpointset[k].spinor_wave_functions().pw_coeffs(ispn)[:] = v
            if on_device:
                kpointset[k].spinor_wave_functions().copy_to_gpu()
    else:
        kpointset[ki].spinor_wave_functions().pw_coeffs(ispn)[:] = cn
        if on_device:
            kpointset[ki].spinor_wave_functions().copy_to_gpu()


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
