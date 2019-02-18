def save(fh5, name, obj, kset):
    """
    Keyword Arguments:
    fh5  --
    name --
    obj  -- np.array like / CoefficientArray
    kset -- (Default None) optional
    """

    from ..coefficient_array import CoefficientArray
    from ..helpers import kpoint_index

    if isinstance(obj, CoefficientArray):
        grp = fh5.create_group(name)
        for key, val in obj.items():
            k, _ = key
            dset = grp.create_dataset(
                name=','.join(map(str, key)),
                shape=val.shape,
                dtype=val.dtype,
                data=val)
            dset.attrs['ki'] = kpoint_index(kset[k], kset.ctx())
    else:
        grp = fh5.create_dataset(name=name, data=obj)
    return grp


def load(fh5, name, obj):
    """
    Keyword Arguments:
    fh5  --
    name --
    obj  --
    """

    from ..coefficient_array import CoefficientArray
    import numpy as np

    out = type(obj)(dtype=obj.dtype)
    if isinstance(obj, CoefficientArray):
        for key, val in obj.items():
            dname = ','.join(map(str, key))
            out[key] = np.array(fh5[name][dname])
    else:
        out = np.array(fh5[name])
    return out
