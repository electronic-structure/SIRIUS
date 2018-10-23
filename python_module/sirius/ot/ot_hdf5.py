def save(fh5, name, obj):
    """
    Keyword Arguments:
    fh5  --
    name --
    obj  --
    """

    from ..coefficient_array import CoefficientArray

    if isinstance(obj, CoefficientArray):
        grp = fh5.create_group(name)
        for key, val in obj.items():
            grp.create_dataset(
                name=','.join(map(str, key)), shape=val.shape, dtype=val.dtype, data=val)
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

    out = type(obj)(dtype=obj.dtype)
    if isinstance(obj, CoefficientArray):
        for key, val in obj.items():
            dname = ','.join(map(str, key))
            out[key] = fh5[name][dname]
    else:
        out = np.array(fh5[name])
    return out
