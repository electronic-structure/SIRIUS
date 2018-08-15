def save(fh5, name, obj):
    """
    Keyword Arguments:
    fh5  --
    name --
    obj  --
    """

    from .coefficient_array import CoefficientArray

    if isinstance(obj, CoefficientArray):
        grp = fh5.create_group(name)
        for key, val in obj.items():
            grp.create_dataset(
                name=','.join(map(str, key)), shape=val.shape, dtype=val.dtype, data=val)
    else:
        fh5.create_dataset(name=name, data=obj)
