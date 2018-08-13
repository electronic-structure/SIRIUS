
def store_pw_coeffs(kpointset, cn, ki=None, ispn=None):
    """
    kpoint -- K_point
    cn     -- numpy array
    ispn   -- spin component
    """
    from .coefficient_array import PwCoeffs
    from .py_sirius import DeviceEnum

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



def make_dict(ctx, ks, x_ticks, x_axis):
    dict = {}
    dict["header"] = {}
    dict["header"]["x_axis"] = x_axis
    dict["header"]["x_ticks"] = []
    dict["header"]["num_bands"] = ctx.num_bands()
    dict["header"]["num_mag_dims"] = ctx.num_mag_dims()

    for e in enumerate(x_ticks):
        j = {}
        j["x"] = e[1][0]
        j["label"] = e[1][1]
        dict["header"]["x_ticks"].append(j)

    dict["bands"] = []

    for ik in range(ks.num_kpoints()):
        bnd_k = {}
        bnd_k["kpoint"] = [0.0, 0.0, 0.0]
        for x in range(3):
            bnd_k["kpoint"][x] = ks(ik).vk()(x)
        bnd_e = []

        bnd_e = ks.get_band_energies(ik, 0)

        bnd_k["values"] = bnd_e
        dict["bands"].append(bnd_k)
    return dict
