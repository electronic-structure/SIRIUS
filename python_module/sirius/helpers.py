def make_dict(ctx, ks, x_ticks, x_axis):
    dict = {}
    dict["header"] = {}
    dict["header"]["x_axis"] = x_axis
    dict["header"]["x_ticks"]=[]
    dict["header"]["num_bands"]=ctx.num_bands()
    dict["header"]["num_mag_dims"] = ctx.num_mag_dims()

    for e in enumerate(x_ticks):
        j = {}
        j["x"] = e[1][0]
        j["label"] = e[1][1]
        dict["header"]["x_ticks"].append(j)

    dict["bands"] = []

    for ik in range(ks.num_kpoints()):
        bnd_k = {}
        bnd_k["kpoint"] = [0.0,0.0,0.0]
        for x in range(3):
            bnd_k["kpoint"][x] = ks(ik).vk()(x)
        bnd_e = []

        bnd_e = ks.get_band_energies(ik, 0)

        bnd_k["values"] = bnd_e
        dict["bands"].append(bnd_k)
    return dict
