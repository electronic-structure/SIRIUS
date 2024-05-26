from __future__ import print_function
import numpy as np
import h5py
from mpi4py import MPI
import json
from .coefficient_array import PwCoeffs, CoefficientArray
from .logger import Logger
from .py_sirius import Simulation_context, K_point_set, DFT_ground_state, MemoryEnum


logger = Logger()


def save_state(objs_dict, kset, prefix="fail"):
    """
    Arguments:
    objs_dict = dictionary(string: CoefficientArray), example: {'Z': Z, 'G': G}
    dump current state to HDF5
    """
    logger("save state")
    rank = MPI.COMM_WORLD.rank
    import sirius.ot as ot

    with h5py.File(prefix + "%d.h5" % rank, "w") as fh5:
        for key in objs_dict:
            # assume it is a string
            name = key
            ot.save(fh5, name, objs_dict[key], kset)


def load_state(filename, kset, name, dtype):
    """
    Keyword Arguments:
    fh5  --
    name --
    """
    import glob

    ctype = np.matrix
    out = CoefficientArray(dtype=dtype, ctype=np.matrix)

    idx_to_k = {}
    for i, kp in enumerate(kset):
        kindex = kpoint_index(kp, kset.ctx())
        idx_to_k[kindex] = i

    if "*" in filename:
        files = glob.glob(filename)
    else:
        files = [filename]
    if not len(files) > 0:
        raise Exception("no files found: ", filename)
    for fi in files:
        with h5py.File(fi, "r") as fh5:
            for key in fh5[name].keys():
                ki = tuple(fh5[name][key].attrs["ki"])
                if ki not in idx_to_k:
                    # looping over all k-points,
                    # skip if k-point is not present on this rank
                    continue
                k = idx_to_k[ki]
                _, s = eval(key)
                if key in fh5[name].keys():
                    out[(k, s)] = ctype(fh5[name][key])
    return out


def store_pw_coeffs(kpointset, cn, ki=None, ispn=None):
    """
    kpoint -- K_point
    cn     -- numpy array
    ispn   -- spin component
    """
    ctx = kpointset.ctx()
    pmem_t = ctx.processing_unit_memory_t()

    if isinstance(cn, PwCoeffs):
        assert ki is None
        assert ispn is None
        for key, v in cn.items():
            k, ispn = key
            psi = kpointset[k].spinor_wave_functions()
            psi.pw_coeffs(ispn)[:, : v.shape[1]] = v
            on_device = pmem_t == MemoryEnum.device
            if on_device:
                psi.copy_to_gpu()
    else:
        psi = kpointset[ki].spinor_wave_functions()
        on_device = pmem_t == MemoryEnum.device
        psi.pw_coeffs(ispn)[:, : cn.shape[1]] = cn
        if on_device:
            psi.copy_to_gpu()


def DFT_ground_state_find(num_dft_iter=1, config="sirius.json"):
    """
    run DFT_ground_state

    Keyword Arguments:
    num_dft_iter -- (Default 1) number of SCF iterations
    config       -- json configuration / or dictionary (from json)

    Returns:
    A dictionary with keys
    - E (total energy)
    - dft_gs (DFT_ground_state instance)
    - kpointset (K_point_set instance)
    - hamiltonian (Hamiltonian instance)
    - density (Density instance)
    - potential (Potential instance)
    - ctx (Simulation_context instance)
    """
    if isinstance(config, dict):
        ctx = Simulation_context(json.dumps(config))
        siriusJson = config
    else:
        siriusJson = json.load(open(config))
        ctx = Simulation_context(json.dumps(siriusJson))
    ctx.initialize()
    if "vk" in siriusJson["parameters"]:
        vk = siriusJson["parameters"]["vk"]
        kPointSet = K_point_set(ctx, vk)
    else:
        if "shiftk" in siriusJson["parameters"]:
            # make sure shiftk is not a list of floats
            shiftk = [int(x) for x in siriusJson["parameters"]["shiftk"]]
        else:
            shiftk = [0, 0, 0]
        if "ngridk" in siriusJson["parameters"]:
            gridk = siriusJson["parameters"]["ngridk"]
        if "use_symmetry" in siriusJson["parameters"]:
            use_symmetry = siriusJson["parameters"]["use_symmetry"]
        else:
            use_symmetry = True
        kPointSet = K_point_set(ctx, gridk, shiftk, use_symmetry)

    dft_gs = DFT_ground_state(kPointSet)
    dft_gs.initial_state()

    if "density_tol" not in siriusJson["parameters"]:
        density_tol = 1e-5
    else:
        density_tol = siriusJson["parameters"]["density_tol"]

    if "energy_tol" not in siriusJson["parameters"]:
        energy_tol = 1e-5
    else:
        energy_tol = siriusJson["parameters"]["energy_tol"]
    write_status = False

    initial_tol = 1e-2  # TODO: magic number
    E0 = dft_gs.find(density_tol, energy_tol, initial_tol, num_dft_iter, write_status)

    return {
        "E": E0,
        "dft_gs": dft_gs,
        "kpointset": kPointSet,
        "density": dft_gs.density(),
        "potential": dft_gs.potential(),
        "ctx": ctx,
    }


def dphk_factory(config="sirius.json"):
    """
    create Density, Potential, Hamiltonian, K_point_set
    K_point_set is initialized by Band.initialize_subspace

    Keyword Arguments:
    config -- (Default sirius.json) json configuration
    """

    from . import Band, K_point_set, Potential, Density, Hamiltonian, Simulation_context
    import json

    siriusJson = json.load(open(config))
    ctx = Simulation_context(json.dumps(siriusJson))
    ctx.initialize()
    density = Density(ctx)
    potential = Potential(ctx)
    hamiltonian = Hamiltonian(potential)

    if "vk" in siriusJson["parameters"]:
        vk = siriusJson["parameters"]["vk"]
        kPointSet = K_point_set(ctx, vk)
    else:
        if "shiftk" in siriusJson["parameters"]:
            shiftk = siriusJson["parameters"]["shiftk"]
        else:
            shiftk = [0, 0, 0]
        if "ngridk" in siriusJson["parameters"]:
            gridk = siriusJson["parameters"]["ngridk"]
        use_symmetry = siriusJson["parameters"]["use_symmetry"]
        kPointSet = K_point_set(ctx, gridk, shiftk, use_symmetry)

    Band(ctx).initialize_subspace(kPointSet, hamiltonian)

    return {
        "kpointset": kPointSet,
        "density": density,
        "potential": potential,
    }


def kpoint_index(kp, ctx):
    """
    Returns tuple (idx, idy, idz), which corresponds
    to the position of the kpoint in the K-point grid

    Keyword Arguments:
    kp -- K point
    ctx -- simulation context
    """
    import numpy as np

    pm = ctx.cfg.parameters
    shiftk = np.array(pm.shiftk, dtype=np.int)
    ngridk = np.array(pm.ngridk, dtype=np.int)
    ik = np.array(kp.vk) * ngridk - shiftk / 2
    if not np.isclose(ik - ik.astype(np.int), 0).all():
        # single k-point given in vk
        # TODO: give proper coordinates
        ik = [0, 0, 0]
    else:
        ik = ik.astype(np.int)

    return tuple(ik)
