from __future__ import print_function

import h5py
from mpi4py import MPI
from .coefficient_array import PwCoeffs, CoefficientArray
from .py_sirius import MemoryEnum
from .ot import matview
from .logger import Logger
import numpy as np

logger = Logger()


def save_state(objs_dict, kset, prefix='fail'):
    """
    Arguments:
    objs_dict = dictionary(string: CoefficientArray), example: {'Z': Z, 'G': G}
    dump current state to HDF5
    """
    logger('save state')
    rank = MPI.COMM_WORLD.rank
    import sirius.ot as ot
    with h5py.File(prefix+'%d.h5' % rank, 'w') as fh5:
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

    ctype=np.matrix
    out = CoefficientArray(dtype=dtype, ctype=np.matrix)

    idx_to_k = {}
    for i, kp in enumerate(kset):
        kindex = kpoint_index(kp, kset.ctx())
        idx_to_k[kindex] = i

    if '*' in filename:
        files = glob.glob(filename)
    else:
        files = [filename]

    for fi in files:
        with h5py.File(fi, 'r') as fh5:
            for key in fh5[name].keys():
                ki = tuple(fh5[name][key].attrs['ki'])
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

    if isinstance(cn, PwCoeffs):
        assert (ki is None)
        assert (ispn is None)
        for key, v in cn.items():
            k, ispn = key
            n, m = v.shape
            assert (np.isclose(matview(v).H * v, np.eye(m, m)).all())
            psi = kpointset[k].spinor_wave_functions()
            psi.pw_coeffs(ispn)[:, :v.shape[1]] = v
            on_device = psi.preferred_memory_t() == MemoryEnum.device
            if on_device:
                psi.copy_to_gpu()
    else:
        psi = kpointset[ki].spinor_wave_functions()
        on_device = psi.preferred_memory_t() == MemoryEnum.device
        psi.pw_coeffs(ispn)[:, :cn.shape[1]] = cn
        if on_device:
            psi.copy_to_gpu()


def DFT_ground_state_find(num_dft_iter=1, config='sirius.json', load=False):
    """
    run DFT_ground_state

    Keyword Arguments:
    num_dft_iter -- (Default 1) number of SCF interations
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
    from . import (Simulation_context,
                   K_point_set,
                   Band,
                   DFT_ground_state,
                   initialize_subspace,
                   vector3d_double)
    import json
    if isinstance(config, dict):
        ctx = Simulation_context(json.dumps(config))
        siriusJson = config
    else:
        siriusJson = json.load(open(config))
        ctx = Simulation_context(json.dumps(siriusJson))
    ctx.initialize()
    if 'vk' in siriusJson['parameters']:
        vk = siriusJson['parameters']['vk']
        kPointSet = K_point_set(ctx, [vector3d_double(x) for x in vk])
    else:
        if 'shiftk' in siriusJson['parameters']:
            # make sure shiftk is not a list of floats
            shiftk = [int(x) for x in siriusJson['parameters']['shiftk']]
        else:
            shiftk = [0, 0, 0]
        if 'ngridk' in siriusJson['parameters']:
           gridk = siriusJson['parameters']['ngridk']
        if 'use_symmetry' in siriusJson['parameters']:
            use_symmetry = siriusJson['parameters']['use_symmetry']
        else:
            use_symmetry = True
        kPointSet = K_point_set(ctx, gridk, shiftk, use_symmetry)

    dft_gs = DFT_ground_state(kPointSet)
    if load:
        density = dft_gs.density()
        potential = dft_gs.potential()
        density.load()
        if ctx.use_symmetry():
            density.symmetrize()
        density.generate_paw_loc_density()
        density.fft_transform(1)
        potential.generate(density)
        if ctx.use_symmetry():
            potential.symmetrize()
        potential.fft_transform(1)

        # dft_gs.potential().load()
        initialize_subspace(dft_gs, ctx)
        # find wfct
        Band(ctx).solve(kPointSet, dft_gs.hamiltonian())
        # get band occupancies according to band energies
        kPointSet.find_band_occupancies()
        E0 = dft_gs.total_energy()
    else:
        dft_gs.initial_state()

        if 'potential_tol' not in siriusJson['parameters']:
            potential_tol = 1e-5
        else:
            potential_tol = siriusJson['parameters']['potential_tol']

        if 'energy_tol' not in siriusJson['parameters']:
            energy_tol = 1e-5
        else:
            energy_tol = siriusJson['parameters']['energy_tol']
        write_status = False

        initial_tol = 1e-2 # TODO: magic number
        E0 = dft_gs.find(potential_tol, energy_tol, initial_tol, num_dft_iter, write_status)
        ks = dft_gs.k_point_set()
        hamiltonian = dft_gs.hamiltonian()

    return {
        'E': E0,
        'dft_gs': dft_gs,
        'kpointset': kPointSet,
        'hamiltonian': dft_gs.hamiltonian(),
        'density': dft_gs.density(),
        'potential': dft_gs.potential(),
        'ctx': ctx
    }


def dphk_factory(config='sirius.json'):
    """
    create Density, Potential, Hamiltonian, K_point_set
    K_point_set is initialized by Band.initialize_subspace

    Keyword Arguments:
    config -- (Default sirius.json) json configuration
    """

    from . import Band, K_point_set, Potential, Density, Hamiltonian, Simulation_context, vector3d_double
    import json

    siriusJson = json.load(open(config))
    ctx = Simulation_context(json.dumps(siriusJson))
    ctx.initialize()
    density = Density(ctx)
    potential = Potential(ctx)
    hamiltonian = Hamiltonian(ctx, potential)

    if 'vk' in siriusJson['parameters']:
        vk = siriusJson['parameters']['vk']
        kPointSet = K_point_set(ctx, [vector3d_double(x) for x in vk])
    else:
        if 'shiftk' in siriusJson['parameters']:
            shiftk = siriusJson['parameters']['shiftk']
        else:
            shiftk = [0, 0, 0]
        if 'ngridk' in siriusJson['parameters']:
            gridk = siriusJson['parameters']['ngridk']
        use_symmetry = siriusJson['parameters']['use_symmetry']
        kPointSet = K_point_set(ctx, gridk, shiftk, use_symmetry)

    Band(ctx).initialize_subspace(kPointSet, hamiltonian)

    return {
        'kpointset': kPointSet,
        'density': density,
        'potential': potential,
        'hamiltonian': hamiltonian
    }


def get_c0_x(kpointset, eps=0):
    """

    """
    from .coefficient_array import PwCoeffs
    import numpy as np

    c0 = PwCoeffs(kpointset)
    x = PwCoeffs(dtype=np.complex)
    for key, c0_loc in c0.items():
        x_loc = np.zeros_like(c0_loc)
        x[key] = x_loc

    return c0, x


def kpoint_index(kp, ctx):
    """
    Returns tuple (idx, idy, idz), which corresponds
    to the position of the kpoint in the K-point grid

    Keyword Arguments:
    kp -- K point
    ctx -- simulation context
    """
    import numpy as np

    pm = ctx.parameters_input()
    shiftk = np.array(pm.shiftk, dtype=np.int)
    ngridk = np.array(pm.ngridk, dtype=np.int)
    ik = np.array(kp.vk) * ngridk - shiftk/2
    if not np.isclose(ik-ik.astype(np.int), 0).all():
        # single k-point given in vk
        print('WARNING: could not identify k-point index')
        ik = [0, 0, 0]
    else:
        ik = ik.astype(np.int)

    return tuple(ik)
