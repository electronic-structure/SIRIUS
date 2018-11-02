from __future__ import print_function


def store_pw_coeffs(kpointset, cn, ki=None, ispn=None):
    """
    kpoint -- K_point
    cn     -- numpy array
    ispn   -- spin component
    """
    from .coefficient_array import PwCoeffs
    from .py_sirius import MemoryEnum
    from .ot import matview
    import numpy as np


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


def DFT_ground_state_find(num_dft_iter=1, config='sirius.json'):
    """
    run DFT_ground_state

    Keyword Arguments:
    num_dft_iter -- (Default 1) number of SCF interations
    config       -- json configuration
    """

    from . import Simulation_context, K_point_set, DFT_ground_state
    import json

    siriusJson = json.load(open(config))
    ctx = Simulation_context(json.dumps(siriusJson))
    ctx.initialize()

    if 'shiftk' in siriusJson['parameters']:
        shiftk = siriusJson['parameters']['shiftk']
    else:
        shiftk = [0, 0, 0]
    if 'ngridk' in siriusJson['parameters']:
        gridk = siriusJson['parameters']['ngridk']
    use_symmetry = siriusJson['parameters']['use_symmetry']

    kPointSet = K_point_set(ctx, gridk, shiftk, use_symmetry)

    dft_gs = DFT_ground_state(kPointSet)

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

    dft_gs.find(potential_tol, energy_tol, num_dft_iter, write_status)
    ks = dft_gs.k_point_set()
    hamiltonian = dft_gs.hamiltonian()

    return {
        'dft_gs': dft_gs,
        'kpointset': ks,
        'hamiltonian': hamiltonian,
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

    from . import Band, K_point_set, Potential, Density, Hamiltonian, Simulation_context
    import json

    siriusJson = json.load(open(config))
    ctx = Simulation_context(json.dumps(siriusJson))
    ctx.initialize()
    density = Density(ctx)
    potential = Potential(ctx)
    hamiltonian = Hamiltonian(ctx, potential)

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
    from . import PwCoeffs
    import numpy as np

    c0 = PwCoeffs(kpointset)
    x = PwCoeffs(dtype=np.complex)
    for key, c0_loc in c0.items():
        x_loc = np.zeros_like(c0_loc)
        x[key] = x_loc

    return c0, x



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

    for ik in range(len(ks)):
        bnd_k = {}
        bnd_k["kpoint"] = [0.0, 0.0, 0.0]
        for x in range(3):
            bnd_k["kpoint"][x] = ks(ik).vk()(x)
        bnd_e = []

        bnd_e = ks.get_band_energies(ik, 0)

        bnd_k["values"] = bnd_e
        dict["bands"].append(bnd_k)
    return dict


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs)
        return cls._instances[cls]


class Logger:
    __metaclass__ = Singleton

    def __init__(self, fout=None, comm=None, all_print=False):
        from mpi4py import MPI
        self.fout = fout
        self._all_print = all_print
        if self.fout is not None:
            with open(self.fout, 'w'):
                print('')
        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm

    def log(self, arg1, *args):
        """

        """
        if self.comm.rank == 0 or self._all_print:
            if self.fout is not None:
                with open(self.fout, 'a') as fh:
                    print(arg1, *args, file=fh)
            else:
                print(arg1, *args)
