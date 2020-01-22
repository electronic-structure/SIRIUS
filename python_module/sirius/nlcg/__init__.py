import yaml
import argparse
import time

from sirius import save_state, Logger
from sirius.ot import ApplyHamiltonian, Energy

logger = Logger()

def validate_config(dd):
    """
    Using voluptuous to make sure that the config is valid and
    populate missing entries by their default values.
    """
    from voluptuous import Schema, Optional, Required, Any, Length, Coerce

    teter_precond = {Required('type'): Any('teter')}
    kinetic_precond = {Required('type'): Any('kinetic'),
                       Optional('eps', default=1e-3): Coerce(float)}
    identity_precond = {Required('type', default='identity'): Any('identity')}
    precond = Any(identity_precond, kinetic_precond, teter_precond)

    marzari = {Required('type'): Any('Marzari'),
               Optional('inner', default=2): int,
               Optional('fd_slope_check', default=False): bool}
    neugebaur = {Required('type'): Any('Neugebaur'), Optional('kappa', default=0.3): Coerce(float)}

    cg = {Required('method'): Any(marzari, neugebaur),
          Optional('type', default='FR'): Any('FR', 'PR'),
          Optional('tol', default=1e-9): float,
          Optional('maxiter', default=300): int,
          Optional('restart', default=20): int,
          Optional('nscf', default=4): int,
          Optional('tau', default=0.1): Coerce(float),
          Optional('precond'): precond,
          Optional('callback_interval', default=50): int
    }

    schema = Schema(cg)
    # validate schema, populate missing entries with default values
    return schema(dd)


def initial_state(sirius_input, nscf):
    from sirius import DFT_ground_state_find
    res = DFT_ground_state_find(nscf, config=sirius_input)
    ctx = res['ctx']
    m = ctx.max_occupancy()
    # not yet implemented for single spin channel system
    # assert m == 1
    kset = res['kpointset']
    potential = res['potential']
    density = res['density']
    H = ApplyHamiltonian(potential, kset)
    E = Energy(kset, potential, density, H)

    fn = kset.fn
    X = kset.C

    return X, fn, E, ctx, kset


def make_smearing(label, T, ctx, kset):
    """
    smearing factory
    """
    from sirius.edft import make_fermi_dirac_smearing, make_gaussian_spline_smearing
    if label == 'fermi-dirac':
        return make_fermi_dirac_smearing(T, ctx, kset)
    elif label == 'gaussian-spline':
        return make_gaussian_spline_smearing(T, ctx, kset)
    else:
        raise NotImplementedError('invalid smearing: ', label)


def make_precond(cg_config, kset):
    """
    preconditioner factory
    """
    from sirius.edft.preconditioner import make_kinetic_precond, make_kinetic_precond2
    if cg_config['precond']['type'].lower() == 'teter':
        print('teter precond')
        return make_kinetic_precond2(kset)
    elif cg_config['precond']['type'].lower() == 'kinetic':
        print('kinetic precond')
        return make_kinetic_precond(kset, eps=cg_config['precond']['eps'])
    else:
        raise NotImplementedError('this preconditioner does not exist:', str(cg_config['precond']))


def run_marzari(config, sirius_config, callback=None, final_callback=None):
    """
    Keyword Arguments:
    config        -- dictionary
    sirius_config -- /path/to/sirius.json
    """
    from sirius.edft import MarzariCG as CG, FreeEnergy

    cg_config = config['CG']

    X, fn, E, ctx, kset = initial_state(sirius_config, cg_config['nscf'])

    T = config['System']['T']
    smearing = make_smearing(config['System']['smearing'], T, ctx, kset)
    M = FreeEnergy(E=E, T=T, smearing=smearing)
    method_config = config['CG']['method']
    cg = CG(M, fd_slope_check=method_config['fd_slope_check'])
    K = make_precond(cg_config, kset)

    tstart = time.time()
    FE, X, fn, success = cg.run(X, fn,
                                tol=cg_config['tol'],
                                maxiter=cg_config['maxiter'],
                                ninner=cg_config['method']['inner'],
                                K=K,
                                callback=callback(kset, E=E))
    assert success
    tstop = time.time()
    logger('cg.run took: ', tstop-tstart, ' seconds')
    if final_callback is not None:
        final_callback(kset, E=E)(X=X, fn=fn)
    return X, fn, FE


def run_neugebaur(config, sirius_config, callback=None, final_callback=None):
    """
    Keyword Arguments:
    config        -- dictionary
    sirius_config -- /path/to/sirius.json
    """
    from sirius.edft import NeugebaurCG as CG, FreeEnergy

    cg_config = config['CG']
    X, fn, E, ctx, kset = initial_state(sirius_config, cg_config['nscf'])
    T = config['System']['T']
    smearing = make_smearing(config['System']['smearing'], T, ctx, kset)
    M = FreeEnergy(E=E, T=T, smearing=smearing)
    cg = CG(M)
    K = make_precond(cg_config, kset)

    tstart = time.time()
    X, fn, FE, success = cg.run(X, fn,
                                tol=cg_config['tol'],
                                K=K,
                                maxiter=cg_config['maxiter'],
                                kappa=cg_config['method']['kappa'],
                                restart=cg_config['restart'],
                                cgtype=cg_config['type'],
                                tau=cg_config['tau'],
                                callback=callback(kset, E=E))
    assert success
    tstop = time.time()
    logger('cg.run took: ', tstop-tstart, ' seconds')
    if final_callback is not None:
        final_callback(kset, E=E)(X=X, fn=fn)
    return X, fn, FE



def run(ycfg, sirius_input, callback=None, final_callback=None):
    """
    Keyword Arguments:
    ycfg         -- EDFT config (dict)
    sirius_input -- /path/to/sirius.json
    """
    method = ycfg['CG']['method']['type'].lower()
    if method == 'marzari':
        X, fn, FE = run_marzari(ycfg,
                                sirius_input,
                                callback, final_callback)
    elif method == 'neugebaur':
        X, fn, FE = run_neugebaur(ycfg,
                                  sirius_input,
                                  callback, final_callback)
    logger('Final free energy: %.10f' % FE)
    return X, fn, FE


def store_density_potential(density, potential):
    density.save()
    potential.save()
