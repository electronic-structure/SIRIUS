import time

from sirius import Logger, load_state
from sirius.ot import ApplyHamiltonian, Energy
from sirius.coefficient_array import PwCoeffs, diag
from ..edft import make_smearing
import numpy as np

logger = Logger()


def make_pwcoeffs(coefficient_array):
    out = PwCoeffs()
    out._data = coefficient_array._data
    return out


def validate_config(dd):
    """
    Using voluptuous to make sure that the config is valid and
    populate missing entries by their default values.
    """
    from voluptuous import Schema, Optional, Required, Any, Coerce

    teter_precond = {Required("type"): Any("teter")}
    kinetic_precond = {
        Required("type"): Any("kinetic"),
        Optional("eps", default=1e-3): Coerce(float),
    }
    identity_precond = {Required("type", default="identity"): Any("identity")}
    precond = Any(identity_precond, kinetic_precond, teter_precond)

    marzari = {
        Required("type"): Any("Marzari"),
        Optional("inner", default=2): int,
        Optional("fd_slope_check", default=False): bool,
    }
    neugebauer = {
        Required("type"): Any("Neugebauer"),
        Optional("kappa", default=0.3): Coerce(float),
    }
    restart = {Required("fname"): str}

    cg = {
        Required("method"): Any(marzari, neugebauer),
        Optional("restart"): Any(restart),
        Optional("type", default="FR"): Any("FR", "PR", "SD"),
        Optional("tol", default=1e-9): float,
        Optional("maxiter", default=300): int,
        Optional("restart", default=20): int,
        Optional("nscf", default=4): int,
        Optional("tau", default=0.1): Coerce(float),
        Optional("precond"): precond,
        Optional("callback_interval", default=50): int,
    }

    schema = Schema(cg)
    # validate schema, populate missing entries with default values
    return schema(dd)


def initial_state(sirius_input, nscf):
    from sirius import DFT_ground_state_find

    res = DFT_ground_state_find(nscf, config=sirius_input)
    ctx = res["ctx"]
    # m = ctx.max_occupancy()
    # not yet implemented for single spin channel system
    kset = res["kpointset"]
    potential = res["potential"]
    density = res["density"]
    H = ApplyHamiltonian(potential, kset)
    E = Energy(kset, potential, density, H)

    fn = kset.fn
    X = kset.C

    return X, fn, E, ctx, kset



def make_precond(cg_config, kset):
    """
    preconditioner factory
    """
    from sirius.edft.preconditioner import (
        make_kinetic_precond,
        make_kinetic_precond2,
        IdentityPreconditioner,
    )

    if cg_config["precond"]["type"].lower() == "teter":
        print("teter precond")
        return make_kinetic_precond2(kset)
    elif cg_config["precond"]["type"].lower() == "kinetic":
        print("kinetic precond")
        return make_kinetic_precond(kset, eps=cg_config["precond"]["eps"])
    elif cg_config["precond"]["type"].lower() == "identity":
        return IdentityPreconditioner()
    else:
        raise NotImplementedError(
            "this preconditioner does not exist:", str(cg_config["precond"])
        )


def run_marzari(config, sirius_config, callback=None, final_callback=None):
    """
    Keyword Arguments:
    config        -- dictionary
    sirius_config -- /path/to/sirius.json
    """
    from sirius.edft import MarzariCG as CG, FreeEnergy
    from sirius import smearing
    from sirius.edft.smearing import kb

    cg_config = config["CG"]
    if "restart" in config:
        nscf = 1
    else:
        nscf = cg_config["nscf"]

    X, fn, E, ctx, kset = initial_state(sirius_config, nscf)
    T = config["System"]["T"]
    smearing = make_smearing(config["System"]["smearing"], kT=kb*T, ctx=ctx, kset=kset)
    M = FreeEnergy(E=E, T=T, smearing=smearing)

    # load state
    if "restart" in config:
        logger("restart loading from " + config["restart"]["fname"])
        fname = config["restart"]["fname"]
        X = make_pwcoeffs(load_state(fname, kset, "X", np.complex128))
        fn = load_state(fname, kset, "fn", np.float64).asarray().flatten()
        M(X, fn)  # make sure band energies are set

    method_config = config["CG"]["method"]
    cg = CG(M, fd_slope_check=method_config["fd_slope_check"])
    K = make_precond(cg_config, kset)

    tstart = time.time()
    FE, X, fn, success = cg.run(
        X,
        fn,
        tol=cg_config["tol"],
        maxiter=cg_config["maxiter"],
        ninner=cg_config["method"]["inner"],
        K=K,
        callback=callback(kset, E=E),
    )
    assert success
    tstop = time.time()
    logger("cg.run took: ", tstop - tstart, " seconds")
    if final_callback is not None:
        final_callback(kset, E=E)(X=X, fn=fn)
    return X, fn, FE


def run_neugebauer(config, sirius_config, callback, final_callback, error_callback):
    """
    Keyword Arguments:
    config        -- dictionary
    sirius_config -- /path/to/sirius.json
    """
    from sirius.edft import NeugebauerCG as CG, FreeEnergy
    from sirius.edft.smearing import kb

    cg_config = config["CG"]
    if "restart" in config:
        nscf = 1
    else:
        nscf = cg_config["nscf"]

    X, fn, E, ctx, kset = initial_state(sirius_config, nscf)
    T = config["System"]["T"]
    smearing = make_smearing(config["System"]["smearing"], kT=kb*T, ctx=ctx, kset=kset)
    M = FreeEnergy(E=E, smearing=smearing)

    # load state, Neugebauer method requires X, eta (pseudo band-energies)
    if "restart" in config:
        logger("restart loading from " + config["restart"]["fname"])
        fname = config["restart"]["fname"]
        X = make_pwcoeffs(load_state(fname, kset, "X", np.complex128))
        # the diagoal of eta is stored in dumps, therefore it is real-valued
        ek = diag(load_state(fname, kset, "eta", np.float64)).asarray().flatten()
    else:
        ek = kset.e

    cg = CG(M)
    K = make_precond(cg_config, kset)

    tstart = time.time()
    X, fn, FE, success = cg.run(
        X,
        ek,
        tol=cg_config["tol"],
        K=K,
        maxiter=cg_config["maxiter"],
        kappa=cg_config["method"]["kappa"],
        restart=cg_config["restart"],
        cgtype=cg_config["type"],
        tau=cg_config["tau"],
        callback=callback(kset, E=E),
        error_callback=error_callback(kset, E=E),
    )
    if not success:
        logger("NOT converged.")
    else:
        logger("SUCCESSFULLY converged")
    tstop = time.time()
    logger("cg.run took: ", tstop - tstart, " seconds")
    if final_callback is not None:
        final_callback(kset, E=E)(X=X, fn=fn)
    return X, fn, FE


def run(ycfg, sirius_input, callback=None, final_callback=None, error_callback=None):
    """
    Keyword Arguments:
    ycfg         -- EDFT config (dict)
    sirius_input -- /path/to/sirius.json
    """

    def EmptyCallbackFactory(*args, **kw):
        return lambda **_: None

    if callback is None:
        callback = EmptyCallbackFactory
    if final_callback is None:
        final_callback = EmptyCallbackFactory
    if error_callback is None:
        error_callback = EmptyCallbackFactory

    method = ycfg["CG"]["method"]["type"].lower()
    if method == "marzari":
        if error_callback is not None:
            Logger()("WARNING: error callback is ignored in this method.")
        X, fn, FE = run_marzari(ycfg, sirius_input, callback, final_callback)
    elif method == "neugebauer":
        X, fn, FE = run_neugebauer(
            ycfg, sirius_input, callback, final_callback, error_callback=error_callback
        )
    else:
        raise Exception("invalid method given")

    logger("Final free energy: %.10f" % FE)
    return X, fn, FE


def store_density_potential(density, potential):
    density.save()
    potential.save()
