from .ortho import loewdin
from types import SimpleNamespace
import numpy as np
from ..coefficient_array import einsum, diag
from ..coefficient_array import CoefficientArray as ca


def grad_eta(Hij, fn, ek, mu, smearing, kw):
    """
    Computes ∂L/∂η

    Arguments:
    Hij       -- subspace Hamiltonian
    fn        -- occupations
    ek        -- Fermi parameters ϵ_n
    mu        -- chemical potential
    smearing  -- occupation numbers
    kw        -- kpoint weights

    Returns:
    g_eta -- gradient wrt η of the free-energy Lagrangian
    """
    delta = smearing.delta(mu - ek)

    # g_eta_1: diagonal term, changes in the occupation numbers
    g_eta_1 = -1.0 * diag(diag(Hij) - kw * ek) * delta

    # g_eta_2: changes in the chemical potential
    dFdmu = np.sum(np.real(einsum("i,i", (diag(Hij) - kw * ek), delta)))
    k_sum_delta = np.sum(kw * delta)
    if np.abs(k_sum_delta) < 1e-10:
        g_eta_2 = 0
    else:
        g_eta_2 = diag(kw * delta) / k_sum_delta * dFdmu
    # off-diagonal terms -> subspace rotations
    II = diag(ca.ones_like(fn))
    Eij = ek - ek.T + II
    Fij = fn - fn.T
    for k in Eij.keys():
        EEc = np.abs(Eij[k]) < 1e-10
        Eij[k] = np.where(EEc, 1, Eij[k])
        Fij[k] = np.where(EEc, 0, Fij[k])

    g_eta_3 = Fij / Eij * Hij * (1 - II)
    g_eta = g_eta_1 + g_eta_2 + g_eta_3
    return g_eta


class LineEvaluator:
    """
    Evaluate free energy along a fixed direction.
    """

    def __init__(self, X, eta, M, G_X, G_eta, overlap):
        """
        Keyword Arguments:
        X     -- plane-wave coefficients
        eta   -- pseudo-Hamiltonian (must be diagonal)
        M     -- free energy evaluator
        G_X   -- direction X
        G_eta -- direction eta
        """

        self.X = X
        self.eta = eta
        self.M = M
        # search direction
        self.G_X = G_X
        self.G_eta = G_eta
        self.overlap = overlap

    def __call__(self, t):
        """
        Evaluate along line

        Returns:
        F  -- free energy
        Hx -- gradient (free of occupation, k-point-weight)
        X  -- plane-wave coefficients
        fn -- occupation numbers
        U  -- subspace rotation matrix
        """
        X_new = self.X + t * self.G_X
        eta_new = self.eta + t * self.G_eta
        ek, Ul = eta_new.eigh()
        X = loewdin(X_new, self.overlap) @ Ul
        fn, mu = self.M.smearing.fn(ek)

        # check fn

        FE, Hx = self.M(X, fn=fn, mu=mu, ek=ek)
        return SimpleNamespace(
            **{
                "free_energy": FE,
                "Hx": Hx,
                "X": X,
                "fn": fn,
                "mu": mu,
                "ek": ek,
                "Ul": Ul,
            }
        )
