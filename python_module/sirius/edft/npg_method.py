import numpy as np
from sirius import inner, l2norm, CoefficientArray, Logger, diag
import h5py
from copy import deepcopy
from scipy.constants import physical_constants
from mpi4py import MPI
from .free_energy import FreeEnergy, s


logger = Logger()


def _constraint34(u, tau, T, pe, vkw):
    """
    Keywoard Arguments:
    u        --
    tau      --
    T        -- temperature
    pe       -- number of electrons
    vkw      -- k-point weights (same shape as u)
    """
    from scipy.optimize import minimize, Bounds
    kb = (physical_constants['Boltzmann constant in eV/K'][0] /
          physical_constants['Hartree energy in eV'][0])

    res = minimize(
        lambda x: np.real(np.linalg.norm(x - u)**2 + 2*tau*kb*T*np.sum(s(x))),
        x0=u,
        bounds=Bounds(np.zeros_like(u), np.ones_like(u)),
        constraints={
            'fun': lambda y: np.sum(vkw*y**2) - pe,
            'type': 'eq'
        },
        options={'ftol': 1e-9, 'maxiter': 20, 'disp': False}
    )
    if not res['success']:
        raise ValueError('optimization did not converge')
    return res['x'], res


def constraint34(u, tau, T, pe, comm, kweights):
    """
    it is called 34 because this is subproblem (3.4) from the Ulbrich paper
    """

    if isinstance(u, CoefficientArray):
        # stack to numpy array
        # allgather and stack to columns
        kw = deepcopy(u)
        for k in kw._data.keys():
            kw[k] = np.ones_like(kw[k]) * kweights[k]

        vu_tmp = comm.allgather(u.to_array())
        # count number of elements per rank
        proc_sizes = list(map(np.size, vu_tmp))
        # get offset per rank
        offsets = np.hstack([0, np.cumsum(proc_sizes[:-1])]).astype(np.int)
        # allgather occupation numbers and gradient (because constraint is global)
        vu = np.hstack(vu_tmp)
        vkw = np.hstack(comm.allgather(kw.to_array()))

        # solve constrained lsq problem
        xs, res = _constraint34(vu, tau, T, pe, vkw)
        offset = offsets[comm.rank]
        lsize = proc_sizes[comm.rank]
        x_loc = xs[offset:offset+lsize]
        # unpack results
        x = deepcopy(u)
        x.from_array(x_loc)
        return x, res
    else:
        return _constraint34(u, tau, T, pe)


class NPGMethod:
    def __init__(self, M, H):
        """
        Keyword Arguments:
        M -- free energy
        H -- Hamiltonian
        """
        self.M = M
        self.H = H
        # num spins
        self.ns = 2 if M.energy.kpointset.ctx().num_mag_dims() == 0 else 1
        self.comm = M.energy.kpointset.ctx().comm_k()
        self.kweights = M.energy.kpointset.w

    def run(self, X0, f0, maxiter=100, delta=0.7, eta=0.5, gamma=0.5):
        """
        Keyword Arguments:
        self --
        x0   --
        f0   --
        """
        Zk = X0 * np.sqrt(f0)
        Qk = 1
        Ck = self.M(X0, f0)
        hmax = 20

        # number of electrons in the system
        pe = self.M.energy.kpointset.ctx().unit_cell().num_electrons()
        tau = 1

        UZ, SZ, VhZ = Zk.svd(full_matrices=False)
        self.E = self.M(UZ, SZ**2)
        dEk = 2 * self.H(UZ, scale=False) @ (diag(SZ) @ VhZ) * self.kweights

        taus = []
        es = []
        for h in range(hmax):
            logger('tau:', tau)
            # query energy before evaluating the gradient
            Wk = Zk - tau * dEk
            U, y, Vh = Wk.svd(full_matrices=False)
            # constrain occupation numbers
            try:
                z, res = constraint34(y / np.sqrt(self.ns), tau, self.M.T,
                                      pe / self.ns,
                                      comm=self.comm,
                                      kweights=self.kweights)
                # print('constraint34 did not converge, skip')
            except ValueError:
                # constraint34 did not converge go to next iteration
                logger('skip line-search iteration')
                tau *= delta
                continue
            z *= np.sqrt(self.ns)
            z = z.asarray()
            Yk = U @ diag(z) @ Vh
            # break condition
            local_energy = self.M(U, z**2)
            tau *= delta
            taus.append(tau)
            es.append(local_energy)
            if local_energy <= Ck - gamma / 2 * l2norm(Yk - Zk)**2:
                logger('found admissible energy', local_energy)
                self.E = local_energy
                break
        else:
            taus.append(0)
            es.append(self.E)
            # TODO: add debug option to activate
            save_state(Zk, UZ, SZ**2, self.M.energy.kpointset)
            assert False

        Zkm = deepcopy(Zk)
        Zk = Yk
        Skm = Zk - Zkm
        # compute step size tau
        Qkm = Qk
        Qk = eta * Qkm + 1
        Ck = (eta * Qkm * Ck + self.M(U, z**2)) / Qk

        for i in range(1, maxiter):
            UZ, SZ, VhZ = Zk.svd(full_matrices=False)
            fn = SZ**2
            self.E = self.M(UZ, fn)
            dEkm = deepcopy(dEk)
            HX = self.H(UZ, scale=False)
            dEk = 2 * HX @ (diag(SZ) @ VhZ) * self.kweights
            Vkm = dEk - dEkm
            tau = np.abs(inner(Skm, Vkm)) / np.real(inner(Vkm, Vkm))
            # compute residual
            HXf = HX @ diag(fn)
            res = np.sqrt(l2norm(HXf - UZ @ (UZ.H @ HXf)))
            logger('%04d energy: %.8f res %.8f' % (i, self.E, res))
            taus = []
            es = []
            trial_points = []
            for h in range(hmax):
                # logger('tau:', tau)
                Wk = Zk - tau * dEk
                U, y, Vh = Wk.svd(full_matrices=False)
                # constrain occupation numbers
                try:
                    z, res = constraint34(y / np.sqrt(self.ns), tau, self.M.T,
                                          pe / self.ns,
                                          comm=self.comm,
                                          kweights=self.kweights)
                    # print('constraint34 did not converge, skip')
                except ValueError:
                    # constraint34 did not converge go to next iteration
                    tau *= delta
                    continue
                z *= np.sqrt(self.ns)
                z = z.asarray()
                Yk = U @ diag(z) @ Vh
                # break condition
                local_energy = self.M(U, z**2)
                trial_points.append({'E': local_energy,
                                     'tau': tau,
                                     'h': h,
                                     'z': z})
                es.append(local_energy)
                taus.append(tau)
                tau *= delta
                if local_energy <= Ck - gamma / 2 * l2norm(Yk - Zk)**2:
                    self.E = local_energy
                    break
            else:
                logger('line search failed')
                save_state(Zk, UZ, SZ**2, self.M.energy.kpointset)
                logger('save_state energy:', self.M(UZ, SZ**2))
                best = min(trial_points, key=lambda x: x['E'])
                if best['E'] < self.E:
                    logger('is recoverable with', best['E'], self.E)
                    tau = best['tau']
                    Wk = Zk - tau * dEk
                    U, y, Vh = Wk.svd(full_matrices=False)
                    z, res = constraint34(y / np.sqrt(self.ns), tau, self.M.T,
                                          pe / self.ns,
                                          comm=self.comm,
                                          kweights=self.kweights)
                    z *= np.sqrt(self.ns)
                    z = z.asarray()
                    Yk = U @ diag(z) @ Vh
                    # break condition
                    local_energy = self.M(U, z**2)
                    assert np.isclose(local_energy, best['E'])
                else:
                    logger('is NOT recoverable')
                    taus.append(0)
                    es.append(self.E)
                    assert False
            # copy
            Zkm = deepcopy(Zk)
            Zk = Yk
            Skm = Zk - Zkm
            Vkm = dEk - dEkm
            # compute step size tau
            Qkm = Qk
            Qk = eta * Qkm + 1
            Ck = (eta * Qkm * Ck + self.M(U, z**2)) / Qk
            # update tau
            tau = np.abs(inner(Skm, Vkm)) / inner(Vkm, Vkm)

        UZ, SZ, VhZ = Zk.svd(full_matrices=False)
        return UZ, SZ**2
