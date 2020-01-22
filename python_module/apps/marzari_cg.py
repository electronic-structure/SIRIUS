#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse

import numpy as np
from mpi4py import MPI
from sirius import DFT_ground_state_find, Logger
from sirius.ot import ApplyHamiltonian, Energy

from sirius import save_state
from sirius.edft import MarzariCG as CG, FreeEnergy, kb
from sirius.edft.smearing import GaussianSplineSmearing
import time


def callback(kset, interval=50, **kwargs):
    def _callback(fn, it, **kwargs):
        if it % interval == 0:
            save_state({'f': fn}, kset=kset, prefix='fn_%05d_' % it)

    return _callback

parser = argparse.ArgumentParser('')
parser.add_argument('input', default='sirius.json', nargs='?')
parser.add_argument('-q', '--tol', dest='tol', type=float, default=1e-10)
parser.add_argument('-e', '--epsilon', type=float,
                    default=0.001, dest='eps', help='wfct prec')
parser.add_argument('-r', '--restart', type=int, default=20,
                    dest='restart', help='CG restart')
parser.add_argument('-m', '--maxiter', type=int, default=100, dest='maxiter')
parser.add_argument('-p', '--precond', type=str, default='ekin1', dest='precond')
parser.add_argument('-d', '--dd', type=float, default=1e-5, dest='dd')
parser.add_argument('--check-slope', action='store_true', default=False, dest='check_slope')
parser.add_argument('-i', '--ni', type=int, default=2, dest='ni')
parser.add_argument('-u', '--tau', type=float, default=0.5, dest='tau', help='bt-search parameter')
parser.add_argument('-T', '--temperature', type=float,
                    default=300, dest='T', help='temperature')
parser.add_argument('-s', '--scf', type=int, default=4,
                    dest='nscf', help='init with nscf iterations')

args = parser.parse_args()

logger = Logger()

logger('input file         : ', args.input)
logger('temperature        : ', args.T, 'K')
logger('CG restart         : ', args.restart)
logger('CG update          : Fletcher-Reeves')
logger('Wfct prec          : ', args.eps)
logger('nscf               : ', args.nscf)
logger('delta (entropy)    :', args.dd)
logger('num inner          :', args.ni)
logger('precond            :', args.precond)
logger('tolerance          :', args.tol)

np.set_printoptions(precision=4, linewidth=120)

res = DFT_ground_state_find(args.nscf, config=args.input)
ctx = res['ctx']
m = ctx.max_occupancy()
# not yet implemented for single spin channel system

assert m == 1
kset = res['kpointset']
potential = res['potential']
density = res['density']
H = ApplyHamiltonian(potential, kset)
E = Energy(kset, potential, density, H)
T = args.T
kT = kb*T
nel = ctx.unit_cell().num_valence_electrons()
smearing = GaussianSplineSmearing(T=T, nel=nel, nspin=2, kw=kset.w)
# smearing = RegularizedFermiDiracSmearing(T=T, nel=nel, nspin=2, kw=kset.w)

fn = kset.fn
X = kset.C

M = FreeEnergy(E=E, T=T, smearing=smearing)
cg = CG(M, fd_slope_check=args.check_slope)
tstart = time.time()
FE, X, fn, success = cg.run(X, fn,
                            maxiter=args.maxiter,
                            ninner=args.ni,
                            prec_type=args.precond,
                            tol=args.tol,
                            callback=callback(kset))
tstop = time.time()
logger('cg.run took: ', tstop-tstart, ' seconds')
if not success:
    logger('!!! CG DID NOT CONVERGE !!!')
logger('final energy:', M(X, fn))
save_state({'X': X, 'f': fn}, kset=kset, prefix='marzari_final')
