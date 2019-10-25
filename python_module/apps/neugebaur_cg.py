#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse

import numpy as np
from mpi4py import MPI
from sirius import DFT_ground_state_find, Logger, PwCoeffs
from sirius.ot import ApplyHamiltonian, Energy

from sirius import save_state
from sirius.edft import (NeugebaurCG as CG,
                         kb,
                         make_fermi_dirac_smearing,
                         make_gaussian_spline_smearing)
from sirius.edft.free_energy import FreeEnergy
import time

parser = argparse.ArgumentParser('')
parser.add_argument('input', default='sirius.json', nargs='?')
parser.add_argument('--tol', '-q', dest='tol', type=float, default=1e-10)
parser.add_argument('--kappa', '-k',  type=float, default=0.3,
                    dest='kappa', help='eta prec')
parser.add_argument('--epsilon', '-e',  type=float,
                    default=0.001, dest='eps', help='wfct prec')
parser.add_argument('--restart', '-r', type=int, default=20,
                    dest='restart', help='CG restart')
parser.add_argument('--maxiter', '-m',  type=int, default=1000, dest='maxiter')
parser.add_argument('--tau', '-u',  type=float, default=0.5, dest='tau', help='bt-search parameter')
parser.add_argument('--temperature', '-T',  type=float,
                    default=300, dest='T', help='temperature')
parser.add_argument('-p', '--precond', dest='precond', default='ekin1', help='wfct prec')
parser.add_argument('--g_eta', '-g',  action='store_true', default=False,
                    dest='g_eta', help='g_eta')
parser.add_argument('--scf', '-s',  type=int, default=4,
                    dest='nscf', help='init with nscf iterations')
parser.add_argument('--cg', '-c',  type=str, default='FR',
                    dest='cg_type', help='CG update parameter')
parser.add_argument('--load', type=str,
                    default='neuge_final*.h5',
                    help='load initial state from files')


args = parser.parse_args()

logger = Logger()

_cgnames = {'FR': 'Fletcher-Reeves',
            'SD': 'Steepest-descent',
            'PR': 'Polak-Ribiere'}

if args.precond in ['ekin1', 'ekin2']:
    use_prec = True
else:
    use_prec = False

logger('input file         : ', args.input)
logger('temperature        : ', args.T, 'K')
logger('wfct preconditioner: ', use_prec)
if use_prec:
    logger('use_g_eta          : ', args.g_eta)
logger('CG restart         : ', args.restart)
logger('CG update          : ', _cgnames[args.cg_type])
logger('Wfct prec          : ', args.eps)
logger('kappa              : ', args.kappa)
logger('nscf               : ', args.nscf)

np.set_printoptions(precision=4, linewidth=120)

res = DFT_ground_state_find(args.nscf, config=args.input)
ctx = res['ctx']
m = ctx.max_occupancy()
# not yet implemented for single spin channel system
assert m == 1
kset = res['kpointset']
potential = res['potential']
density = res['density']
E = Energy(kset, potential, density, ApplyHamiltonian(potential, kset))
T = args.T
kT = kb*T

fn = kset.fn
X = kset.C

# smearing = make_fermi_dirac_smearing(T, ctx, kset)
smearing = make_gaussian_spline_smearing(T, ctx, kset)
M = FreeEnergy(E=E, T=T, smearing=smearing)
cg = CG(M)
tstart = time.time()

def callback(kset, interval=50, **kwargs):
    def _callback(fn, it, **kwargs):
        if it % interval == 0:
            save_state({'f': fn}, kset=kset, prefix='fn_%05d_' % it)

    return _callback


X, fn = cg.run(X, fn,
               tol=args.tol,
               prec=use_prec,
               prec_type=args.precond,
               maxiter=args.maxiter,
               kappa=args.kappa,
               eps=args.eps,
               restart=args.restart,
               cgtype=args.cg_type,
               tau=args.tau,
               callback=callback(kset))
tstop = time.time()
logger('cg.run took: ', tstop-tstart, ' seconds')
logger('final energy:', M(X, fn))
save_state({'X': X, 'f': fn}, kset=kset, prefix='neuge_final')
