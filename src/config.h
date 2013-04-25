/** \file config.h

    \brief Main configuration header
*/

#ifndef __CONFIG_H__
#define __CONFIG_H__

#define FORTRAN(x) x##_

const bool test_spinor_wf = false;

const bool hdf5_trace_errors = false;

const bool check_pseudo_charge = false;

//** const bool full_relativistic_core = false;

/// level of internal debugging and checking

/** debug_level = 0 : nothing to do \n
    debug_level > 0 : check symmetry of Hamiltonian radial integrals, check hermiticity of the Hamiltonian matrix \n
    debug_level > 1 : check orthonormaliztion of the wave-functions \n
    debug_level > 2 : check scalapack vs. lapack diagonalization   
*/
const int debug_level = 2;

const int verbosity_level = 0;

const basis_t basis_type = apwlo;

#endif // __CONFIG_H__
