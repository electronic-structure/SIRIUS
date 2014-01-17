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
    debug_level >= 1 : check symmetry of Hamiltonian radial integrals, check hermiticity of the Hamiltonian matrix, 
                       check plane wave matching condition, check local orbital linear independence \n
    debug_level >= 2 : check orthonormaliztion of the wave-functions \n
    debug_level >= 3 : check scalapack vs. lapack diagonalization   
*/
const int debug_level = 0;

/// verbosity level
/** Controls the ammount of information printed to standard output. 
    verbosity_level = 0 : silent mode, nothing is printed \n
    verbosity_level >= 1 : print global parameters of the calculation \n
    verbosity_level >= 2 : (suggested default) print information of any initialized k_set \n
    verbosity_level >= 3 : print extended information about band distribution \n
    verbosity_level >= 4 : print linearization energies \n
    verbosity_level >= 5 : print lowest eigen-values \n
*/
const int verbosity_level = 5;

const bool fix_apwlo_linear_dependence = false;

const basis_t basis_type = apwlo;

const radial_grid_t default_radial_grid_t = linear_exponential_grid;

const bool use_second_variation = true; 

#endif // __CONFIG_H__
