#ifndef __CONFIG_H__
#define __CONFIG_H__

#define FORTRAN(x) x##_

const bool check_evecfv = false;

const bool test_scalar_wf = false;

const bool test_spinor_wf = false;

const bool hdf5_trace_errors = false;

const bool check_pseudo_charge = false;

const bool full_relativistic_core = false;

const int scalapack_nb = 2;

const eigen_value_solver_t eigen_value_solver = lapack;

const int debug_level = 0;

const int verbosity_level = 1;

#endif // __CONFIG_H__
