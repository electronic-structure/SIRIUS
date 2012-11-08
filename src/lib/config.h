#ifndef __CONFIG_H__
#define __CONFIG_H__

#define FORTRAN(x) x##_

const bool check_evecfv = false;

const bool test_scalar_wf = false;

const bool test_spinor_wf = false;

const bool hdf5_trace_errors = false;

const bool check_pseudo_charge = false;

const bool full_relativistic_core = false;

/// true if MPI_Init should be called
const bool call_mpi_init = false;

#endif // __CONFIG_H__
