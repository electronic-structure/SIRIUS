#ifndef __SIRIUS_H__
#define __SIRIUS_H__
#include <mpi.h>
#include <assert.h>
#include <stdio.h>
#include <sys/time.h>
#include <signal.h>
#include <omp.h>
#include <stdint.h>

#include <vector>
#include <map>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <complex>

extern "C" {
#include <spglib.h>
}
#include <fftw3.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_coupling.h>
#include <gsl/gsl_sf_erf.h>
#include <xc.h>
// libxc should do that
#undef FLOAT

#include <hdf5.h>
#include <libjson.h>

#include "typedefs.h"
#include "config.h"
#include "LebedevLaikov.h"
#ifdef _GPU_
#include "gpu_interface.h"
#endif
#include "platform.h"

// ===========================
// low-level stuff
// ===========================
#include "version.h"
#include "error_handling.h"
#include "timer.h"
#include "constants.h"
#include "mdarray.h"
#include "linalg.h"
#include "utils.h"
#include "radial_grid.h"
#include "spline.h"
#include "radial_solver.h"
#include "sht.h"
#include "gaunt.h"
#include "fft3d.h"
#include "json_tree.h"
#include "hdf5_tree.h" 
#include "libxc_interface.h"
#include "run_time_info.h"
#include "mpi_grid.h"
#include "splindex.h"
#include "sirius_io.h"

// =============================
// atoms        
// =============================
#include "atom_type.h"
#include "atom_symmetry_class.h"
#include "atom.h"

// =================================
// stack of classes for Global class
// =================================
#include "unit_cell.h"
#include "reciprocal_lattice.h"
#include "step_function.h"
#include "global.h"

// ===========================
// main classes
// ===========================
#include "sbessel_pw.h"
#include "periodic_function.h"
#include "band.h"
#include "kpoint.h"
#include "kpoint_set.h"
#include "potential.h"
#include "density.h"

#endif // __SIRIUS_H__
