#ifndef __SIRIUS_H__
#define __SIRIUS_H__
#include <mpi.h>
#include <assert.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/stat.h>
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
#include <typeinfo>

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

//============================
// low-level stuff
//============================
#include "version.h"
#include "error_handling.h"
#include "timer.h"

#define stop_here Timer::print(); error_local(__FILE__, __LINE__, "stop_here macros is called");

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
#include "mpi_grid.h"
#include "splindex.h"
#include "sirius_io.h"
#include "descriptors.h"

//==============================
// atoms        
//==============================
#include "atom_type.h"
#include "atom_symmetry_class.h"
#include "atom.h"

//==================================
// stack of classes for Global class
//==================================
#include "unit_cell.h"
#include "reciprocal_lattice.h"
#include "step_function.h"
#include "global.h"

//============================
// main classes
//============================
#include "sbessel_pw.h"
#include "spheric_function.h"
#include "spheric_function_vector.h"
#include "spheric_function_gradient.h"
#include "periodic_function.h"
#include "k_point.h"
#include "band.h"
#include "potential.h"
#include "k_set.h"
#include "density.h"
#include "force.h"
#include "mixer.h"
#include "dft_ground_state.h"


#endif // __SIRIUS_H__
