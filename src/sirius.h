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

#include <hdf5.h>
#include <libjson.h>

#include "typedefs.h"
#include "LebedevLaikov.h"
#ifdef _GPU_
#include "gpu_interface.h"
#endif
#include "platform.h"
#include "timer.h"
#include "error_handling.h"


#include "constants.h"
#include "mdarray.h"
#include "linalg.h"
#include "utils.h"
#include "radial_grid.h"
#include "spline.h"
#include "radial_solver.h"
#include "sht.h"
#include "gaunt.h"
#include "splindex.h"
#include "json_tree.h"
#include "hdf5_tree.h" 
#include "libxc_interface.h"
#include "mpi_grid.h"
#include "sirius_io.h"
#include "descriptors.h"
#include "mixer.h"
#include "atom_type.h"
#include "atom_symmetry_class.h"
#include "atom.h"
#include "unit_cell.h"
#include "sbessel_pw.h"
#include "reciprocal_lattice.h"
#include "step_function.h"
#include "global.h"
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
#include "dft_ground_state.h"

#endif // __SIRIUS_H__

/** \mainpage Welcome to SIRIUS
    \section intro Introduction
    SIRIUS is ...
    \section install Installation
    ...
*/

/** \page coding Coding style
        - Page width: approximately 120 characters. Screens are wide nowdays and 80 characters is an 
          obsolete restriction. Going slightly over 120 characters is allowed if it is requird for the line continuity.
        - Identation: 4 spaces (no tabs)
        - Spaces between most operators:
            \code{.cpp}
                if (i < 5) j = 5;  

                for (int k = 0; k < 3; k++)

                int lm = l * l + l + m;

                double d = fabs(e);

                int k = idx[3];
            \endcode
        - Spaces between function arguments:
            \code{.cpp}
                double d = some_func(a, b, c);
            \endcode
        - Curly braces start at the new line:
            \code{.cpp}
                for (int i = 0; i < 10; i++)
                {
                   ...
                }

                inline int num_points()
                {
                    return num_points_;
                }
            \endcode


*/
