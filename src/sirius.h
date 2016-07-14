// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that 
// the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the 
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file sirius.h
 *   
 *  \brief "All-in-one" include file.
 */

#ifndef __SIRIUS_H__
#define __SIRIUS_H__

#include "input.h"
#include "cmd_args.h"
#include "constants.h"
#include "linalg.h"
#include "radial_grid.h"
#include "spline.h"
#include "radial_solver.h"
#include "sht.h"
#include "gaunt.h"
#include "splindex.h"
#include "json_tree.h"
#include "hdf5_tree.h" 
#include "xc_functional.h"
#include "mpi_grid.h"
#include "sirius_io.h"
#include "descriptors.h"
#include "mixer.h"
#include "atom_type.h"
#include "atom_symmetry_class.h"
#include "atom.h"
#include "unit_cell.h"
#include "step_function.h"
#include "periodic_function.h"
#include "k_point.h"
#include "band.h"
#include "potential.h"
#include "k_set.h"
#include "density.h"
#include "force.h"
#include "dft_ground_state.h"
#include "simulation_context.h"
#include "simulation_parameters.h"

#endif // __SIRIUS_H__

/** \mainpage Welcome to SIRIUS
    \section intro Introduction
    SIRIUS is ...
    \section install Installation
    ...
*/

/** \page coding Coding style
       
    Below are some basic style rules we try to follow:
        - Page width is approximately 120 characters. Screens are wide nowdays and 80 characters is an 
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
