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

#include "json.hpp"
using json = nlohmann::json;

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
 *  \section intro Introduction
 *  SIRIUS is ...
 *  \section install Installation
 *   ...
 */

/** \page coding Coding style
 *      
 *  Below are some basic style rules we try to follow:
 *      - Page width is approximately 120 characters. Screens are wide nowdays and 80 characters is an 
 *        obsolete restriction. Going slightly over 120 characters is allowed if it is requird for the line continuity.
 *      - Identation: 4 spaces (no tabs)
 *      - Spaces between most operators:
 *        \code{.cpp}
 *            if (i < 5) j = 5;  
 *
 *            for (int k = 0; k < 3; k++)
 *
 *            int lm = l * l + l + m;
 *
 *            double d = std::abs(e);
 *
 *            int k = idx[3];
 *        \endcode
 *      - Spaces between function arguments:
 *        \code{.cpp}
 *            double d = some_func(a, b, c);
 *        \endcode
 *      - Curly braces start at the new line:
 *        \code{.cpp}
 *            for (int i = 0; i < 10; i++)
 *            {
 *                ...
 *            }
 * 
 *            inline int num_points()
 *            {
 *                return num_points_;
 *            }
 *        \endcode
 *      - Single line 'if' statements and 'for' loops don't have curly braces:
 *        \code{.cpp}  
 *            if (i == 4) some_variable = 5;
 *
 *            for (int k = 0; k < 10; k++) do_something(k);
 *        \endcode
 *        or if it doesn't fit into 120 characters:
 *        \code{.cpp}
 *            if (i == 4) 
 *                some_variable = 5;
 *
 *            for (int k = 0; k < 10; k++) 
 *                do_something(k);
 *        \endcode
 *        but
 *        \code{.cpp}
 *            if (i == 4)
 *            {
 *                some_variable = 5;
 *            }
 *            else
 *            {
 *                some_variable = 6;
 *            }
 *        \endcode
 * 
 * 
 *  Class naming convention.
 *       
 *  Problem: all 'standard' naming conventions are not satisfactory. For example, we have a class 
 *  which does a DFT ground state. Following the common naming conventions it could be named like this:
 *  DFTGroundState, DftGroundState, dft_ground_state. Last two are bad, because DFT (and not Dft or dft)
 *  is a well recognized abbreviation. First one is band because capital G adds to DFT and we automaticaly
 *  read DFTG round state.
 *  
 *  Solution: we can propose the following: DFTgroundState or DFT_ground_state. The first variant still 
 *  doens't look very good because one of the words is captalized (State) and one (ground) - is not. So we pick 
 *  the second variant: DFT_ground_state (by the way, this is close to the Bjarne Stroustrup's naiming convention,
 *  where he uses first capital letter and underscores, for example class Io_obj).
 *
 *  Some other examples:
 *      - class Ground_state (composed of two words) 
 *      - class FFT_interface (composed of an abbreviation and a word)
 *      - class Interface_XC (composed of a word and abbreviation)
 *      - class Spline (single word)
 *      
 *  Exceptions are allowed if it makes sense. For example, low level utility classes like 'mdarray' (multi-dimentional
 *  array) or 'pstdout' (parallel standard output) are named with small letters. 
 *
 */
