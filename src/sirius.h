// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

#include "sirius_internal.h"
#include "input.h"
#include "cmd_args.h"
#include "constants.h"
#include "radial_grid.h"
#include "spline.h"
#include "radial_solver.h"
#include "sht.h"
#include "gaunt.h"
#include "sddk.hpp"
#include "hdf5_tree.hpp" 
#include "xc_functional.h"
#include "descriptors.h"
#include "mixer.h"
#include "Unit_cell/atom_type.h"
#include "Unit_cell/atom_symmetry_class.h"
#include "Unit_cell/atom.h"
#include "Unit_cell/free_atom.hpp"
#include "Unit_cell/unit_cell.h"
#include "step_function.h"
#include "periodic_function.h"
#include "k_point.h"
#include "band.h"
#include "potential.h"
#include "k_point_set.h"
#include "density.h"
#include "Geometry/stress.hpp"
#include "Geometry/force.hpp"
#include "dft_ground_state.h"

#endif // __SIRIUS_H__
