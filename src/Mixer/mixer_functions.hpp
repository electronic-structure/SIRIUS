// Copyright (c) 2013-2019 Simon Frasch, Anton Kozhevnikov, Thomas Schulthess
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

/** \file mixer_functions.hpp
 *
 *  \brief Contains declarations of functions required for mixing.
 */

#ifndef __MIXER_FUNCTIONS_HPP__
#define __MIXER_FUNCTIONS_HPP__

#include "periodic_function.hpp"
#include "SDDK/memory.hpp"
#include "Mixer/mixer.hpp"
#include "Density/paw_density.hpp"

namespace sirius {

namespace mixer {

FunctionProperties<Periodic_function<double>> periodic_function_property();

FunctionProperties<Periodic_function<double>> periodic_function_property_modified(bool use_coarse_gvec__);

FunctionProperties<sddk::mdarray<double_complex, 4>> density_function_property();

FunctionProperties<paw_density> paw_density_function_property();

} // namespace mixer

} // namespace sirius

#endif
