// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

/** \file wf_ortho.hpp
 *
 *  \brief Wave-function orthonormalization.
 */
#ifndef __WF_OUTER_HPP__
#define __WF_OUTER_HPP__

#include "SDDK/wave_functions.hpp"

namespace sddk {
    void orthogonalize(memory_t mem__,
                       linalg_t la__,
                       int ispn__,
                       Wave_functions &phi__,
                       Wave_functions &hphi__,
                       int N__,
                       int n__,
                       dmatrix<double> &o__,
                       Wave_functions &tmp__);

    void orthogonalize(memory_t mem__,
                       linalg_t la__,
                       int ispn__,
                       Wave_functions &phi__,
                       Wave_functions &hphi__,
                       int N__,
                       int n__,
                       dmatrix <double_complex> &o__,
                       Wave_functions &tmp__);

    void orthogonalize(memory_t mem__,
                       linalg_t la__,
                       int ispn__,
                       Wave_functions &phi__,
                       Wave_functions &hphi__,
                       Wave_functions &ophi__,
                       int N__,
                       int n__,
                       dmatrix<double> &o__,
                       Wave_functions &tmp__);

    void orthogonalize(memory_t mem__,
                       linalg_t la__,
                       int ispn__,
                       Wave_functions &phi__,
                       Wave_functions &hphi__,
                       Wave_functions &ophi__,
                       int N__,
                       int n__,
                       dmatrix <double_complex> &o__,
                       Wave_functions &tmp__);
}

#endif