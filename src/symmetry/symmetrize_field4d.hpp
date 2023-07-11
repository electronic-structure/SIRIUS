// Copyright (c) 2013-2023 Anton Kozhevnikov, Thomas Schulthess
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

/** \file symmetrize_field4d.hpp
 *
 *  \brief Symmetrize density and potential fields (scalar + vector).
 */

#ifndef __SYMMETRIZE_FIELD4D_HPP__
#define __SYMMETRIZE_FIELD4D_HPP__

#include "symmetrize_mt_function.hpp"
#include "symmetrize_pw_function.hpp"

namespace sirius {

inline void
symmetrize_field4d(Field4D& f__)
{
    auto& ctx = f__.ctx();

    /* quick exit: the only symmetry operation is identity */
    if (ctx.unit_cell().symmetry().size() == 1) {
        return;
    }

    /* symmetrize PW components */
    symmetrize_pw_function(ctx.unit_cell().symmetry(), ctx.remap_gvec(), ctx.sym_phase_factors(),
        ctx.num_mag_dims(), f__.pw_components());

    if (ctx.full_potential()) {
        symmetrize_mt_function(ctx.unit_cell().symmetry(), ctx.comm(), ctx.num_mag_dims(), f__.mt_components());
    }
}

}

#endif
