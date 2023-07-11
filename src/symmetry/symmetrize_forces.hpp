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

/** \file symmetrize_forces.hpp
 *
 *  \brief Symmetrize atomic forces.
 */

#ifndef __SYMMETRIZE_FORCES_HPP__
#define __SYMMETRIZE_FORCES_HPP__

#include "crystal_symmetry.hpp"

namespace sirius {

inline void
symmetrize_forces(Unit_cell const& uc__, sddk::mdarray<double, 2>& f__)
{
    auto& sym = uc__.symmetry();

    if (sym.size() == 1) {
        return;
    }

    sddk::mdarray<double, 2> sym_forces(3, uc__.spl_num_atoms().local_size());
    sym_forces.zero();

    for (int isym = 0; isym < sym.size(); isym++) {
        auto const& Rc = sym[isym].spg_op.Rc;

        for (int ia = 0; ia < uc__.num_atoms(); ia++) {
            r3::vector<double> force_ia(&f__(0, ia));
            int ja        = sym[isym].spg_op.sym_atom[ia];
            auto location = uc__.spl_num_atoms().location(ja);
            if (location.rank == uc__.comm().rank()) {
                auto force_ja = dot(Rc, force_ia);
                for (int x : {0, 1, 2}) {
                    sym_forces(x, location.local_index) += force_ja[x];
                }
            }
        }
    }

    double alpha = 1.0 / double(sym.size());
    for (int ia = 0; ia < uc__.spl_num_atoms().local_size(); ia++) {
        for (int x: {0, 1, 2}) {
            sym_forces(x, ia) *= alpha;
        }
    }
    double* sbuf = uc__.spl_num_atoms().local_size() ? sym_forces.at(sddk::memory_t::host) : nullptr;
    uc__.comm().allgather(sbuf, f__.at(sddk::memory_t::host), 3 * uc__.spl_num_atoms().local_size(),
        3 * uc__.spl_num_atoms().global_offset());
}

}

#endif
