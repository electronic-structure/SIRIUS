// Copyright (c) 2013-2019 Anton Kozhevnikov, Thomas Schulthess
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

/** \file check_gvec.hpp
 *
 *  \brief Check G-vector symmetry.
 */

#ifndef __CHECK_GVEC_HPP__
#define __CHECK_GVEC_HPP__

#include "symmetry/crystal_symmetry.hpp"

namespace sirius {

inline void
check_gvec(fft::Gvec const& gvec__, Crystal_symmetry const& sym__)
{
    PROFILE("sirius::check_gvec");

    #pragma omp parallel for
    for (int isym = 0; isym < sym__.size(); isym++) {
        auto sm = sym__[isym].spg_op.R;

        for (int igloc = 0; igloc < gvec__.count(); igloc++) {
            auto gv = gvec__.gvec(gvec_index_t::local(igloc));
            /* apply symmetry operation to the G-vector */
            auto gv_rot = dot(gv, sm);

            //== /* check limits */
            //== for (int x: {0, 1, 2}) {
            //==     auto limits = gvec__.fft_box().limits(x);
            //==     /* check boundaries */
            //==     if (gv_rot[x] < limits.first || gv_rot[x] > limits.second) {
            //==         std::stringstream s;
            //==         s << "rotated G-vector is outside of grid limits" << std::endl
            //==           << "original G-vector: " << gv << ", length: " << gvec__.cart(ig).length() << std::endl
            //==           << "rotation matrix: " << std::endl
            //==           << sm(0, 0) << " " << sm(0, 1) << " " << sm(0, 2) << std::endl
            //==           << sm(1, 0) << " " << sm(1, 1) << " " << sm(1, 2) << std::endl
            //==           << sm(2, 0) << " " << sm(2, 1) << " " << sm(2, 2) << std::endl
            //==           << "rotated G-vector: " << gv_rot << std::endl
            //==           << "limits: "
            //==           << gvec__.fft_box().limits(0).first << " " <<  gvec__.fft_box().limits(0).second << " "
            //==           << gvec__.fft_box().limits(1).first << " " <<  gvec__.fft_box().limits(1).second << " "
            //==           << gvec__.fft_box().limits(2).first << " " <<  gvec__.fft_box().limits(2).second;

            //==           TERMINATE(s);
            //==     }
            //== }
            int ig_rot = gvec__.index_by_gvec(gv_rot);
            /* special case where -G is equal to G */
            if (gvec__.reduced() && ig_rot < 0) {
                gv_rot = gv_rot * (-1);
                ig_rot = gvec__.index_by_gvec(gv_rot);
            }
            if (ig_rot < 0 || ig_rot >= gvec__.num_gvec()) {
                std::stringstream s;
                s << "rotated G-vector index is wrong" << std::endl
                  << "original G-vector: " << gv << std::endl
                  << "rotation matrix: " << std::endl
                  << sm << std::endl
                  << "rotated G-vector: " << gv_rot << std::endl
                  << "rotated G-vector index: " << ig_rot << std::endl
                  << "number of G-vectors: " << gvec__.num_gvec();
                RTE_THROW(s);
            }
        }
    }
}

inline void
check_gvec(fft::Gvec_shells const& gvec_shells__, Crystal_symmetry const& sym__)
{
    /* check G-vector symmetries */
    for (int igloc = 0; igloc < gvec_shells__.gvec_count_remapped(); igloc++) {
        auto G = gvec_shells__.gvec_remapped(igloc);

        for (int i = 0; i < sym__.size(); i++) {
            auto& invRT = sym__[i].spg_op.invRT;
            auto gv_rot = dot(invRT, G);

            /* local index of a rotated G-vector */
            int ig_rot = gvec_shells__.index_by_gvec(gv_rot);

            if (ig_rot == -1) {
                gv_rot = gv_rot * (-1);
                ig_rot = gvec_shells__.index_by_gvec(gv_rot);
                if (ig_rot == -1) {
                    std::stringstream s;
                    s << "Failed to find a rotated G-vector in the list" << std::endl
                      << "  local index of original vector: " << igloc << std::endl
                      << "  global index of G-shell: " << gvec_shells__.gvec_shell_remapped(igloc) << std::endl
                      << "  original G-vector: " << G << std::endl
                      << "  rotated G-vector: " << gv_rot;
                    RTE_THROW(s);
                }
            }
            if (ig_rot >= gvec_shells__.gvec_count_remapped()) {
                std::stringstream s;
                s << "G-vector index is above the boundary";
                RTE_THROW(s);
            }
        }
    }
}

} // namespace sirius

#endif // __CHECK_GVEC_HPP__
