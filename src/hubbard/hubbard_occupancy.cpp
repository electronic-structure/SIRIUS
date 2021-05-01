// Copyright (c) 2013-2018 Mathieu Taillefumier, Anton Kozhevnikov, Thomas Schulthess
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

/** \file hubbard_occupancy.hpp
 *
 *  \brief Generate occupation matrix for Hubbard orbitals.
 */

/** Compute the occupation numbers associated to the hubbard wavefunctions (locally centered orbitals, wannier
 *  functions, etc) that are relevant for the hubbard correction.
 *
 * These quantities are defined by
 * \f[
 *    n_{m,m'}^I \sigma = \sum_{kv} f(\varepsilon_{kv}) |<\psi_{kv}| phi_I_m>|^2
 * \f]
 * where \f[m=-l\cdot l$ (same for m')\f], I is the atom.
 *
 * Requires symmetrization. */

#include "hubbard.hpp"
#include "symmetry/symmetrize.hpp"

namespace sirius {
//void Hubbard::symmetrize_occupancy_matrix(sddk::mdarray<double_complex, 4>& om__)
//{
//    //auto& sym = unit_cell_.symmetry();
//
//    STOP();
//
//    //// check if we have some symmetries
//    //if (sym.num_mag_sym()) {
//    //    int lmax  = unit_cell_.lmax();
//    //    int lmmax = utils::lmmax(lmax);
//
//    //    mdarray<double, 2> rotm(lmmax, lmmax);
//    //    mdarray<double_complex, 4> dm(om__.size(0), om__.size(1), om__.size(2), unit_cell_.num_atoms());
//    //    double alpha = 1.0 / static_cast<double>(sym.num_mag_sym());
//
//    //    dm.zero();
//
//    //    for (int i = 0; i < sym.num_mag_sym(); i++) {
//    //        int  pr   = sym.magnetic_group_symmetry(i).spg_op.proper;
//    //        auto eang = sym.magnetic_group_symmetry(i).spg_op.euler_angles;
//    //        int isym  = sym.magnetic_group_symmetry(i).isym;
//    //        SHT::rotation_matrix(lmax, eang, pr, rotm);
//    //        auto spin_rot_su2 = rotation_matrix_su2(sym.magnetic_group_symmetry(i).spin_rotation);
//
//    //        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
//    //            auto& atom_type = unit_cell_.atom(ia).type();
//    //            int   ja        = sym.sym_table(ia, isym);
//    //            if (atom_type.hubbard_correction()) {
//    //                sirius::symmetrize(om__, unit_cell_.atom(ia).type().indexb_hub(), ia, ja,
//    //                                   ctx_.num_mag_comp(), rotm, spin_rot_su2, dm, true);
//    //            }
//    //        }
//    //    }
//
//    //    for (auto d3 = 0u; d3 < dm.size(3); d3++) {
//    //        for(auto d1 = 0u; d1 < dm.size(1); d1++) {
//    //            for(auto d0 = 0u; d0 < dm.size(0); d0++) {
//    //                dm(d0, d1, 0, d3) = dm(d0, d1, 0, d3) * alpha;
//    //                dm(d0, d1, 1, d3) = dm(d0, d1, 1, d3) * alpha;
//    //                dm(d0, d1, 2, d3) = std::conj(dm(d0, d1, 2, d3))  * alpha;
//    //                dm(d0, d1, 3, d3) = dm(d0, d1, 2, d3) * alpha;
//    //            }
//    //        }
//    //    }
//    //    dm >> om__;
//    //}
//}

}
