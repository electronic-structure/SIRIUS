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

/** \file symmetrize.hpp
 *
 *  \brief Symmetrize scalar and vector functions and various matrices.
 *
 *  The following objects are need to be symmetrized:
 *    - scalar and vector functions
 *    - PAW functions (defined for a subset of atoms)
 *    - LAPW density matrix (on-site)
 *    - density matrix of the pseudopotential formalism (on-site)
 *    - Hubbard occupation matrix (local and non-local)
 */

#ifndef __SYMMETRIZE_HPP__
#define __SYMMETRIZE_HPP__

#include "crystal_symmetry.hpp"
#include "fft/gvec.hpp"
#include "SDDK/omp.hpp"
#include "typedefs.hpp"
#include "sht/sht.hpp"
#include "utils/profiler.hpp"
#include "utils/rte.hpp"
#include "function3d/spheric_function_set.hpp"

namespace sirius {

/// Symmetrize density or occupancy matrix according to a given list of basis functions.
/** Density matrix arises in LAPW or PW methods. In PW it is computed in the basis of beta-projectors. Occupancy
 *  matrix is computed for the Hubbard-U correction. In both cases the matrix has the same structure and is
 *  symmetrized in the same way The symmetrization does depend explicitly on the beta or wfc. The last
 *  parameter is on when the atom has spin-orbit coupling and hubbard correction in
 *  that case, we must skip half of the indices because of the averaging of the
 *  radial integrals over the total angular momentum
 */
inline void
symmetrize(const sddk::mdarray<std::complex<double>, 4>& ns_, basis_functions_index const& indexb, const int ia, const int ja,
           const int ndm, sddk::mdarray<double, 2> const& rotm, sddk::mdarray<std::complex<double>, 2> const& spin_rot_su2,
           sddk::mdarray<std::complex<double>, 4>& dm_, const bool hubbard_)
{
    for (int xi1 = 0; xi1 < indexb.size(); xi1++) {
        int l1  = indexb[xi1].am.l();
        int lm1 = indexb[xi1].lm;
        int o1  = indexb[xi1].order;

        if ((hubbard_) && (xi1 >= (2 * l1 + 1))) {
            break;
        }

        for (int xi2 = 0; xi2 < indexb.size(); xi2++) {
            int l2  = indexb[xi2].am.l();
            int lm2 = indexb[xi2].lm;
            int o2  = indexb[xi2].order;
            std::array<std::complex<double>, 3> dm_rot_spatial = {0, 0, 0};

            //} the hubbard treatment when spin orbit coupling is present is
            // foundamentally wrong since we consider the full hubbard
            // correction with a averaged wave function (meaning we neglect the
            // L.S correction within hubbard). A better option (although still
            // wrong from physics pov) would be to consider a multi orbital case.

            if ((hubbard_) && (xi2 >= (2 * l2 + 1))) {
                break;
            }

            //      if (l1 == l2) {
            // the rotation matrix of the angular momentum is block
            // diagonal and does not couple different l.
            for (int j = 0; j < ndm; j++) {
                for (int m3 = -l1; m3 <= l1; m3++) {
                    int lm3 = utils::lm(l1, m3);
                    int xi3 = indexb.index_by_lm_order(lm3, o1);
                    for (int m4 = -l2; m4 <= l2; m4++) {
                        int lm4 = utils::lm(l2, m4);
                        int xi4 = indexb.index_by_lm_order(lm4, o2);
                        dm_rot_spatial[j] += ns_(xi3, xi4, j, ia) * rotm(lm1, lm3) * rotm(lm2, lm4);
                    }
                }
            }

            /* magnetic symmetrization */
            if (ndm == 1) {
                dm_(xi1, xi2, 0, ja) += dm_rot_spatial[0];
            } else {
                std::complex<double> spin_dm[2][2] = {{dm_rot_spatial[0], dm_rot_spatial[2]},
                                                {std::conj(dm_rot_spatial[2]), dm_rot_spatial[1]}};

                /* spin blocks of density matrix are: uu, dd, ud
                   the mapping from linear index (0, 1, 2) of density matrix components is:
                   for the first spin index: k & 1, i.e. (0, 1, 2) -> (0, 1, 0)
                   for the second spin index: min(k, 1), i.e. (0, 1, 2) -> (0, 1, 1)
                */
                for (int k = 0; k < ndm; k++) {
                    for (int is = 0; is < 2; is++) {
                        for (int js = 0; js < 2; js++) {
                            dm_(xi1, xi2, k, ja) +=
                                spin_rot_su2(k & 1, is) * spin_dm[is][js] * std::conj(spin_rot_su2(std::min(k, 1), js));
                        }
                    }
                }
            }
        }
    }
}

inline void
symmetrize(std::function<sddk::mdarray<std::complex<double>, 3>&(int ia__)> dm__, int num_mag_comp__,
           Crystal_symmetry const& sym__, std::function<basis_functions_index const*(int)> indexb__)
{
    /* quick exit */
    if (sym__.size() == 1) {
        return;
    }

    std::vector<sddk::mdarray<std::complex<double>, 3>> dmsym(sym__.num_atoms());
    for (int ia = 0; ia < sym__.num_atoms(); ia++) {
        int iat = sym__.atom_type(ia);
        if (indexb__(iat)) {
            dmsym[ia] = sddk::mdarray<std::complex<double>, 3>(indexb__(iat)->size(), indexb__(iat)->size(), 4);
            dmsym[ia].zero();
        }
    }

    int lmax{0};
    for (int iat = 0; iat < sym__.num_atom_types(); iat++) {
        if (indexb__(iat)) {
            lmax = std::max(lmax, indexb__(iat)->indexr().lmax());
        }
    }

    /* loop over symmetry operations */
    for (int isym = 0; isym < sym__.size(); isym++) {
        int pr            = sym__[isym].spg_op.proper;
        auto eang         = sym__[isym].spg_op.euler_angles;
        auto rotm         = sht::rotation_matrix<double>(lmax, eang, pr);
        auto spin_rot_su2 = rotation_matrix_su2(sym__[isym].spin_rotation);

        for (int ia = 0; ia < sym__.num_atoms(); ia++) {
            int iat = sym__.atom_type(ia);

            if (!indexb__(iat)) {
                continue;
            }

            int ja = sym__[isym].spg_op.inv_sym_atom[ia];

            auto& indexb = *indexb__(iat);
            auto& indexr = indexb.indexr();

            int mmax = 2 * indexb.indexr().lmax() + 1;
            sddk::mdarray<std::complex<double>, 3> dm_ia(mmax, mmax, num_mag_comp__);

            /* loop over radial functions */
            for (auto e1 : indexr) {
                /* angular momentum of radial function */
                auto am1     = e1.am;
                auto ss1     = am1.subshell_size();
                auto offset1 = indexb.index_of(e1.idxrf);
                for (auto e2 : indexr) {
                    /* angular momentum of radial function */
                    auto am2     = e2.am;
                    auto ss2     = am2.subshell_size();
                    auto offset2 = indexb.index_of(e2.idxrf);

                    dm_ia.zero();
                    for (int j = 0; j < num_mag_comp__; j++) {
                        /* apply spatial rotation */
                        for (int m1 = 0; m1 < ss1; m1++) {
                            for (int m2 = 0; m2 < ss2; m2++) {
                                for (int m1p = 0; m1p < ss1; m1p++) {
                                    for (int m2p = 0; m2p < ss2; m2p++) {
                                        dm_ia(m1, m2, j) += rotm[am1.l()](m1, m1p) *
                                                            dm__(ja)(offset1 + m1p, offset2 + m2p, j) *
                                                            rotm[am2.l()](m2, m2p);
                                    }
                                }
                            }
                        }
                    }
                    /* magnetic symmetry */
                    if (num_mag_comp__ == 1) { /* trivial non-magnetic case */
                        for (int m1 = 0; m1 < ss1; m1++) {
                            for (int m2 = 0; m2 < ss2; m2++) {
                                dmsym[ia](m1 + offset1, m2 + offset2, 0) += dm_ia(m1, m2, 0);
                            }
                        }
                    } else {
                        int const map_s[3][2] = {{0, 0}, {1, 1}, {0, 1}};
                        for (int j = 0; j < num_mag_comp__; j++) {
                            int s1 = map_s[j][0];
                            int s2 = map_s[j][1];

                            for (int m1 = 0; m1 < ss1; m1++) {
                                for (int m2 = 0; m2 < ss2; m2++) {
                                    std::complex<double> dm[2][2] = {{dm_ia(m1, m2, 0), 0}, {0, dm_ia(m1, m2, 1)}};
                                    if (num_mag_comp__ == 3) {
                                        dm[0][1] = dm_ia(m1, m2, 2);
                                        dm[1][0] = std::conj(dm[0][1]);
                                    }

                                    for (int s1p = 0; s1p < 2; s1p++) {
                                        for (int s2p = 0; s2p < 2; s2p++) {
                                            dmsym[ia](m1 + offset1, m2 + offset2, j) +=
                                                spin_rot_su2(s1, s1p) * dm[s1p][s2p] * std::conj(spin_rot_su2(s2, s2p));
                                        }
                                    }
                                }
                            }
                        }
                        if (num_mag_comp__ == 3) {
                            for (int m1 = 0; m1 < ss1; m1++) {
                                for (int m2 = 0; m2 < ss2; m2++) {
                                    dmsym[ia](m1 + offset1, m2 + offset2, 3) =
                                        std::conj(dmsym[ia](m1 + offset1, m2 + offset2, 2));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    double alpha = 1.0 / sym__.size();

    for (int ia = 0; ia < sym__.num_atoms(); ia++) {
        int iat = sym__.atom_type(ia);
        if (indexb__(iat)) {
            for (size_t i = 0; i < dm__(ia).size(); i++) {
                dm__(ia)[i] = dmsym[ia][i] * alpha;
            }
        }
    }
}

} // namespace sirius

#endif // __SYMMETRIZE_HPP__
