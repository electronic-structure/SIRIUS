/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file symmetrize_occupation_matrix.hpp
 *
 *  \brief Symmetrize occupation matrix of the LDA+U method.
 */

#ifndef __SYMMETRIZE_OCCUPATION_MATRIX_HPP__
#define __SYMMETRIZE_OCCUPATION_MATRIX_HPP__

#include "density/occupation_matrix.hpp"

namespace sirius {

inline void
symmetrize_occupation_matrix(Occupation_matrix& om__)
{
    auto& ctx = om__.ctx();
    auto& uc  = ctx.unit_cell();

    if (!ctx.hubbard_correction()) {
        return;
    }

    auto& sym      = uc.symmetry();
    const double f = 1.0 / sym.size();
    std::vector<mdarray<std::complex<double>, 3>> local_tmp;

    local_tmp.resize(om__.local().size());

    for (int at_lvl = 0; at_lvl < static_cast<int>(om__.local().size()); at_lvl++) {
        const int ia          = om__.atomic_orbitals(at_lvl).first;
        auto const& atom_type = uc.atom(ia).type();
        /* We can skip the symmetrization for this atomic level since it does not contribute
         * to the Hubbard correction (or U = 0) */
        if (atom_type.lo_descriptor_hub(om__.atomic_orbitals(at_lvl).second).use_for_calculation()) {
            local_tmp[at_lvl] =
                    mdarray<std::complex<double>, 3>({om__.local(at_lvl).size(0), om__.local(at_lvl).size(1), 4});
            copy(om__.local(at_lvl), local_tmp[at_lvl]);
        }
    }

    for (int at_lvl = 0; at_lvl < static_cast<int>(om__.local().size()); at_lvl++) {
        const int ia     = om__.atomic_orbitals(at_lvl).first;
        const auto& atom = uc.atom(ia);
        om__.local(at_lvl).zero();
        /* We can skip the symmetrization for this atomic level since it does not contribute
         * to the Hubbard correction (or U = 0) */
        if (atom.type().lo_descriptor_hub(om__.atomic_orbitals(at_lvl).second).use_for_calculation()) {
            const int il       = atom.type().lo_descriptor_hub(om__.atomic_orbitals(at_lvl).second).l();
            const int lmmax_at = 2 * il + 1;
            // local_[at_lvl].zero();
            mdarray<std::complex<double>, 3> dm_ia({lmmax_at, lmmax_at, 4});
            for (int isym = 0; isym < sym.size(); isym++) {
                int pr            = sym[isym].spg_op.proper;
                auto eang         = sym[isym].spg_op.euler_angles;
                auto rotm         = sht::rotation_matrix<double>(4, eang, pr);
                auto spin_rot_su2 = rotation_matrix_su2(sym[isym].spin_rotation);

                int iap = sym[isym].spg_op.inv_sym_atom[ia];
                dm_ia.zero();

                int at_lvl1 = om__.find_orbital_index(
                        iap, atom.type().lo_descriptor_hub(om__.atomic_orbitals(at_lvl).second).n(),
                        atom.type().lo_descriptor_hub(om__.atomic_orbitals(at_lvl).second).l());

                for (int ispn = 0; ispn < (ctx.num_mag_dims() == 3 ? 4 : ctx.num_spins()); ispn++) {
                    for (int m1 = 0; m1 < lmmax_at; m1++) {
                        for (int m2 = 0; m2 < lmmax_at; m2++) {
                            for (int m1p = 0; m1p < lmmax_at; m1p++) {
                                for (int m2p = 0; m2p < lmmax_at; m2p++) {
                                    dm_ia(m1, m2, ispn) += rotm[il](m1, m1p) * rotm[il](m2, m2p) *
                                                           local_tmp[at_lvl1](m1p, m2p, ispn) * f;
                                }
                            }
                        }
                    }
                }

                if (ctx.num_mag_dims() == 0) {
                    for (int m1 = 0; m1 < lmmax_at; m1++) {
                        for (int m2 = 0; m2 < lmmax_at; m2++) {
                            om__.local(at_lvl)(m1, m2, 0) += dm_ia(m1, m2, 0);
                        }
                    }
                }

                if (ctx.num_mag_dims() == 1) {
                    int const map_s[3][2] = {{0, 0}, {1, 1}, {0, 1}};
                    for (int j = 0; j < 2; j++) {
                        int s1 = map_s[j][0];
                        int s2 = map_s[j][1];

                        for (int m1 = 0; m1 < lmmax_at; m1++) {
                            for (int m2 = 0; m2 < lmmax_at; m2++) {
                                std::complex<double> dm[2][2] = {{dm_ia(m1, m2, 0), 0}, {0, dm_ia(m1, m2, 1)}};

                                for (int s1p = 0; s1p < 2; s1p++) {
                                    for (int s2p = 0; s2p < 2; s2p++) {
                                        om__.local(at_lvl)(m1, m2, j) +=
                                                dm[s1p][s2p] * spin_rot_su2(s1, s1p) * std::conj(spin_rot_su2(s2, s2p));
                                    }
                                }
                            }
                        }
                    }
                }

                if (ctx.num_mag_dims() == 3) {
                    int s_idx[2][2] = {{0, 3}, {2, 1}};
                    for (int m1 = 0; m1 < lmmax_at; m1++) {
                        for (int m2 = 0; m2 < lmmax_at; m2++) {

                            std::complex<double> dm[2][2];
                            std::complex<double> dm1[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
                            for (int s1 = 0; s1 < ctx.num_spins(); s1++) {
                                for (int s2 = 0; s2 < ctx.num_spins(); s2++) {
                                    dm[s1][s2] = dm_ia(m1, m2, s_idx[s1][s2]);
                                }
                            }

                            for (int i = 0; i < 2; i++) {
                                for (int j = 0; j < 2; j++) {
                                    for (int s1p = 0; s1p < 2; s1p++) {
                                        for (int s2p = 0; s2p < 2; s2p++) {
                                            dm1[i][j] += dm[s1p][s2p] * spin_rot_su2(i, s1p) *
                                                         std::conj(spin_rot_su2(j, s2p));
                                        }
                                    }
                                }
                            }

                            for (int s1 = 0; s1 < ctx.num_spins(); s1++) {
                                for (int s2 = 0; s2 < ctx.num_spins(); s2++) {
                                    om__.local(at_lvl)(m1, m2, s_idx[s1][s2]) += dm1[s1][s2];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (ctx.cfg().hubbard().nonlocal().size() && ctx.num_mag_dims() == 3) {
        RTE_THROW("non-collinear nonlocal occupancy symmetrization is not implemented");
    }

    /* a pair of "total number, offests" for the Hubbard orbitals indexing */
    auto r = uc.num_hubbard_wf();

    for (int i = 0; i < static_cast<int>(ctx.cfg().hubbard().nonlocal().size()); i++) {
        auto nl = ctx.cfg().hubbard().nonlocal(i);
        int ia  = nl.atom_pair()[0];
        int ja  = nl.atom_pair()[1];
        int il  = nl.l()[0];
        int jl  = nl.l()[1];
        int n1  = nl.n()[0];
        int n2  = nl.n()[1];
        int ib  = 2 * il + 1;
        int jb  = 2 * jl + 1;
        auto T  = nl.T();
        om__.nonlocal(i).zero();
        for (int isym = 0; isym < sym.size(); isym++) {
            int pr            = sym[isym].spg_op.proper;
            auto eang         = sym[isym].spg_op.euler_angles;
            auto rotm         = sht::rotation_matrix<double>(4, eang, pr);
            auto spin_rot_su2 = rotation_matrix_su2(sym[isym].spin_rotation);

            int iap = sym[isym].spg_op.inv_sym_atom[ia];
            int jap = sym[isym].spg_op.inv_sym_atom[ja];

            auto Ttot = sym[isym].spg_op.inv_sym_atom_T[ja] - sym[isym].spg_op.inv_sym_atom_T[ia] +
                        dot(sym[isym].spg_op.invR, r3::vector<int>(T));

            /* we must search for the right hubbard subspace since we may have
             * multiple orbitals involved in the hubbard correction */

            /* NOTE : the atom order is important here. */
            int at1_lvl          = om__.find_orbital_index(iap, n1, il);
            int at2_lvl          = om__.find_orbital_index(jap, n2, jl);
            auto const& occ_mtrx = om__.occ_mtrx_T(Ttot);

            mdarray<std::complex<double>, 3> dm_ia_ja({2 * il + 1, 2 * jl + 1, ctx.num_spins()});
            dm_ia_ja.zero();
            /* apply spatial rotation */
            for (int ispn = 0; ispn < ctx.num_spins(); ispn++) {
                for (int m1 = 0; m1 < ib; m1++) {
                    for (int m2 = 0; m2 < jb; m2++) {
                        for (int m1p = 0; m1p < ib; m1p++) {
                            for (int m2p = 0; m2p < jb; m2p++) {
                                dm_ia_ja(m1, m2, ispn) +=
                                        rotm[il](m1, m1p) * rotm[jl](m2, m2p) *
                                        occ_mtrx(om__.offset(at1_lvl) + m1p, om__.offset(at2_lvl) + m2p, ispn) * f;
                            }
                        }
                    }
                }
            }

            if (ctx.num_mag_dims() == 0) {
                for (int m1 = 0; m1 < ib; m1++) {
                    for (int m2 = 0; m2 < jb; m2++) {
                        om__.nonlocal(i)(m1, m2, 0) += dm_ia_ja(m1, m2, 0);
                    }
                }
            }
            if (ctx.num_mag_dims() == 1) {
                int const map_s[3][2] = {{0, 0}, {1, 1}, {0, 1}};
                for (int j = 0; j < 2; j++) {
                    int s1 = map_s[j][0];
                    int s2 = map_s[j][1];

                    for (int m1 = 0; m1 < ib; m1++) {
                        for (int m2 = 0; m2 < jb; m2++) {
                            std::complex<double> dm[2][2] = {{dm_ia_ja(m1, m2, 0), 0}, {0, dm_ia_ja(m1, m2, 1)}};

                            for (int s1p = 0; s1p < 2; s1p++) {
                                for (int s2p = 0; s2p < 2; s2p++) {
                                    om__.nonlocal(i)(m1, m2, j) +=
                                            dm[s1p][s2p] * spin_rot_su2(s1, s1p) * std::conj(spin_rot_su2(s2, s2p));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace sirius

#endif
