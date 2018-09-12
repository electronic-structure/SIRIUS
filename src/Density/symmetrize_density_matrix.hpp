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

/** \file symmetrize_density_matrix.hpp
 *
 *  \brief Symmetrization of a density matrix.
 */

inline void Density::symmetrize_density_matrix()
{
    PROFILE("sirius::Density::symmetrize_density_matrix");

    auto& sym = unit_cell_.symmetry();

    int ndm = ctx_.num_mag_comp();

    mdarray<double_complex, 4> dm(unit_cell_.max_mt_basis_size(), unit_cell_.max_mt_basis_size(),
                                  ndm, unit_cell_.num_atoms());
    dm.zero();

    int lmax  = unit_cell_.lmax();
    int lmmax = utils::lmmax(lmax);

    mdarray<double, 2> rotm(lmmax, lmmax);

    double alpha = 1.0 / double(sym.num_mag_sym());

    for (int i = 0; i < sym.num_mag_sym(); i++) {
        int  pr   = sym.magnetic_group_symmetry(i).spg_op.proper;
        auto eang = sym.magnetic_group_symmetry(i).spg_op.euler_angles;
        int  isym = sym.magnetic_group_symmetry(i).isym;
        SHT::rotation_matrix(lmax, eang, pr, rotm);
        auto spin_rot_su2 = SHT::rotation_matrix_su2(sym.magnetic_group_symmetry(i).spin_rotation);

        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto& atom_type = unit_cell_.atom(ia).type();
            int   ja        = sym.sym_table(ia, isym);

            for (int xi1 = 0; xi1 < unit_cell_.atom(ia).mt_basis_size(); xi1++) {
                int l1  = atom_type.indexb(xi1).l;
                int lm1 = atom_type.indexb(xi1).lm;
                int o1  = atom_type.indexb(xi1).order;

                for (int xi2 = 0; xi2 < unit_cell_.atom(ia).mt_basis_size(); xi2++) {
                    int l2  = atom_type.indexb(xi2).l;
                    int lm2 = atom_type.indexb(xi2).lm;
                    int o2  = atom_type.indexb(xi2).order;

                    std::array<double_complex, 3> dm_rot_spatial = {0, 0, 0};

                    for (int j = 0; j < ndm; j++) {
                        for (int m3 = -l1; m3 <= l1; m3++) {
                            int lm3 = utils::lm(l1, m3);
                            int xi3 = atom_type.indexb().index_by_lm_order(lm3, o1);
                            for (int m4 = -l2; m4 <= l2; m4++) {
                                int lm4 = utils::lm(l2, m4);
                                int xi4 = atom_type.indexb().index_by_lm_order(lm4, o2);
                                dm_rot_spatial[j] += density_matrix_(xi3, xi4, j, ja) * rotm(lm1, lm3) * rotm(lm2, lm4) * alpha;
                            }
                        }
                    }

                    /* magnetic symmetrization */
                    if (ndm == 1) {
                        dm(xi1, xi2, 0, ia) += dm_rot_spatial[0];
                    } else {
                        double_complex spin_dm[2][2] = {
                            {dm_rot_spatial[0], dm_rot_spatial[2]},
                            {std::conj(dm_rot_spatial[2]), dm_rot_spatial[1]}};

                        /* spin blocks of density matrix are: uu, dd, ud
                           the mapping from linear index (0, 1, 2) of density matrix components is:
                             for the first spin index: k & 1, i.e. (0, 1, 2) -> (0, 1, 0)
                             for the second spin index: min(k, 1), i.e. (0, 1, 2) -> (0, 1, 1)
                        */
                        for (int k = 0; k < ndm; k++) {
                            for (int is = 0; is < 2; is++) {
                                for (int js = 0; js < 2; js++) {
                                    dm(xi1, xi2, k, ia) += spin_rot_su2(k & 1, is) * spin_dm[is][js] * std::conj(spin_rot_su2(std::min(k, 1), js));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    dm >> density_matrix_;

    if (ctx_.control().print_checksum_ && ctx_.comm().rank() == 0) {
        auto cs = dm.checksum();
        utils::print_checksum("density_matrix", cs);
        //for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        //    auto cs = mdarray<double_complex, 1>(&dm(0, 0, 0, ia), dm.size(0) * dm.size(1) * dm.size(2)).checksum();
        //    DUMP("checksum(density_matrix(%i)): %20.14f %20.14f", ia, cs.real(), cs.imag());
        //}
    }

    if (ctx_.control().print_hash_ && ctx_.comm().rank() == 0) {
        auto h = dm.hash();
        utils::print_hash("density_matrix", h);
    }
}
