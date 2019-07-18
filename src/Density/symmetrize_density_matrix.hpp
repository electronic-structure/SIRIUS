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

    mdarray<double_complex, 4> dm(unit_cell_.max_mt_basis_size(),
                                  unit_cell_.max_mt_basis_size(),
                                  ndm,
                                  unit_cell_.num_atoms());
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
        auto spin_rot_su2 = rotation_matrix_su2(sym.magnetic_group_symmetry(i).spin_rotation);

        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            int   ja        = sym.sym_table(ia, isym);

            Symmetrize(density_matrix_,
                       unit_cell_.atom(ia).type().indexb(),
                       ia,
                       ja,
                       ndm,
                       rotm,
                       spin_rot_su2,
                       dm,
                       false);
        }
    }

    // multiply by alpha which is the inverse of the number of symmetries.
    std::complex<double> *a = dm.at(memory_t::host);
    for (auto i = 0u; i < dm.size(); i++) {
        a[i] *= alpha;
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
