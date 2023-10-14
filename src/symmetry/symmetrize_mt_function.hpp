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

/** \file symmetrize_mt_function.hpp
 *
 *  \brief Symmetrize muffin-tin spheric functions.
 */

#ifndef __SYMMETRIZE_MT_FUNCTION_HPP__
#define __SYMMETRIZE_MT_FUNCTION_HPP__

#include "crystal_symmetry.hpp"

namespace sirius {

template <typename Index_t>
inline void
symmetrize_mt_function(Crystal_symmetry const& sym__, mpi::Communicator const& comm__, int num_mag_dims__,
        std::vector<Spheric_function_set<double, Index_t>*> frlm__)
{
    PROFILE("sirius::symmetrize_mt_function");

    /* first (scalar) component is always available */
    auto& frlm = *frlm__[0];

    /* compute maximum lm size */
    int lmmax{0};
    for (auto ia : frlm.atoms()) {
        lmmax = std::max(lmmax, frlm[ia].angular_domain_size());
    }
    int lmax = sf::lmax(lmmax);

    /* split atoms between MPI ranks */
    splindex_block<Index_t> spl_atoms(frlm.atoms().size(), n_blocks(comm__.size()), block_id(comm__.rank()));

    /* space for real Rlm rotation matrix */
    sddk::mdarray<double, 2> rotm(lmmax, lmmax);

    /* symmetry-transformed functions */
    sddk::mdarray<double, 4> fsym_loc(lmmax, frlm.unit_cell().max_num_mt_points(), num_mag_dims__ + 1,
            spl_atoms.local_size());
    fsym_loc.zero();

    sddk::mdarray<double, 3> ftmp(lmmax, frlm.unit_cell().max_num_mt_points(), num_mag_dims__ + 1);

    double alpha = 1.0 / sym__.size();

    /* loop over crystal symmetries */
    for (int i = 0; i < sym__.size(); i++) {
        /* full space-group symmetry operation is S{R|t} */
        auto S = sym__[i].spin_rotation;
        /* compute Rlm rotation matrix */
        sht::rotation_matrix(lmax, sym__[i].spg_op.euler_angles, sym__[i].spg_op.proper, rotm);

        for (auto it : spl_atoms) {
            /* get global index of the atom */
            int ia = frlm.atoms()[it.i];
            int lmmax_ia = frlm[ia].angular_domain_size();
            int nrmax_ia = frlm.unit_cell().atom(ia).num_mt_points();
            int ja = sym__[i].spg_op.inv_sym_atom[ia];
            /* apply {R|t} part of symmetry operation to all components */
            for (int j = 0; j < num_mag_dims__ + 1; j++) {
                la::wrap(la::lib_t::blas).gemm('N', 'N', lmmax_ia, nrmax_ia, lmmax_ia, &alpha,
                    rotm.at(sddk::memory_t::host), rotm.ld(), (*frlm__[j])[ja].at(sddk::memory_t::host),
                    (*frlm__[j])[ja].ld(), &la::constant<double>::zero(),
                    ftmp.at(sddk::memory_t::host, 0, 0, j), ftmp.ld());
            }
            /* always symmetrize the scalar component */
            for (int ir = 0; ir < nrmax_ia; ir++) {
                for (int lm = 0; lm < lmmax_ia; lm++) {
                    fsym_loc(lm, ir, 0, it.li) += ftmp(lm, ir, 0);
                }
            }
            /* apply S part to [0, 0, z] collinear vector */
            if (num_mag_dims__ == 1) {
                for (int ir = 0; ir < nrmax_ia; ir++) {
                    for (int lm = 0; lm < lmmax_ia; lm++) {
                        fsym_loc(lm, ir, 1, it.li) += ftmp(lm, ir, 1) * S(2, 2);
                    }
                }
            }
            /* apply 3x3 S-matrix to [x, y, z] vector */
            if (num_mag_dims__ == 3) {
                for (int k : {0, 1, 2}) {
                    for (int j : {0, 1, 2}) {
                        for (int ir = 0; ir < nrmax_ia; ir++) {
                            for (int lm = 0; lm < lmmax_ia; lm++) {
                                fsym_loc(lm, ir, 1 + k, it.li) += ftmp(lm, ir, 1 + j) * S(k, j);
                            }
                        }
                    }
                }
            }
        }
    }

    /* gather full function */
    double* sbuf = spl_atoms.local_size() ? fsym_loc.at(sddk::memory_t::host) : nullptr;
    auto ld = static_cast<int>(fsym_loc.size(0) * fsym_loc.size(1) * fsym_loc.size(2));

    sddk::mdarray<double, 4> fsym_glob(lmmax, frlm.unit_cell().max_num_mt_points(), num_mag_dims__ + 1,
            frlm.atoms().size());

    comm__.allgather(sbuf, fsym_glob.at(sddk::memory_t::host), ld * spl_atoms.local_size(),
            ld * spl_atoms.global_offset());

    /* copy back the result */
    for (int i = 0; i < static_cast<int>(frlm.atoms().size()); i++) {
        int ia = frlm.atoms()[i];
        for (int j = 0; j < num_mag_dims__ + 1; j++) {
            for (int ir = 0; ir < frlm.unit_cell().atom(ia).num_mt_points(); ir++) {
                for (int lm = 0; lm < frlm[ia].angular_domain_size(); lm++) {
                    (*frlm__[j])[ia](lm, ir) = fsym_glob(lm, ir, j, i);
                }
            }
        }
    }
}

}

#endif

