/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file symmetrize_mt_function.hpp
 *
 *  \brief Symmetrize muffin-tin spheric functions.
 */

#ifndef __SYMMETRIZE_MT_FUNCTION_HPP__
#define __SYMMETRIZE_MT_FUNCTION_HPP__

#include "crystal_symmetry.hpp"
#include "function3d/spheric_function_set.hpp"

namespace sirius {

/// Symmetrize spherical expansion coefficients of the scalar and vector function for atoms of the same symmetry class.
template <typename Index_t>
inline void
symmetrize_mt_function(Crystal_symmetry const& sym__, std::vector<mdarray<double, 2>> const& rotm__,
                       Atom_symmetry_class const& atom_class__, mpi::Grid const& mpi_grid__, int num_mag_dims__,
                       std::vector<Spheric_function_set<double, Index_t>*> frlm__)
{
    PROFILE("sirius::symmetrize_mt_function");

    /* first (scalar) component is always available */
    auto& frlm = *frlm__[0];

    /* get lmax */
    int lmax = frlm.lmax(atom_class__.atom_id(0));

    /* compute maximum lm size */
    int lmmax = sf::lmmax(lmax);

    /* split atoms of the same class over this communicator */
    auto& comm_a = mpi_grid__.communicator(1 << 0);
    /* split radial grid points over this communicator */
    auto& comm_r = mpi_grid__.communicator(1 << 1);

    /* number of atoms belonging to the same symmetry class */
    int na = atom_class__.num_atoms();
    /* number of muffin-tin points */
    int nr = atom_class__.atom_type().num_mt_points();

    /* split atoms of a given symmetry class between MPI ranks */
    splindex_block<Index_t> spl_atoms(na, n_blocks(comm_a.size()), block_id(comm_a.rank()));
    /* split radial grid points */
    splindex_block spl_rgrid(nr, n_blocks(comm_r.size()), block_id(comm_r.rank()));
    int nr_loc = spl_rgrid.local_size();
    int i0_loc = spl_rgrid.global_offset();

    /* symmetry-transformed functions */
    mdarray<double, 4> fsym_loc({lmmax, nr, num_mag_dims__ + 1, spl_atoms.local_size()},
                                get_memory_pool(memory_t::host));
    fsym_loc.zero();

    double alpha = 1.0 / sym__.size();

    if (nr_loc) {
        mdarray<double, 3> ftmp({lmmax, nr_loc, num_mag_dims__ + 1}, get_memory_pool(memory_t::host));
        /* loop over crystal symmetries */
        for (int i = 0; i < sym__.size(); i++) {
            /* full space-group symmetry operation is S{R|t} */
            auto S = sym__[i].spin_rotation;

            for (auto it : spl_atoms) {
                /* get global index of the atom */
                int ia = atom_class__.atom_id(it.i);
                int ja = sym__[i].spg_op.inv_sym_atom[ia];
                /* apply {R|t} part of symmetry operation to all components */
                for (int j = 0; j < num_mag_dims__ + 1; j++) {
                    la::wrap(la::lib_t::blas)
                            .gemm('N', 'N', lmmax, nr_loc, lmmax, &alpha, rotm__[i].at(memory_t::host), rotm__[i].ld(),
                                  (*frlm__[j])[ja].at(memory_t::host, 0, i0_loc), (*frlm__[j])[ja].ld(),
                                  &la::constant<double>::zero(), ftmp.at(memory_t::host, 0, 0, j), ftmp.ld());
                }
                /* always symmetrize the scalar component */
                for (int ir = 0; ir < nr_loc; ir++) {
                    for (int lm = 0; lm < lmmax; lm++) {
                        fsym_loc(lm, ir + i0_loc, 0, it.li) += ftmp(lm, ir, 0);
                    }
                }
                /* apply S part to [0, 0, z] collinear vector */
                if (num_mag_dims__ == 1) {
                    for (int ir = 0; ir < nr_loc; ir++) {
                        for (int lm = 0; lm < lmmax; lm++) {
                            fsym_loc(lm, ir + i0_loc, 1, it.li) += ftmp(lm, ir, 1) * S(2, 2);
                        }
                    }
                }
                /* apply 3x3 S-matrix to [x, y, z] vector */
                if (num_mag_dims__ == 3) {
                    for (int k : {0, 1, 2}) {
                        for (int j : {0, 1, 2}) {
                            for (int ir = 0; ir < nr_loc; ir++) {
                                for (int lm = 0; lm < lmmax; lm++) {
                                    fsym_loc(lm, ir + i0_loc, 1 + k, it.li) += ftmp(lm, ir, 1 + j) * S(k, j);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /* gather radial grid points */
    for (int i = 0; i < spl_atoms.local_size(); i++) {
        for (int j = 0; j < num_mag_dims__ + 1; j++) {
            double* sbuf = fsym_loc.at(memory_t::host, 0, 0, j, i);
            comm_r.allgather(sbuf, lmmax * nr_loc, lmmax * i0_loc);
        }
    }

    /* gather full function */
    double* sbuf = spl_atoms.local_size() ? fsym_loc.at(memory_t::host) : nullptr;
    auto ld      = lmmax * nr * (num_mag_dims__ + 1);

    mdarray<double, 4> fsym_glob({lmmax, nr, num_mag_dims__ + 1, na}, get_memory_pool(memory_t::host));

    comm_a.allgather(sbuf, fsym_glob.at(memory_t::host), ld * spl_atoms.local_size(), ld * spl_atoms.global_offset());

    /* copy back the result */
    for (int i = 0; i < na; i++) {
        int ia = atom_class__.atom_id(i);
        for (int j = 0; j < num_mag_dims__ + 1; j++) {
            for (int ir = 0; ir < nr; ir++) {
                for (int lm = 0; lm < lmmax; lm++) {
                    (*frlm__[j])[ia](lm, ir) = fsym_glob(lm, ir, j, i);
                }
            }
        }
    }
}

template <typename Index_t>
inline void
symmetrize_mt_function(Unit_cell const& uc__, std::vector<mdarray<double, 2>> const& rotm__,
                       std::vector<std::unique_ptr<mpi::Grid>> const& mpi_grid__, int num_mag_dims__,
                       std::vector<Spheric_function_set<double, Index_t>*> frlm__)
{
    for (int ic = 0; ic < uc__.num_atom_symmetry_classes(); ic++) {
        if (mpi_grid__[ic]) {
            symmetrize_mt_function(uc__.symmetry(), rotm__, uc__.atom_symmetry_class(ic), *mpi_grid__[ic],
                                   num_mag_dims__, frlm__);
        }
    }
}

} // namespace sirius

#endif
