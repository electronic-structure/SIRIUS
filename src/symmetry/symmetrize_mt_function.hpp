#ifndef __SYMMETRIZE_MT_FUNCTION_HPP__
#define __SYMMETRIZE_MT_FUNCTION_HPP__

#include "crystal_symmetry.hpp"

namespace sirius {

inline void
symmetrize_mt_function(Crystal_symmetry const& sym__, mpi::Communicator const& comm__, int num_mag_dims__,
        std::vector<Spheric_function_set<double>*> frlm__)
{
    PROFILE("sirius::symmetrize_mt_function");

    /* first (scalar) component is always available */
    auto& frlm = *frlm__[0];

    /* compute maximum lm size */
    int lmmax{0};
    for (auto ia : frlm.atoms()) {
        lmmax = std::max(lmmax, frlm[ia].angular_domain_size());
    }
    int lmax = utils::lmax(lmmax);

    /* split atoms between MPI ranks */
    sddk::splindex<sddk::splindex_t::block> spl_atoms(frlm.atoms().size(), comm__.size(), comm__.rank());

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

        for (int ialoc = 0; ialoc < spl_atoms.local_size(); ialoc++) {
            /* get global index of the atom */
            int ia = frlm.atoms()[spl_atoms[ialoc]];
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
                    fsym_loc(lm, ir, 0, ialoc) += ftmp(lm, ir, 0);
                }
            }
            /* apply S part to [0, 0, z] collinear vector */
            if (num_mag_dims__ == 1) {
                for (int ir = 0; ir < nrmax_ia; ir++) {
                    for (int lm = 0; lm < lmmax_ia; lm++) {
                        fsym_loc(lm, ir, 1, ialoc) += ftmp(lm, ir, 1) * S(2, 2);
                    }
                }
            }
            /* apply 3x3 S-matrix to [x, y, z] vector */
            if (num_mag_dims__ == 3) {
                for (int k : {0, 1, 2}) {
                    for (int j : {0, 1, 2}) {
                        for (int ir = 0; ir < nrmax_ia; ir++) {
                            for (int lm = 0; lm < lmmax_ia; lm++) {
                                fsym_loc(lm, ir, 1 + k, ialoc) += ftmp(lm, ir, 1 + j) * S(k, j);
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

