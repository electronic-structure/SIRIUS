/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file non_local_functor.hpp
 *
 *  \brief Common operation for forces and stress tensor.
 */

#ifndef __NON_LOCAL_FUNCTOR_HPP__
#define __NON_LOCAL_FUNCTOR_HPP__

#include "function3d/periodic_function.hpp"
#include "k_point/k_point.hpp"
#include "density/augmentation_operator.hpp"
#include "beta_projectors/beta_projectors_base.hpp"

namespace sirius {

/** \tparam T  Precision type of the wave-functions */
template <typename T, typename F>
void
add_k_point_contribution_nonlocal(Simulation_context& ctx__, Beta_projectors_base<T>& bp_base__, K_point<T>& kp__,
                                  mdarray<real_type<F>, 2>& collect_res__)
{
    PROFILE("sirius::add_k_point_contribution_nonlocal");

    auto& uc = ctx__.unit_cell();

    if (uc.max_mt_basis_size() == 0) {
        return;
    }

    auto& bp = kp__.beta_projectors();

    double main_two_factor{-2};

    auto mt          = ctx__.processing_unit_memory_t();
    auto bp_gen      = bp.make_generator(mt);
    auto beta_coeffs = bp_gen.prepare();

    auto bp_base_gen      = bp_base__.make_generator(mt);
    auto beta_coeffs_base = bp_base_gen.prepare();

    for (int icnk = 0; icnk < bp_base__.num_chunks(); icnk++) {

        /* generate chunk for inner product of beta */
        bp_gen.generate(beta_coeffs, icnk);

        /* store <beta|psi> for spin up and down */
        matrix<F> beta_phi_chunks[2];

        for (int ispn = 0; ispn < ctx__.num_spins(); ispn++) {
            int nbnd              = kp__.num_occupied_bands(ispn);
            beta_phi_chunks[ispn] = inner_prod_beta<F>(ctx__.spla_context(), mt, ctx__.host_memory_t(),
                                                       is_device_memory(mt), beta_coeffs, kp__.spinor_wave_functions(),
                                                       wf::spin_index(ispn), wf::band_range(0, nbnd));
        }

        for (int x = 0; x < bp_base__.num_comp(); x++) {
            /* generate chunk for inner product of beta gradient */
            bp_base_gen.generate(beta_coeffs_base, icnk, x);

            for (int ispn = 0; ispn < ctx__.num_spins(); ispn++) {
                int spin_factor = (ispn == 0 ? 1 : -1);

                int nbnd = kp__.num_occupied_bands(ispn);

                /* inner product of beta gradient and WF */
                auto bp_base_phi_chunk = inner_prod_beta<F>(
                        ctx__.spla_context(), mt, ctx__.host_memory_t(), is_device_memory(mt), beta_coeffs_base,
                        kp__.spinor_wave_functions(), wf::spin_index(ispn), wf::band_range(0, nbnd));

                splindex_block<> spl_nbnd(nbnd, n_blocks(kp__.comm().size()), block_id(kp__.comm().rank()));

                int nbnd_loc = spl_nbnd.local_size();

                #pragma omp parallel for
                for (int ia_chunk = 0; ia_chunk < beta_coeffs_base.beta_chunk_->num_atoms_; ia_chunk++) {
                    int ia   = beta_coeffs_base.beta_chunk_->desc_(beta_desc_idx::ia, ia_chunk);
                    int offs = beta_coeffs_base.beta_chunk_->desc_(beta_desc_idx::offset, ia_chunk);
                    int nbf  = beta_coeffs_base.beta_chunk_->desc_(beta_desc_idx::nbf, ia_chunk);
                    int iat  = uc.atom(ia).type_id();

                    if (uc.atom(ia).type().spin_orbit_coupling()) {
                        RTE_THROW("stress and forces with SO coupling are not upported");
                    }

                    /* helper lambda to calculate for sum loop over bands for different beta_phi and dij combinations*/
                    auto for_bnd = [&](int ibf, int jbf, std::complex<real_type<F>> dij, real_type<F> qij,
                                       matrix<F>& beta_phi_chunk) {
                        /* gather everything = - 2  Re[ occ(k,n) weight(k) beta_phi*(i,n) [Dij - E(n)Qij]
                         * beta_base_phi(j,n) ]*/
                        for (int ibnd_loc = 0; ibnd_loc < nbnd_loc; ibnd_loc++) {
                            int ibnd = spl_nbnd.global_index(ibnd_loc);

                            auto d1 = main_two_factor * kp__.band_occupancy(ibnd, ispn) * kp__.weight();
                            auto z2 = dij - static_cast<real_type<F>>(kp__.band_energy(ibnd, ispn) * qij);
                            auto z1 = z2 * std::conj(beta_phi_chunk(offs + jbf, ibnd)) *
                                      bp_base_phi_chunk(offs + ibf, ibnd);

                            auto scalar_part = static_cast<real_type<F>>(d1) * z1;

                            /* get real part and add to the result array*/
                            collect_res__(x, ia) += scalar_part.real();
                        }
                    };

                    for (int ibf = 0; ibf < nbf; ibf++) {
                        int lm2    = uc.atom(ia).type().indexb(ibf).lm;
                        int idxrf2 = uc.atom(ia).type().indexb(ibf).idxrf;
                        for (int jbf = 0; jbf < nbf; jbf++) {
                            int lm1    = uc.atom(ia).type().indexb(jbf).lm;
                            int idxrf1 = uc.atom(ia).type().indexb(jbf).idxrf;

                            /* Qij exists only in the case of ultrasoft/PAW */
                            real_type<F> qij{0};
                            if (uc.atom(ia).type().augment()) {
                                qij = ctx__.augmentation_op(iat).q_mtrx(ibf, jbf);
                            }
                            std::complex<real_type<F>> dij{0};

                            /* get non-magnetic or collinear spin parts of dij*/
                            switch (ctx__.num_spins()) {
                                case 1: {
                                    dij = uc.atom(ia).d_mtrx(ibf, jbf, 0);
                                    if (lm1 == lm2) {
                                        dij += uc.atom(ia).type().d_mtrx_ion()(idxrf1, idxrf2);
                                    }
                                    break;
                                }

                                case 2: {
                                    /* Dij(00) = dij + dij_Z ;  Dij(11) = dij - dij_Z*/
                                    dij = (uc.atom(ia).d_mtrx(ibf, jbf, 0) +
                                           spin_factor * uc.atom(ia).d_mtrx(ibf, jbf, 1));
                                    if (lm1 == lm2) {
                                        dij += uc.atom(ia).type().d_mtrx_ion()(idxrf1, idxrf2);
                                    }
                                    break;
                                }

                                default: {
                                    RTE_THROW("Error in non_local_functor, D_aug_mtrx. ");
                                    break;
                                }
                            }

                            /* add non-magnetic or diagonal spin components (or collinear part) */
                            for_bnd(ibf, jbf, dij, qij, beta_phi_chunks[ispn]);

                            /* for non-collinear case*/
                            if (ctx__.num_mag_dims() == 3) {
                                /* Dij(10) = dij_X + i dij_Y ; Dij(01) = dij_X - i dij_Y */
                                dij = std::complex<real_type<F>>(uc.atom(ia).d_mtrx(ibf, jbf, 2),
                                                                 spin_factor * uc.atom(ia).d_mtrx(ibf, jbf, 3));
                                /* add non-diagonal spin components*/
                                for_bnd(ibf, jbf, dij, 0.0, beta_phi_chunks[ispn + spin_factor]);
                            }
                        } // jbf
                    }     // ibf
                }         // ia_chunk
            }             // ispn
        }                 // x
    }

    // bp_base__.dismiss();
}

} // namespace sirius

#endif /* __NON_LOCAL_FUNCTOR_HPP__ */
