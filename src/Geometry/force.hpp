// Copyright (c) 2013-2018 Anton Kozhevnikov, Ilia Sivkov, Thomas Schulthess
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

/** \file force.hpp
 *
 *  \brief Contains defintion and implementation of sirius::Force class.
 */


#ifndef __FORCE_HPP__
#define __FORCE_HPP__

#include "../simulation_context.h"
#include "../periodic_function.h"
#include "../augmentation_operator.h"
#include "../Beta_projectors/beta_projectors.h"
#include "../Beta_projectors/beta_projectors_gradient.h"
#include "../potential.h"
#include "../density.h"
#include "non_local_functor.hpp"

namespace sirius {

    using namespace geometry3d;

    /// Compute atomic forces.
    class Force
    {
    private:
        Simulation_context& ctx_;

        Density& density_;

        Potential& potential_;

        K_point_set& kset_;

        Hamiltonian& hamiltonian_;

        mdarray<double, 2> forces_vloc_;

        mdarray<double, 2> forces_us_;

        mdarray<double, 2> forces_nonloc_;

        mdarray<double, 2> forces_core_;

        mdarray<double, 2> forces_ewald_;

        mdarray<double, 2> forces_scf_corr_;

        mdarray<double, 2> forces_hubbard_;

        mdarray<double, 2> forces_total_;

        template <typename T>
        void add_k_point_contribution(K_point& kpoint, mdarray<double, 2>& forces__) const
        {
            Beta_projectors_gradient bp_grad(ctx_, kpoint.gkvec(), kpoint.igk_loc(), kpoint.beta_projectors());
#ifdef __GPU
            if (ctx_.processing_unit() == GPU && !keep_wf_on_gpu) {
                int nbnd = ctx_.num_bands();
                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    /* allocate GPU memory */
                    kpoint.spinor_wave_functions().pw_coeffs(ispn).allocate_on_device();
                    kpoint.spinor_wave_functions().pw_coeffs(ispn).copy_to_device(0, nbnd);
                }
            }
#endif
            Non_local_functor<T, 3> nlf(ctx_, bp_grad);

            nlf.add_k_point_contribution(kpoint, forces__);
#ifdef __GPU
            if (ctx_.processing_unit() == GPU && !keep_wf_on_gpu) {
                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    /* deallocate GPU memory */
                    kpoint.spinor_wave_functions().pw_coeffs(ispn).deallocate_on_device();
                }
            }
#endif
        }

        inline void symmetrize(mdarray<double, 2>& forces__) const
        {
            if (!ctx_.use_symmetry()) {
                return;
            }

            mdarray<double, 2> sym_forces(3, ctx_.unit_cell().num_atoms());
            sym_forces.zero();

            auto& lattice_vectors         = ctx_.unit_cell().symmetry().lattice_vectors();
            auto& inverse_lattice_vectors = ctx_.unit_cell().symmetry().inverse_lattice_vectors();

            sym_forces.zero();

            #pragma omp parallel for
            for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
                vector3d<double> cart_force(&forces__(0, ia));
                vector3d<double> lat_force =
                    inverse_lattice_vectors * (cart_force / (double)ctx_.unit_cell().symmetry().num_mag_sym());

                for (int isym = 0; isym < ctx_.unit_cell().symmetry().num_mag_sym(); isym++) {
                    int ja                     = ctx_.unit_cell().symmetry().sym_table(ia, isym);
                    auto& R                    = ctx_.unit_cell().symmetry().magnetic_group_symmetry(isym).spg_op.R;
                    vector3d<double> rot_force = lattice_vectors * (R * lat_force);

                    #pragma omp atomic update
                    sym_forces(0, ja) += rot_force[0];

                    #pragma omp atomic update
                    sym_forces(1, ja) += rot_force[1];

                    #pragma omp atomic update
                    sym_forces(2, ja) += rot_force[2];
                }
            }
            sym_forces >> forces__;
        }

        /** In the second-variational approach we need to compute the following expression for the k-dependent
         *  contribution to the forces:
         *  \f[
         *      {\bf F}_{\rm IBS}^{\alpha}=\sum_{\bf k}w_{\bf k}\sum_{l\sigma}n_{l{\bf k}}
         *      \sum_{ij}c_{\sigma i}^{l{\bf k}*}c_{\sigma j}^{l{\bf k}}
         *      {\bf F}_{ij}^{\alpha{\bf k}}
         *  \f]
         *  First, we sum over band and spin indices to get the "density matrix":
         *  \f[
         *      q_{ij} = \sum_{l\sigma}n_{l{\bf k}} c_{\sigma i}^{l{\bf k}*}c_{\sigma j}^{l{\bf k}}
         *  \f]
         */
        void compute_dmat(K_point*                 kp__,
                          dmatrix<double_complex>& dm__)
        {
            dm__.zero();

            STOP();

            ///* trivial case */
            //if (!parameters__.need_sv())
            //{
            //    for (int i = 0; i < parameters__.num_fv_states(); i++) dm__.set(i, i, double_complex(kp__->band_occupancy(i), 0));
            //}
            //else
            //{
            //    if (parameters__.num_mag_dims() != 3)
            //    {
            //        dmatrix<double_complex> ev1(parameters__.num_fv_states(), parameters__.num_fv_states(), kp__->blacs_grid(), parameters__.cyclic_block_size(), parameters__.cyclic_block_size());
            //        for (int ispn = 0; ispn < parameters__.num_spins(); ispn++)
            //        {
            //            auto& ev = kp__->sv_eigen_vectors(ispn);
            //            /* multiply second-variational eigen-vectors with band occupancies */
            //            for (int j = 0; j < ev.num_cols_local(); j++)
            //            {
            //                /* up- or dn- band index */
            //                int jb = ev.icol(j);
            //                for (int i = 0; i < ev.num_rows_local(); i++)
            //                    ev1(i, j) = conj(ev(i, j)) * kp__->band_occupancy(jb + ispn * parameters__.num_fv_states());
            //            }

            //            linalg<CPU>::gemm(0, 1, parameters__.num_fv_states(), parameters__.num_fv_states(), parameters__.num_fv_states(),
            //                              linalg_const<double_complex>::one(), ev1, ev, linalg_const<double_complex>::one(), dm__);
            //        }
            //    }
            //    else
            //    {
            //        dmatrix<double_complex> ev1(parameters__.num_bands(), parameters__.num_bands(), kp__->blacs_grid(), parameters__.cyclic_block_size(), parameters__.cyclic_block_size());
            //        auto& ev = kp__->sv_eigen_vectors(0);
            //        /* multiply second-variational eigen-vectors with band occupancies */
            //        for (int j = 0; j < ev.num_cols_local(); j++)
            //        {
            //            /* band index */
            //            int jb = ev.icol(j);
            //            for (int i = 0; i < ev.num_rows_local(); i++) ev1(i, j) = conj(ev(i, j)) * kp__->band_occupancy(jb);
            //        }
            //        for (int ispn = 0; ispn < parameters__.num_spins(); ispn++)
            //        {
            //            int offs = ispn * parameters__.num_fv_states();

            //            linalg<CPU>::gemm(0, 1, parameters__.num_fv_states(), parameters__.num_fv_states(), parameters__.num_bands(),
            //                              linalg_const<double_complex>::one(), ev1, offs, 0, ev, offs, 0, linalg_const<double_complex>::one(), dm__, 0, 0);
            //        }
            //    }
            //}
        }

        // compute the forces for the simplex LDA+U method not the fully
        // rotationally invariant one. It can not be used for LDA+U+SO either

        // It is based on this reference : PRB 84, 161102(R) (2011)

        void hubbard_force_add_k_contribution_colinear(K_point &kp__, mdarray<double, 2>& force__)
        {
            if (ctx_.so_correction() || (ctx_.num_mag_dims() == 3)) {
                TERMINATE("Hubbard forces are yet implemented for the full lda+U corrections");
            }

            // temporary wave functions
            Wave_functions phi(kp__.gkvec_partition(), hamiltonian_.U().number_of_hubbard_orbitals(), ctx_.num_spins());
            Wave_functions phitmp(kp__.gkvec_partition(), hamiltonian_.U().number_of_hubbard_orbitals(), ctx_.num_spins());
            Wave_functions dphi(kp__.gkvec_partition(), hamiltonian_.U().number_of_hubbard_orbitals(), ctx_.num_spins());

            // We generate the atomic orbitals again since there is no waranty
            // that atomic orbitals of different atoms are not mixed
            // together. Since we need the gradient of these guys compared to the
            // positions of the atoms better calculate them again

            hamiltonian_.U().GenerateAtomicOrbitals(kp__, phi);

            // check if we have a norm conserving pseudo potential only
            bool augment = false;
            for (auto ia = 0; (ia < ctx_.unit_cell().num_atom_types()) && (!augment); ia++) {
                augment = ctx_.unit_cell().atom_type(ia).augment();
            }

            if (ctx_.gamma_point() && (ctx_.so_correction() == false)) {
                hamiltonian_.prepare<double>();
            } else {
                hamiltonian_.prepare<double_complex>();
            }

#ifdef __GPU
            if (ctx_.processing_unit() == GPU) {
                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    /* allocate GPU memory */
                    phi.pw_coeffs(ispn).prime().allocate(memory_t::device);
                    phi.pw_coeffs(ispn).copy_to_device(0, hamiltonian_.U().number_of_hubbard_orbitals());
                }
            }
#endif

            auto &q_op = hamiltonian_.Q<double_complex>();

            if (!ctx_.full_potential() && augment) {
                /* apply the s matrix here on the orbitals (ultra soft pseudo potential) */
                /* Not the block functions. */
                // Put it in dphi
                for (int i = 0; i < kp__.beta_projectors().num_chunks(); i++) {
                    /* generate beta-projectors for a block of atoms */
                    kp__.beta_projectors().generate(i);
                    /* non-collinear case */
                    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {

                        auto beta_phi = kp__.beta_projectors().inner<double_complex>(i, phi, ispn, 0, hamiltonian_.U().number_of_hubbard_orbitals());

                        /* apply Q operator (diagonal in spin) */
                        q_op.apply(i, ispn, dphi, 0, hamiltonian_.U().number_of_hubbard_orbitals(), kp__.beta_projectors(),
                                   beta_phi);
                        /* apply non-diagonal spin blocks */
                        if (ctx_.so_correction()) {
                            q_op.apply(i, ispn ^ 3, dphi, 0, hamiltonian_.U().number_of_hubbard_orbitals(), kp__.beta_projectors(), beta_phi);
                        }
                    }
                }
            }

            // I will need to multiply by the conjugate because I compute
            // <psi | S | phi> instead.

            int HowManyBands = kp__.num_occupied_bands(0);
            if (ctx_.num_spins() == 2)
                HowManyBands = kp__.num_occupied_bands(1);

            dmatrix<double_complex> PhiSPsi(hamiltonian_.U().number_of_hubbard_orbitals(), HowManyBands);

            // now we compute the derivative of <phi | S | psi> compared to r_alpha
            // we have two terms <d phi| S | psi> + <phi | d S | psi>

            // the second term is split into two terms (only present for the ultra soft case)
            //<phi | d S | psi> = \sum_ij q_ij (<phi | dbeta_i > < beta_j|psi> + <phi | beta_i > < dbeta_j|psi>

            dmatrix<double_complex> dPhiSPsi(hamiltonian_.U().number_of_hubbard_orbitals(), HowManyBands);

            // I need now the derivatives of the beta projectors
            Beta_projectors_gradient bp_grad(ctx_, kp__.gkvec(), kp__.igk_loc(), kp__.beta_projectors());

            for (int dir = 0; dir < 3; dir++) {
                // Ultra soft case

                // compute \partial phi/\partial_{r_a}

                hamiltonian_.U().ComputeDerivatives(kp__, phi, dphi, dir);

                // compute  S | dphi>

                if (!ctx_.full_potential() && augment) {
#ifdef __GPU
                    if (ctx_.processing_unit() == GPU) {
                        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                            phitmp.pw_coeffs(ispn).prime().allocate(memory_t::device);
                            phitmp.pw_coeffs(ispn).copy_to_device(0, hamiltonian_.U().number_of_hubbard_orbitals());
                        }
                    }
#endif
                    for (int s = 0; s < ctx_.num_spins(); s++) {
                        // I need to consider the case where all atoms are norm
                        // conserving. In that case the S operator is diagonal in orbital space
                        phitmp.copy_from(ctx_.processing_unit(), hamiltonian_.U().number_of_hubbard_orbitals(), dphi, s, 0, s, 0);
                    }

                    // Now add the contribution from the beta projectors
                    for (int i = 0; i < kp__.beta_projectors().num_chunks(); i++) {
                        // Ultra soft case
                        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                            auto beta_dphi = kp__.beta_projectors().inner<double_complex>(i, phitmp, ispn, 0, hamiltonian_.U().number_of_hubbard_orbitals());

                            // compute  S | dphi>

                            /* apply Q operator (diagonal in spin) */
                            q_op.apply(i, ispn, dphi, 0, hamiltonian_.U().number_of_hubbard_orbitals(), kp__.beta_projectors(),
                                       beta_dphi);

                            // compute dS | phi> and add it it d(S|phi>)
                            // dS contains two terms Q_ij (|d \beta_i><beta_j| + |beta_i><d beta_j|)

                            // first term Q_{ij} |d beta_i><beta_j|

                            // compute the derivatives of the beta projectors and compute
                            // the overlap with the atomic orbitals.
                            bp_grad.generate(i, dir);
                            auto dbeta_phi = bp_grad.template inner<double_complex>(i, phi, ispn, 0, hamiltonian_.U().number_of_hubbard_orbitals());

                            // Q_ij <d\beta_i|phi> |beta_j>
                            // add it to dphi

                            /* apply Q operator (diagonal in spin) */
                            q_op.apply(i, ispn, dphi, 0, hamiltonian_.U().number_of_hubbard_orbitals(), kp__.beta_projectors(), dbeta_phi);

                            //second term Q_{ij} |beta_i><d beta_j|

                            // Q_ij <\beta_i|phi> |d beta_j>

                            /* apply Q operator (diagonal in spin) */
                            q_op.apply(i, ispn, dphi, 0, hamiltonian_.U().number_of_hubbard_orbitals(), kp__.beta_projectors(),
                                       beta_dphi);
                        }
                    }
                }

#ifdef __GPU
                if (ctx_.processing_unit() == GPU) {
                    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                        kp__.hubbard_wave_functions().pw_coeffs(ispn).prime().allocate(memory_t::device);
                        kp__.hubbard_wave_functions().pw_coeffs(ispn).copy_to_device(0, hamiltonian_.U().number_of_hubbard_orbitals());
                    }
                }
#endif

                // now compute the product with psi and add it to <psi | d (S | phi>)
                dPhiSPsi.zero();
                for (int ispn = 0; ispn < 2; ispn++) {
                    inner(ctx_.processing_unit(),
                          ispn,
                          kp__.spinor_wave_functions(),
                          0,
                          kp__.num_occupied_bands(ispn),
                          dphi,
                          0,
                          hamiltonian_.U().number_of_hubbard_orbitals(),
                          dPhiSPsi,
                          0,
                          (ispn == 1) * kp__.num_occupied_bands(1));
                }

#ifdef __GPU
                if (ctx_.processing_unit() == GPU) {
                    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                        dphi.deallocate_on_device(ispn);
                    }

                    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                        phi.pw_coeffs(ispn).prime().allocate(memory_t::device);
                        phi.pw_coeffs(ispn).copy_to_device(0, hamiltonian_.U().number_of_hubbard_orbitals());
                    }
                }
#endif

                if (!ctx_.full_potential() && augment) {

                    for (int s = 0; s < ctx_.num_spins(); s++) {
                        // I need to consider the case where all atoms are norm
                        // conserving. In that case the S operator is diagonal in orbital space
                        phitmp.copy_from(ctx_.processing_unit(), hamiltonian_.U().number_of_hubbard_orbitals(), phi, s, 0, s, 0);
                    }

                    for (int i = 0; i < kp__.beta_projectors().num_chunks(); i++) {
                        // Ultra soft case
                        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                            auto beta_dphi = kp__.beta_projectors().template inner<double_complex>(i, phitmp, ispn,
                                                                                                   0, hamiltonian_.U().number_of_hubbard_orbitals());
                            // compute  S | phi>

                            /* apply Q operator (diagonal in spin) */
                            hamiltonian_.Q<double_complex>().apply(i, ispn, phi, 0, hamiltonian_.U().number_of_hubbard_orbitals(), kp__.beta_projectors(),
                                                                   beta_dphi);
                        }
                    }
                }

                // compute <phi|S|psi>
                PhiSPsi.zero();
                for (int ispn = 0; ispn < 2; ispn++) {
                    inner(ctx_.processing_unit(),
                          ispn,
                          kp__.spinor_wave_functions(),
                          0,
                          kp__.num_occupied_bands(ispn),
                          phi,
                          0,
                          hamiltonian_.U().number_of_hubbard_orbitals(),
                          PhiSPsi,
                          0,
                          (ispn == 1) * kp__.num_occupied_bands(1));
                }

#ifdef __GPU
                if (ctx_.processing_unit() == GPU) {
                    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                        kp__.hubbard_wave_functions().deallocate_on_device(ispn);
                        phi.deallocate_on_device(ispn);
                        phitmp.deallocate_on_device(ispn);
                    }
                }
#endif
                // compute d ns. See Eq.3 of ref PRB 84, 161102(R) (2011)
                // https://journals.aps.org/prb/pdf/10.1103/PhysRevB.84.161102

                mdarray<double_complex, 4> dn (2 * hamiltonian_.U().hubbard_lmax() + 1,
                                               2 * hamiltonian_.U().hubbard_lmax() + 1,
                                               ctx_.num_spins(), ctx_.unit_cell().num_atoms());
                #pragma omp parallel for
                for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ++ia) {
                    const auto& atom = ctx_.unit_cell().atom(ia);
                    const int lmax_at = 2 * atom.type().hubbard_l() + 1;
                    if (atom.type().hubbard_correction()) {
                        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                            size_t bnd_offset = 0;
                            if (ispn == 1)
                                bnd_offset = kp__.num_occupied_bands(0);
                            for (int m1 = 0; m1 < lmax_at; m1++) {
                                for (int m2 = 0; m2 < lmax_at; m2++) {
                                    for (int nbnd = 0; nbnd < kp__.num_occupied_bands(ispn); nbnd++) {
                                        dn(m1, m2, ispn, ia) += (std::conj(PhiSPsi(hamiltonian_.U().offset[ia] + m1, nbnd + bnd_offset)) *
                                                                 dPhiSPsi(hamiltonian_.U().offset[ia] + m2, nbnd + bnd_offset) +
                                                                 std::conj(dPhiSPsi(hamiltonian_.U().offset[ia] + m1, nbnd + bnd_offset)) *
                                                                 PhiSPsi(hamiltonian_.U().offset[ia] + m2, nbnd + bnd_offset)) *
                                            kp__.weight() * kp__.band_occupancy(nbnd, ispn);
                                    }
                                }
                            }
                        }
                    }
                }
                #pragma omp parallel for
                for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ++ia) {
                    const auto& atom = ctx_.unit_cell().atom(ia);
                    const int lmax_at = 2 * atom.type().hubbard_l() + 1;
                    if (atom.type().hubbard_correction()) {
                        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                            for (int m1 = 0; m1 < lmax_at; m1++) {
                                for (int m2 = 0; m2 < lmax_at; m2++) {
                                    force__(dir, ia) -= (hamiltonian_.U().U(m2, m1, ispn, ia).real() * dn(m1, m2, ispn, ia)).real();
                                }
                            }
                        }
                    }
                }
            }
        }

        void ibs_force(K_point* kp__,
                       mdarray<double, 2>& ffac__,
                       mdarray<double, 2>& forcek__)
        {

            PROFILE("sirius::Force::ibs_force");

            auto& uc = ctx_.unit_cell();
            //auto rl = ctx__.reciprocal_lattice();

            forcek__.zero();

            dmatrix<double_complex> dm(ctx_.num_fv_states(), ctx_.num_fv_states(), ctx_.blacs_grid(),
                                       ctx_.cyclic_block_size(), ctx_.cyclic_block_size());
            compute_dmat(kp__, dm);

            auto& fv_evec = kp__->fv_eigen_vectors();

            dmatrix<double_complex> h(kp__->gklo_basis_size(), kp__->gklo_basis_size(), ctx_.blacs_grid(),
                                      ctx_.cyclic_block_size(), ctx_.cyclic_block_size());
            dmatrix<double_complex> o(kp__->gklo_basis_size(), kp__->gklo_basis_size(), ctx_.blacs_grid(),
                                      ctx_.cyclic_block_size(), ctx_.cyclic_block_size());

            dmatrix<double_complex> h1(kp__->gklo_basis_size(), kp__->gklo_basis_size(), ctx_.blacs_grid(),
                                       ctx_.cyclic_block_size(), ctx_.cyclic_block_size());
            dmatrix<double_complex> o1(kp__->gklo_basis_size(), kp__->gklo_basis_size(), ctx_.blacs_grid(),
                                       ctx_.cyclic_block_size(), ctx_.cyclic_block_size());

            dmatrix<double_complex> zm1(kp__->gklo_basis_size(), ctx_.num_fv_states(), ctx_.blacs_grid(),
                                        ctx_.cyclic_block_size(), ctx_.cyclic_block_size());
            dmatrix<double_complex> zf(ctx_.num_fv_states(), ctx_.num_fv_states(), ctx_.blacs_grid(),
                                       ctx_.cyclic_block_size(), ctx_.cyclic_block_size());

            mdarray<double_complex, 2> alm_row(kp__->num_gkvec_row(), uc.max_mt_aw_basis_size());
            mdarray<double_complex, 2> alm_col(kp__->num_gkvec_col(), uc.max_mt_aw_basis_size());
            mdarray<double_complex, 2> halm_col(kp__->num_gkvec_col(), uc.max_mt_aw_basis_size());

            for (int ia = 0; ia < uc.num_atoms(); ia++)
                {
                    h.zero();
                    o.zero();

                    auto& atom = uc.atom(ia);
                    auto& type = atom.type();

                    /* generate matching coefficients for current atom */
                    kp__->alm_coeffs_row().generate(ia, alm_row);
                    kp__->alm_coeffs_col().generate(ia, alm_col);

                    /* setup apw-lo and lo-apw blocks */
                    hamiltonian_.set_fv_h_o_apw_lo(kp__, type, atom, ia, alm_row, alm_col, h, o);

                    /* apply MT Hamiltonian to column coefficients */
                    hamiltonian_.apply_hmt_to_apw<spin_block_t::nm>(atom, kp__->num_gkvec_col(), alm_col, halm_col);

                    /* conjugate row (<bra|) matching coefficients */
                    for (int i = 0; i < type.mt_aw_basis_size(); i++) {
                        for (int igk = 0; igk < kp__->num_gkvec_row(); igk++) {
                            alm_row(igk, i) = std::conj(alm_row(igk, i));
                        }
                    }

                    /* apw-apw block of the overlap matrix */
                    linalg<CPU>::gemm(0, 1, kp__->num_gkvec_row(), kp__->num_gkvec_col(), type.mt_aw_basis_size(),
                                      alm_row.at<CPU>(), alm_row.ld(), alm_col.at<CPU>(), alm_col.ld(), o.at<CPU>(), o.ld());

                    /* apw-apw block of the Hamiltonian matrix */
                    linalg<CPU>::gemm(0, 1, kp__->num_gkvec_row(), kp__->num_gkvec_col(), type.mt_aw_basis_size(),
                                      alm_row.at<CPU>(), alm_row.ld(), halm_col.at<CPU>(), halm_col.ld(), h.at<CPU>(), h.ld());

                    int iat = type.id();

                    for (int igk_col = 0; igk_col < kp__->num_gkvec_col(); igk_col++) { // loop over columns
                        auto gvec_col       = kp__->gkvec().gvec(kp__->igk_col(igk_col));
                        auto gkvec_col_cart = kp__->gkvec().gkvec_cart(kp__->igk_col(igk_col));
                        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) { // for each column loop over rows
                            auto gvec_row       = kp__->gkvec().gvec(kp__->igk_row(igk_row));
                            auto gkvec_row_cart = kp__->gkvec().gkvec_cart(kp__->igk_row(igk_row));

                            int ig12 = ctx_.gvec().index_g12(gvec_row, gvec_col);

                            int igs = ctx_.gvec().shell(ig12);

                            double_complex zt = std::conj(ctx_.gvec_phase_factor(ig12, ia)) * ffac__(iat, igs) * fourpi / uc.omega();

                            double t1 = 0.5 * dot(gkvec_row_cart, gkvec_col_cart);

                            h(igk_row, igk_col) -= t1 * zt;
                            o(igk_row, igk_col) -= zt;
                        }
                    }

                    for (int x = 0; x < 3; x++) {
                        for (int igk_col = 0; igk_col < kp__->num_gkvec_col(); igk_col++) { // loop over columns
                            auto gvec_col = kp__->gkvec().gvec(kp__->igk_col(igk_col));
                            for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) { // for each column loop over rows
                                auto gvec_row = kp__->gkvec().gvec(kp__->igk_row(igk_row));
                                int ig12 = ctx_.gvec().index_g12(gvec_row, gvec_col);

                                vector3d<double> vg = ctx_.gvec().gvec_cart(ig12);
                                h1(igk_row, igk_col) = double_complex(0.0, vg[x]) * h(igk_row, igk_col);
                                o1(igk_row, igk_col) = double_complex(0.0, vg[x]) * o(igk_row, igk_col);
                            }
                        }

                        for (int icol = 0; icol < kp__->num_lo_col(); icol++) {
                            for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) {
                                auto gkvec_row_cart = kp__->gkvec().gkvec_cart(kp__->igk_row(igk_row));
                                h1(igk_row, icol + kp__->num_gkvec_col()) = double_complex(0.0, gkvec_row_cart[x]) * h(igk_row, icol + kp__->num_gkvec_col());
                                o1(igk_row, icol + kp__->num_gkvec_col()) = double_complex(0.0, gkvec_row_cart[x]) * o(igk_row, icol + kp__->num_gkvec_col());
                            }
                        }

                        for (int irow = 0; irow < kp__->num_lo_row(); irow++) {
                            for (int igk_col = 0; igk_col < kp__->num_gkvec_col(); igk_col++) {
                                auto gkvec_col_cart = kp__->gkvec().gkvec_cart(kp__->igk_col(igk_col));
                                h1(irow + kp__->num_gkvec_row(), igk_col) = double_complex(0.0, -gkvec_col_cart[x]) * h(irow + kp__->num_gkvec_row(), igk_col);
                                o1(irow + kp__->num_gkvec_row(), igk_col) = double_complex(0.0, -gkvec_col_cart[x]) * o(irow + kp__->num_gkvec_row(), igk_col);
                            }
                        }

                        /* zm1 = H * V */
                        linalg<CPU>::gemm(0, 0, kp__->gklo_basis_size(), ctx_.num_fv_states(), kp__->gklo_basis_size(),
                                          linalg_const<double_complex>::one(), h1, fv_evec, linalg_const<double_complex>::zero(), zm1);

                        /* F = V^{+} * zm1 = V^{+} * H * V */
                        linalg<CPU>::gemm(2, 0, ctx_.num_fv_states(), ctx_.num_fv_states(), kp__->gklo_basis_size(),
                                          linalg_const<double_complex>::one(), fv_evec, zm1, linalg_const<double_complex>::zero(), zf);

                        /* zm1 = O * V */
                        linalg<CPU>::gemm(0, 0, kp__->gklo_basis_size(), ctx_.num_fv_states(), kp__->gklo_basis_size(),
                                          linalg_const<double_complex>::one(), o1, fv_evec, linalg_const<double_complex>::zero(), zm1);

                        STOP();
                        ///* multiply by energy */
                        //for (int i = 0; i < (int)kp__->spl_fv_states().local_size(); i++)
                        //{
                        //    int ist = kp__->spl_fv_states(i);
                        //    for (int j = 0; j < kp__->gklo_basis_size_row(); j++) zm1(j, i) = zm1(j, i) * kp__->fv_eigen_value(ist);
                        //}

                        /* F = F - V^{+} * zm1 = F - V^{+} * O * (E*V) */
                        linalg<CPU>::gemm(2, 0, ctx_.num_fv_states(), ctx_.num_fv_states(), kp__->gklo_basis_size(),
                                          double_complex(-1, 0), fv_evec, zm1, double_complex(1, 0), zf);

                        for (int i = 0; i < dm.num_cols_local(); i++) {
                            for (int j = 0; j < dm.num_rows_local(); j++) {
                                forcek__(x, ia) += kp__->weight() * real(dm(j, i) * zf(j, i));
                            }
                        }
                    }
                } //ia
        }

    public:
        Force(Simulation_context& ctx__, Density& density__, Potential& potential__, Hamiltonian &h__, K_point_set& kset__)
            : ctx_(ctx__)
            , density_(density__)
            , potential_(potential__)
            , kset_(kset__)
            , hamiltonian_(h__)
        {
        }

        inline void calc_forces_vloc()
        {
            PROFILE("sirius::Force::calc_forces_vloc");

            forces_vloc_ = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
            forces_vloc_.zero();

            auto& valence_rho = density_.rho();

            Radial_integrals_vloc<false> ri(ctx_.unit_cell(), ctx_.pw_cutoff(), ctx_.settings().nprii_vloc_);

            Unit_cell& unit_cell = ctx_.unit_cell();

            Gvec const& gvecs = ctx_.gvec();

            int gvec_count  = gvecs.gvec_count(ctx_.comm().rank());
            int gvec_offset = gvecs.gvec_offset(ctx_.comm().rank());

            double fact = valence_rho.gvec().reduced() ? 2.0 : 1.0;

            /* here the calculations are in lattice vectors space */
            #pragma omp parallel for
            for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
                Atom& atom = unit_cell.atom(ia);

                int iat = atom.type_id();

                for (int igloc = 0; igloc < gvec_count; igloc++) {
                    int ig  = gvec_offset + igloc;

                    /* cartesian form for getting cartesian force components */
                    vector3d<double> gvec_cart = gvecs.gvec_cart(ig);

                    /* scalar part of a force without multipying by G-vector */
                    double_complex z = fact * fourpi * ri.value(iat, gvecs.gvec_len(ig)) *
                        std::conj(valence_rho.f_pw_local(igloc)) *
                        std::conj(ctx_.gvec_phase_factor(ig, ia));

                    /* get force components multiplying by cartesian G-vector  */
                    for (int x: {0, 1, 2}) {
                        forces_vloc_(x, ia) -= (gvec_cart[x] * z).imag();
                    }
                }
            }

            ctx_.comm().allreduce(&forces_vloc_(0, 0), 3 * ctx_.unit_cell().num_atoms());
        }

        inline mdarray<double, 2> const& forces_vloc() const
        {
            return forces_vloc_;
        }

        inline void calc_forces_nonloc()
        {
            PROFILE("sirius::Force::calc_forces_nonloc");

            forces_nonloc_ = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
            forces_nonloc_.zero();

            auto& spl_num_kp = kset_.spl_num_kpoints();

            for (int ikploc = 0; ikploc < spl_num_kp.local_size(); ikploc++) {
                K_point* kp = kset_[spl_num_kp[ikploc]];

                if (ctx_.gamma_point()) {
                    add_k_point_contribution<double>(*kp, forces_nonloc_);
                } else {
                    add_k_point_contribution<double_complex>(*kp, forces_nonloc_);
                }
            }

            ctx_.comm().allreduce(&forces_nonloc_(0, 0), 3 * ctx_.unit_cell().num_atoms());

            symmetrize(forces_nonloc_);
        }

        inline mdarray<double, 2> const& forces_nonloc() const
        {
            return forces_nonloc_;
        }

        inline void calc_forces_core()
        {
            PROFILE("sirius::Force::calc_forces_core");

            forces_core_ = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
            forces_core_.zero();

            /* get main arrays */
            auto xc_pot = potential_.xc_potential();

            /* transform from real space to reciprocal */
            xc_pot->fft_transform(-1);

            Unit_cell& unit_cell = ctx_.unit_cell();

            Gvec const& gvecs = ctx_.gvec();

            int gvec_count  = gvecs.count();
            int gvec_offset = gvecs.offset();

            double fact = gvecs.reduced() ? 2.0 : 1.0;

            Radial_integrals_rho_core_pseudo<false> ri(ctx_.unit_cell(), ctx_.pw_cutoff(), ctx_.settings().nprii_rho_core_);

            /* here the calculations are in lattice vectors space */
            #pragma omp parallel for
            for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
                Atom& atom = unit_cell.atom(ia);

                int iat = atom.type_id();

                for (int igloc = 0; igloc < gvec_count; igloc++) {
                    int ig = gvec_offset + igloc;

                    if (ig == 0) {
                        continue;
                    }

                    /* cartesian form for getting cartesian force components */
                    vector3d<double> gvec_cart = gvecs.gvec_cart(ig);

                    /* scalar part of a force without multipying by G-vector */
                    double_complex z = fact * fourpi * ri.value<int>(iat, gvecs.gvec_len(ig)) *
                        std::conj(xc_pot->f_pw_local(igloc) * ctx_.gvec_phase_factor(ig, ia));

                    /* get force components multiplying by cartesian G-vector */
                    for (int x: {0, 1, 2}) {
                        forces_core_(x, ia) -= (gvec_cart[x] * z).imag();
                    }
                }
            }
            ctx_.comm().allreduce(&forces_core_(0, 0), 3 * ctx_.unit_cell().num_atoms());
        }

        inline mdarray<double, 2> const& forces_core() const
        {
            return forces_core_;
        }

        inline void calc_forces_scf_corr()
        {
            PROFILE("sirius::Force::calc_forces_scf_corr");

            forces_scf_corr_ = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
            forces_scf_corr_.zero();

            /* get main arrays */
            auto& dveff = potential_.dveff();

            Unit_cell& unit_cell = ctx_.unit_cell();

            Gvec const& gvec = ctx_.gvec();

            int gvec_count  = gvec.count();
            int gvec_offset = gvec.offset();

            double fact = gvec.reduced() ? 2.0 : 1.0;

            int ig0 = (ctx_.comm().rank() == 0) ? 1 : 0;

            Radial_integrals_rho_pseudo ri(ctx_.unit_cell(), ctx_.pw_cutoff(), 20);

            #pragma omp parallel for
            for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
                Atom& atom = unit_cell.atom(ia);

                int iat = atom.type_id();

                for (int igloc = ig0; igloc < gvec_count; igloc++) {
                    int ig = gvec_offset + igloc;

                    /* cartesian form for getting cartesian force components */
                    vector3d<double> gvec_cart = gvec.gvec_cart(ig);

                    /* scalar part of a force without multipying by G-vector */
                    double_complex z = fact * fourpi * ri.value<int>(iat, gvec.gvec_len(ig)) *
                        std::conj(dveff.f_pw_local(igloc) * ctx_.gvec_phase_factor(ig, ia));

                    /* get force components multiplying by cartesian G-vector */
                    for (int x: {0, 1, 2}) {
                        forces_scf_corr_(x, ia) -= (gvec_cart[x] * z).imag();
                    }
                }
            }
            ctx_.comm().allreduce(&forces_scf_corr_(0, 0), 3 * ctx_.unit_cell().num_atoms());
        }

        inline mdarray<double, 2> const& forces_scf_corr() const
        {
            return forces_scf_corr_;
        }

        inline void calc_forces_us()
        {
            PROFILE("sirius::Force::calc_forces_us");

            forces_us_ = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
            forces_us_.zero();

            /* pack v effective in one array of pointers*/
            Periodic_function<double>* vfield_eff[4];
            vfield_eff[0] = potential_.effective_potential();
            vfield_eff[0]->fft_transform(-1);
            for (int imagn = 0; imagn < ctx_.num_mag_dims(); imagn++){
                vfield_eff[imagn + 1] = potential_.effective_magnetic_field(imagn);
                vfield_eff[imagn + 1]->fft_transform(-1);
            }

            Unit_cell& unit_cell = ctx_.unit_cell();

            double reduce_g_fact = ctx_.gvec().reduced() ? 2.0 : 1.0;

            /* over atom types */
            for (int iat = 0; iat < unit_cell.num_atom_types(); iat++){
                auto& atom_type = unit_cell.atom_type(iat);

                if (!atom_type.augment()) {
                    continue;
                }

                const Augmentation_operator& aug_op = ctx_.augmentation_op(iat);

                int nbf = atom_type.mt_basis_size();

                /* get auxiliary density matrix */
                auto dm = density_.density_matrix_aux(iat);

                //mdarray<double, 2> q_tmp(nbf * (nbf + 1) / 2, ctx_.gvec().count() * 2);
                mdarray<double, 2> v_tmp(atom_type.num_atoms(), ctx_.gvec().count() * 2);
                mdarray<double, 2> tmp(nbf * (nbf + 1) / 2, atom_type.num_atoms());

                /* over spin components, can be from 1 to 4*/
                for (int ispin = 0; ispin < ctx_.num_mag_dims() + 1; ispin++ ){
                    /* over 3 components of the force/G - vectors */
                    for (int ivec = 0; ivec < 3; ivec++ ){
                        /* over local rank G vectors */
                        #pragma omp parallel for schedule(static)
                        for (int igloc = 0; igloc < ctx_.gvec().count(); igloc++) {
                            int ig = ctx_.gvec().offset() + igloc;
                            auto gvc = ctx_.gvec().gvec_cart(ig);
                            for (int ia = 0; ia < atom_type.num_atoms(); ia++) {
                                /* here we write in v_tmp  -i * G * exp[ iGRn] Veff(G)
                                 * but in formula we have   i * G * exp[-iGRn] Veff*(G)
                                 * the differences because we unfold complex array in the real one
                                 * and need negative imagine part due to a multiplication law of complex numbers */
                                auto z = double_complex(0, -gvc[ivec]) * ctx_.gvec_phase_factor(ig, atom_type.atom_id(ia)) *
                                    vfield_eff[ispin]->f_pw_local(igloc);
                                v_tmp(ia, 2 * igloc)     = z.real();
                                v_tmp(ia, 2 * igloc + 1) = z.imag();
                            }
                        }

                        /* multiply tmp matrices, or sum over G*/
                        linalg<CPU>::gemm(0, 1, nbf * (nbf + 1) / 2, atom_type.num_atoms(), 2 * ctx_.gvec().count(),
                                          aug_op.q_pw(), v_tmp, tmp);


                        #pragma omp parallel for
                        for (int ia = 0; ia < atom_type.num_atoms(); ia++) {
                            for (int i = 0; i < nbf * (nbf + 1) / 2; i++) {
                                forces_us_(ivec, atom_type.atom_id(ia)) += ctx_.unit_cell().omega() * reduce_g_fact *
                                    dm(i, ia, ispin) * aug_op.sym_weight(i) *
                                    tmp(i, ia);
                            }
                        }
                    }
                }
            }

            ctx_.comm().allreduce(&forces_us_(0, 0), 3 * ctx_.unit_cell().num_atoms());
        }

        inline mdarray<double, 2> const& forces_us() const
        {
            return forces_us_;
        }

        inline void calc_forces_ewald()
        {
            PROFILE("sirius::Force::calc_forces_ewald");

            forces_ewald_ = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
            forces_ewald_.zero();

            Unit_cell& unit_cell = ctx_.unit_cell();

            double alpha = ctx_.ewald_lambda();

            double prefac = (ctx_.gvec().reduced() ? 4.0 : 2.0) * (twopi / unit_cell.omega());

            int ig0{0};
            if (ctx_.comm().rank() == 0) {
                ig0 = 1;
            }

            mdarray<double_complex, 1> rho_tmp(ctx_.gvec().count());
            rho_tmp.zero();
            #pragma omp parallel for schedule(static)
            for (int igloc = ig0; igloc < ctx_.gvec().count(); igloc++) {
                int ig = ctx_.gvec().offset() + igloc;

                double_complex rho(0, 0);

                for (int ja = 0; ja < unit_cell.num_atoms(); ja++) {
                    rho += ctx_.gvec_phase_factor(ig, ja) * static_cast<double>(unit_cell.atom(ja).zn());
                }

                rho_tmp[igloc] = std::conj(rho);
            }

            #pragma omp parallel for
            for (int ja = 0; ja < unit_cell.num_atoms(); ja++) {
                for (int igloc = ig0; igloc < ctx_.gvec().count(); igloc++) {
                    int ig = ctx_.gvec().offset() + igloc;

                    double g2 = std::pow(ctx_.gvec().gvec_len(ig), 2);

                    /* cartesian form for getting cartesian force components */
                    vector3d<double> gvec_cart = ctx_.gvec().gvec_cart(ig);
                    double_complex rho(0, 0);

                    double scalar_part = prefac * (rho_tmp[igloc] * ctx_.gvec_phase_factor(ig, ja)).imag() *
                        static_cast<double>(unit_cell.atom(ja).zn()) * std::exp(-g2 / (4 * alpha)) / g2;

                    for (int x : {0, 1, 2}) {
                        forces_ewald_(x, ja) += scalar_part * gvec_cart[x];
                    }
                }
            }

            ctx_.comm().allreduce(&forces_ewald_(0, 0), 3 * ctx_.unit_cell().num_atoms());

            double invpi = 1. / pi;

            #pragma omp parallel for
            for (int ia = 0; ia < unit_cell.num_atoms(); ia++) {
                for (int i = 1; i < unit_cell.num_nearest_neighbours(ia); i++) {
                    int ja = unit_cell.nearest_neighbour(i, ia).atom_id;

                    double d  = unit_cell.nearest_neighbour(i, ia).distance;
                    double d2 = d * d;

                    vector3d<double> t = unit_cell.lattice_vectors() * vector3d<int>(unit_cell.nearest_neighbour(i, ia).translation);

                    double scalar_part =
                        static_cast<double>(unit_cell.atom(ia).zn() * unit_cell.atom(ja).zn()) / d2 *
                        (gsl_sf_erfc(std::sqrt(alpha) * d) / d + 2.0 * std::sqrt(alpha * invpi) * std::exp(-d2 * alpha));

                    for (int x : {0, 1, 2}) {
                        forces_ewald_(x, ia) += scalar_part * t[x];
                    }
                }
            }
        }

        inline void calc_forces_hubbard()
        {

            forces_hubbard_ = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
            forces_hubbard_.zero();

            for (int ikloc = 0; ikloc < kset_.spl_num_kpoints().local_size(); ikloc++) {

                int ik  = kset_.spl_num_kpoints(ikloc);
                auto kp = kset_[ik];
                if (ctx_.num_mag_dims() == 3)
                    TERMINATE("Hubbard forces are only implemented for the simple hubbard correction.");

                hubbard_force_add_k_contribution_colinear(*kp, forces_hubbard_);
            }

            // global reduction
            ctx_.comm().allreduce<double, mpi_op_t::sum>(forces_hubbard_.at<CPU>(),
                                                         static_cast<int>(forces_hubbard_.size()));
        }

        inline mdarray<double, 2> const& forces_ewald() const
        {
            return forces_ewald_;
        }

        inline void calc_forces_total()
        {
            calc_forces_vloc();
            calc_forces_us();
            calc_forces_nonloc();
            calc_forces_core();
            calc_forces_ewald();
            calc_forces_scf_corr();

            if (ctx_.hubbard_correction())
                calc_forces_hubbard();

            forces_total_ = mdarray<double, 2>(3, ctx_.unit_cell().num_atoms());
            for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
                for (int x: {0, 1, 2}) {
                    forces_total_(x, ia) = forces_vloc_(x, ia) + forces_us_(x, ia) + forces_nonloc_(x, ia) +
                        forces_core_(x, ia) + forces_ewald_(x, ia) + forces_scf_corr_(x, ia);
                }

                if (ctx_.hubbard_correction()) {
                    for (int x: {0, 1, 2}) {
                        forces_total_(x, ia) += forces_hubbard_(x, ia);
                    }
                }
            }
        }

        inline mdarray<double, 2> const& forces_total() const
        {
            return forces_total_;
        }

        inline mdarray<double, 2> const& forces_hubbard() const
        {
            return forces_hubbard_;
        }

        inline void print_info()
        {
            PROFILE("sirius::DFT_ground_state::forces");

            if (ctx_.comm().rank() == 0) {
                auto print_forces = [&](mdarray<double, 2> const& forces)
                    {
                        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
                            printf("atom %4i    force = %15.7f  %15.7f  %15.7f \n", ctx_.unit_cell().atom(ia).type_id(),
                                   forces(0, ia), forces(1, ia), forces(2, ia));
                        }
                    };

                printf("===== total Forces in Ha/bohr =====\n");
                print_forces(forces_total());

                printf("===== ultrasoft contribution from Qij =====\n");
                print_forces(forces_us());

                printf("===== non-local contribution from Beta-projectors =====\n");
                print_forces(forces_nonloc());

                printf("===== contribution from local potential =====\n");
                print_forces(forces_vloc());

                printf("===== contribution from core density =====\n");
                print_forces(forces_core());

                printf("===== Ewald forces from ions =====\n");
                print_forces(forces_ewald());

                if (ctx_.hubbard_correction()) {
                    printf("===== Ewald forces from hubbard correction =====\n");
                    print_forces(forces_hubbard());
                }
            }
        }

        void total_force(Hamiltonian* hamiltonian_,
                         Potential* potential__,
                         Density* density__,
                         K_point_set* ks__,
                         mdarray<double, 2>& force__)
        {
            PROFILE("sirius::Force::total_force");

            auto& uc = ctx_.unit_cell();

            //auto ffac = ctx__.step_function().get_step_function_form_factors(ctx__.gvec().num_shells(), ctx__.unit_cell(), ctx__.gvec(), ctx__.comm());
            STOP();

            force__.zero();

            mdarray<double, 2> forcek(3, uc.num_atoms());
            for (int ikloc = 0; ikloc < ks__->spl_num_kpoints().local_size(); ikloc++) {
                //int ik = ks__->spl_num_kpoints(ikloc);
                //ibs_force(ctx__, band_, (*ks__)[ik], ffac, forcek);
                STOP();
                for (int ia = 0; ia < uc.num_atoms(); ia++) {
                    for (int x: {0, 1, 2}) {
                        force__(x, ia) += forcek(x, ia);
                    }
                }
            }
            ctx_.comm().allreduce(&force__(0, 0), (int)force__.size());

            if (ctx_.control().verbosity_ > 2 && ctx_.comm().rank() == 0) {
                printf("\n");
                printf("Forces\n");
                printf("\n");
                for (int ia = 0; ia < uc.num_atoms(); ia++) {
                    printf("ia : %i, IBS : %12.6f %12.6f %12.6f\n", ia, force__(0, ia), force__(1, ia), force__(2, ia));
                }
            }

            mdarray<double, 2> forcehf(3, uc.num_atoms());

            forcehf.zero();
            for (int ialoc = 0; ialoc < (int)uc.spl_num_atoms().local_size(); ialoc++)
                {
                    int ia = uc.spl_num_atoms(ialoc);
                    auto g = gradient(potential__->hartree_potential_mt(ialoc));
                    for (int x = 0; x < 3; x++) forcehf(x, ia) = uc.atom(ia).zn() * g[x](0, 0) * y00;
                }
            ctx_.comm().allreduce(&forcehf(0, 0), (int)forcehf.size());

            if (ctx_.control().verbosity_ > 2 && ctx_.comm().rank() == 0) {
                printf("\n");
                for (int ia = 0; ia < uc.num_atoms(); ia++) {
                    printf("ia : %i, Hellmann–Feynman : %12.6f %12.6f %12.6f\n", ia, forcehf(0, ia), forcehf(1, ia), forcehf(2, ia));
                }
            }

            mdarray<double, 2> forcerho(3, uc.num_atoms());
            forcerho.zero();
            for (int ialoc = 0; ialoc < (int)uc.spl_num_atoms().local_size(); ialoc++)
                {
                    int ia = uc.spl_num_atoms(ialoc);
                    auto g = gradient(density__->density_mt(ialoc));
                    for (int x = 0; x < 3; x++) forcerho(x, ia) = inner(potential__->effective_potential_mt(ialoc), g[x]);
                }
            ctx_.comm().allreduce(&forcerho(0, 0), (int)forcerho.size());

            if (ctx_.control().verbosity_ > 2 && ctx_.comm().rank() == 0) {
                printf("\n");
                printf("rho force\n");
                for (int ia = 0; ia < uc.num_atoms(); ia++) {
                    printf("ia : %i, density contribution : %12.6f %12.6f %12.6f\n", ia, forcerho(0, ia), forcerho(1, ia), forcerho(2, ia));
                }
            }

            for (int ia = 0; ia < uc.num_atoms(); ia++) {
                for (int x = 0; x < 3; x++) {
                    force__(x, ia) += (forcehf(x, ia) + forcerho(x, ia));
                }
            }
        }
    };
}

#endif // __FORCES_HAMILTONIAN__
