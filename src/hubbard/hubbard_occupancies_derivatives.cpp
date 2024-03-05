// Copyright (c) 2013-2022 Mathieu Taillefumier, Anton Kozhevnikov, Thomas Schulthess
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

/** \file hubbard_occupancies_derivatives.hpp
 *
 *  \brief Generate derivatives of occupancy matrix.
 *
 * Compute the forces for the simple LDA+U method not the fully rotationally invariant one.
 * It can not be used for LDA+U+SO either.
 *
 * This code is based on two papers :
 *   - PRB 84, 161102(R) (2011) : https://journals.aps.org/prb/abstract/10.1103/PhysRevB.84.161102
 *   - PRB 102, 235159 (2020) : https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.235159
 *
 * \note This code only applies to the collinear case.
 */

#include "beta_projectors/beta_projectors_base.hpp"
#include "hubbard.hpp"
#include "core/la/inverse_sqrt.hpp"
#include "geometry/wavefunction_strain_deriv.hpp"

namespace sirius {

static void
update_density_matrix_deriv(la::lib_t la__, memory_t mt__, int nwfh__, int nbnd__, std::complex<double>* alpha__,
                            la::dmatrix<std::complex<double>> const& phi_hub_s_psi_deriv__,
                            la::dmatrix<std::complex<double>> const& psi_s_phi_hub__, std::complex<double>* dn__,
                            int ld__)
{
    la::wrap(la__).gemm('N', 'N', nwfh__, nwfh__, nbnd__, alpha__, phi_hub_s_psi_deriv__.at(mt__, 0, 0),
                        phi_hub_s_psi_deriv__.ld(), psi_s_phi_hub__.at(mt__, 0, 0), psi_s_phi_hub__.ld(),
                        &la::constant<std::complex<double>>::one(), dn__, ld__);

    la::wrap(la__).gemm('C', 'C', nwfh__, nwfh__, nbnd__, alpha__, psi_s_phi_hub__.at(mt__, 0, 0), psi_s_phi_hub__.ld(),
                        phi_hub_s_psi_deriv__.at(mt__, 0, 0), phi_hub_s_psi_deriv__.ld(),
                        &la::constant<std::complex<double>>::one(), dn__, ld__);
}

static void
build_phi_hub_s_psi_deriv(Simulation_context const& ctx__, int nbnd__, int nawf__,
                          la::dmatrix<std::complex<double>> const& ovlp__,
                          la::dmatrix<std::complex<double>> const& inv_sqrt_O__,
                          la::dmatrix<std::complex<double>> const& phi_atomic_s_psi__,
                          la::dmatrix<std::complex<double>> const& phi_atomic_ds_psi__,
                          std::vector<int> const& atomic_wf_offset__, std::vector<int> const& hubbard_wf_offset__,
                          la::dmatrix<std::complex<double>>& phi_hub_s_psi_deriv__)
{
    phi_hub_s_psi_deriv__.zero();

    for (int ia = 0; ia < ctx__.unit_cell().num_atoms(); ia++) {
        auto& type = ctx__.unit_cell().atom(ia).type();

        if (type.hubbard_correction()) {
            /* loop over Hubbard orbitals of the atom */
            for (auto e : type.indexr_hub()) {
                auto& hd = type.lo_descriptor_hub(e.idxrf);
                int l    = e.am.l();
                int mmax = 2 * l + 1;

                int idxr_wf       = hd.idx_wf();
                int offset_in_wf  = atomic_wf_offset__[ia] + type.indexb_wfs().index_of(rf_index(idxr_wf));
                int offset_in_hwf = hubbard_wf_offset__[ia] + type.indexb_hub().index_of(e.idxrf);

                if (ctx__.cfg().hubbard().hubbard_subspace_method() == "full_orthogonalization") {
                    /* compute \sum_{m} d/d r_{alpha} O^{-1/2}_{m,i} <phi_atomic_{m} | S | psi_{jk} > */
                    la::wrap(la::lib_t::blas)
                            .gemm('C', 'N', mmax, nbnd__, nawf__, &la::constant<std::complex<double>>::one(),
                                  ovlp__.at(memory_t::host, 0, offset_in_wf), ovlp__.ld(),
                                  phi_atomic_s_psi__.at(memory_t::host), phi_atomic_s_psi__.ld(),
                                  &la::constant<std::complex<double>>::one(),
                                  phi_hub_s_psi_deriv__.at(memory_t::host, offset_in_hwf, 0),
                                  phi_hub_s_psi_deriv__.ld());

                    la::wrap(la::lib_t::blas)
                            .gemm('C', 'N', mmax, nbnd__, nawf__, &la::constant<std::complex<double>>::one(),
                                  inv_sqrt_O__.at(memory_t::host, 0, offset_in_wf), inv_sqrt_O__.ld(),
                                  phi_atomic_ds_psi__.at(memory_t::host), phi_atomic_ds_psi__.ld(),
                                  &la::constant<std::complex<double>>::one(),
                                  phi_hub_s_psi_deriv__.at(memory_t::host, offset_in_hwf, 0),
                                  phi_hub_s_psi_deriv__.ld());
                } else {
                    /* just copy part of the matrix elements in the order in which
                     * Hubbard wfs are defined */
                    for (int ibnd = 0; ibnd < nbnd__; ibnd++) {
                        for (int m = 0; m < mmax; m++) {
                            phi_hub_s_psi_deriv__(offset_in_hwf + m, ibnd) =
                                    phi_atomic_ds_psi__(offset_in_wf + m, ibnd);
                        }
                    }
                }
            } // idxrf
        }
    } // ia
}

static void
compute_inv_sqrt_O_deriv(la::dmatrix<std::complex<double>>& O_deriv__, la::dmatrix<std::complex<double>>& evec_O__,
                         std::vector<double>& eval_O__, int nawf__)
{
    /* compute \tilde O' = U^{H}O'U */
    unitary_similarity_transform(1, O_deriv__, evec_O__, nawf__);

    for (int i = 0; i < nawf__; i++) {
        for (int j = 0; j < nawf__; j++) {
            O_deriv__(j, i) /= -(eval_O__[i] * std::sqrt(eval_O__[j]) + eval_O__[j] * std::sqrt(eval_O__[i]));
        }
    }
    /* compute d/dr O^{-1/2} */
    unitary_similarity_transform(0, O_deriv__, evec_O__, nawf__);
}

void
Hubbard::compute_occupancies_derivatives(K_point<double>& kp__, Q_operator<double>& q_op__,
                                         mdarray<std::complex<double>, 5>& dn__)
{
    PROFILE("sirius::Hubbard::compute_occupancies_derivatives");

    auto la = la::lib_t::none;
    auto mt = memory_t::none;
    switch (ctx_.processing_unit()) {
        case device_t::CPU: {
            la = la::lib_t::blas;
            mt = memory_t::host;
            break;
        }
        case device_t::GPU: {
            la = la::lib_t::gpublas;
            mt = memory_t::device;
            break;
        }
    }
    auto alpha = std::complex<double>(kp__.weight(), 0.0);

    // TODO: check if we have a norm conserving pseudo potential;
    // TODO: distribute (MPI) all matrices in the basis of atomic orbitals
    // only derivatives of the atomic wave functions are needed.
    auto& phi_atomic   = kp__.atomic_wave_functions();
    auto& phi_atomic_S = kp__.atomic_wave_functions_S();

    auto num_ps_atomic_wf = ctx_.unit_cell().num_ps_atomic_wf();
    auto num_hubbard_wf   = ctx_.unit_cell().num_hubbard_wf();

    /* number of atomic wave-functions */
    int nawf = num_ps_atomic_wf.first;
    /* number of Hubbard wave-functions */
    int nhwf = num_hubbard_wf.first;

    RTE_ASSERT(nawf == phi_atomic.num_wf().get());
    RTE_ASSERT(nawf == phi_atomic_S.num_wf().get());

    if (ctx_.processing_unit() == device_t::GPU) {
        dn__.allocate(memory_t::device);
    }

    /* compute overlap matrix */
    la::dmatrix<std::complex<double>> ovlp;
    std::unique_ptr<la::dmatrix<std::complex<double>>> inv_sqrt_O;
    std::unique_ptr<la::dmatrix<std::complex<double>>> evec_O;
    std::vector<double> eval_O;
    if (ctx_.cfg().hubbard().hubbard_subspace_method() == "full_orthogonalization") {
        ovlp = la::dmatrix<std::complex<double>>(nawf, nawf);
        wf::inner(ctx_.spla_context(), mt, wf::spin_range(0), phi_atomic, wf::band_range(0, nawf), phi_atomic_S,
                  wf::band_range(0, nawf), ovlp, 0, 0);

        /* a tuple of O^{-1/2}, U, \lambda */
        auto result = inverse_sqrt(ovlp, nawf);
        inv_sqrt_O  = std::move(std::get<0>(result));
        evec_O      = std::move(std::get<1>(result));
        eval_O      = std::get<2>(result);
    }

    /* compute < psi_{ik} | S | phi_hub > */
    /* this is used in the final expression for the occupation matrix derivative */
    std::array<la::dmatrix<std::complex<double>>, 2> psi_s_phi_hub;
    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        psi_s_phi_hub[ispn] = la::dmatrix<std::complex<double>>(kp__.num_occupied_bands(ispn), nhwf);
        wf::inner(ctx_.spla_context(), mt, wf::spin_range(ispn), kp__.spinor_wave_functions(),
                  wf::band_range(0, kp__.num_occupied_bands(ispn)), kp__.hubbard_wave_functions_S(),
                  wf::band_range(0, nhwf), psi_s_phi_hub[ispn], 0, 0);
    }

    /* temporary storage */
    auto phi_atomic_tmp   = wave_function_factory(ctx_, kp__, phi_atomic.num_wf(), wf::num_mag_dims(0), false);
    auto s_phi_atomic_tmp = wave_function_factory(ctx_, kp__, phi_atomic_S.num_wf(), wf::num_mag_dims(0), false);
    auto mg1              = phi_atomic_tmp->memory_guard(mt);
    auto mg2              = s_phi_atomic_tmp->memory_guard(mt);

    /* compute < d phi_atomic / d r_{j} | S | psi_{ik} > and < d phi_atomic / d r_{j} | S | phi_atomic > */
    std::array<std::array<la::dmatrix<std::complex<double>>, 2>, 3> grad_phi_atomic_s_psi;
    std::array<la::dmatrix<std::complex<double>>, 3> grad_phi_atomic_s_phi_atomic;

    auto& bp       = kp__.beta_projectors();
    auto bp_gen    = bp.make_generator(mt);
    auto bp_coeffs = bp_gen.prepare();

    for (int x = 0; x < 3; x++) {
        /* compute |phi_atomic_tmp> = |d phi_atomic / d r_{alpha} > for all atoms */
        for (int i = 0; i < nawf; i++) {
            for (int igloc = 0; igloc < kp__.num_gkvec_loc(); igloc++) {
                /* G+k vector in Cartesian coordinates */
                auto gk = kp__.gkvec().gkvec_cart(gvec_index_t::local(igloc));
                /* gradient of phi_atomic */
                phi_atomic_tmp->pw_coeffs(igloc, wf::spin_index(0), wf::band_index(i)) =
                        std::complex<double>(0.0, -gk[x]) *
                        phi_atomic.pw_coeffs(igloc, wf::spin_index(0), wf::band_index(i));
            }
        }
        if (is_device_memory(mt)) {
            phi_atomic_tmp->copy_to(mt);
        }
        /* apply S to |d phi_atomic / d r_{alpha} > */
        apply_S_operator<double, std::complex<double>>(mt, wf::spin_range(0), wf::band_range(0, nawf), bp_gen,
                                                       bp_coeffs, *phi_atomic_tmp, &q_op__, *s_phi_atomic_tmp);

        /* compute < d phi_atomic / d r_{alpha} | S | phi_atomic >
         * used to compute derivative of the inverse square root of the overlap matrix */
        if (ctx_.cfg().hubbard().hubbard_subspace_method() == "full_orthogonalization") {
            grad_phi_atomic_s_phi_atomic[x] = la::dmatrix<std::complex<double>>(nawf, nawf);
            wf::inner(ctx_.spla_context(), mt, wf::spin_range(0), *s_phi_atomic_tmp, wf::band_range(0, nawf),
                      phi_atomic, wf::band_range(0, nawf), grad_phi_atomic_s_phi_atomic[x], 0, 0);
        }

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            /* allocate space */
            grad_phi_atomic_s_psi[x][ispn] = la::dmatrix<std::complex<double>>(nawf, kp__.num_occupied_bands(ispn));
            /* compute < d phi_atomic / d r_{j} | S | psi_{ik} > for all atoms */
            wf::inner(ctx_.spla_context(), mt, wf::spin_range(ispn), *s_phi_atomic_tmp, wf::band_range(0, nawf),
                      kp__.spinor_wave_functions(), wf::band_range(0, kp__.num_occupied_bands(ispn)),
                      grad_phi_atomic_s_psi[x][ispn], 0, 0);
        }
    }

    /* compute <phi_atomic | S | psi_{ik} > */
    std::array<la::dmatrix<std::complex<double>>, 2> phi_atomic_s_psi;
    if (ctx_.cfg().hubbard().hubbard_subspace_method() == "full_orthogonalization") {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            phi_atomic_s_psi[ispn] = la::dmatrix<std::complex<double>>(nawf, kp__.num_occupied_bands(ispn));
            /* compute < phi_atomic | S | psi_{ik} > for all atoms */
            wf::inner(ctx_.spla_context(), mt, wf::spin_range(ispn), phi_atomic_S, wf::band_range(0, nawf),
                      kp__.spinor_wave_functions(), wf::band_range(0, kp__.num_occupied_bands(ispn)),
                      phi_atomic_s_psi[ispn], 0, 0);
        }
    }

    Beta_projectors_gradient<double> bp_grad(ctx_, kp__.gkvec(), kp__.beta_projectors());
    auto bp_grad_gen    = bp_grad.make_generator(mt);
    auto bp_grad_coeffs = bp_grad_gen.prepare();

    dn__.zero(mt);

    for (int ichunk = 0; ichunk < kp__.beta_projectors().num_chunks(); ichunk++) {

        // store <beta|x> on device if `mt` is device memory.
        bool copy_back_innerb = is_device_memory(mt);

        bp_gen.generate(bp_coeffs, ichunk);
        auto& beta_chunk = bp_coeffs.beta_chunk_;

        /* <beta | phi_atomic> for this chunk */
        // auto beta_phi_atomic = kp__.beta_projectors().inner<std::complex<double>>(mt, ichunk, phi_atomic,
        // wf::spin_index(0),
        //         wf::band_range(0, nawf));
        auto beta_phi_atomic = inner_prod_beta<std::complex<double>>(ctx_.spla_context(), mt, ctx_.host_memory_t(),
                                                                     copy_back_innerb, bp_coeffs, phi_atomic,
                                                                     wf::spin_index(0), wf::band_range(0, nawf));

        for (int x = 0; x < 3; x++) {
            bp_grad_gen.generate(bp_grad_coeffs, ichunk, x);

            /* <dbeta | phi> for this chunk */
            // auto grad_beta_phi_atomic = bp_grad.inner<std::complex<double>>(mt, ichunk, phi_atomic,
            // wf::spin_index(0),
            //         wf::band_range(0, nawf));
            auto grad_beta_phi_atomic = inner_prod_beta<std::complex<double>>(
                    ctx_.spla_context(), mt, ctx_.host_memory_t(), copy_back_innerb, bp_grad_coeffs, phi_atomic,
                    wf::spin_index(0), wf::band_range(0, nawf));

            for (int i = 0; i < beta_chunk->num_atoms_; i++) {
                /* this is a displacement atom */
                int ja = beta_chunk->desc_(beta_desc_idx::ia, i);

                /* build |phi_atomic_tmp> = | d S / d r_{j} | phi_atomic > */
                /* it consists of two contributions:
                 *   | beta >        Q < d beta / dr | phi_atomic > and
                 *   | d beta / dr > Q < beta        | phi_atomic > */
                phi_atomic_tmp->zero(mt, wf::spin_index(0), wf::band_range(0, nawf));
                if (ctx_.unit_cell().atom(ja).type().augment()) {
                    q_op__.apply(mt, ichunk, atom_index_t::local(i), 0, *phi_atomic_tmp, wf::band_range(0, nawf),
                                 bp_grad_coeffs, beta_phi_atomic);
                    q_op__.apply(mt, ichunk, atom_index_t::local(i), 0, *phi_atomic_tmp, wf::band_range(0, nawf),
                                 bp_coeffs, grad_beta_phi_atomic);
                }

                /* compute O' = d O / d r_{alpha} */
                /* from O = <phi | S | phi > we get
                 * O' = <phi' | S | phi> + <phi | S' |phi> + <phi | S | phi'> */

                if (ctx_.cfg().hubbard().hubbard_subspace_method() == "full_orthogonalization") {
                    /* <phi | S' | phi> */
                    wf::inner(ctx_.spla_context(), mt, wf::spin_range(0), phi_atomic, wf::band_range(0, nawf),
                              *phi_atomic_tmp, wf::band_range(0, nawf), ovlp, 0, 0);
                    /* add <phi' | S | phi> and <phi | S | phi'> */
                    /* |phi' > = d|phi> / d r_{alpha} which is non-zero for a current displacement atom only */
                    auto& type = ctx_.unit_cell().atom(ja).type();
                    for (int xi = 0; xi < type.indexb_wfs().size(); xi++) {
                        int i = num_ps_atomic_wf.second[ja] + xi;
                        for (int j = 0; j < nawf; j++) {
                            ovlp(i, j) += grad_phi_atomic_s_phi_atomic[x](i, j);
                            ovlp(j, i) += std::conj(grad_phi_atomic_s_phi_atomic[x](i, j));
                        }
                    }
                    compute_inv_sqrt_O_deriv(ovlp, *evec_O, eval_O, nawf);
                }

                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    /* compute <phi_atomic | dS/dr_j | psi_{ik}> */
                    la::dmatrix<std::complex<double>> phi_atomic_ds_psi(nawf, kp__.num_occupied_bands(ispn));
                    wf::inner(ctx_.spla_context(), mt, wf::spin_range(ispn), *phi_atomic_tmp, wf::band_range(0, nawf),
                              kp__.spinor_wave_functions(), wf::band_range(0, kp__.num_occupied_bands(ispn)),
                              phi_atomic_ds_psi, 0, 0);

                    /* add <d phi / d r_{alpha} | S | psi_{jk}> which is diagonal (in atom index) */
                    for (int ibnd = 0; ibnd < kp__.num_occupied_bands(ispn); ibnd++) {
                        for (int xi = 0; xi < ctx_.unit_cell().atom(ja).type().indexb_wfs().size(); xi++) {
                            int i = num_ps_atomic_wf.second[ja] + xi;
                            phi_atomic_ds_psi(i, ibnd) += grad_phi_atomic_s_psi[x][ispn](i, ibnd);
                        }
                    }

                    /* build the full d <phi_hub | S | psi_ik> / d r_{alpha} matrix */
                    la::dmatrix<std::complex<double>> phi_hub_s_psi_deriv(num_hubbard_wf.first,
                                                                          kp__.num_occupied_bands(ispn));

                    build_phi_hub_s_psi_deriv(ctx_, kp__.num_occupied_bands(ispn), nawf, ovlp, *inv_sqrt_O,
                                              phi_atomic_s_psi[ispn], phi_atomic_ds_psi, num_ps_atomic_wf.second,
                                              num_hubbard_wf.second, phi_hub_s_psi_deriv);

                    /* multiply by eigen-energy */
                    for (int ibnd = 0; ibnd < kp__.num_occupied_bands(ispn); ibnd++) {
                        for (int j = 0; j < num_hubbard_wf.first; j++) {
                            phi_hub_s_psi_deriv(j, ibnd) *= kp__.band_occupancy(ibnd, ispn);
                        }
                    }

                    if (ctx_.processing_unit() == device_t::GPU) {
                        phi_hub_s_psi_deriv.allocate(get_memory_pool(memory_t::device)).copy_to(memory_t::device);
                        psi_s_phi_hub[ispn].allocate(get_memory_pool(memory_t::device)).copy_to(memory_t::device);
                    }

                    /* update the density matrix derivative */
                    update_density_matrix_deriv(la, mt, num_hubbard_wf.first, kp__.num_occupied_bands(ispn), &alpha,
                                                phi_hub_s_psi_deriv, psi_s_phi_hub[ispn],
                                                dn__.at(mt, 0, 0, ispn, x, ja), dn__.ld());
                } // ispn
            }     // i
        }         // x
    }             // ichunk

    if (ctx_.processing_unit() == device_t::GPU) {
        dn__.copy_to(memory_t::host);
        dn__.deallocate(memory_t::device);
    }
}

void // TODO: rename to strain_deriv, rename previous func. to displacement_deriv
Hubbard::compute_occupancies_stress_derivatives(K_point<double>& kp__, Q_operator<double>& q_op__,
                                                mdarray<std::complex<double>, 4>& dn__)
{
    PROFILE("sirius::Hubbard::compute_occupancies_stress_derivatives");

    auto la = la::lib_t::none;
    auto mt = memory_t::none;
    switch (ctx_.processing_unit()) {
        case device_t::CPU: {
            la = la::lib_t::blas;
            mt = memory_t::host;
            break;
        }
        case device_t::GPU: {
            la = la::lib_t::gpublas;
            mt = memory_t::device;
            break;
        }
    }
    auto alpha = std::complex<double>(kp__.weight(), 0.0);

    Beta_projectors_strain_deriv<double> bp_strain_deriv(ctx_, kp__.gkvec());
    auto bp_strain_gen = bp_strain_deriv.make_generator();
    /* initialize the beta projectors and derivatives */
    auto bp_strain_coeffs = bp_strain_gen.prepare();

    auto bp_gen    = kp__.beta_projectors().make_generator();
    auto bp_coeffs = bp_gen.prepare();

    const int lmax  = ctx_.unit_cell().lmax();
    const int lmmax = sf::lmmax(lmax);

    mdarray<double, 2> rlm_g({lmmax, kp__.num_gkvec_loc()});
    mdarray<double, 3> rlm_dg({lmmax, 3, kp__.num_gkvec_loc()});

    /* array of real spherical harmonics and derivatives for each G-vector */
    #pragma omp parallel for schedule(static)
    for (int igkloc = 0; igkloc < kp__.num_gkvec_loc(); igkloc++) {
        /* gvs = {r, theta, phi} */
        auto gvc = kp__.gkvec().gkvec_cart(gvec_index_t::local(igkloc));
        auto rtp = r3::spherical_coordinates(gvc);

        sf::spherical_harmonics(lmax, rtp[1], rtp[2], &rlm_g(0, igkloc));
        mdarray<double, 2> rlm_dg_tmp({lmmax, 3}, &rlm_dg(0, 0, igkloc));
        sf::dRlm_dr(lmax, gvc, rlm_dg_tmp);
    }

    /* atomic wave functions  */
    auto& phi_atomic   = kp__.atomic_wave_functions();
    auto& phi_atomic_S = kp__.atomic_wave_functions_S();
    auto& phi_hub_S    = kp__.hubbard_wave_functions_S();

    auto num_ps_atomic_wf = ctx_.unit_cell().num_ps_atomic_wf();
    auto num_hubbard_wf   = ctx_.unit_cell().num_hubbard_wf();

    /* number of atomic wave-functions */
    int nawf = num_ps_atomic_wf.first;
    /* number of Hubbard wave-functions */
    int nhwf = num_hubbard_wf.first;

    RTE_ASSERT(nawf == phi_atomic.num_wf().get());
    RTE_ASSERT(nawf == phi_atomic_S.num_wf().get());
    RTE_ASSERT(nhwf == phi_hub_S.num_wf().get());

    auto dphi_atomic   = wave_function_factory(ctx_, kp__, phi_atomic.num_wf(), wf::num_mag_dims(0), false);
    auto s_dphi_atomic = wave_function_factory(ctx_, kp__, phi_atomic.num_wf(), wf::num_mag_dims(0), false);
    auto ds_phi_atomic = wave_function_factory(ctx_, kp__, phi_atomic.num_wf(), wf::num_mag_dims(0), false);

    /* compute < psi_{ik} | S | phi_hub > */
    /* this is used in the final expression for the occupation matrix derivative */
    std::array<la::dmatrix<std::complex<double>>, 2> psi_s_phi_hub;
    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        psi_s_phi_hub[ispn] = la::dmatrix<std::complex<double>>(kp__.num_occupied_bands(ispn), nhwf);
        wf::inner(ctx_.spla_context(), mt, wf::spin_range(ispn), kp__.spinor_wave_functions(),
                  wf::band_range(0, kp__.num_occupied_bands(ispn)), phi_hub_S, wf::band_range(0, nhwf),
                  psi_s_phi_hub[ispn], 0, 0);
    }

    /* compute overlap matrix */
    la::dmatrix<std::complex<double>> ovlp;
    std::unique_ptr<la::dmatrix<std::complex<double>>> inv_sqrt_O;
    std::unique_ptr<la::dmatrix<std::complex<double>>> evec_O;
    std::vector<double> eval_O;
    if (ctx_.cfg().hubbard().hubbard_subspace_method() == "full_orthogonalization") {
        ovlp = la::dmatrix<std::complex<double>>(nawf, nawf);
        wf::inner(ctx_.spla_context(), mt, wf::spin_range(0), phi_atomic, wf::band_range(0, nawf), phi_atomic_S,
                  wf::band_range(0, nawf), ovlp, 0, 0);

        /* a tuple of O^{-1/2}, U, \lambda */
        auto result = inverse_sqrt(ovlp, nawf);
        inv_sqrt_O  = std::move(std::get<0>(result));
        evec_O      = std::move(std::get<1>(result));
        eval_O      = std::get<2>(result);
    }

    /* compute <phi_atomic | S | psi_{ik} > */
    std::array<la::dmatrix<std::complex<double>>, 2> phi_atomic_s_psi;
    if (ctx_.cfg().hubbard().hubbard_subspace_method() == "full_orthogonalization") {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            phi_atomic_s_psi[ispn] = la::dmatrix<std::complex<double>>(nawf, kp__.num_occupied_bands(ispn));
            /* compute < phi_atomic | S | psi_{ik} > for all atoms */
            wf::inner(ctx_.spla_context(), mt, wf::spin_range(ispn), phi_atomic_S, wf::band_range(0, nawf),
                      kp__.spinor_wave_functions(), wf::band_range(0, kp__.num_occupied_bands(ispn)),
                      phi_atomic_s_psi[ispn], 0, 0);
        }
    }

    auto mg1 = dphi_atomic->memory_guard(mt);
    auto mg2 = s_dphi_atomic->memory_guard(mt);
    auto mg3 = ds_phi_atomic->memory_guard(mt);
    for (int nu = 0; nu < 3; nu++) {
        for (int mu = 0; mu < 3; mu++) {
            /* compute |d phi_atomic / d epsilon_{mu, nu} > */
            wavefunctions_strain_deriv(ctx_, kp__, *dphi_atomic, rlm_g, rlm_dg, nu, mu);

            if (is_device_memory(mt)) {
                dphi_atomic->copy_to(mt);
            }

            /* compute S |d phi_atomic / d epsilon_{mu, nu} > */
            sirius::apply_S_operator<double, std::complex<double>>(mt, wf::spin_range(0), wf::band_range(0, nawf),
                                                                   bp_gen, bp_coeffs, *dphi_atomic, &q_op__,
                                                                   *s_dphi_atomic);

            ds_phi_atomic->zero(mt, wf::spin_index(0), wf::band_range(0, nawf));
            sirius::apply_S_operator_strain_deriv(mt, 3 * nu + mu, bp_gen, bp_coeffs, bp_strain_gen, bp_strain_coeffs,
                                                  phi_atomic, q_op__, *ds_phi_atomic);

            if (ctx_.cfg().hubbard().hubbard_subspace_method() == "full_orthogonalization") {
                /* compute <phi_atomic | ds/d epsilon | phi_atomic> */
                wf::inner(ctx_.spla_context(), mt, wf::spin_range(0), phi_atomic, wf::band_range(0, nawf),
                          *ds_phi_atomic, wf::band_range(0, nawf), ovlp, 0, 0);

                /* compute <d phi_atomic / d epsilon | S | phi_atomic > */
                la::dmatrix<std::complex<double>> tmp(nawf, nawf);
                wf::inner(ctx_.spla_context(), mt, wf::spin_range(0), *s_dphi_atomic, wf::band_range(0, nawf),
                          phi_atomic, wf::band_range(0, nawf), tmp, 0, 0);

                for (int i = 0; i < nawf; i++) {
                    for (int j = 0; j < nawf; j++) {
                        ovlp(i, j) += tmp(i, j) + std::conj(tmp(j, i));
                    }
                }
                compute_inv_sqrt_O_deriv(ovlp, *evec_O, eval_O, nawf);
            }

            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                la::dmatrix<std::complex<double>> dphi_atomic_s_psi(nawf, kp__.num_occupied_bands(ispn));
                wf::inner(ctx_.spla_context(), mt, wf::spin_range(ispn), *s_dphi_atomic, wf::band_range(0, nawf),
                          kp__.spinor_wave_functions(), wf::band_range(0, kp__.num_occupied_bands(ispn)),
                          dphi_atomic_s_psi, 0, 0);

                la::dmatrix<std::complex<double>> phi_atomic_ds_psi(nawf, kp__.num_occupied_bands(ispn));
                wf::inner(ctx_.spla_context(), mt, wf::spin_range(ispn), *ds_phi_atomic, wf::band_range(0, nawf),
                          kp__.spinor_wave_functions(), wf::band_range(0, kp__.num_occupied_bands(ispn)),
                          phi_atomic_ds_psi, 0, 0);

                for (int ibnd = 0; ibnd < kp__.num_occupied_bands(ispn); ibnd++) {
                    for (int j = 0; j < nawf; j++) {
                        phi_atomic_ds_psi(j, ibnd) += dphi_atomic_s_psi(j, ibnd);
                    }
                }

                /* build the full d <phi_hub | S | psi_ik> / d epsilon_{mu,nu} matrix */
                la::dmatrix<std::complex<double>> phi_hub_s_psi_deriv(num_hubbard_wf.first,
                                                                      kp__.num_occupied_bands(ispn));
                build_phi_hub_s_psi_deriv(ctx_, kp__.num_occupied_bands(ispn), nawf, ovlp, *inv_sqrt_O,
                                          phi_atomic_s_psi[ispn], phi_atomic_ds_psi, num_ps_atomic_wf.second,
                                          num_hubbard_wf.second, phi_hub_s_psi_deriv);

                /* multiply by eigen-energy */
                for (int ibnd = 0; ibnd < kp__.num_occupied_bands(ispn); ibnd++) {
                    for (int j = 0; j < num_hubbard_wf.first; j++) {
                        phi_hub_s_psi_deriv(j, ibnd) *= kp__.band_occupancy(ibnd, ispn);
                    }
                }

                if (ctx_.processing_unit() == device_t::GPU) {
                    psi_s_phi_hub[ispn].allocate(get_memory_pool(memory_t::device)).copy_to(memory_t::device);
                    phi_hub_s_psi_deriv.allocate(get_memory_pool(memory_t::device)).copy_to(memory_t::device);
                }

                /* update the density matrix derivative */
                update_density_matrix_deriv(la, mt, num_hubbard_wf.first, kp__.num_occupied_bands(ispn), &alpha,
                                            phi_hub_s_psi_deriv, psi_s_phi_hub[ispn],
                                            dn__.at(mt, 0, 0, ispn, 3 * nu + mu), dn__.ld());
            }
        }
    }

    if (ctx_.processing_unit() == device_t::GPU) {
        dn__.copy_to(memory_t::host);
    }
}

} // namespace sirius
