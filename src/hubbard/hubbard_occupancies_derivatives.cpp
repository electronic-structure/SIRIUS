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

#include "hubbard.hpp"
#include "SDDK/wf_inner.hpp"
#include "SDDK/wf_trans.hpp"
#include "symmetry/crystal_symmetry.hpp"
#include "linalg/inverse_sqrt.hpp"
#include "geometry/wavefunction_strain_deriv.hpp"

namespace sirius {

static void
update_density_matrix_deriv(linalg_t la__, memory_t mt__, int nwfh__, int nbnd__, std::complex<double>* alpha__,
    dmatrix<std::complex<double>> const& phi_hub_s_psi_deriv__, dmatrix<std::complex<double>> const& psi_s_phi_hub__,
    std::complex<double>* dn__, int ld__)
{
    linalg(la__).gemm('N', 'N', nwfh__, nwfh__, nbnd__, alpha__,
                      phi_hub_s_psi_deriv__.at(mt__, 0, 0), phi_hub_s_psi_deriv__.ld(),
                      psi_s_phi_hub__.at(mt__, 0, 0), psi_s_phi_hub__.ld(),
                      &linalg_const<double_complex>::one(), dn__, ld__);

    linalg(la__).gemm('C', 'C', nwfh__, nwfh__, nbnd__, alpha__,
                      psi_s_phi_hub__.at(mt__, 0, 0), psi_s_phi_hub__.ld(),
                      phi_hub_s_psi_deriv__.at(mt__, 0, 0), phi_hub_s_psi_deriv__.ld(),
                      &linalg_const<double_complex>::one(), dn__, ld__);
}

static void
build_phi_hub_s_psi_deriv(Simulation_context const& ctx__, int nbnd__, int nawf__,
        dmatrix<std::complex<double>> const& ovlp__, dmatrix<std::complex<double>> const& inv_sqrt_O__,
        dmatrix<std::complex<double>> const& phi_atomic_s_psi__,
        dmatrix<std::complex<double>> const& phi_atomic_ds_psi__,
        std::vector<int> const& atomic_wf_offset__, std::vector<int> const& hubbard_wf_offset__,
        dmatrix<std::complex<double>>& phi_hub_s_psi_deriv__)
{
    phi_hub_s_psi_deriv__.zero();

    for (int ia = 0; ia < ctx__.unit_cell().num_atoms(); ia++) {
        auto& type = ctx__.unit_cell().atom(ia).type();

        if (type.hubbard_correction()) {
            /* loop over Hubbard orbitals of the atom */
            for (int idxrf = 0; idxrf < type.indexr_hub().size(); idxrf++) {
                auto& hd = type.lo_descriptor_hub(idxrf);
                int l = type.indexr_hub().am(idxrf).l();
                int mmax = 2 * l + 1;

                int idxr_wf = hd.idx_wf();
                int offset_in_wf = atomic_wf_offset__[ia] + type.indexb_wfs().offset(idxr_wf);
                int offset_in_hwf = hubbard_wf_offset__[ia] + type.indexb_hub().offset(idxrf);

                if (ctx__.cfg().hubbard().full_orthogonalization()) {
                    /* compute \sum_{m} d/d r_{alpha} O^{-1/2}_{m,i} <phi_atomic_{m} | S | psi_{jk} > */
                    linalg(linalg_t::blas).gemm('C', 'N', mmax, nbnd__, nawf__,
                        &linalg_const<double_complex>::one(),
                        ovlp__.at(memory_t::host, 0, offset_in_wf), ovlp__.ld(),
                        phi_atomic_s_psi__.at(memory_t::host), phi_atomic_s_psi__.ld(),
                        &linalg_const<double_complex>::one(),
                        phi_hub_s_psi_deriv__.at(memory_t::host, offset_in_hwf, 0), phi_hub_s_psi_deriv__.ld());

                    linalg(linalg_t::blas).gemm('C', 'N', mmax, nbnd__, nawf__,
                        &linalg_const<double_complex>::one(),
                        inv_sqrt_O__.at(memory_t::host, 0, offset_in_wf), inv_sqrt_O__.ld(),
                        phi_atomic_ds_psi__.at(memory_t::host), phi_atomic_ds_psi__.ld(),
                        &linalg_const<double_complex>::one(),
                        phi_hub_s_psi_deriv__.at(memory_t::host, offset_in_hwf, 0), phi_hub_s_psi_deriv__.ld());
                } else {
                    /* just copy part of the matrix elements in the order in which
                     * Hubbard wfs are defined */
                    for (int ibnd = 0; ibnd < nbnd__; ibnd++) {
                        for (int m = 0; m < mmax; m++) {
                             phi_hub_s_psi_deriv__(offset_in_hwf + m, ibnd) = phi_atomic_ds_psi__(offset_in_wf + m, ibnd);
                        }
                    }
                }
            } // idxrf
        }
    } // ia
}

void
Hubbard::compute_occupancies_derivatives(K_point<double>& kp__, Q_operator<double>& q_op__,
                                         sddk::mdarray<std::complex<double>, 5>& dn__)
{
    PROFILE("sirius::Hubbard::compute_occupancies_derivatives");

    auto la = linalg_t::none;
    auto mt = memory_t::none;
    switch (ctx_.processing_unit()) {
        case device_t::CPU: {
            la = linalg_t::blas;
            mt = memory_t::host;
            break;
        }
        case device_t::GPU: {
            la = linalg_t::gpublas;
            mt = memory_t::device;
            break;
        }
    }
    auto alpha = double_complex(kp__.weight(), 0.0);

    // TODO: check if we have a norm conserving pseudo potential;
    // TODO: distrribute (MPI) all matrices in the basis of atomic orbitals
    // only derivatives of the atomic wave functions are needed.
    auto& phi_atomic   = kp__.atomic_wave_functions();
    auto& phi_atomic_S = kp__.atomic_wave_functions_S();
    auto& phi_hub_S    = kp__.hubbard_wave_functions_S();

    auto num_ps_atomic_wf = ctx_.unit_cell().num_ps_atomic_wf();
    auto num_hubbard_wf   = ctx_.unit_cell().num_hubbard_wf();

    int nawf = phi_atomic.num_wf();

    if (ctx_.processing_unit() == device_t::GPU) {
        dn__.allocate(memory_t::device);
        phi_atomic.prepare(spin_range(0), true, &ctx_.mem_pool(memory_t::device));
        phi_atomic_S.prepare(spin_range(0), true, &ctx_.mem_pool(memory_t::device));
        phi_hub_S.prepare(spin_range(0), true, &ctx_.mem_pool(memory_t::device));
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            kp__.spinor_wave_functions().prepare(spin_range(ispn), true, &ctx_.mem_pool(memory_t::device));
        }
    }

    /* compute overlap matrix */
    sddk::dmatrix<std::complex<double>> ovlp;
    std::unique_ptr<sddk::dmatrix<std::complex<double>>> inv_sqrt_O;
    std::unique_ptr<sddk::dmatrix<std::complex<double>>> evec_O;
    std::vector<double> eval_O;
    if (ctx_.cfg().hubbard().full_orthogonalization()) {
        ovlp = sddk::dmatrix<std::complex<double>>(phi_atomic.num_wf(), phi_atomic.num_wf());
        sddk::inner(ctx_.spla_context(), spin_range(0), phi_atomic, 0, phi_atomic.num_wf(),
                phi_atomic_S, 0, phi_atomic_S.num_wf(), ovlp, 0, 0);

        /* a tuple of O^{-1/2}, U, \lambda */
        auto result = inverse_sqrt(ovlp, phi_atomic.num_wf());
        inv_sqrt_O = std::move(std::get<0>(result));
        evec_O = std::move(std::get<1>(result));
        eval_O = std::get<2>(result);
    }

    /* compute < psi_{ik} | S | phi_hub > */
    /* this is used in the final expression for the occupation matrix derivative */
    std::array<sddk::dmatrix<double_complex>, 2> psi_s_phi_hub;
    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        psi_s_phi_hub[ispn] = sddk::dmatrix<double_complex>(kp__.num_occupied_bands(ispn), phi_hub_S.num_wf());
        if (ctx_.processing_unit() == device_t::GPU) {
            psi_s_phi_hub[ispn].allocate(ctx_.mem_pool(memory_t::device));
        }
        inner(ctx_.spla_context(), spin_range(ispn), kp__.spinor_wave_functions(), 0, kp__.num_occupied_bands(ispn),
              phi_hub_S, 0, phi_hub_S.num_wf(), psi_s_phi_hub[ispn], 0, 0);
    }

    /* temporary storage */
    Wave_functions<double> phi_atomic_tmp(kp__.gkvec_partition(), phi_atomic.num_wf(), ctx_.preferred_memory_t(), 1);

    Wave_functions<double> s_phi_atomic_tmp(kp__.gkvec_partition(), phi_atomic.num_wf(), ctx_.preferred_memory_t(), 1);

    /* compute < d phi_atomic / d r_{j} | S | psi_{ik} > and < d phi_atomic / d r_{j} | S | phi_atomic > */
    std::array<std::array<sddk::dmatrix<double_complex>, 2>, 3> grad_phi_atomic_s_psi;
    std::array<sddk::dmatrix<double_complex>, 3> grad_phi_atomic_s_phi_atomic;

    for (int x = 0; x < 3; x++) {
        /* compute |phi_atomic_tmp> = |d phi_atomic / d r_{alpha} > for all atoms */
        for (int i = 0; i < phi_atomic.num_wf(); i++) {
            for (int igloc = 0; igloc < kp__.num_gkvec_loc(); igloc++) {
                /* G+k vector in Cartesian coordinates */
                auto gk = kp__.gkvec().template gkvec_cart<index_domain_t::local>(igloc);
                /* gradient of phi_atomic */
                phi_atomic_tmp.pw_coeffs(0).prime(igloc, i) = std::complex<double>(0.0, -gk[x]) *
                    phi_atomic.pw_coeffs(0).prime(igloc, i);
            }
        }
        if (ctx_.processing_unit() == device_t::GPU) {
            phi_atomic_tmp.prepare(spin_range(0), true, &ctx_.mem_pool(memory_t::device));
            s_phi_atomic_tmp.prepare(spin_range(0), false, &ctx_.mem_pool(memory_t::device));
        }
        /* apply S to |d phi_atomic / d r_{alpha} > */
        apply_S_operator<std::complex<double>>(ctx_.processing_unit(), spin_range(0), 0, phi_atomic_tmp.num_wf(),
                kp__.beta_projectors(), phi_atomic_tmp, &q_op__, s_phi_atomic_tmp);

        /* compute < d phi_atomic / d r_{alpha} | S | phi_atomic >
         * used to compute derivative of the inverse square root of the overlap matrix */
        if (ctx_.cfg().hubbard().full_orthogonalization()) {
            grad_phi_atomic_s_phi_atomic[x] = sddk::dmatrix<double_complex>(s_phi_atomic_tmp.num_wf(), phi_atomic.num_wf());
            inner(ctx_.spla_context(), spin_range(0), s_phi_atomic_tmp, 0, s_phi_atomic_tmp.num_wf(),
                  phi_atomic, 0, phi_atomic.num_wf(), grad_phi_atomic_s_phi_atomic[x], 0, 0);
        }

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            /* allocate space */
            grad_phi_atomic_s_psi[x][ispn] =
                sddk::dmatrix<double_complex>(s_phi_atomic_tmp.num_wf(), kp__.num_occupied_bands(ispn));
            /* compute < d phi_atomic / d r_{j} | S | psi_{ik} > for all atoms */
            inner(ctx_.spla_context(), spin_range(ispn), s_phi_atomic_tmp, 0, s_phi_atomic_tmp.num_wf(),
                kp__.spinor_wave_functions(), 0, kp__.num_occupied_bands(ispn), grad_phi_atomic_s_psi[x][ispn], 0, 0);
        }
    }

    /* compute <phi_atomic | S | psi_{ik} > */
    std::array<sddk::dmatrix<double_complex>, 2> phi_atomic_s_psi;
    if (ctx_.cfg().hubbard().full_orthogonalization()) {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            phi_atomic_s_psi[ispn] = sddk::dmatrix<double_complex>(phi_atomic_S.num_wf(), kp__.num_occupied_bands(ispn));
            /* compute < phi_atomic | S | psi_{ik} > for all atoms */
            inner(ctx_.spla_context(), spin_range(ispn), phi_atomic_S, 0, phi_atomic_S.num_wf(),
                kp__.spinor_wave_functions(), 0, kp__.num_occupied_bands(ispn), phi_atomic_s_psi[ispn], 0, 0);
        }
    }

    Beta_projectors_gradient<double> bp_grad(ctx_, kp__.gkvec(), kp__.igk_loc(), kp__.beta_projectors());
    bp_grad.prepare();

    dn__.zero(mt);

    for (int ichunk = 0; ichunk < kp__.beta_projectors().num_chunks(); ichunk++) {
        kp__.beta_projectors().generate(ichunk);

        /* <beta | phi_atomic> for this chunk */
        auto beta_phi_atomic = kp__.beta_projectors().inner<double_complex>(ichunk, phi_atomic, 0, 0, phi_atomic.num_wf());

        for (int x = 0; x < 3; x++) {
            bp_grad.generate(ichunk, x);

            /* <dbeta | phi> for this chunk */
            auto grad_beta_phi_atomic = bp_grad.inner<double_complex>(ichunk, phi_atomic, 0, 0, phi_atomic.num_wf());

            for (int i = 0; i < kp__.beta_projectors().chunk(ichunk).num_atoms_; i++) {
                /* this is a displacement atom */
                int ja = kp__.beta_projectors().chunk(ichunk).desc_(static_cast<int>(beta_desc_idx::ia), i);

                /* build |phi_atomic_tmp> = | d S / d r_{j} | phi_atomic > */
                /* it consists of two contributions:
                 *   | beta >        Q < d beta / dr | phi_atomic > and
                 *   | d beta / dr > Q < beta        | phi_atomic > */
                phi_atomic_tmp.zero(ctx_.processing_unit());
                if (ctx_.unit_cell().atom(ja).type().augment()) {
                    q_op__.apply(ichunk, i, 0, phi_atomic_tmp, 0, phi_atomic_tmp.num_wf(), bp_grad, beta_phi_atomic);
                    q_op__.apply(ichunk, i, 0, phi_atomic_tmp, 0, phi_atomic_tmp.num_wf(), kp__.beta_projectors(),
                            grad_beta_phi_atomic);
                }

                /* compute O' = d O / d r_{alpha} */
                /* from O = <phi | S | phi > we get
                 * O' = <phi' | S | phi> + <phi | S' |phi> + <phi | S | phi'> */

                if (ctx_.cfg().hubbard().full_orthogonalization()) {
                    /* <phi | S' | phi> */
                    sddk::inner(ctx_.spla_context(), spin_range(0), phi_atomic, 0, phi_atomic.num_wf(),
                        phi_atomic_tmp, 0, phi_atomic_tmp.num_wf(), ovlp, 0, 0);
                    /* add <phi' | S | phi> and <phi | S | phi'> */
                    /* |phi' > = d|phi> / d r_{alpha} which is non-zero for a current displacement atom only */
                    auto& type = ctx_.unit_cell().atom(ja).type();
                    for (int xi = 0; xi < type.indexb_wfs().size(); xi++) {
                        int i = num_ps_atomic_wf.second[ja] + xi;
                        for (int j = 0; j < phi_atomic.num_wf(); j++) {
                            ovlp(i, j) += grad_phi_atomic_s_phi_atomic[x](i, j);
                            ovlp(j, i) += std::conj(grad_phi_atomic_s_phi_atomic[x](i, j));
                        }
                    }

                    /* compute \tilde O' = U^{H}O'U */
                    unitary_similarity_transform(1, ovlp, *evec_O, phi_atomic.num_wf());

                    for (int i = 0; i < phi_atomic.num_wf(); i++) {
                        for (int j = 0; j < phi_atomic.num_wf(); j++) {
                            ovlp(j, i) /= -(eval_O[i] * std::sqrt(eval_O[j]) + eval_O[j] * std::sqrt(eval_O[i]));
                        }
                    }
                    /* compute d/dr O^{-1/2} */
                    unitary_similarity_transform(0, ovlp, *evec_O, phi_atomic.num_wf());
                }

                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    /* compute <phi_atomic | dS/dr_j | psi_{ik}> */
                    sddk::dmatrix<double_complex> phi_atomic_ds_psi(phi_atomic_tmp.num_wf(), kp__.num_occupied_bands(ispn));
                    inner(ctx_.spla_context(), spin_range(ispn), phi_atomic_tmp, 0, phi_atomic_tmp.num_wf(),
                        kp__.spinor_wave_functions(), 0, kp__.num_occupied_bands(ispn), phi_atomic_ds_psi, 0, 0);

                    /* add <d phi / d r_{alpha} | S | psi_{jk}> which is diagonal (in atom index) */
                    for (int ibnd = 0; ibnd < kp__.num_occupied_bands(ispn); ibnd++) {
                        for (int xi = 0; xi < ctx_.unit_cell().atom(ja).type().indexb_wfs().size(); xi++) {
                            int i = num_ps_atomic_wf.second[ja] + xi;
                            phi_atomic_ds_psi(i, ibnd) += grad_phi_atomic_s_psi[x][ispn](i, ibnd);
                        }
                    }

                    /* build the full d <phi_hub | S | psi_ik> / d r_{alpha} matrix */
                    sddk::dmatrix<double_complex> phi_hub_s_psi_deriv(num_hubbard_wf.first, kp__.num_occupied_bands(ispn));

                    build_phi_hub_s_psi_deriv(ctx_, kp__.num_occupied_bands(ispn), nawf, ovlp, *inv_sqrt_O,
                            phi_atomic_s_psi[ispn], phi_atomic_ds_psi, num_ps_atomic_wf.second, num_hubbard_wf.second,
                            phi_hub_s_psi_deriv);

                    //phi_hub_s_psi_deriv.zero();

                    //for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
                    //    auto& type = ctx_.unit_cell().atom(ia).type();

                    //    if (type.hubbard_correction()) {
                    //        /* loop over Hubbard orbitals of the atom */
                    //        for (int idxrf = 0; idxrf < type.indexr_hub().size(); idxrf++) {
                    //            auto& hd = type.lo_descriptor_hub(idxrf);
                    //            int l = type.indexr_hub().am(idxrf).l();
                    //            int mmax = 2 * l + 1;

                    //            int idxr_wf = hd.idx_wf();
                    //            int offset_in_wf = num_ps_atomic_wf.second[ia] + type.indexb_wfs().offset(idxr_wf);
                    //            int offset_in_hwf = num_hubbard_wf.second[ia] + type.indexb_hub().offset(idxrf);

                    //            if (ctx_.cfg().hubbard().full_orthogonalization()) {
                    //                /* compute \sum_{m} d/d r_{alpha} O^{-1/2}_{m,i} <phi_atomic_{m} | S | psi_{jk} > */
                    //                linalg(linalg_t::blas).gemm('C', 'N', mmax, kp__.num_occupied_bands(ispn), phi_atomic.num_wf(),
                    //                    &linalg_const<double_complex>::one(),
                    //                    ovlp.at(memory_t::host, 0, offset_in_wf), ovlp.ld(),
                    //                    phi_atomic_s_psi[ispn].at(memory_t::host), phi_atomic_s_psi[ispn].ld(),
                    //                    &linalg_const<double_complex>::one(),
                    //                    phi_hub_s_psi_deriv.at(memory_t::host, offset_in_hwf, 0), phi_hub_s_psi_deriv.ld());

                    //                linalg(linalg_t::blas).gemm('C', 'N', mmax, kp__.num_occupied_bands(ispn), phi_atomic.num_wf(),
                    //                    &linalg_const<double_complex>::one(),
                    //                    inv_sqrt_O->at(memory_t::host, 0, offset_in_wf), inv_sqrt_O->ld(),
                    //                    phi_atomic_ds_psi.at(memory_t::host), phi_atomic_ds_psi.ld(),
                    //                    &linalg_const<double_complex>::one(),
                    //                    phi_hub_s_psi_deriv.at(memory_t::host, offset_in_hwf, 0), phi_hub_s_psi_deriv.ld());
                    //            } else {
                    //                /* just copy part of the matrix elements in the order in which
                    //                 * Hubbard wfs are defined */
                    //                for (int ibnd = 0; ibnd < kp__.num_occupied_bands(ispn); ibnd++) {
                    //                    for (int m = 0; m < mmax; m++) {
                    //                         phi_hub_s_psi_deriv(offset_in_hwf + m, ibnd) = phi_atomic_ds_psi(offset_in_wf + m, ibnd);
                    //                    }
                    //                }
                    //            }
                    //        } // idxrf
                    //    }
                    //} // ia

                    /* multiply by eigen-energy */
                    for (int ibnd = 0; ibnd < kp__.num_occupied_bands(ispn); ibnd++) {
                        for (int j = 0; j < num_hubbard_wf.first; j++) {
                            phi_hub_s_psi_deriv(j, ibnd) *= kp__.band_occupancy(ibnd, ispn);
                        }
                    }

                    if (ctx_.processing_unit() == device_t::GPU) {
                        phi_hub_s_psi_deriv.allocate(ctx_.mem_pool(memory_t::device)).copy_to(memory_t::device);
                    }

                    /* update the density matrix derivative */
                    update_density_matrix_deriv(la, mt, num_hubbard_wf.first, kp__.num_occupied_bands(ispn),
                            &alpha, phi_hub_s_psi_deriv, psi_s_phi_hub[ispn], dn__.at(mt, 0, 0, ispn, x, ja),
                            dn__.ld());

                    //linalg(la).gemm('N', 'N', num_hubbard_wf.first, num_hubbard_wf.first,
                    //                kp__.num_occupied_bands(ispn), &alpha,
                    //                phi_hub_s_psi_deriv.at(mt, 0, 0), phi_hub_s_psi_deriv.ld(),
                    //                psi_s_phi_hub[ispn].at(mt, 0, 0), psi_s_phi_hub[ispn].ld(),
                    //                &linalg_const<double_complex>::one(),
                    //                dn__.at(mt, 0, 0, ispn, x, ja), dn__.ld());

                    //linalg(la).gemm('C', 'C', num_hubbard_wf.first, num_hubbard_wf.first,
                    //                kp__.num_occupied_bands(ispn), &alpha,
                    //                psi_s_phi_hub[ispn].at(mt, 0, 0), psi_s_phi_hub[ispn].ld(),
                    //                phi_hub_s_psi_deriv.at(mt, 0, 0), phi_hub_s_psi_deriv.ld(),
                    //                &linalg_const<double_complex>::one(),
                    //                dn__.at(mt, 0, 0, ispn, x, ja), dn__.ld());
                } // ispn
            } //i
        } // x
    } // ichunk

    if (ctx_.processing_unit() == device_t::GPU) {
        dn__.copy_to(memory_t::host);
        dn__.deallocate(memory_t::device);
        phi_atomic.dismiss(spin_range(0), false);
        phi_atomic_S.dismiss(spin_range(0), false);
        phi_hub_S.dismiss(spin_range(0), false);
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            kp__.spinor_wave_functions().dismiss(spin_range(ispn), false);
        }
    }
}

void
Hubbard::compute_occupancies_stress_derivatives(K_point<double>& kp__, Q_operator<double>& q_op__,
                                                mdarray<std::complex<double>, 4>& dn__)
{
    PROFILE("sirius::Hubbard::compute_occupancies_stress_derivatives");

    auto la = linalg_t::none;
    auto mt = memory_t::none;
    switch (ctx_.processing_unit()) {
        case device_t::CPU: {
            la = linalg_t::blas;
            mt = memory_t::host;
            break;
        }
        case device_t::GPU: {
            la = linalg_t::gpublas;
            mt = memory_t::device;
            break;
        }
    }
    auto alpha = double_complex(kp__.weight(), 0.0);

    Beta_projectors_strain_deriv<double> bp_strain_deriv(ctx_, kp__.gkvec(), kp__.igk_loc());
    /* initialize the beta projectors and derivatives */
    bp_strain_deriv.prepare();

    const int lmax  = ctx_.unit_cell().lmax();
    const int lmmax = utils::lmmax(lmax);

    sddk::mdarray<double, 2> rlm_g(lmmax, kp__.num_gkvec_loc());
    sddk::mdarray<double, 3> rlm_dg(lmmax, 3, kp__.num_gkvec_loc());

    /* array of real spherical harmonics and derivatives for each G-vector */
    #pragma omp parallel for schedule(static)
    for (int igkloc = 0; igkloc < kp__.num_gkvec_loc(); igkloc++) {
        /* gvs = {r, theta, phi} */
        auto gvc = kp__.gkvec().gkvec_cart<index_domain_t::local>(igkloc);
        auto rtp = SHT::spherical_coordinates(gvc);

        sf::spherical_harmonics(lmax, rtp[1], rtp[2], &rlm_g(0, igkloc));
        sddk::mdarray<double, 2> rlm_dg_tmp(&rlm_dg(0, 0, igkloc), lmmax, 3);
        sf::dRlm_dr(lmax, gvc, rlm_dg_tmp);
    }

    /* atomic wave functions  */
    auto& phi_atomic    = kp__.atomic_wave_functions();
    auto& phi_atomic_S  = kp__.atomic_wave_functions_S();
    auto& phi_hub_S     = kp__.hubbard_wave_functions_S();

    /* total number of atomic wave-functions */
    int nawf = phi_atomic.num_wf();

    auto num_ps_atomic_wf = ctx_.unit_cell().num_ps_atomic_wf();
    auto num_hubbard_wf   = ctx_.unit_cell().num_hubbard_wf();

    Wave_functions<double> dphi_atomic(kp__.gkvec_partition(), nawf, ctx_.preferred_memory_t(), 1);
    Wave_functions<double> s_dphi_atomic(kp__.gkvec_partition(), nawf, ctx_.preferred_memory_t(), 1);
    Wave_functions<double> ds_phi_atomic(kp__.gkvec_partition(), nawf, ctx_.preferred_memory_t(), 1);

    if (ctx_.processing_unit() == device_t::GPU) {
        dn__.allocate(memory_t::device);
        dn__.copy_to(memory_t::device);
        phi_atomic.prepare(spin_range(0), true, &ctx_.mem_pool(memory_t::device));
        phi_atomic_S.prepare(spin_range(0), true, &ctx_.mem_pool(memory_t::device));
        phi_hub_S.prepare(spin_range(0), true, &ctx_.mem_pool(memory_t::device));
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            kp__.spinor_wave_functions().prepare(spin_range(ispn), true, &ctx_.mem_pool(memory_t::device));
        }
        s_dphi_atomic.prepare(spin_range(0), false, &ctx_.mem_pool(memory_t::device));
        ds_phi_atomic.prepare(spin_range(0), false, &ctx_.mem_pool(memory_t::device));
    }

    /* compute < psi_{ik} | S | phi_hub > */
    /* this is used in the final expression for the occupation matrix derivative */
    std::array<sddk::dmatrix<double_complex>, 2> psi_s_phi_hub;
    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        psi_s_phi_hub[ispn] = sddk::dmatrix<double_complex>(kp__.num_occupied_bands(ispn), phi_hub_S.num_wf());
        if (ctx_.processing_unit() == device_t::GPU) {
            psi_s_phi_hub[ispn].allocate(ctx_.mem_pool(memory_t::device));
        }
        inner(ctx_.spla_context(), spin_range(ispn), kp__.spinor_wave_functions(), 0, kp__.num_occupied_bands(ispn),
              phi_hub_S, 0, phi_hub_S.num_wf(), psi_s_phi_hub[ispn], 0, 0);
    }

    /* compute overlap matrix */
    sddk::dmatrix<std::complex<double>> ovlp;
    std::unique_ptr<sddk::dmatrix<std::complex<double>>> inv_sqrt_O;
    std::unique_ptr<sddk::dmatrix<std::complex<double>>> evec_O;
    std::vector<double> eval_O;
    if (ctx_.cfg().hubbard().full_orthogonalization()) {
        ovlp = sddk::dmatrix<std::complex<double>>(nawf, nawf);
        sddk::inner(ctx_.spla_context(), spin_range(0), phi_atomic, 0, nawf, phi_atomic_S, 0, nawf, ovlp, 0, 0);

        /* a tuple of O^{-1/2}, U, \lambda */
        auto result = inverse_sqrt(ovlp, nawf);
        inv_sqrt_O = std::move(std::get<0>(result));
        evec_O = std::move(std::get<1>(result));
        eval_O = std::get<2>(result);
    }

    /* compute <phi_atomic | S | psi_{ik} > */
    std::array<sddk::dmatrix<double_complex>, 2> phi_atomic_s_psi;
    if (ctx_.cfg().hubbard().full_orthogonalization()) {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            phi_atomic_s_psi[ispn] = sddk::dmatrix<double_complex>(nawf, kp__.num_occupied_bands(ispn));
            /* compute < phi_atomic | S | psi_{ik} > for all atoms */
            inner(ctx_.spla_context(), spin_range(ispn), phi_atomic_S, 0, nawf, kp__.spinor_wave_functions(),
                    0, kp__.num_occupied_bands(ispn), phi_atomic_s_psi[ispn], 0, 0);
        }
    }

    for (int nu = 0; nu < 3; nu++) {
        for (int mu = 0; mu < 3; mu++) {
            /* compute |d phi_atomic / d epsilon_{mu, nu} > */
            wavefunctions_strain_deriv(ctx_, kp__, dphi_atomic, rlm_g, rlm_dg, nu, mu);
            if (ctx_.processing_unit() == device_t::GPU) {
                dphi_atomic.prepare(spin_range(0), true, &ctx_.mem_pool(memory_t::device));
            }
            /* compute S |d phi_atomic / d epsilon_{mu, nu} > */
            sirius::apply_S_operator<double_complex>(ctx_.processing_unit(), spin_range(0), 0,
                    nawf, kp__.beta_projectors(), dphi_atomic, &q_op__, s_dphi_atomic);

            if (ctx_.processing_unit() == device_t::GPU) {
                ds_phi_atomic.prepare(spin_range(0), false, &ctx_.mem_pool(memory_t::device));
            }
            ds_phi_atomic.zero(ctx_.processing_unit());
            sirius::apply_S_operator_strain_deriv(ctx_.processing_unit(), 3 * nu + mu, kp__.beta_projectors(),
                         bp_strain_deriv, phi_atomic, q_op__, ds_phi_atomic);

            if (ctx_.cfg().hubbard().full_orthogonalization()) {
                /* compute <phi_atomic | ds/d epsilon | phi_atomic> */
                sddk::inner(ctx_.spla_context(), spin_range(0), phi_atomic, 0, nawf, ds_phi_atomic, 0, nawf, ovlp, 0, 0);

                /* compute <d phi_atomic / d epsilon | S | phi_atomic > */
                sddk::dmatrix<std::complex<double>> tmp(nawf, nawf);
                sddk::inner(ctx_.spla_context(), spin_range(0), s_dphi_atomic, 0, nawf, phi_atomic, 0, nawf, tmp, 0, 0);

                for (int i = 0; i < nawf; i++) {
                    for (int j = 0; j < nawf; j++) {
                        ovlp(i, j) += tmp(i, j) + std::conj(tmp(j, i));
                    }
                }
                /* compute \tilde O' = U^{H}O'U */
                unitary_similarity_transform(1, ovlp, *evec_O, nawf);

                for (int i = 0; i < nawf; i++) {
                    for (int j = 0; j < nawf; j++) {
                        ovlp(j, i) /= -(eval_O[i] * std::sqrt(eval_O[j]) + eval_O[j] * std::sqrt(eval_O[i]));
                    }
                }
                /* compute d/dr O^{-1/2} */
                unitary_similarity_transform(0, ovlp, *evec_O, nawf);
            }

            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                sddk::dmatrix<double_complex> dphi_atomic_s_psi(nawf, kp__.num_occupied_bands(ispn));
                inner(ctx_.spla_context(), spin_range(ispn), s_dphi_atomic, 0, nawf,
                    kp__.spinor_wave_functions(), 0, kp__.num_occupied_bands(ispn), dphi_atomic_s_psi, 0, 0);

                sddk::dmatrix<double_complex> phi_atomic_ds_psi(nawf, kp__.num_occupied_bands(ispn));
                inner(ctx_.spla_context(), spin_range(ispn), ds_phi_atomic, 0, nawf,
                    kp__.spinor_wave_functions(), 0, kp__.num_occupied_bands(ispn), phi_atomic_ds_psi, 0, 0);

                for (int ibnd = 0; ibnd < kp__.num_occupied_bands(ispn); ibnd++) {
                    for (int j = 0; j < nawf; j++) {
                        phi_atomic_ds_psi(j, ibnd) += dphi_atomic_s_psi(j, ibnd);
                    }
                }

                /* build the full d <phi_hub | S | psi_ik> / d epsilon_{mu,nu} matrix */
                sddk::dmatrix<double_complex> phi_hub_s_psi_deriv(num_hubbard_wf.first, kp__.num_occupied_bands(ispn));
                build_phi_hub_s_psi_deriv(ctx_, kp__.num_occupied_bands(ispn), nawf, ovlp, *inv_sqrt_O,
                        phi_atomic_s_psi[ispn], phi_atomic_ds_psi, num_ps_atomic_wf.second, num_hubbard_wf.second,
                        phi_hub_s_psi_deriv);
                //phi_hub_s_psi_deriv.zero();

                //for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
                //    auto& type = ctx_.unit_cell().atom(ia).type();

                //    if (type.hubbard_correction()) {
                //        /* loop over Hubbard orbitals of the atom */
                //        for (int idxrf = 0; idxrf < type.indexr_hub().size(); idxrf++) {
                //            auto& hd = type.lo_descriptor_hub(idxrf);
                //            int l = type.indexr_hub().am(idxrf).l();
                //            int mmax = 2 * l + 1;

                //            int idxr_wf = hd.idx_wf();
                //            int offset_in_wf = num_ps_atomic_wf.second[ia] + type.indexb_wfs().offset(idxr_wf);
                //            int offset_in_hwf = num_hubbard_wf.second[ia] + type.indexb_hub().offset(idxrf);

                //            if (ctx_.cfg().hubbard().full_orthogonalization()) {
                //                /* compute \sum_{m} d/d r_{alpha} O^{-1/2}_{m,i} <phi_atomic_{m} | S | psi_{jk} > */
                //                linalg(linalg_t::blas).gemm('C', 'N', mmax, kp__.num_occupied_bands(ispn), nawf,
                //                    &linalg_const<double_complex>::one(),
                //                    ovlp.at(memory_t::host, 0, offset_in_wf), ovlp.ld(),
                //                    phi_atomic_s_psi[ispn].at(memory_t::host), phi_atomic_s_psi[ispn].ld(),
                //                    &linalg_const<double_complex>::one(),
                //                    phi_hub_s_psi_deriv.at(memory_t::host, offset_in_hwf, 0), phi_hub_s_psi_deriv.ld());

                //                linalg(linalg_t::blas).gemm('C', 'N', mmax, kp__.num_occupied_bands(ispn), nawf,
                //                    &linalg_const<double_complex>::one(),
                //                    inv_sqrt_O->at(memory_t::host, 0, offset_in_wf), inv_sqrt_O->ld(),
                //                    phi_atomic_ds_psi.at(memory_t::host), phi_atomic_ds_psi.ld(),
                //                    &linalg_const<double_complex>::one(),
                //                    phi_hub_s_psi_deriv.at(memory_t::host, offset_in_hwf, 0), phi_hub_s_psi_deriv.ld());
                //            } else {
                //                /* just copy part of the matrix elements in the order in which
                //                 * Hubbard wfs are defined */
                //                for (int ibnd = 0; ibnd < kp__.num_occupied_bands(ispn); ibnd++) {
                //                    for (int m = 0; m < mmax; m++) {
                //                         phi_hub_s_psi_deriv(offset_in_hwf + m, ibnd) = phi_atomic_ds_psi(offset_in_wf + m, ibnd);
                //                    }
                //                }
                //            }
                //        } // idxrf
                //    }
                //} // ia

                /* multiply by eigen-energy */
                for (int ibnd = 0; ibnd < kp__.num_occupied_bands(ispn); ibnd++) {
                    for (int j = 0; j < num_hubbard_wf.first; j++) {
                        phi_hub_s_psi_deriv(j, ibnd) *= kp__.band_occupancy(ibnd, ispn);
                    }
                }

                if (ctx_.processing_unit() == device_t::GPU) {
                    phi_hub_s_psi_deriv.allocate(ctx_.mem_pool(memory_t::device)).copy_to(memory_t::device);
                }

                /* update the density matrix derivative */
                update_density_matrix_deriv(la, mt, num_hubbard_wf.first, kp__.num_occupied_bands(ispn),
                        &alpha, phi_hub_s_psi_deriv, psi_s_phi_hub[ispn], dn__.at(mt, 0, 0, ispn, 3 * nu + mu),
                        dn__.ld());


                //linalg(la).gemm('N', 'N', num_hubbard_wf.first, num_hubbard_wf.first,
                //                kp__.num_occupied_bands(ispn), &alpha,
                //                phi_hub_s_psi_deriv.at(mt, 0, 0), phi_hub_s_psi_deriv.ld(),
                //                psi_s_phi_hub[ispn].at(mt, 0, 0), psi_s_phi_hub[ispn].ld(),
                //                &linalg_const<double_complex>::one(),
                //                dn__.at(mt, 0, 0, ispn, 3 * nu + mu), dn__.ld());

                //linalg(la).gemm('C', 'C', num_hubbard_wf.first, num_hubbard_wf.first,
                //                kp__.num_occupied_bands(ispn), &alpha,
                //                psi_s_phi_hub[ispn].at(mt, 0, 0), psi_s_phi_hub[ispn].ld(),
                //                phi_hub_s_psi_deriv.at(mt, 0, 0), phi_hub_s_psi_deriv.ld(),
                //                &linalg_const<double_complex>::one(),
                //                dn__.at(mt, 0, 0, ispn, 3 * nu + mu), dn__.ld());
            }
        }
    }

    if (ctx_.processing_unit() == device_t::GPU) {
        dn__.copy_to(memory_t::host);
        dn__.deallocate(memory_t::device);
        phi_atomic.dismiss(spin_range(0), false);
        //phi_atomic_S.dismiss(spin_range(0), false);
        phi_hub_S.dismiss(spin_range(0), false);
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            kp__.spinor_wave_functions().dismiss(spin_range(ispn), false);
        }
    }
}

} // namespace sirius
