// Copyright (c) 2013-2018 Mathieu Taillefumier, Anton Kozhevnikov, Thomas Schulthess
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
 */

// compute the forces for the simple LDA+U method not the fully
// rotationally invariant one. It can not be used for LDA+U+SO either

// this code is based on two papers :
//
// - PRB 84, 161102(R) (2011) : https://journals.aps.org/prb/abstract/10.1103/PhysRevB.84.161102
// - PRB 102, 235159 (2020) : https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.235159
//
// note that this code only apply for the colinear case
#include "hubbard.hpp"

namespace sirius {

void
Hubbard::compute_occupancies_derivatives(K_point& kp, Q_operator& q_op, sddk::mdarray<double_complex, 6>& dn__)
{
    PROFILE("sirius::Hubbard::compute_occupancies_derivatives");

    dn__.zero();
    // check if we have a norm conserving pseudo potential only. Only
    // derivatives of the hubbard wave functions are needed.
    auto& phi = kp.hubbard_wave_functions_without_S();

    Beta_projectors_gradient bp_grad(ctx_, kp.gkvec(), kp.igk_loc(), kp.beta_projectors());
    bp_grad.prepare();

    bool augment = ctx_.unit_cell().augment();

    /*
      Compute the derivatives of the occupancies in two cases.

      - the atom is pp norm conserving or

      - the atom is ppus (in that case the derivative the beta projectors
      compared to the atomic displacements gives a non zero contribution)
    */

    /* temporary wave functions */
    Wave_functions dphi(kp.gkvec_partition(), this->number_of_hubbard_orbitals(), ctx_.preferred_memory_t(), 1);
    /* temporary wave functions */
    Wave_functions phitmp(kp.gkvec_partition(), this->number_of_hubbard_orbitals(), ctx_.preferred_memory_t(), 1);

    int HowManyBands = kp.num_occupied_bands(0);
    if (ctx_.num_spins() == 2) {
        HowManyBands = std::max(kp.num_occupied_bands(1), kp.num_occupied_bands(0));
    }
    /*
      d_phitmp contains the derivatives of the hubbard wave functions
      corresponding to the displacement r^I_a.
    */
    dmatrix<double_complex> dphi_s_psi(HowManyBands, this->number_of_hubbard_orbitals() * ctx_.num_spins());
    dmatrix<double_complex> phi_s_psi(HowManyBands, this->number_of_hubbard_orbitals() * ctx_.num_spins());
    mdarray<double_complex, 5> dn_tmp(max_number_of_orbitals_per_atom(),
                                      max_number_of_orbitals_per_atom(),
                                      ctx_.num_spins(),
                                      ctx_.unit_cell().num_atoms(),
                                      3);

    if (ctx_.processing_unit() == device_t::GPU) {
        dn_tmp.allocate(memory_t::device);

        /* allocation of the overlap matrices on GPU */
        phi_s_psi.allocate(memory_t::device);
        dphi_s_psi.allocate(memory_t::device);

        /* wave functions */
        phitmp.allocate(spin_range(0), memory_t::device);
        phi.allocate(spin_range(0), memory_t::device);
        dphi.allocate(spin_range(0), memory_t::device);
        phi.copy_to(spin_range(0), memory_t::device, 0, this->number_of_hubbard_orbitals());
        kp.spinor_wave_functions().allocate(spin_range(ctx_.num_spins()), memory_t::device);
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            kp.spinor_wave_functions().copy_to(spin_range(ispn), memory_t::device, 0, kp.num_occupied_bands(ispn));
        }
    }
    phi_s_psi.zero(memory_t::host);
    phi_s_psi.zero(memory_t::device);

    sirius::apply_S_operator<double_complex>(ctx_.processing_unit(), spin_range(0), 0, this->number_of_hubbard_orbitals(),
                             kp.beta_projectors(), phi, &q_op, dphi);

    /* compute <phi^I_m| S | psi_{nk}> */
    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        inner(ctx_.spla_context(), spin_range(ispn), kp.spinor_wave_functions(), 0, kp.num_occupied_bands(ispn), dphi, 0,
              this->number_of_hubbard_orbitals(), phi_s_psi, 0, ispn * this->number_of_hubbard_orbitals());
    }


    for (int atom_id = 0; atom_id < ctx_.unit_cell().num_atoms(); atom_id++) { // loop over the atom displacement.
        dn_tmp.zero(memory_t::host);
        dn_tmp.zero(memory_t::device);
        for (int dir = 0; dir < 3; dir++) {
            // reset dphi
            dphi.pw_coeffs(0).prime().zero(memory_t::host);
            dphi.pw_coeffs(0).prime().zero(memory_t::device);

            // compute S|d\phi>. Will be zero if the atom has no hubbard
            // correction
            if (ctx_.unit_cell().atom(atom_id).type().hubbard_correction()) {
                phitmp.pw_coeffs(0).prime().zero(memory_t::host);
                phitmp.pw_coeffs(0).prime().zero(memory_t::device);

                // atom atom_id has hubbard correction so we need to compute the
                // derivatives of the hubbard orbitals associated to the atom
                // atom_id, the derivatives of the others hubbard orbitals been
                // zero compared to the displacement of atom atom_id

                // compute the derivative of |phi> corresponding to the
                // atom atom_id
                const int lmax_at = 2 * ctx_.unit_cell().atom(atom_id).type().lo_descriptor_hub(0).l + 1;

                // compute the derivatives of the hubbard wave functions
                // |phi_m^J> (J = atom_id) compared to a displacement of atom J.

                kp.compute_gradient_wave_functions(phi, this->offset_[atom_id], lmax_at, phitmp, this->offset_[atom_id], dir);

                // phitmp contains the derivatives of the hubbard wave functions of atom I compared to a displacement of atom I.
                if (ctx_.processing_unit() == device_t::GPU) {
                    phitmp.copy_to(spin_range(0), memory_t::device, 0, this->number_of_hubbard_orbitals());
                }

                /* for norm conserving pp, it is enough to have the derivatives
                 * of |phi^J_m> (J = atom_id) and the next line will be
                 * ignored */
                sirius::apply_S_operator<double_complex>(ctx_.processing_unit(), spin_range(0), 0,
                                                         this->number_of_hubbard_orbitals(), kp.beta_projectors(),
                                                         phitmp, &q_op, dphi);
            }

            // compute (d S/ d R_K) |phi> and add to dphi. It is Eq.18 of Ref PRB 102, 235159 (2020)
            if (!ctx_.full_potential() && augment) {
                // it is equal to
                // \sum Q^I_ij <d \beta^I_i|phi> |\beta^I_j> + < \beta^I_i|phi> |d\beta^I_j>
                for (int chunk__ = 0; chunk__ < kp.beta_projectors().num_chunks(); chunk__++) {
                    for (int i = 0; i < kp.beta_projectors().chunk(chunk__).num_atoms_; i++) {
                        // need to find the right atom in the chunks.
                        if (kp.beta_projectors().chunk(chunk__).desc_(static_cast<int>(beta_desc_idx::ia), i) == atom_id) {
                            kp.beta_projectors().generate(chunk__);
                            bp_grad.generate(chunk__, dir);

                            // compute Q_ij <\beta_i|\phi> |d \beta_j> and add it to d\phi
                            {
                                /* <beta | phi> for this chunk */
                                auto beta_phi = kp.beta_projectors().inner<double_complex>(chunk__, phi, 0, 0,
                                                                                           this->number_of_hubbard_orbitals());
                                q_op.apply(chunk__, i, 0, dphi, 0, this->number_of_hubbard_orbitals(), bp_grad, beta_phi);
                            }

                            // compute Q_ij <d \beta_i|\phi> |\beta_j> and add it to d\phi
                            {
                                /* <dbeta | phi> for this chunk */
                                auto dbeta_phi = bp_grad.inner<double_complex>(chunk__, phi, 0, 0,
                                                                                this->number_of_hubbard_orbitals());

                                /* apply Q operator (diagonal in spin) */
                                /* Effectively compute Q_ij <d beta_i| phi> |beta_j> and add it dphi */
                                q_op.apply(chunk__, i, 0, dphi, 0, this->number_of_hubbard_orbitals(),
                                           kp.beta_projectors(), dbeta_phi);
                            }
                        }
                    }
                }
            }

            // d\phi now contains (\partial_{R^K} S) |\phi> + S d|\phi>. The
            // second term is only present if the atom has hubbard correction.
            compute_occupancies(kp,
                                phi_s_psi,
                                dphi,
                                dn_tmp,
                                dir);
        } // direction x, y, z

        /* use a memcpy here */
        std::memcpy(dn__.at(memory_t::host, 0, 0, 0, 0, 0, atom_id), dn_tmp.at(memory_t::host),
                    sizeof(double_complex) * dn_tmp.size());
    } // atom_id

    if (ctx_.processing_unit() == device_t::GPU) {
        phi.deallocate(spin_range(0), memory_t::device);
        kp.spinor_wave_functions().deallocate(spin_range(ctx_.num_spins()), memory_t::device);
    }
}

void
Hubbard::compute_occupancies_stress_derivatives(K_point& kp__, Q_operator& q_op__, mdarray<double_complex, 5>& dn__)
{
    PROFILE("sirius::Hubbard::compute_occupancies_stress_derivatives");

    /* this is the original atomic wave functions without the operator S applied */
    auto& phi = kp__.atomic_wave_functions_hub();

    /*
      dphi contains this

      \f[
      \left| d \phi\right> = \partial_{\mu\nu} (S \left| \phi \right>)
      ]

    */
    Wave_functions dphi(kp__.gkvec_partition(), phi.num_wf(), ctx_.preferred_memory_t(), 1);

    Wave_functions phitmp(kp__.gkvec_partition(), phi.num_wf(), ctx_.preferred_memory_t(), 1);

    Beta_projectors_strain_deriv bp_strain_deriv(ctx_, kp__.gkvec(), kp__.igk_loc());

    /* maximum number of occupied bands */
    int nbnd = (ctx_.num_mag_dims() == 1) ? std::max(kp__.num_occupied_bands(0), kp__.num_occupied_bands(1))
                                          : kp__.num_occupied_bands(0);

    bool augment = ctx_.unit_cell().augment();

    const int lmax  = ctx_.unit_cell().lmax();
    const int lmmax = utils::lmmax(lmax);

    sddk::mdarray<double, 2> rlm_g(lmmax, kp__.num_gkvec_loc());
    sddk::mdarray<double, 3> rlm_dg(lmmax, 3, kp__.num_gkvec_loc());

    /* overlap between psi_nk and phi_ik <psi|S|phi> */
    dmatrix<double_complex> psi_s_phi(nbnd, phi.num_wf() * ctx_.num_spinors());

    /* initialize the beta projectors and derivatives */
    bp_strain_deriv.prepare();

    auto sr = spin_range(ctx_.num_spins() == 2 ? 2 : 0);
    kp__.spinor_wave_functions().prepare(sr, true);
    kp__.atomic_wave_functions_S_hub().prepare(sr, true);
    kp__.atomic_wave_functions_hub().prepare(sr, true);


    if (ctx_.processing_unit() == device_t::GPU) {
        psi_s_phi.allocate(memory_t::device);

        dphi.allocate(spin_range(0), memory_t::device);

        phitmp.allocate(spin_range(0), memory_t::device);
    }

    /* compute <psi_nk | S | phi_mk>
     * treat non-magnetic, collinear and non-collinear cases;
     * in collinear case psi_s_phi contains two blocks for up- and dn- spin channels */
    for (int is = 0; is < ctx_.num_spinors(); is++) {
        auto sr = ctx_.num_mag_dims() == 3 ? spin_range(2) : spin_range(is);

        inner(ctx_.spla_context(), sr, kp__.spinor_wave_functions(), 0, kp__.num_occupied_bands(is),
            kp__.atomic_wave_functions_S_hub(), 0, kp__.atomic_wave_functions_S_hub().num_wf(),
            psi_s_phi, 0, is * kp__.atomic_wave_functions_S_hub().num_wf());
    }

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

    /* the expressions below assume that the hubbard wave-functions are actually non-orthogonalised,
       spin-independent atomic orbiatals */

    for (int nu = 0; nu < 3; nu++) {
        for (int mu = 0; mu < 3; mu++) {

            // compute the derivatives of all hubbard wave functions
            // |phi_m^J> compared to the strain

            /* Compute derivatives of Hubbard atomic functions w.r.t. lattice strain:
             * - the functions indexing is compatible with K_point::generate_atomic_wave_functions();
             * - only non-orthogonal atomic orbitals are handled
             * - S operator is not applied
             */
            wavefunctions_strain_deriv(kp__, phitmp, rlm_g, rlm_dg, nu, mu);

            ///* ths phitmp functions are non-magnetic at the moment, so copy spin-up channel to spin dn;
            // * this is done in order to have a general case of spin-dependent Hubbard orbitals */
            //if (ctx_.num_spins() == 2) {
            //    phitmp.copy_from(device_t::CPU, phitmp.num_wf(), phitmp, 0, 0, 1, (ctx_.num_mag_dims() == 3) ? phitmp.num_wf() : 0);
            //}

            /* Just to keep in mind:
             *
             * - kp.hubbard_wave_functions() are the hubbard functions with S applied
             * - kp.atomic_wave_functions_S_hub() are the non-orthogonal (initial) atomic functions with S applied
             * - kp.atomic_wave_functions_hub() are the non-orthogonal (initial) atomic functions
             * - both sets are spin-dependent, even if the orbitals have no spin label
             * - atom_type.indexr_hub() is the new index of Hubbard radial functions
             * - atom_type.indexb_hub() is the new index of Hubbard orbitals (radial function times a spherical harmonic)
             * - wavefunctions_strain_deriv() and K_point::generate_atomic_wave_functions() work with the new indices
             */

            if (ctx_.processing_unit() == device_t::GPU) {
               phitmp.copy_to(spin_range(0), memory_t::device, 0, this->number_of_hubbard_orbitals());
            }

            /* dphi is the strain derivative of the hubbard orbitals (with S applied). Derivation imply this

               d(S phi) = (dS) phi + S (d\phi) = (d\phi) - \sum_{ij} Q_{ij} |beta_i><beta_j| d(phi) -
               \sum_{ij} Q_{ij} |d beta_i><beta_j|phi> - \sum_{ij} Q_{ij} |beta_i><d beta_j|phi>

               dphi contains the full expression phitmp contains (d phi).
            */

            dphi.copy_from(ctx_.processing_unit(), this->number_of_hubbard_orbitals(), phitmp, 0, 0, 0, 0);

            // dphi = |phitmp> - \sum_{ij} Q_{ij} | beta_i><beta_j|phitmp> + |d beta_i><beta_j|phi> + |beta_i><d beta_j|phi>
            // dphi = (1 - \sum_{ij} Q_{ij} | beta_i><beta_j|) |phitmp> - (sum_{ij} Q_{ij} |d beta_i><beta_j|  + |beta_i><d beta_j|) | phi>

            if (!ctx_.full_potential() && augment) {
               for (int i = 0; i < kp__.beta_projectors().num_chunks(); i++) {
                   /* generate beta-projectors for a block of atoms */
                   kp__.beta_projectors().generate(i);
                   /* generate derived beta-projectors for a block of atoms */
                   bp_strain_deriv.generate(i, 3 * nu + mu);

                   /* basically apply the second term of the S projector to
                    * phitmp = (strain derivatives of the original atomic
                    * orbitals )*/
                   {
                       /* < beta | dphi > */
                       auto beta_dphi = kp__.beta_projectors().inner<double_complex>(i,
                                                                                     phitmp, // contains the strain derivative of the hubbard orbitals.
                                                                                     0, 0,
                                                                                     this->number_of_hubbard_orbitals());
                       /* apply Q operator (diagonal in spin) */

                       // compute Q_{ij}<beta_i | dphi> |beta_j> and add it to dphi
                       q_op__.apply(i, 0,
                                    dphi, 0,
                                    this->number_of_hubbard_orbitals(),
                                    kp__.beta_projectors(),
                                    beta_dphi);
                   }

                   {
                       /* <d(beta) | phi> */
                       auto dbeta_phi = bp_strain_deriv.inner<double_complex>(i,
                                                                              phi,
                                                                              0, 0,
                                                                              this->number_of_hubbard_orbitals());
                       /* apply Q operator (diagonal in spin) */

                       // compute <d (beta) | phi> |beta> and add it to dphi
                       q_op__.apply(i, 0,
                                    dphi, 0,
                                    this->number_of_hubbard_orbitals(),
                                    kp__.beta_projectors(),
                                    dbeta_phi);
                   }

                   {
                       // <beta|phi> |d beta>
                       auto beta_phi = kp__.beta_projectors().inner<double_complex>(i,
                                                                                    phi,
                                                                                    0,
                                                                                    0,
                                                                                    this->number_of_hubbard_orbitals());
                       /* apply Q operator (diagonal in spin) */
                       q_op__.apply(i, 0, dphi, 0, this->number_of_hubbard_orbitals(), bp_strain_deriv, beta_phi);
                   }
               }
            }
            compute_occupancies(kp__, psi_s_phi, dphi, dn__, 3 * nu + mu);
        }
    }

    kp__.spinor_wave_functions().dismiss(sr, false);
    kp__.atomic_wave_functions_S_hub().dismiss(sr, false);
    kp__.atomic_wave_functions_hub().dismiss(sr, false);
}

void
Hubbard::wavefunctions_strain_deriv(K_point& kp__, Wave_functions& dphi__, sddk::mdarray<double, 2> const& rlm_g__,
                                    sddk::mdarray<double, 3> const& rlm_dg__, int nu__, int mu__)
{
    PROFILE("sirius::Hubbard::wavefunctions_strain_deriv");
    #pragma omp parallel for schedule(static)
    for (int igkloc = 0; igkloc < kp__.num_gkvec_loc(); igkloc++) {
        /* global index of G+k vector */
        const int igk = kp__.idxgk(igkloc);
        /* Cartesian coordinats of G-vector */
        auto gvc = kp__.gkvec().gkvec_cart<index_domain_t::local>(igkloc);
        /* vs = {r, theta, phi} */
        auto gvs = SHT::spherical_coordinates(gvc);

        std::vector<sddk::mdarray<double, 1>> ri_values(unit_cell_.num_atom_types());
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            ri_values[iat] = ctx_.hubbard_wf_ri().values(iat, gvs[0]);
        }

        std::vector<sddk::mdarray<double, 1>> ridjl_values(unit_cell_.num_atom_types());
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            ridjl_values[iat] = ctx_.hubbard_wf_djl().values(iat, gvs[0]);
        }

        const double p = (mu__ == nu__) ? 0.5 : 0.0;

        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto& atom_type = ctx_.unit_cell().atom(ia).type();
            if (atom_type.hubbard_correction()) {
                // TODO: this can be optimized, check k_point::generate_atomic_wavefunctions()
                auto phase        = twopi * dot(kp__.gkvec().gkvec(igk), unit_cell_.atom(ia).position());
                auto phase_factor = std::exp(double_complex(0.0, phase));
                int offset = this->offset_[ia];
                for (int xi = 0; xi < atom_type.indexb_hub().size(); xi++) {
                    /*  orbital quantum  number of this atomic orbital */
                    int l = atom_type.indexb_hub().l(xi);
                    /*  composite l,m index */
                    int lm = atom_type.indexb_hub().lm(xi);
                    /* index of the radial function */
                    int idxrf = atom_type.indexb_hub().idxrf(xi);

                    auto z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(unit_cell_.omega());

                    /* case |g+k| = 0 */
                    if (gvs[0] < 1e-10) {
                        if (l == 0) {
                            auto d1 = ri_values[atom_type.id()][idxrf] * p * y00;

                            dphi__.pw_coeffs(0).prime(igkloc, offset + xi) = -z * d1 * phase_factor;
                        } else {
                            dphi__.pw_coeffs(0).prime(igkloc, offset + xi) = 0.0;
                        }
                    } else {
                        auto d1 = ri_values[atom_type.id()][idxrf] * (gvc[mu__] * rlm_dg__(lm, nu__, igkloc) +
                                                                      p * rlm_g__(lm, igkloc));
                        auto d2 = ridjl_values[atom_type.id()][idxrf] * rlm_g__(lm, igkloc) * gvc[mu__] * gvc[nu__] / gvs[0];

                        dphi__.pw_coeffs(0).prime(igkloc, offset + xi) = -z * (d1 + d2) * std::conj(phase_factor);
                    }
                } // xi
            }
        }
    }
}

void
Hubbard::compute_occupancies(K_point& kp__, dmatrix<double_complex>& psi_s_phi__, Wave_functions& dphi__,
    sddk::mdarray<double_complex, 5>& dn__, const int index__)
{
    PROFILE("sirius::Hubbard::compute_occupancies");

    /* maximum number of occupied bands */
    int nbnd = (ctx_.num_mag_dims() == 1) ? std::max(kp__.num_occupied_bands(0), kp__.num_occupied_bands(1))
                                          : kp__.num_occupied_bands(0);

    /* overlap between psi_{nk} and dphi */
    dmatrix<double_complex> psi_s_dphi(nbnd, this->number_of_hubbard_orbitals(), ctx_.mem_pool(memory_t::host));

    dmatrix<double_complex> dm(this->number_of_hubbard_orbitals(), this->number_of_hubbard_orbitals(),
                               ctx_.mem_pool(memory_t::host));

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
            dm.allocate(ctx_.mem_pool(memory_t::device));
            psi_s_dphi.allocate(ctx_.mem_pool(memory_t::device));
            break;
        }
    }

    //if (ctx_.processing_unit() == device_t::GPU) {
    //    phi_s_psi__.copy_to(memory_t::host);
    //}

    /* include the occupancy directly in dphi_s_psi */

    //for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
    //    for (int n_orb = 0; n_orb < this->number_of_hubbard_orbitals(); n_orb++) {
    //        for (int ibnd = 0; ibnd < kp__.num_occupied_bands(ispn); ibnd++) {
    //            dphi_s_psi__(ibnd, ispn * this->number_of_hubbard_orbitals() + n_orb) *=
    //                kp__.band_occupancy(ibnd, ispn);
    //        }
    //    }
    //}
    //if (ctx_.processing_unit() == device_t::GPU) {
    //    dphi_s_psi__.copy_to(memory_t::device);
    //}

    auto alpha = double_complex(kp__.weight(), 0.0);
    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        /* compute <psi_{ik}^{sigma}|S|dphi> */
        /* dphi don't have a spin index; they are derived from scalar atomic orbitals */
        inner(ctx_.spla_context(), spin_range(ispn), kp__.spinor_wave_functions(), 0, kp__.num_occupied_bands(ispn),
              dphi__, 0, this->number_of_hubbard_orbitals(), psi_s_dphi, 0, 0);

        for (int i = 0; i < this->number_of_hubbard_orbitals(); i++) {
            for (int ibnd = 0; ibnd < kp__.num_occupied_bands(ispn); ibnd++) {
                psi_s_dphi(ibnd, i) *= kp__.band_occupancy(ibnd, ispn);
            }
        }

        if (ctx_.processing_unit() == device_t::GPU) {
            psi_s_dphi.copy_to(memory_t::device);
        }

        linalg(la).gemm('C', 'N',
                         this->number_of_hubbard_orbitals(),
                         this->number_of_hubbard_orbitals(),
                         kp__.num_occupied_bands(ispn),
                         &alpha,
                         psi_s_dphi.at(mt), psi_s_dphi.ld(),
                         psi_s_phi__.at(mt, 0, ispn * this->number_of_hubbard_orbitals()), psi_s_phi__.ld(),
                         &linalg_const<double_complex>::zero(),
                         dm.at(mt), dm.ld());

        linalg(la).gemm('C', 'N',
                         this->number_of_hubbard_orbitals(),
                         this->number_of_hubbard_orbitals(),
                         kp__.num_occupied_bands(ispn),
                         &alpha,
                         psi_s_phi__.at(mt, 0, ispn * this->number_of_hubbard_orbitals()), psi_s_phi__.ld(),
                         psi_s_dphi.at(mt), psi_s_dphi.ld(),
                         &linalg_const<double_complex>::one(),
                         dm.at(mt), dm.ld());

        switch (ctx_.processing_unit()) {
            case device_t::CPU: {
                break;
            }
            case device_t::GPU: {
                dm.copy_to(memory_t::host);
                break;
            }
        }

        #pragma omp parallel for schedule(static)
        for (int ia1 = 0; ia1 < ctx_.unit_cell().num_atoms(); ++ia1) {
            const auto& atom = ctx_.unit_cell().atom(ia1);
            if (atom.type().hubbard_correction()) {
                const int lmax_at = 2 * atom.type().lo_descriptor_hub(0).l + 1;
                for (int m2 = 0; m2 < lmax_at; m2++) {
                    for (int m1 = 0; m1 < lmax_at; m1++) {
                        dn__(m1, m2, ispn, ia1, index__) = dm(this->offset_[ia1] + m1, this->offset_[ia1] + m2);
                    }
                }
            }
        }
    }
}

}
