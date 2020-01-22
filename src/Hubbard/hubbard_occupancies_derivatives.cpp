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

// compute the forces for the simplex LDA+U method not the fully
// rotationally invariant one. It can not be used for LDA+U+SO either

// It is based on this reference : PRB 84, 161102(R) (2011)

// gradient of beta projectors. Needed for the computations of the forces

#include "hubbard.hpp"

namespace sirius {

void
Hubbard::compute_occupancies_derivatives(K_point& kp,
                                         Q_operator& q_op, // overlap operator
                                         mdarray<double_complex, 6>& dn__)  // Atom we shift
{
    dn__.zero();
    // check if we have a norm conserving pseudo potential only. OOnly
    // derivatives of the hubbard wave functions are needed.
    auto& phi = kp.hubbard_wave_functions();

    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        auto& atom_type   = unit_cell_.atom(ia).type();
        if (atom_type.hubbard_correction()) {

            kp.generate_atomic_wave_functions(atom_type.hubbard_indexb_wfc(),
                                              ia,
                                              this->offset_[ia],
                                              true,
                                              phi);
        }
    }

    Beta_projectors_gradient bp_grad_(ctx_, kp.gkvec(), kp.igk_loc(), kp.beta_projectors());
    //kp.beta_projectors().prepare();
    bp_grad_.prepare();

    bool augment = false;

    for (auto ia = 0; (ia < ctx_.unit_cell().num_atom_types()) && (!augment); ia++) {
        augment = ctx_.unit_cell().atom_type(ia).augment();
    }

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
    matrix<double_complex>  dm(this->number_of_hubbard_orbitals() * ctx_.num_spins(),
                               this->number_of_hubbard_orbitals() * ctx_.num_spins());
    mdarray<double_complex, 5> dn_tmp(max_number_of_orbitals_per_atom(),
                                      max_number_of_orbitals_per_atom(),
                                      ctx_.num_spins(),
                                      ctx_.unit_cell().num_atoms(),
                                      3);

    if (ctx_.processing_unit() == device_t::GPU) {
        dm.allocate(memory_t::device);
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

    memory_t mem{memory_t::host};
    linalg_t la{linalg_t::blas};
    if (ctx_.processing_unit() == device_t::GPU) {
        mem = memory_t::device;
        la = linalg_t::gpublas;
    }

    /* compute <phi^I_m| S | psi_{nk}> */
    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        inner(mem, la, ispn, kp.spinor_wave_functions(), 0, kp.num_occupied_bands(ispn), dphi, 0,
              this->number_of_hubbard_orbitals(), phi_s_psi, 0, ispn * this->number_of_hubbard_orbitals());
    }


    for (int atom_id = 0; atom_id < ctx_.unit_cell().num_atoms(); atom_id++) {
        dn_tmp.zero(memory_t::host);
        dn_tmp.zero(memory_t::device);
        for (int dir = 0; dir < 3; dir++) {
            // reset dphi
            dphi.pw_coeffs(0).prime().zero(memory_t::host);
            dphi.pw_coeffs(0).prime().zero(memory_t::device);

            if (ctx_.unit_cell().atom(atom_id).type().hubbard_correction()) {
                // atom atom_id has hubbard correction so we need to compute the
                // derivatives of the hubbard orbitals associated to the atom
                // atom_id, the derivatives of the others hubbard orbitals been
                // zero compared to the displacement of atom atom_id

                // compute the derivative of |phi> corresponding to the
                // atom atom_id
                const int lmax_at = 2 * ctx_.unit_cell().atom(atom_id).type().hubbard_orbital(0).l + 1;

                // compute the derivatives of the hubbard wave functions
                // |phi_m^J> (J = atom_id) compared to a displacement of atom J.

                kp.compute_gradient_wave_functions(phi, this->offset_[atom_id], lmax_at, phitmp, this->offset_[atom_id], dir);

                if (ctx_.processing_unit() == device_t::GPU) {
                    phitmp.copy_to(spin_range(0), memory_t::device, 0, this->number_of_hubbard_orbitals());
                }

                // For norm conserving pp, it is enough to have the derivatives
                // of |phi^J_m> (J = atom_id)
                //apply_S_operator(kp, q_op, phitmp, dphi, this->offset_[atom_id], lmax_at);
                sirius::apply_S_operator<double_complex>(ctx_.processing_unit(), spin_range(0), 0,
                                                         this->number_of_hubbard_orbitals(), kp.beta_projectors(),
                                                         phitmp, &q_op, dphi);
            }

            // compute d S/ dr^I_a |phi> and add to dphi
            if (!ctx_.full_potential() && augment) {
                // it is equal to
                // \sum Q^I_ij <d \beta^I_i|phi> |\beta^I_j> + < \beta^I_i|phi> |d\beta^I_j>
                for (int chunk__ = 0; chunk__ < kp.beta_projectors().num_chunks(); chunk__++) {
                    for (int i = 0; i < kp.beta_projectors().chunk(chunk__).num_atoms_; i++) {
                        // need to find the right atom in the chunks.
                        if (kp.beta_projectors().chunk(chunk__).desc_(static_cast<int>(beta_desc_idx::ia), i) == atom_id) {
                            kp.beta_projectors().generate(chunk__);
                            bp_grad_.generate(chunk__, dir);

                            // compute Q_ij <\beta_i|\phi> |d \beta_j> and add it to d\phi
                            {
                                /* <beta | phi> for this chunk */
                                auto beta_phi = kp.beta_projectors().inner<double_complex>(chunk__, phi, 0, 0,
                                                                                           this->number_of_hubbard_orbitals());
                                q_op.apply(chunk__, i, 0, dphi, 0, this->number_of_hubbard_orbitals(), bp_grad_, beta_phi);
                            }

                            // compute Q_ij <d \beta_i|\phi> |\beta_j> and add it to d\phi
                            {
                                /* <dbeta | phi> for this chunk */
                                auto dbeta_phi = bp_grad_.inner<double_complex>(chunk__, phi, 0, 0,
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

            compute_occupancies(kp,
                                phi_s_psi,
                                dphi_s_psi,
                                dphi,
                                dn_tmp,
                                dm, // temporary table
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

    //kp.beta_projectors().dismiss();
    //bp_grad_.dismiss();
}

void
Hubbard::compute_occupancies_stress_derivatives(K_point&                    kp__,
                                                Q_operator& q_op__, // Compensnation operator or overlap operator
                                                mdarray<double_complex, 5>& dn__)  // derivative of the occupation number compared to displacement of atom aton_id
{
    auto& phi = kp__.hubbard_wave_functions();

    Wave_functions dphi(kp__.gkvec_partition(), this->number_of_hubbard_orbitals(), ctx_.preferred_memory_t(), 1);
    Wave_functions phitmp(kp__.gkvec_partition(), this->number_of_hubbard_orbitals(), ctx_.preferred_memory_t(), 1);

    Beta_projectors_strain_deriv bp_strain_deriv(ctx_, kp__.gkvec(), kp__.igk_loc());

    dmatrix<double_complex> dm(this->number_of_hubbard_orbitals() * ctx_.num_spins(),
                               this->number_of_hubbard_orbitals() * ctx_.num_spins());

    // maximum number of occupied bands
    int HowManyBands = kp__.num_occupied_bands(0);
    if (ctx_.num_spins() == 2) {
        HowManyBands = std::max(kp__.num_occupied_bands(1), kp__.num_occupied_bands(0));
    }

    bool augment = false;

    const int lmax  = ctx_.unit_cell().lmax();
    const int lmmax = utils::lmmax(lmax);

    mdarray<double, 2> rlm_g(lmmax, kp__.num_gkvec_loc());
    mdarray<double, 3> rlm_dg(lmmax, 3, kp__.num_gkvec_loc());

    // overlap between dphi and psi_{nk}
    dmatrix<double_complex> dphi_s_psi(HowManyBands, this->number_of_hubbard_orbitals() * ctx_.num_spins());
    // overlap between phi and psi_nk
    dmatrix<double_complex> phi_s_psi(HowManyBands, this->number_of_hubbard_orbitals() * ctx_.num_spins());

    // check if the pseudo potential is norm conserving or not
    for (auto ia = 0; (ia < ctx_.unit_cell().num_atom_types()) && (!augment); ia++) {
        augment = ctx_.unit_cell().atom_type(ia).augment();
    }

    /* initialize the beta projectors and derivatives */
    //kp__.beta_projectors().prepare();
    bp_strain_deriv.prepare();

    /* compute the hubbard orbitals */
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        auto& atom_type   = unit_cell_.atom(ia).type();
        if (atom_type.hubbard_correction()) {

            kp__.generate_atomic_wave_functions(atom_type.hubbard_indexb_wfc(),
                                                ia,
                                                this->offset_[ia],
                                                true,
                                                phi);
        }
    }

    if (ctx_.processing_unit() == device_t::GPU) {
        dm.allocate(memory_t::device);
        phi_s_psi.allocate(memory_t::device);
        dphi_s_psi.allocate(memory_t::device);

        phi.allocate(spin_range(0), memory_t::device);
        phi.copy_to(spin_range(0), memory_t::device, 0, this->number_of_hubbard_orbitals());

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            kp__.spinor_wave_functions().allocate(spin_range(ispn), memory_t::device);
            kp__.spinor_wave_functions().copy_to(spin_range(ispn), memory_t::device, 0, kp__.num_occupied_bands(ispn));
        }

        dphi.allocate(spin_range(0), memory_t::device);

        phitmp.allocate(spin_range(0), memory_t::device);
    }
    /* compute the S|phi^I_ia> */
    //apply_S_operator(kp__, q_op__, phi, dphi, 0, this->number_of_hubbard_orbitals());
    sirius::apply_S_operator<double_complex>(ctx_.processing_unit(), spin_range(0), 0, this->number_of_hubbard_orbitals(),
                             kp__.beta_projectors(), phi, &q_op__, dphi);

    phi_s_psi.zero(memory_t::host);
    phi_s_psi.zero(memory_t::device);

    memory_t mem{memory_t::host};
    linalg_t la{linalg_t::blas};
    if (ctx_.processing_unit() == device_t::GPU) {
        mem = memory_t::device;
        la = linalg_t::gpublas;
    }

    /* compute <phi^I_m| S | psi_{nk}> */
    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        inner(mem, la, ispn, kp__.spinor_wave_functions(), 0, kp__.num_occupied_bands(ispn), dphi, 0,
              this->number_of_hubbard_orbitals(), phi_s_psi, 0, ispn * this->number_of_hubbard_orbitals());
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

    for (int nu = 0; nu < 3; nu++) {
        for (int mu = 0; mu < 3; mu++) {

            // compute the derivatives of all hubbard wave functions
            // |phi_m^J> compared to the strain

            wavefunctions_strain_deriv(kp__, phitmp, rlm_g, rlm_dg, nu, mu);
            if (ctx_.processing_unit() == device_t::GPU) {
                phitmp.copy_to(spin_range(0), memory_t::device, 0, this->number_of_hubbard_orbitals());
            }
            // computes the S|d phi^I_ia>. It just happens that doing
            // this is equivalent to
            dphi.copy_from(ctx_.processing_unit(), this->number_of_hubbard_orbitals(), phitmp, 0, 0, 0, 0);

            // dphi = -0.5 * \delta_{\nu \mu} phi - d_e \phi - \sum_{ij} Q_{ij} <dphi| beta_i><beta_j|

            if (!ctx_.full_potential() && augment) {
                for (int i = 0; i < kp__.beta_projectors().num_chunks(); i++) {
                    /* generate beta-projectors for a block of atoms */
                    kp__.beta_projectors().generate(i);
                    /* generate derived beta-projectors for a block of atoms */
                    bp_strain_deriv.generate(i, 3 * nu + mu);

                    {
                        /* <d phi | beta> */
                        auto beta_dphi = kp__.beta_projectors().inner<double_complex>(i,
                                                                                      phitmp,
                                                                                      0, 0,
                                                                                      this->number_of_hubbard_orbitals());
                        /* apply Q operator (diagonal in spin) */
                        q_op__.apply(i, 0,
                                     dphi, 0,
                                     this->number_of_hubbard_orbitals(),
                                     kp__.beta_projectors(),
                                     beta_dphi);
                    }

                    {
                        /* <phi | d beta> */
                        auto dbeta_phi = bp_strain_deriv.inner<double_complex>(i,
                                                                               phi,
                                                                               0, 0,
                                                                               this->number_of_hubbard_orbitals());
                        /* apply Q operator (diagonal in spin) */
                        q_op__.apply(i, 0,
                                     dphi, 0,
                                     this->number_of_hubbard_orbitals(),
                                     kp__.beta_projectors(),
                                     dbeta_phi);
                    }

                    {
                        /* non-collinear case */
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

            compute_occupancies(kp__,
                                phi_s_psi,
                                dphi_s_psi,
                                dphi,
                                dn__,
                                dm,
                                3 * nu + mu);
        }
    }

    if (ctx_.processing_unit() == device_t::GPU) {
        phi.deallocate(spin_range(0), memory_t::device);
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            kp__.spinor_wave_functions().deallocate(spin_range(ispn), memory_t::device);
        }
    }

    //kp__.beta_projectors().dismiss();
    //bp_strain_deriv.dismiss();
}

void
Hubbard::wavefunctions_strain_deriv(K_point& kp__, Wave_functions& dphi, mdarray<double, 2> const& rlm_g,
                                    mdarray<double, 3> const& rlm_dg, const int nu, const int mu)
{
    #pragma omp parallel for schedule(static)
    for (int igkloc = 0; igkloc < kp__.num_gkvec_loc(); igkloc++) {
        /* global index of G+k vector */
        const int igk = kp__.idxgk(igkloc);
        auto gvc = kp__.gkvec().gkvec_cart<index_domain_t::local>(igkloc);
        /* vs = {r, theta, phi} */
        auto gvs = SHT::spherical_coordinates(gvc);
        std::vector<mdarray<double, 1>> ri_values(unit_cell_.num_atom_types());
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            ri_values[iat] = ctx_.atomic_wf_ri().values(iat, gvs[0]);
        }

        std::vector<mdarray<double, 1>> ridjl_values(unit_cell_.num_atom_types());
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            ridjl_values[iat] = ctx_.atomic_wf_djl().values(iat, gvs[0]);
        }

        const double p = (mu == nu) ? 0.5 : 0.0;
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto& atom_type = ctx_.unit_cell().atom(ia).type();
            if (atom_type.hubbard_correction()) {
                int offset__ = this->offset_[ia];
                for (auto&& orb : atom_type.hubbard_orbitals()) {
                    const int i            = orb.rindex();
                    const int l            = orb.l;
                    auto      phase        = twopi * dot(kp__.gkvec().gkvec(igk), unit_cell_.atom(ia).position());
                    auto      phase_factor = std::exp(double_complex(0.0, phase));
                    auto      z            = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(unit_cell_.omega());

                    // case |g+k| = 0
                    if (gvs[0] < 1e-10) {
                        if (l == 0) {
                            auto d1 = ri_values[atom_type.id()][i] * p * y00;

                            dphi.pw_coeffs(0).prime(igkloc, offset__) = -z * d1 * phase_factor;
                        } else {
                            for (int m = -l; m <= l; m++) {
                                dphi.pw_coeffs(0).prime(igkloc, offset__ + l + m) = 0.0;
                            }
                        }
                    } else {
                        for (int m = -l; m <= l; m++) {
                            int  lm = utils::lm(l, m);
                            auto d1 = ri_values[atom_type.id()][i] * (gvc[mu] * rlm_dg(lm, nu, igkloc) +
                                                                      p * rlm_g(lm, igkloc));
                            auto d2 = ridjl_values[atom_type.id()][i] * rlm_g(lm, igkloc) * gvc[mu] * gvc[nu] / gvs[0];

                            dphi.pw_coeffs(0).prime(igkloc, offset__ + l + m) = -z * (d1 + d2) * std::conj(phase_factor);
                        }
                    }
                    offset__ += 2 * l + 1;
                }
            }
        }
    }
}

void
Hubbard::compute_occupancies(K_point&                    kp,
                             dmatrix<double_complex>&    phi_s_psi,
                             dmatrix<double_complex>&    dphi_s_psi,
                             Wave_functions&             dphi,
                             mdarray<double_complex, 5>& dn__,
                             matrix<double_complex>&     dm__,
                             const int                   index)
{
    // it is actually <psi | d(S|phi>)
    dphi_s_psi.zero(memory_t::host);
    dphi_s_psi.zero(memory_t::device);
    int HowManyBands = kp.num_occupied_bands(0);
    if (ctx_.num_spins() == 2) {
        HowManyBands = std::max(kp.num_occupied_bands(1), kp.num_occupied_bands(0));
    }

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

    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        inner(mt, la, ispn, kp.spinor_wave_functions(), 0, kp.num_occupied_bands(ispn),
              dphi, //   S d |phi>
              0, this->number_of_hubbard_orbitals(), dphi_s_psi, 0, ispn * this->number_of_hubbard_orbitals());
    }

    if (ctx_.processing_unit() == device_t::GPU) {
        dphi_s_psi.copy_to(memory_t::host);
        phi_s_psi.copy_to(memory_t::host);
    }

    /* include the occupancy directly in dphi_s_psi */

    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        for (int n_orb = 0; n_orb < this->number_of_hubbard_orbitals(); n_orb++) {
            for (int nbnd = 0; nbnd < kp.num_occupied_bands(ispn); nbnd++) {
                dphi_s_psi(nbnd, ispn * this->number_of_hubbard_orbitals() + n_orb) *= kp.band_occupancy(nbnd, ispn);
            }
        }
    }
    if (ctx_.processing_unit() == device_t::GPU) {
        dphi_s_psi.copy_to(memory_t::device);
    }

    auto alpha = double_complex(kp.weight(), 0.0);
    linalg(la).gemm('C', 'N',
                     this->number_of_hubbard_orbitals() * ctx_.num_spins(),
                     this->number_of_hubbard_orbitals() * ctx_.num_spins(),
                     HowManyBands,
                     &alpha,
                     dphi_s_psi.at(mt),dphi_s_psi.ld(),
                     phi_s_psi.at(mt), phi_s_psi.ld(),
                     &linalg_const<double_complex>::zero(),
                     dm__.at(mt), dm__.ld());

    linalg(la).gemm('C', 'N',
                     this->number_of_hubbard_orbitals() * ctx_.num_spins(),
                     this->number_of_hubbard_orbitals() * ctx_.num_spins(),
                     HowManyBands,
                     &alpha,
                     phi_s_psi.at(mt), phi_s_psi.ld(),
                     dphi_s_psi.at(mt), dphi_s_psi.ld(),
                     &linalg_const<double_complex>::one(),
                     dm__.at(mt), dm__.ld());

    switch (ctx_.processing_unit()) {
        case device_t::CPU: {
            break;
        }
        case device_t::GPU: {
            dm__.copy_to(memory_t::host);
            break;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int ia1 = 0; ia1 < ctx_.unit_cell().num_atoms(); ++ia1) {
        const auto& atom = ctx_.unit_cell().atom(ia1);
        if (atom.type().hubbard_correction()) {
            const int lmax_at = 2 * atom.type().hubbard_orbital(0).l + 1;
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                const int ispn_offset = ispn * this->number_of_hubbard_orbitals() + this->offset_[ia1];
                for (int m2 = 0; m2 < lmax_at; m2++) {
                    for (int m1 = 0; m1 < lmax_at; m1++) {
                        dn__(m1, m2, ispn, ia1, index) = dm__(ispn_offset + m1, ispn_offset + m2);
                    }
                }
            }
        }
    }
}

}
