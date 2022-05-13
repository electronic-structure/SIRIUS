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
 */

// compute the forces for the simple LDA+U method not the fully
// rotationally invariant one. It can not be used for LDA+U+SO either

// this code is based on two papers :
//
// - PRB 84, 161102(R) (2011) : https://journals.aps.org/prb/abstract/10.1103/PhysRevB.84.161102
// - PRB 102, 235159 (2020) : https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.235159
//
// note that this code only apply for the collinear case
#include "hubbard.hpp"
#include "SDDK/wf_inner.hpp"
#include "SDDK/wf_trans.hpp"
#include "symmetry/crystal_symmetry.hpp"
#include "linalg/inverse_sqrt.hpp"

namespace sirius {

/* compute this |dphi> = dS | phi> + |dphi>, where the derivative is taken
 * compared to atom_id displacement. we can also use lambda */
void
Hubbard::apply_dS(K_point<double>& kp, Q_operator<double>& q_op, Beta_projectors_gradient<double>& bp_grad,
                  const int atom_id, const int dir, Wave_functions<double>& phi, Wave_functions<double>& dphi)
{
    // compute d S/ dr^I_a |phi> and add to dphi
    if (!ctx_.full_potential() && ctx_.unit_cell().augment()) {
        // it is equal to
        // \sum Q^I_ij <d \beta^I_i|phi> |\beta^I_j> + < \beta^I_i|phi> |d\beta^I_j>
        for (int ichunk = 0; ichunk < kp.beta_projectors().num_chunks(); ichunk++) {

          // check if this group of beta projector contains some beta projectors associated to atom_id

          bool beta_generate_ = false;
          for (int i = 0; i < kp.beta_projectors().chunk(ichunk).num_atoms_ && !beta_generate_; i++) {
            // need to find the right atom in the chunks.
            if (kp.beta_projectors().chunk(ichunk).desc_(static_cast<int>(beta_desc_idx::ia), i) == atom_id) {
              beta_generate_ = true;
            }
          }

          if (beta_generate_) {
            kp.beta_projectors().generate(ichunk);
            bp_grad.generate(ichunk, dir);
            auto beta_phi = kp.beta_projectors().inner<double_complex>(ichunk, phi, 0, 0, phi.num_wf());
            auto dbeta_phi = bp_grad.inner<double_complex>(ichunk, phi, 0, 0, phi.num_wf());

            for (int i = 0; i < kp.beta_projectors().chunk(ichunk).num_atoms_; i++) {

              if (atom_id == kp.beta_projectors().chunk(ichunk).desc_(static_cast<int>(beta_desc_idx::ia), i))
                // compute Q_ij <\beta_i|\phi> |d \beta_j> and add it to d\phi
                {
                  /* <beta | phi> |d \beta> for this chunk */
                  q_op.apply(ichunk, i, 0, dphi, 0, dphi.num_wf(), bp_grad, beta_phi);

                  /* Effectively compute Q_ij <d beta_i| phi> |beta_j> and add it dphi */
                  q_op.apply(ichunk, i, 0, dphi, 0, dphi.num_wf(), kp.beta_projectors(), dbeta_phi);
                }
            }
          }
        }
    }
}

  void
  Hubbard::compute_occupancies_derivatives_ortho(K_point<double>& kp__, Q_operator<double>& q_op__,
                                                 sddk::mdarray<std::complex<double>, 5>& dn__)
  {

    int num_hub_wf = ctx_.unit_cell().num_hubbard_wf().first;
    //int BS = ctx_.cyclic_block_size();

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
    default:
        break;
    }
    /*
      Compute the derivatives of the occupancies in two cases.

      - the atom is pp norm conserving or

      - the atom is ppus (in that case the derivative of the S operator gives a
      non zero contribution)
    */

    /* atomic wave functions  */
    auto& phi_atomic  = kp__.atomic_wave_functions();
    auto& sphi_atomic  = kp__.atomic_wave_functions_S();

    /* temporary wave functions */
    Wave_functions<double> grad_phi(kp__.gkvec_partition(), phi_atomic.num_wf(), ctx_.preferred_memory_t(), 1);
    /* temporary wave functions */
    Wave_functions<double> grad_phi_atomic(kp__.gkvec_partition(), phi_atomic.num_wf(), ctx_.preferred_memory_t(), 1);
    /* temporary wave functions */
    Wave_functions<double> s_grad_phi_atomic(kp__.gkvec_partition(), phi_atomic.num_wf(), ctx_.preferred_memory_t(), 1);
    /* temporary wave functions */
    Wave_functions<double> phi_tmp(kp__.gkvec_partition(), phi_atomic.num_wf(), ctx_.preferred_memory_t(), 1);
    /* derivative of the hubbard wave functions */
    Wave_functions<double> grad_phi_hub(kp__.gkvec_partition(), kp__.hubbard_wave_functions_S().num_wf(), ctx_.preferred_memory_t(), 1);
    auto num_hubbard_wf = unit_cell_.num_hubbard_wf();
    auto num_ps_atomic_wf = unit_cell_.num_ps_atomic_wf();

    std::array<sddk::dmatrix<double_complex>, 2>  phi_s_psi;

    /* <atomic_orbitals | S | atomic_orbitals> */
    dmatrix<double_complex> overlap__(phi_atomic.num_wf(), phi_atomic.num_wf());

    std::vector<double> eigenvalues(phi_atomic.num_wf());

    /* transformation matrix going from overlap to diagonal form */
    dmatrix<double_complex> U__(phi_atomic.num_wf(), phi_atomic.num_wf());

    /* Matrix orthogonalizing the wave functions set */
    dmatrix<double_complex> O__(phi_atomic.num_wf(), phi_atomic.num_wf());

    /* derivative of the Matrix orthogonalizing the wave functions set */
    dmatrix<double_complex> d_O_(phi_atomic.num_wf(), phi_atomic.num_wf());

    mdarray<double_complex, 4> dn_tmp(num_hub_wf, num_hub_wf, ctx_.num_spins(), 3);

    /* compute the overlap matrix and diagonalize it */
    overlap__.zero(memory_t::host);
    grad_phi.pw_coeffs(0).prime().zero(memory_t::host);

    if (ctx_.processing_unit() == device_t::GPU) {
        dn_tmp.allocate(memory_t::device);
        overlap__.allocate(memory_t::device);
        d_O_.allocate(memory_t::device);
        O__.allocate(memory_t::device);
        U__.allocate(memory_t::device);
        /* allocation of the overlap matrices on GPU */
        // dphi_s_psi.allocate(memory_t::device);

        phi_atomic.prepare(spin_range(0), true, &ctx_.mem_pool(memory_t::device));
        sphi_atomic.prepare(spin_range(0), true, &ctx_.mem_pool(memory_t::device));

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            kp__.spinor_wave_functions().prepare(spin_range(ispn), true, &ctx_.mem_pool(memory_t::device));
        }

        kp__.hubbard_wave_functions_S().prepare(spin_range(0), true, &ctx_.mem_pool(memory_t::device));

        /* wave functions */
        phi_tmp.prepare(spin_range(0), false, &ctx_.mem_pool(memory_t::device));
        grad_phi.prepare(spin_range(0), false, &ctx_.mem_pool(memory_t::device));
        grad_phi_atomic.prepare(spin_range(0), false, &ctx_.mem_pool(memory_t::device));
        s_grad_phi_atomic.prepare(spin_range(0), false, &ctx_.mem_pool(memory_t::device));

        overlap__.zero(memory_t::device);
    }

    kp__.compute_orthogonalization_operator(0, phi_atomic, sphi_atomic, O__, U__, eigenvalues);

    if (ctx_.processing_unit() == device_t::GPU) {
        U__.copy_to(memory_t::device);
        O__.copy_to(memory_t::device);
    }

    Beta_projectors_gradient<double> bp_grad(ctx_, kp__.gkvec(), kp__.igk_loc(), kp__.beta_projectors());
    bp_grad.prepare();

    /* compute <phi (O)^(-1/2)| S | psi_{nk}> where |(O)^(-1/2) phi> are the hubbard wavefunctions */
    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        phi_s_psi[ispn] = dmatrix<double_complex>(kp__.hubbard_wave_functions_S().num_wf(), kp__.num_occupied_bands(ispn));
        if (ctx_.processing_unit() == device_t::GPU) {
            phi_s_psi[ispn].allocate(ctx_.mem_pool(memory_t::device));
        }
        inner(ctx_.spla_context(), spin_range(ispn), kp__.hubbard_wave_functions_S(), 0, kp__.hubbard_wave_functions_S().num_wf(),
              kp__.spinor_wave_functions(), 0, kp__.num_occupied_bands(ispn), phi_s_psi[ispn], 0, 0);
    }


    for (int dir = 0; dir < 3; dir++) {

        for (int i = 0; i < phi_atomic.num_wf(); i++) {
            for (int igloc = 0; igloc < kp__.num_gkvec_loc(); igloc++) {
                /* G+k vector in Cartesian coordinates */
                auto gk = kp__.gkvec().template gkvec_cart<index_domain_t::local>(igloc);
                /* gradient of phi_atomic */
                grad_phi_atomic.pw_coeffs(0).prime(igloc, i) = std::complex<double>(0.0, -gk[dir]) *
                    phi_atomic.pw_coeffs(0).prime(igloc, i);
            }
        }


        apply_S_operator<std::complex<double>>(ctx_.processing_unit(), spin_range(0), 0, phi_atomic.num_wf(),
                                               kp__.beta_projectors(), grad_phi_atomic, &q_op__, s_grad_phi_atomic);

        for (int atom_id = 0; atom_id < ctx_.unit_cell().num_atoms(); atom_id++) { // loop over the atom displacement.
            dn_tmp.zero(memory_t::host);
            dn_tmp.zero(memory_t::device);
            // reset dphi
            grad_phi.zero(ctx_.processing_unit());
            grad_phi_hub.zero(ctx_.processing_unit());
            phi_tmp.zero(ctx_.processing_unit());

            // compute S|d\phi>. Will be zero if the atom has no hubbard
            // correction
            if (ctx_.unit_cell().atom(atom_id).type().hubbard_correction() || ctx_.cfg().hubbard().full_orthogonalization()) {
                // extract the block of atomic wave functions that are specific to atom_id.

                phi_tmp.copy_from(ctx_.processing_unit(),
                                  ctx_.unit_cell().atom(atom_id).type().indexb_wfs().size(),
                                  s_grad_phi_atomic,
                                  0,
                                  num_ps_atomic_wf.second[atom_id],
                                  0,
                                  num_ps_atomic_wf.second[atom_id]);

                if (ctx_.cfg().hubbard().full_orthogonalization()) {
                    /* this section computes the derivatives of O^{-1/2} compared to a
                       atomic displacement.

                       to compute the correction coming from the orthogonalization
                       procedure we need to compute the derivative of the overlap matrix

                       \partial_r (<phi_i|S|phi_j>) = <\partial_r (\phi) | S | phi_j>
                       + <phi_i | \partial_r (S) | \phi_j>
                       + <phi_i | S | \partial_r(phi_j)>

                    */

                    overlap__.zero(memory_t::host);
                    overlap__.zero(memory_t::device);

                    // compute (<\partial_r (\phi) | S) | phi_j>

                    inner(ctx_.spla_context(),
                          sddk::spin_range(0),
                          phi_tmp, 0, phi_tmp.num_wf(),
                          phi_atomic, 0, phi_atomic.num_wf(),
                          overlap__, 0, 0);

                    // // compute (d S/ d R_K) |phi_atomic> and add to S|dphi>. It is Eq.18 of Ref PRB 102, 235159 (2020)
                    apply_dS(kp__, q_op__, bp_grad, atom_id, dir, phi_atomic, phi_tmp);

                    inner(ctx_.spla_context(),
                          sddk::spin_range(0),
                          phi_atomic, 0, phi_atomic.num_wf(),
                          phi_tmp, 0, phi_tmp.num_wf(),
                          d_O_, 0, 0);

                    if (ctx_.processing_unit() == device_t::GPU) {
                        d_O_.copy_to(memory_t::host);
                        overlap__.copy_to(memory_t::host);
                    }

                    for (int ii = 0; ii < phi_atomic.num_wf(); ii++) {
                        for (int ji = 0; ji < phi_atomic.num_wf(); ji++) {
                            overlap__(ji, ii) += d_O_(ji, ii);
                        }
                    }

                    if (ctx_.processing_unit() == device_t::GPU) {
                        overlap__.copy_to(memory_t::device);
                    }
                    /*
                      Starting from that point phi_tmp can used for other calculations

                    we have the derivative of  the overlap matrix. It is Eq.33

                    first compute the (U^dagger dO/dr U) from Eq.32
                    */

                    linalg(la).gemm('N', 'N', phi_atomic.num_wf(), phi_atomic.num_wf(), phi_atomic.num_wf(),
                                    &linalg_const<double_complex>::one(), overlap__.at(mt), overlap__.ld(), U__.at(mt),
                                    U__.ld(), &linalg_const<double_complex>::zero(), d_O_.at(mt), d_O_.ld());

                    linalg(la).gemm('C', 'N', phi_atomic.num_wf(), phi_atomic.num_wf(), phi_atomic.num_wf(),
                                    &linalg_const<double_complex>::one(), U__.at(mt), U__.ld(), d_O_.at(mt), d_O_.ld(),
                                    &linalg_const<double_complex>::zero(), overlap__.at(mt), overlap__.ld());


                    if (ctx_.processing_unit() == device_t::GPU) {
                        overlap__.copy_to(memory_t::host);
                    }

                    /* Eq.32 is a double dgemm product although not written explicitly that way */

                    /* first compute the middle matrix. We just have to multiply
                       overlap__(m3, m4) with appropriate term */
                    for (int m1 = 0; m1 < phi_atomic.num_wf(); m1++) {
                        for (int m2 = 0; m2 < phi_atomic.num_wf(); m2++) {
                            overlap__(m1, m2) *= -
                                1.0 / (eigenvalues[m1] * std::sqrt(eigenvalues[m2]) +
                                       eigenvalues[m2] * std::sqrt(eigenvalues[m1]));
                        }
                    }

                    if (ctx_.processing_unit() == device_t::GPU) {
                        overlap__.copy_to(memory_t::device);
                    }


                    // (d O / dr) * U^dagger
                    linalg(la).gemm('N', 'C', phi_atomic.num_wf(), phi_atomic.num_wf(), phi_atomic.num_wf(),
                                    &linalg_const<double_complex>::one(), overlap__.at(mt), overlap__.ld(), U__.at(mt),
                                    U__.ld(), &linalg_const<double_complex>::zero(), d_O_.at(mt), d_O_.ld());

                    // U (d O / dr * U^dagger)
                    linalg(la).gemm('N', 'N', phi_atomic.num_wf(), phi_atomic.num_wf(), phi_atomic.num_wf(),
                                    &linalg_const<double_complex>::one(), U__.at(mt), U__.ld(), d_O_.at(mt), d_O_.ld(),
                                    &linalg_const<double_complex>::zero(), overlap__.at(mt), overlap__.ld());

                    /* From that point out, we have the Lowen operator stored in O__ and
                       its derivative stored in overlap__. We need to apply these two
                       operators the following way

                       $$
                       \nabla_r \phi_{hub} = S O__ |\nabla_r \phi_atomic>
                       + S d_O__ | phi_atomic>
                       + dS O__ | phi_atomic>
                       $$

                       The only tricky point here is to take the corresponding block of
                       atomic wave functions that corresponds to a given atom.
                    */


                    // apply d O^(-1/2) on the original phi. d O^{-1/2} is contained in overlap__ matrix
                    transform<double_complex>(ctx_.spla_context(), 0, phi_atomic, 0, phi_atomic.num_wf(), overlap__,
                                              0, 0, phi_tmp, 0, phi_tmp.num_wf());

                    // now add O^{-1/2} |d\phi_atomic>

                    /* Firs extract the atomic orbitals corresponding to atom atom_id */
                    grad_phi.copy_from(ctx_.processing_unit(),
                                       ctx_.unit_cell().atom(atom_id).type().indexb_wfs().size(),
                                       grad_phi_atomic,
                                       0,
                                       num_ps_atomic_wf.second[atom_id],
                                       0,
                                       num_ps_atomic_wf.second[atom_id]);

                    // then apply O^{-1/2}
                    transform<double_complex>(ctx_.spla_context(), 0,
                                              linalg_const<double>::one(), {&grad_phi}, 0, grad_phi.num_wf(),
                                              O__, 0, 0,
                                              linalg_const<double>::one(), {&phi_tmp}, 0, phi_tmp.num_wf());

                    // apply S on (d O^(-1/2) |phi_atomic> + O^(-1/2) | d phi_atomic>) and store it to grad_phi
                    sirius::apply_S_operator<double_complex>(ctx_.processing_unit(), spin_range(0), 0, phi_tmp.num_wf(),
                                                             kp__.beta_projectors(), phi_tmp, &q_op__, grad_phi);

                    /*
                      apply dS to O^{-1/2} |phi_atomic> and add result to grad_phi that contains  S \partial_R(O^-1/2) | phi>
                      + S O^{-1/2} |dphi>)
                    */

                    transform<std::complex<double>>(ctx_.spla_context(), 0, phi_atomic, 0, phi_atomic.num_wf(), O__,
                                                    0, 0, phi_tmp, 0, phi_tmp.num_wf());

                    apply_dS(kp__, q_op__, bp_grad, atom_id, dir, phi_tmp, grad_phi);
                } else {
                    grad_phi.copy_from(ctx_.processing_unit(), phi_atomic.num_wf(), phi_tmp, 0, 0, 0, 0);
                }
            }


            if (!ctx_.cfg().hubbard().full_orthogonalization()) {
                apply_dS(kp__, q_op__, bp_grad, atom_id, dir, phi_atomic, grad_phi);
            }

            // we need to extract the hubbard wave function derivatives from grad_phi

            /* loop over Hubbard orbitals of the atom */
            for (int atom_id1 = 0; atom_id1 < ctx_.unit_cell().num_atoms(); atom_id1++) {
                auto& atom1 = ctx_.unit_cell().atom(atom_id1);
                auto& type1 = atom1.type();

                for (int idxrf = 0; idxrf < type1.indexr_hub().size(); idxrf++) {
                    /* Hubbard orbital descriptor */
                    auto& hd = type1.lo_descriptor_hub(idxrf);
                    int l = type1.indexr_hub().am(idxrf).l();
                    int mmax = 2 * l + 1;

                    int idxr_wf = hd.idx_wf();

                    int offset_in_wf = num_ps_atomic_wf.second[atom_id1] + type1.indexb_wfs().offset(idxr_wf);
                    int offset_in_hwf = num_hubbard_wf.second[atom_id1] + type1.indexb_hub().offset(idxrf);

                    grad_phi_hub.copy_from(device_t::CPU, mmax, grad_phi, 0, offset_in_wf, 0, offset_in_hwf);
                }
            }

            compute_occupancies(kp__, phi_s_psi, grad_phi_hub, dn_tmp, dir);

            /* use a memcpy here */
            std::memcpy(dn__.at(memory_t::host, 0, 0, 0, dir, atom_id), dn_tmp.at(memory_t::host, 0, 0, 0, dir),
                        sizeof(double_complex) * dn_tmp.size(0) * dn_tmp.size(1) * dn_tmp.size(2));
        } // atom_id
    } // direction x, y, z

    if (ctx_.processing_unit() == device_t::GPU) {
        phi_atomic.deallocate(spin_range(0), memory_t::device);
        kp__.spinor_wave_functions().deallocate(spin_range(ctx_.num_spins()), memory_t::device);
        sphi_atomic.deallocate(spin_range(0), memory_t::device);
    }
}

void
Hubbard::compute_occupancies_derivatives_non_ortho(K_point<double>& kp__, Q_operator<double>& q_op__,
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
        default:
            break;
    }

    // TODO: check if we have a norm conserving pseudo potential;
    // TODO: distrribute (MPI) all matrices in the basis of atomic orbitals
    // only derivatives of the atomic wave functions are needed.
    auto& phi_atomic    = kp__.atomic_wave_functions();
    auto& phi_atomic_S  = kp__.atomic_wave_functions_S();
    auto& sphi_hub      = kp__.hubbard_wave_functions_S();

    auto num_ps_atomic_wf = ctx_.unit_cell().num_ps_atomic_wf();
    auto num_hubbard_wf   = ctx_.unit_cell().num_hubbard_wf();

    if (ctx_.processing_unit() == device_t::GPU) {
        dn__.allocate(memory_t::device);
        phi_atomic.prepare(spin_range(0), true, &ctx_.mem_pool(memory_t::device));
        sphi_hub.prepare(spin_range(0), true, &ctx_.mem_pool(memory_t::device));
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            kp__.spinor_wave_functions().prepare(spin_range(ispn), true, &ctx_.mem_pool(memory_t::device));
        }
    }

    /* compute overlap matrix */
    sddk::dmatrix<std::complex<double>> ovlp(phi_atomic.num_wf(), phi_atomic.num_wf());
    sddk::inner(ctx_.spla_context(), spin_range(0), phi_atomic, 0, phi_atomic.num_wf(),
            phi_atomic_S, 0, phi_atomic_S.num_wf(), ovlp, 0, 0);
    /* a tuple of O^{-1/2}, U, \lambda */
    auto result = inverse_sqrt(ovlp, phi_atomic.num_wf());
    auto& inv_sqrt_O = std::get<0>(result);
    auto& evec_O = std::get<1>(result);
    auto& eval_O = std::get<2>(result);

    /* compute < psi_{ik} | S | phi_hub > */
    /* this is used in the final expression for the occupation matrix derivative */
    std::array<sddk::dmatrix<double_complex>, 2> psi_s_phi_hub;
    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        psi_s_phi_hub[ispn] = sddk::dmatrix<double_complex>(kp__.num_occupied_bands(ispn), sphi_hub.num_wf());
        if (ctx_.processing_unit() == device_t::GPU) {
            psi_s_phi_hub[ispn].allocate(ctx_.mem_pool(memory_t::device));
        }
        inner(ctx_.spla_context(), spin_range(ispn), kp__.spinor_wave_functions(), 0, kp__.num_occupied_bands(ispn),
              sphi_hub, 0, sphi_hub.num_wf(), psi_s_phi_hub[ispn], 0, 0);
    }

    /* temporary storage */
    Wave_functions<double> phi_atomic_tmp(kp__.gkvec_partition(), phi_atomic.num_wf(), ctx_.preferred_memory_t(), 1);

    Wave_functions<double> s_phi_atomic_tmp(kp__.gkvec_partition(), phi_atomic.num_wf(), ctx_.preferred_memory_t(), 1);

    /* compute < d phi_atomic / d r_{j} | S | psi_{ik} > and < d phi_atomic / d r_{j} | S | phi_atomic > */
    std::array<std::array<sddk::dmatrix<double_complex>, 2>, 3> grad_phi_atomic_s_psi;
    std::array<sddk::dmatrix<double_complex>, 3> grad_phi_atomic_s_phi_atomic;
    for (int x = 0; x < 3; x++) {
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
        apply_S_operator<std::complex<double>>(ctx_.processing_unit(), spin_range(0), 0, phi_atomic_tmp.num_wf(),
                kp__.beta_projectors(), phi_atomic_tmp, &q_op__, s_phi_atomic_tmp);

        grad_phi_atomic_s_phi_atomic[x]= sddk::dmatrix<double_complex>(s_phi_atomic_tmp.num_wf(), phi_atomic.num_wf());
        /* compute < d phi_atomic / d r_{j} | S | phi_atomic > */
        inner(ctx_.spla_context(), spin_range(0), s_phi_atomic_tmp, 0, s_phi_atomic_tmp.num_wf(),
                phi_atomic, 0, phi_atomic.num_wf(), grad_phi_atomic_s_phi_atomic[x], 0, 0);

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            /* allocate space */
            grad_phi_atomic_s_psi[x][ispn] =
                sddk::dmatrix<double_complex>(s_phi_atomic_tmp.num_wf(), kp__.num_occupied_bands(ispn));
            /* compute < d phi_atomic / d r_{j} | S | psi_{ik} > for all atoms */
            inner(ctx_.spla_context(), spin_range(ispn), s_phi_atomic_tmp, 0, s_phi_atomic_tmp.num_wf(),
                kp__.spinor_wave_functions(), 0, kp__.num_occupied_bands(ispn), grad_phi_atomic_s_psi[x][ispn], 0, 0);
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

                /* build | d S / d r_{j} | phi_atomic > */
                /* it consists of two contributions:
                 *   | beta >        Q < d beta / dr | phi_atomic > and
                 *   | d beta / dr > Q < beta        | phi_atomic > */
                phi_atomic_tmp.zero(ctx_.processing_unit());
                q_op__.apply(ichunk, i, 0, phi_atomic_tmp, 0, phi_atomic_tmp.num_wf(), bp_grad, beta_phi_atomic);
                q_op__.apply(ichunk, i, 0, phi_atomic_tmp, 0, phi_atomic_tmp.num_wf(), kp__.beta_projectors(), grad_beta_phi_atomic);


                /* compute O' = d O / d r_{alpha} */
                /* from O = <phi | S | phi > we get
                 * O' = <phi' | S | phi> + <phi | S' |phi> + <phi | S | phi'> */

                /* <phi | S' | phi> */
                sddk::inner(ctx_.spla_context(), spin_range(0), phi_atomic, 0, phi_atomic.num_wf(),
                    phi_atomic_tmp, 0, phi_atomic_tmp.num_wf(), ovlp, 0, 0);
                /* add <phi' | S | phi> and <phi | S | phi'> */
                for (int i = 0; i < phi_atomic.num_wf(); i++) {
                    for (int j = 0; j < phi_atomic.num_wf(); j++) {
                        ovlp(j, i) +=
                            (grad_phi_atomic_s_phi_atomic[x](j, i) + std::conj(grad_phi_atomic_s_phi_atomic[x](i, j)));
                    }
                }
                /* compute \tilde O' = U^{H}O'U */
                unitary_similarity_transform(1, ovlp, evec_O, phi_atomic.num_wf());
                for (int i = 0; i < phi_atomic.num_wf(); i++) {
                    for (int j = 0; j < phi_atomic.num_wf(); j++) {
                        ovlp(j, i) /= -(eval_O[i] * std::sqrt(eval_O[j]) + eval_O[j] * std::sqrt(eval_O[i]));
                    }
                }
                /* compute d/dr O^{-1/2} */
                unitary_similarity_transform(0, ovlp, evec_O, phi_atomic.num_wf());

                /* compute <phi_atomic | dS/dr_j | psi_{ik}> */
                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    sddk::dmatrix<double_complex> phi_atomic_ds_psi(phi_atomic_tmp.num_wf(), kp__.num_occupied_bands(ispn));
                    inner(ctx_.spla_context(), spin_range(ispn), phi_atomic_tmp, 0, phi_atomic_tmp.num_wf(),
                        kp__.spinor_wave_functions(), 0, kp__.num_occupied_bands(ispn), phi_atomic_ds_psi, 0, 0);

                    /* build the full d <phi_atomic | S | psi_ik> / d r_{j} matrix */
                    sddk::dmatrix<double_complex> dphi_hub_s_psi(num_hubbard_wf.first, kp__.num_occupied_bands(ispn));
                    dphi_hub_s_psi.zero();
                    for (int ibnd = 0; ibnd < kp__.num_occupied_bands(ispn); ibnd++) {
                        if (true) {
                            for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
                                auto& atom = ctx_.unit_cell().atom(ia);
                                auto& type = atom.type();
                                if (type.hubbard_correction()) {
                                    /* loop over Hubbard orbitals of the atom */
                                    for (int idxrf = 0; idxrf < type.indexr_hub().size(); idxrf++) {
                                        /* Hubbard orbital descriptor */
                                        auto& hd = type.lo_descriptor_hub(idxrf);
                                        int l = type.indexr_hub().am(idxrf).l();
                                        int mmax = 2 * l + 1;

                                        int idxr_wf = hd.idx_wf();

                                        int offset_in_wf = num_ps_atomic_wf.second[ia] + type.indexb_wfs().offset(idxr_wf);
                                        int offset_in_hwf = num_hubbard_wf.second[ia] + type.indexb_hub().offset(idxrf);
                                        for (int m = 0; m < mmax; m++) {
                                            dphi_hub_s_psi(offset_in_hwf + m, ibnd) = phi_atomic_ds_psi(offset_in_wf + m, ibnd);
                                        }
                                        if (ja == ia) {
                                            for (int m = 0; m < mmax; m++) {
                                                dphi_hub_s_psi(offset_in_hwf + m, ibnd) +=
                                                    grad_phi_atomic_s_psi[x][ispn](offset_in_wf + m, ibnd);
                                            }
                                        }
                                    } // idxrf
                                }
                            } // ia
                        } else {

                        }
                        for (int j = 0; j < num_hubbard_wf.first; j++) {
                            dphi_hub_s_psi(j, ibnd) *= kp__.band_occupancy(ibnd, ispn);
                        }
                    } // ibnd

                    if (ctx_.processing_unit() == device_t::GPU) {
                        dphi_hub_s_psi.allocate(ctx_.mem_pool(memory_t::device)).copy_to(memory_t::device);
                    }
                    auto alpha = double_complex(kp__.weight(), 0.0);

                    /* update the density matrix derivative */

                    linalg(la).gemm('N', 'N', num_hubbard_wf.first, num_hubbard_wf.first,
                                    kp__.num_occupied_bands(ispn), &alpha,
                                    dphi_hub_s_psi.at(mt, 0, 0), dphi_hub_s_psi.ld(),
                                    psi_s_phi_hub[ispn].at(mt, 0, 0), psi_s_phi_hub[ispn].ld(),
                                    &linalg_const<double_complex>::one(),
                                    dn__.at(mt, 0, 0, ispn, x, ja), dn__.ld());

                    linalg(la).gemm('C', 'C', num_hubbard_wf.first, num_hubbard_wf.first,
                                    kp__.num_occupied_bands(ispn), &alpha,
                                    psi_s_phi_hub[ispn].at(mt, 0, 0), psi_s_phi_hub[ispn].ld(),
                                    dphi_hub_s_psi.at(mt, 0, 0), dphi_hub_s_psi.ld(),
                                    &linalg_const<double_complex>::one(),
                                    dn__.at(mt, 0, 0, ispn, x, ja), dn__.ld());
                } // ispn
            } //i
        } // x
    } // ichunk

    if (ctx_.processing_unit() == device_t::GPU) {
        dn__.copy_to(memory_t::host);
        dn__.deallocate(memory_t::device);
        phi_atomic.dismiss(spin_range(0), false);
        sphi_hub.dismiss(spin_range(0), false);
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

    /* this is the original atomic wave functions without the operator S applied */
    auto& phi = kp__.hubbard_wave_functions();

    if (ctx_.processing_unit() == device_t::GPU) {
        dn__.allocate(memory_t::device);
    }

    /*
      dphi contains this

      \f[
      \left| d \phi\right> = \partial_{\mu\nu} (S \left| \phi \right>)
      ]

    */
    Wave_functions<double> dphi(kp__.gkvec_partition(), phi.num_wf(), ctx_.preferred_memory_t(), 1);

    Wave_functions<double> phitmp(kp__.gkvec_partition(), phi.num_wf(), ctx_.preferred_memory_t(), 1);

    Beta_projectors_strain_deriv<double> bp_strain_deriv(ctx_, kp__.gkvec(), kp__.igk_loc());

    bool augment = ctx_.unit_cell().augment();

    const int lmax  = ctx_.unit_cell().lmax();
    const int lmmax = utils::lmmax(lmax);

    sddk::mdarray<double, 2> rlm_g(lmmax, kp__.num_gkvec_loc());
    sddk::mdarray<double, 3> rlm_dg(lmmax, 3, kp__.num_gkvec_loc());

    /* overlap between psi_nk and phi_ik <psi|S|phi> */
    std::array<dmatrix<double_complex>,2> phi_s_psi_hub;

    /* initialize the beta projectors and derivatives */
    bp_strain_deriv.prepare();

    auto sr = spin_range(ctx_.num_spins() == 2 ? 2 : 0);
    kp__.spinor_wave_functions().prepare(sr, true);
    kp__.hubbard_wave_functions_S().prepare(sr, true);
    kp__.hubbard_wave_functions().prepare(sr, true);
    auto &sphi_hub = kp__.hubbard_wave_functions_S();
    if (ctx_.processing_unit() == device_t::GPU) {
      dphi.prepare(spin_range(0), false, &ctx_.mem_pool(memory_t::device));
      phitmp.prepare(spin_range(0), true, &ctx_.mem_pool(memory_t::device));
    }

    /* compute <psi_nk | S | phi_mk>
     * treat non-magnetic, collinear and non-collinear cases;
     * in collinear case psi_s_phi contains two blocks for up- and dn- spin channels */
    for (int is = 0; is < ctx_.num_spinors(); is++) {
        auto sr = ctx_.num_mag_dims() == 3 ? spin_range(2) : spin_range(is);
        phi_s_psi_hub[is] = sddk::dmatrix<double_complex>(sphi_hub.num_wf(), kp__.num_occupied_bands(is));
        if (ctx_.processing_unit() == device_t::GPU) {
          phi_s_psi_hub[is].allocate(ctx_.mem_pool(memory_t::device));
        }
        inner(ctx_.spla_context(), sr,
              kp__.hubbard_wave_functions_S(), 0, kp__.hubbard_wave_functions_S().num_wf(),
              kp__.spinor_wave_functions(), 0, kp__.num_occupied_bands(is),
              phi_s_psi_hub[is], 0, 0);
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
            // if (ctx_.num_spins() == 2) {
            //    phitmp.copy_from(device_t::CPU, phitmp.num_wf(), phitmp, 0, 0, 1, (ctx_.num_mag_dims() == 3) ?
            //    phitmp.num_wf() : 0);
            //}

            /* Just to keep in mind:
             *
             * - kp.hubbard_wave_functions() are the hubbard functions with S applied
             * - kp.atomic_wave_functions_S_hub() are the non-orthogonal (initial) atomic functions with S applied
             * - kp.atomic_wave_functions_hub() are the non-orthogonal (initial) atomic functions
             * - both sets are spin-dependent, even if the orbitals have no spin label
             * - atom_type.indexr_hub() is the new index of Hubbard radial functions
             * - atom_type.indexb_hub() is the new index of Hubbard orbitals (radial function times a spherical
             * harmonic)
             * - wavefunctions_strain_deriv() and K_point::generate_atomic_wave_functions() work with the new indices
             */

            if (ctx_.processing_unit() == device_t::GPU) {
                phitmp.copy_to(spin_range(0), memory_t::device, 0, this->num_hubbard_wf());
            }

            /* dphi is the strain derivative of the hubbard orbitals (with S applied). Derivation imply this

               d(S phi) = (dS) phi + S (d\phi) = (d\phi) - \sum_{ij} Q_{ij} |beta_i><beta_j| d(phi) -
               \sum_{ij} Q_{ij} |d beta_i><beta_j|phi> - \sum_{ij} Q_{ij} |beta_i><d beta_j|phi>

               dphi contains the full expression phitmp contains (d phi).
            */

            dphi.copy_from(ctx_.processing_unit(), this->num_hubbard_wf(), phitmp, 0, 0, 0, 0);

            // dphi = |phitmp> - \sum_{ij} Q_{ij} | beta_i><beta_j|phitmp> + |d beta_i><beta_j|phi> + |beta_i><d
            // beta_j|phi> dphi = (1 - \sum_{ij} Q_{ij} | beta_i><beta_j|) |phitmp> - (sum_{ij} Q_{ij} |d
            // beta_i><beta_j|  + |beta_i><d beta_j|) | phi>

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
                        auto beta_dphi = kp__.beta_projectors().inner<double_complex>(
                            i,
                            phitmp, // contains the strain derivative of the hubbard orbitals.
                            0, 0, this->num_hubbard_wf());
                        /* apply Q operator (diagonal in spin) */

                        // compute Q_{ij}<beta_i | dphi> |beta_j> and add it to dphi
                        q_op__.apply(i, 0, dphi, 0, this->num_hubbard_wf(), kp__.beta_projectors(),
                                     beta_dphi);
                    }

                    {
                        /* <d(beta) | phi> */
                        auto dbeta_phi =
                            bp_strain_deriv.inner<double_complex>(i, phi, 0, 0, this->num_hubbard_wf());
                        /* apply Q operator (diagonal in spin) */

                        // compute <d (beta) | phi> |beta> and add it to dphi
                        q_op__.apply(i, 0, dphi, 0, this->num_hubbard_wf(), kp__.beta_projectors(),
                                     dbeta_phi);
                    }

                    {
                        // <beta|phi> |d beta>
                        auto beta_phi = kp__.beta_projectors().inner<double_complex>(
                            i, phi, 0, 0, this->num_hubbard_wf());
                        /* apply Q operator (diagonal in spin) */
                        q_op__.apply(i, 0, dphi, 0, this->num_hubbard_wf(), bp_strain_deriv, beta_phi);
                    }
                }
            }
            compute_occupancies(kp__, phi_s_psi_hub, dphi, dn__, 3 * nu + mu);
        }
    }

    if (ctx_.processing_unit() == device_t::GPU) {
        dn__.deallocate(memory_t::device);
    }

    kp__.spinor_wave_functions().dismiss(sr, false);
    kp__.hubbard_wave_functions_S().dismiss(sr, false);
    kp__.hubbard_wave_functions().dismiss(sr, false);
}

void
Hubbard::wavefunctions_strain_deriv(K_point<double>& kp__, Wave_functions<double>& dphi__,
                                    sddk::mdarray<double, 2> const& rlm_g__, sddk::mdarray<double, 3> const& rlm_dg__,
                                    int nu__, int mu__)
{
    auto num_hubbard_wf = ctx_.unit_cell().num_hubbard_wf();

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
            ri_values[iat] = ctx_.ps_atomic_wf_ri().values(iat, gvs[0]);
        }

        std::vector<sddk::mdarray<double, 1>> ridjl_values(unit_cell_.num_atom_types());
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            ridjl_values[iat] = ctx_.ps_atomic_wf_ri_djl().values(iat, gvs[0]);
        }

        const double p = (mu__ == nu__) ? 0.5 : 0.0;

        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto& atom_type = ctx_.unit_cell().atom(ia).type();
            if (atom_type.hubbard_correction()) {
                // TODO: this can be optimized, check k_point::generate_atomic_wavefunctions()
                auto phase        = twopi * dot(kp__.gkvec().gkvec(igk), unit_cell_.atom(ia).position());
                auto phase_factor = std::exp(double_complex(0.0, phase));
                int offset        = num_hubbard_wf.second[ia];
                for (int xi = 0; xi < atom_type.indexb_hub().size(); xi++) {
                    /*  orbital quantum  number of this atomic orbital */
                    int l = atom_type.indexb_hub().l(xi);
                    /*  composite l,m index */
                    int lm = atom_type.indexb_hub().lm(xi);
                    /* index of the radial function */
                    int idxrf = atom_type.lo_descriptor_hub(atom_type.indexb_hub().idxrf(xi)).idx_wf();

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
                        auto d1 = ri_values[atom_type.id()][idxrf] *
                                  (gvc[mu__] * rlm_dg__(lm, nu__, igkloc) + p * rlm_g__(lm, igkloc));
                        auto d2 =
                            ridjl_values[atom_type.id()][idxrf] * rlm_g__(lm, igkloc) * gvc[mu__] * gvc[nu__] / gvs[0];

                        dphi__.pw_coeffs(0).prime(igkloc, offset + xi) = -z * (d1 + d2) * std::conj(phase_factor);
                    }
                } // xi
            }
        }
    }
}

void
Hubbard::compute_occupancies(K_point<double>& kp__, std::array<dmatrix<double_complex>, 2>& phi_s_psi__,
                             Wave_functions<double>& dphi__, sddk::mdarray<double_complex, 4>& dn__, const int index__)
{
    PROFILE("sirius::Hubbard::compute_occupancies");

    /* overlap between psi_{nk} and dphi */
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
    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        dmatrix<double_complex> psi_s_dphi(kp__.num_occupied_bands(ispn), dphi__.num_wf(), ctx_.mem_pool(memory_t::host));

        if (ctx_.processing_unit() == device_t::GPU) {
          psi_s_dphi.allocate(ctx_.mem_pool(memory_t::device));
           psi_s_dphi.zero(memory_t::device);
        } else {
          psi_s_dphi.zero(memory_t::host);
        }

        /* compute <psi_{ik}^{sigma}|S|dphi> */
        /* dphi don't have a spin index; they are derived from scalar atomic orbitals */
        inner(ctx_.spla_context(), spin_range(ispn), kp__.spinor_wave_functions(), 0, kp__.num_occupied_bands(ispn),
              dphi__, 0, dphi__.num_wf(), psi_s_dphi, 0, 0);

        if (ctx_.processing_unit() == device_t::GPU) {
            psi_s_dphi.copy_to(memory_t::host);
        }

        for (int i = 0; i < this->num_hubbard_wf(); i++) {
            for (int ibnd = 0; ibnd < kp__.num_occupied_bands(ispn); ibnd++) {
                psi_s_dphi(ibnd, i) *= kp__.band_occupancy(ibnd, ispn);
            }
        }

        if (ctx_.processing_unit() == device_t::GPU) {
            psi_s_dphi.copy_to(memory_t::device);
        }

        linalg(la).gemm('C', 'C', dphi__.num_wf(), dphi__.num_wf(),
                        kp__.num_occupied_bands(ispn), &alpha,
                        psi_s_dphi.at(mt), psi_s_dphi.ld(),
                        phi_s_psi__[ispn].at(mt, 0, 0), phi_s_psi__[ispn].ld(),
                        &linalg_const<double_complex>::zero(), dn__.at(mt, 0, 0, ispn, index__), dn__.ld());

        linalg(la).gemm('N', 'N', dphi__.num_wf(), dphi__.num_wf(),
                        kp__.num_occupied_bands(ispn), &alpha,
                        phi_s_psi__[ispn].at(mt, 0, 0), phi_s_psi__[ispn].ld(),
                        psi_s_dphi.at(mt), psi_s_dphi.ld(), &linalg_const<double_complex>::one(),
                        dn__.at(mt, 0, 0, ispn, index__), dn__.ld());

        if (ctx_.processing_unit() == device_t::GPU) {
            dn__.copy_to(memory_t::host);
        }
    }
}

} // namespace sirius
