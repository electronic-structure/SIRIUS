// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

/** \file get_h_o_diag.hpp
 *
 *  \brief Get diagonal values of Hamiltonian and overlap matrices.
 */

inline mdarray<double, 2>
Hamiltonian::get_h_diag(K_point* kp__,
                        double   v0__,
                        double   theta0__) const
{
    PROFILE("sirius::Hamiltonian::get_h_diag");

    // TODO: code is replicated in o_diag
    splindex<block> spl_num_atoms(unit_cell_.num_atoms(), kp__->comm().size(), kp__->comm().rank());
    int nlo{0};
    for (int ialoc = 0; ialoc < spl_num_atoms.local_size(); ialoc++) {
        int ia = spl_num_atoms[ialoc];
        nlo += unit_cell_.atom(ia).mt_lo_basis_size();
    }

    mdarray<double, 2> h_diag(kp__->num_gkvec_loc() + nlo, 1);
    for (int igloc = 0; igloc < kp__->num_gkvec_loc(); igloc++) {
        auto gvc = kp__->gkvec().gkvec_cart<index_domain_t::local>(igloc);

        double ekin = 0.5 * dot(gvc, gvc);
        h_diag[igloc] = v0__ + ekin * theta0__;
    }

    matrix<double_complex> alm(kp__->num_gkvec_loc(), unit_cell_.max_mt_aw_basis_size());
    matrix<double_complex> halm(kp__->num_gkvec_loc(), unit_cell_.max_mt_aw_basis_size());

    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        auto& atom = unit_cell_.atom(ia);
        int nmt = atom.mt_aw_basis_size();

        kp__->alm_coeffs_loc().generate(ia, alm);
        apply_hmt_to_apw<spin_block_t::nm>(atom, kp__->num_gkvec_loc(), alm, halm);

        for (int xi = 0; xi < nmt; xi++) {
            for (int igloc = 0; igloc < kp__->num_gkvec_loc(); igloc++) {
                h_diag[igloc] += std::real(std::conj(alm(igloc, xi)) * halm(igloc, xi));
            }
        }
    }

    nlo = 0;
    for (int ialoc = 0; ialoc < spl_num_atoms.local_size(); ialoc++) {
        int ia = spl_num_atoms[ialoc];
        auto& atom = unit_cell_.atom(ia);
        auto& type = atom.type();
        for (int ilo = 0; ilo < type.mt_lo_basis_size(); ilo++) {
            int xi_lo = type.mt_aw_basis_size() + ilo;
            /* local orbital indices */
            int lm_lo    = type.indexb(xi_lo).lm;
            int idxrf_lo = type.indexb(xi_lo).idxrf;

            h_diag[kp__->num_gkvec_loc() + nlo] = atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf_lo, idxrf_lo, gaunt_coefs_->gaunt_vector(lm_lo, lm_lo)).real();
            nlo++;
        }
    }
    if (ctx_.processing_unit() == device_t::GPU) {
        h_diag.allocate(memory_t::device).copy_to(memory_t::device);
    }
    return std::move(h_diag);
}

inline mdarray<double, 1>
Hamiltonian::get_o_diag(K_point* kp__,
                        double   theta0__) const
{
    PROFILE("sirius::Hamiltonian::get_o_diag");

    splindex<block> spl_num_atoms(unit_cell_.num_atoms(), kp__->comm().size(), kp__->comm().rank());
    int nlo{0};
    for (int ialoc = 0; ialoc < spl_num_atoms.local_size(); ialoc++) {
        int ia = spl_num_atoms[ialoc];
        nlo += unit_cell_.atom(ia).mt_lo_basis_size();
    }

    mdarray<double, 1> o_diag(kp__->num_gkvec_loc() + nlo);
    for (int igloc = 0; igloc < kp__->num_gkvec_loc(); igloc++) {
        o_diag[igloc] = theta0__;
    }
    for (size_t i = kp__->num_gkvec_loc(); i < o_diag.size(); i++) {
        o_diag[i] = 1;
    }

    matrix<double_complex> alm(kp__->num_gkvec_loc(), unit_cell_.max_mt_aw_basis_size());

    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        auto& atom = unit_cell_.atom(ia);
        int nmt = atom.mt_aw_basis_size();

        kp__->alm_coeffs_loc().generate(ia, alm);

        for (int xi = 0; xi < nmt; xi++) {
            for (int igloc = 0; igloc < kp__->num_gkvec_loc(); igloc++) {
                o_diag[igloc] += std::real(std::conj(alm(igloc, xi)) * alm(igloc, xi));
            }
        }
    }
    if (ctx_.processing_unit() == device_t::GPU) {
        o_diag.allocate(memory_t::device).copy_to(memory_t::device);
    }
    return std::move(o_diag);
}

template <typename T>
inline mdarray<double, 2>
Hamiltonian::get_h_diag(K_point* kp__) const
{
    PROFILE("sirius::Hamiltonian::get_h_diag");

    mdarray<double, 2> h_diag(kp__->num_gkvec_loc(), ctx_.num_spins());

    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {

        /* local H contribution */
        for (int ig_loc = 0; ig_loc < kp__->num_gkvec_loc(); ig_loc++) {
            auto vgk = kp__->gkvec().gkvec_cart<index_domain_t::local>(ig_loc);
            h_diag(ig_loc, ispn) = 0.5 * dot(vgk, vgk) + this->local_op().v0(ispn);
        }

        /* non-local H contribution */
        auto beta_gk_t = kp__->beta_projectors().pw_coeffs_t(0);
        matrix<double_complex> beta_gk_tmp(unit_cell_.max_mt_basis_size(), kp__->num_gkvec_loc());

        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            auto& atom_type = unit_cell_.atom_type(iat);
            int nbf = atom_type.mt_basis_size();
            matrix<double_complex> d_sum(nbf, nbf);
            d_sum.zero();

            for (int i = 0; i < atom_type.num_atoms(); i++) {
                int ia = atom_type.atom_id(i);

                for (int xi2 = 0; xi2 < nbf; xi2++) {
                    for (int xi1 = 0; xi1 < nbf; xi1++) {
                        d_sum(xi1, xi2) += this->D<T>().operator()(xi1, xi2, ispn, ia);
                    }
                }
            }

            int offs = unit_cell_.atom_type(iat).offset_lo();
            for (int ig_loc = 0; ig_loc < kp__->num_gkvec_loc(); ig_loc++) {
                for (int xi = 0; xi < nbf; xi++) {
                    beta_gk_tmp(xi, ig_loc) = beta_gk_t(ig_loc, offs + xi);
                }
            }

            #pragma omp parallel for schedule(static)
            for (int ig_loc = 0; ig_loc < kp__->num_gkvec_loc(); ig_loc++) {
                for (int xi2 = 0; xi2 < nbf; xi2++) {
                    for (int xi1 = 0; xi1 < nbf; xi1++) {
                        /* compute <G+k|beta_xi1> D_{xi1, xi2} <beta_xi2|G+k> contribution from all atoms */
                        auto z = beta_gk_tmp(xi1, ig_loc) * d_sum(xi1, xi2) * std::conj(beta_gk_tmp(xi2, ig_loc));
                        h_diag(ig_loc, ispn) += z.real();
                    }
                }
            }
        }
    }
    if (ctx_.processing_unit() == device_t::GPU) {
        h_diag.allocate(memory_t::device).copy_to(memory_t::device);
    }
    return std::move(h_diag);
}

template <typename T>
inline mdarray<double, 1>
Hamiltonian::get_o_diag(K_point* kp__) const
{
    PROFILE("sirius::Hamiltonian::get_o_diag");

    mdarray<double, 1> o_diag(kp__->num_gkvec_loc());
    for (int ig = 0; ig < kp__->num_gkvec_loc(); ig++) {
        o_diag[ig] = 1;
    }

    /* non-local O contribution */
    auto beta_gk_t = kp__->beta_projectors().pw_coeffs_t(0);
    matrix<double_complex> beta_gk_tmp(unit_cell_.max_mt_basis_size(), kp__->num_gkvec_loc());

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        auto& atom_type = unit_cell_.atom_type(iat);
        if (!atom_type.augment()) {
            continue;
        }

        int nbf = atom_type.mt_basis_size();

        matrix<double_complex> q_sum(nbf, nbf);
        q_sum.zero();

        for (int i = 0; i < atom_type.num_atoms(); i++) {
            int ia = atom_type.atom_id(i);

            for (int xi2 = 0; xi2 < nbf; xi2++) {
                for (int xi1 = 0; xi1 < nbf; xi1++) {
                    q_sum(xi1, xi2) += Q<T>().operator()(xi1, xi2, ia);
                }
            }
        }

        int offs = unit_cell_.atom_type(iat).offset_lo();
        for (int ig_loc = 0; ig_loc < kp__->num_gkvec_loc(); ig_loc++) {
            for (int xi = 0; xi < nbf; xi++) {
                beta_gk_tmp(xi, ig_loc) = beta_gk_t(ig_loc, offs + xi);
            }
        }

        #pragma omp parallel for
        for (int ig_loc = 0; ig_loc < kp__->num_gkvec_loc(); ig_loc++) {
            for (int xi2 = 0; xi2 < nbf; xi2++) {
                for (int xi1 = 0; xi1 < nbf; xi1++) {
                    /* compute <G+k|beta_xi1> Q_{xi1, xi2} <beta_xi2|G+k> contribution from all atoms */
                    auto z = beta_gk_tmp(xi1, ig_loc) * q_sum(xi1, xi2) * std::conj(beta_gk_tmp(xi2, ig_loc));
                    o_diag[ig_loc] += z.real();
                }
            }
        }
    }
    if (ctx_.processing_unit() == device_t::GPU) {
        o_diag.allocate(memory_t::device).copy_to(memory_t::device);
    }
    return std::move(o_diag);
}
