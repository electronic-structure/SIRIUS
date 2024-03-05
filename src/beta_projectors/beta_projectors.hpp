// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

/** \file beta_projectors.hpp
 *
 *  \brief Contains declaration and implementation of sirius::Beta_projectors class.
 */

#ifndef __BETA_PROJECTORS_HPP__
#define __BETA_PROJECTORS_HPP__

#include "beta_projectors_base.hpp"

namespace sirius {

/// Stores <G+k | beta> expansion
template <typename T>
class Beta_projectors : public Beta_projectors_base<T>
{
  protected:
    /// Generate plane-wave coefficients for beta-projectors of atom types.
    void
    generate_pw_coefs_t()
    {
        PROFILE("sirius::Beta_projectors::generate_pw_coefs_t");

        auto& uc = this->ctx_.unit_cell();

        std::vector<int> offset_t(uc.num_atom_types());
        std::generate(offset_t.begin(), offset_t.end(), [n = 0, iat = 0, &uc]() mutable {
            int offs = n;
            n += uc.atom_type(iat++).mt_basis_size();
            return offs;
        });

        auto& comm = this->gkvec_.comm();

        auto& beta_radial_integrals = *this->ctx_.ri().beta_;

        std::vector<std::complex<double>> z(uc.lmax() + 1);
        for (int l = 0; l <= uc.lmax(); l++) {
            z[l] = std::pow(std::complex<double>(0, -1), l) * fourpi / std::sqrt(uc.omega());
        }

        /* compute <G+k|beta> */
        #pragma omp parallel for
        for (int igkloc = 0; igkloc < this->num_gkvec_loc(); igkloc++) {
            /* vs = {r, theta, phi} */
            auto vs = r3::spherical_coordinates(this->gkvec_.gkvec_cart(gvec_index_t::local(igkloc)));
            /* compute real spherical harmonics for G+k vector */
            std::vector<double> gkvec_rlm(sf::lmmax(uc.lmax()));
            sf::spherical_harmonics(uc.lmax(), vs[1], vs[2], &gkvec_rlm[0]);
            for (int iat = 0; iat < uc.num_atom_types(); iat++) {
                auto& atom_type = uc.atom_type(iat);
                /* get all values of radial integrals */
                auto ri_val = beta_radial_integrals.values(iat, vs[0]);
                for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
                    int l     = atom_type.indexb(xi).am.l();
                    int lm    = atom_type.indexb(xi).lm;
                    int idxrf = atom_type.indexb(xi).idxrf;

                    this->pw_coeffs_t_(igkloc, offset_t[atom_type.id()] + xi, 0) =
                            static_cast<std::complex<T>>(z[l] * gkvec_rlm[lm] * ri_val(idxrf));
                }
            }
        }

        if (env::print_checksum()) {
            auto c1 = this->pw_coeffs_t_.checksum();
            comm.allreduce(&c1, 1);
            if (comm.rank() == 0) {
                print_checksum("beta_pw_coeffs_t", c1, std::cout);
            }
        }
    }

  public:
    Beta_projectors(Simulation_context& ctx__, fft::Gvec const& gkvec__)
        : Beta_projectors_base<T>(ctx__, gkvec__, 1)
    {
        /* quick exit */
        if (!this->num_beta_t()) {
            return;
        }

        PROFILE("sirius::Beta_projectors");
        /* generate phase-factor independent projectors for atom types */
        generate_pw_coefs_t();

        if (true) { // current implementation: allocate beta_t(G+k) on GPUs
            if (this->ctx_.processing_unit() == device_t::GPU) {
                this->pw_coeffs_t_.allocate(memory_t::device).copy_to(memory_t::device);
            }
        }

        // TODO: can be improved... nlcglib might ask for beta coefficients on host,
        // create them such that they are there in any case
        if (true) { // current implementation: store all beta-projectors in host memory
            this->pw_coeffs_all_atoms_ = mdarray<std::complex<T>, 3>({this->num_gkvec_loc(), this->num_beta(), 1});
            for (int ichunk = 0; ichunk < this->num_chunks(); ichunk++) {
                /* wrap chunk of beta-projectors */
                matrix<std::complex<T>> tmp({this->num_gkvec_loc(), this->beta_chunks_[ichunk].num_beta_},
                                            &this->pw_coeffs_all_atoms_(0, this->beta_chunks_[ichunk].offset_, 0));
                local::beta_projectors_generate_cpu(tmp, this->pw_coeffs_t_, ichunk, /*j*/ 0,
                                                    this->beta_chunks_[ichunk], ctx__, gkvec__);
            }
            if (ctx__.cfg().control().beta_on_device()) {
                if (this->ctx_.processing_unit() == device_t::GPU) {
                    this->pw_coeffs_all_atoms_.allocate(get_memory_pool(memory_t::device)).copy_to(memory_t::device);
                }
            }
        }
    }
};

} // namespace sirius

#endif
