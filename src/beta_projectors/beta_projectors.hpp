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
class Beta_projectors : public Beta_projectors_base
{
  protected:
    bool prepared_{false};
    matrix<double_complex> beta_pw_all_atoms_;
    /// Generate plane-wave coefficients for beta-projectors of atom types.
    void generate_pw_coefs_t(std::vector<int>& igk__)
    {
        PROFILE("sirius::Beta_projectors::generate_pw_coefs_t");
        if (!num_beta_t()) {
            return;
        }

        auto& comm = gkvec_.comm();

        auto& beta_radial_integrals = ctx_.beta_ri();

        std::vector<double_complex> z(ctx_.unit_cell().lmax() + 1);
        for (int l = 0; l <= ctx_.unit_cell().lmax(); l++) {
            z[l] = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(ctx_.unit_cell().omega());
        }

        /* compute <G+k|beta> */
        #pragma omp parallel for
        for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++) {
            int igk = igk__[igkloc];
            /* vs = {r, theta, phi} */
            auto vs = SHT::spherical_coordinates(gkvec_.gkvec_cart<index_domain_t::global>(igk));
            /* compute real spherical harmonics for G+k vector */
            std::vector<double> gkvec_rlm(utils::lmmax(ctx_.unit_cell().lmax()));
            sf::spherical_harmonics(ctx_.unit_cell().lmax(), vs[1], vs[2], &gkvec_rlm[0]);
            for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
                auto& atom_type = ctx_.unit_cell().atom_type(iat);
                /* get all values of radial integrals */
                auto ri_val = beta_radial_integrals.values(iat, vs[0]);
                for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
                    int l     = atom_type.indexb(xi).l;
                    int lm    = atom_type.indexb(xi).lm;
                    int idxrf = atom_type.indexb(xi).idxrf;

                    pw_coeffs_t_(igkloc, atom_type.offset_lo() + xi, 0) = z[l] * gkvec_rlm[lm] * ri_val(idxrf);
                }
            }
        }

        if (ctx_.control().print_checksum_) {
            auto c1 = pw_coeffs_t_.checksum();
            comm.allreduce(&c1, 1);
            if (comm.rank() == 0) {
                utils::print_checksum("beta_pw_coeffs_t", c1);
            }
        }

    }

  public:
    Beta_projectors(Simulation_context& ctx__, Gvec const& gkvec__, std::vector<int>& igk__)
        : Beta_projectors_base(ctx__, gkvec__, igk__, 1)
    {
        PROFILE("sirius::Beta_projectors");
        /* generate phase-factor independent projectors for atom types */
        generate_pw_coefs_t(igk__);
        /* special treatment for beta-projectors as they are mostly often used */
        switch (ctx_.processing_unit()) {
            /* beta projectors for atom types will be stored on GPU for the entire run */
            case device_t::GPU: {
                reallocate_pw_coeffs_t_on_gpu_ = false;
                pw_coeffs_t_.allocate(memory_t::device).copy_to(memory_t::device);
                break;
            }
            /* generate beta projectors for all atoms */
            case device_t::CPU: {
                beta_pw_all_atoms_ = matrix<double_complex>(num_gkvec_loc(), ctx_.unit_cell().mt_lo_basis_size());
                for (int ichunk = 0; ichunk < num_chunks(); ichunk++) {
                    /* wrap the the pointer in the big array beta_pw_all_atoms */
                    pw_coeffs_a_ = matrix<double_complex>(&beta_pw_all_atoms_(0, chunk(ichunk).offset_),
                                                          num_gkvec_loc(), chunk(ichunk).num_beta_);
                    Beta_projectors_base::generate(ichunk, 0);
                }
                break;
            }
        }
    }

    inline void prepare()
    {
        if (prepared_) {
            TERMINATE("beta projectors are already prepared");
        }
        switch (ctx_.processing_unit()) {
            case device_t::GPU: {
                Beta_projectors_base::prepare();
                break;
            }
            case device_t::CPU: break;
        }
        prepared_ = true;
    }

    inline void dismiss()
    {
        if (!prepared_) {
            TERMINATE("beta projectors are already dismissed");
        }
        switch (ctx_.processing_unit()) {
            case device_t::GPU: {
                Beta_projectors_base::dismiss();
                break;
            }
            case device_t::CPU: break;
        }
        prepared_ = false;
    }

    void generate(int chunk__)
    {
        switch (ctx_.processing_unit()) {
            case device_t::CPU: {
                pw_coeffs_a_ = matrix<double_complex>(&beta_pw_all_atoms_(0, chunk(chunk__).offset_),
                                                      num_gkvec_loc(), chunk(chunk__).num_beta_);
                break;
            }
            case device_t::GPU: {
                Beta_projectors_base::generate(chunk__, 0);
                break;
            }
        }
    }
};

} // namespace sirius

#endif
