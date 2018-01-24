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

/** \file beta_projectors.h
 *
 *  \brief Contains declaration and implementation of sirius::Beta_projectors class.
 */

#ifndef __BETA_PROJECTORS_H__
#define __BETA_PROJECTORS_H__

#include "communicator.hpp"
#include "Unit_cell/unit_cell.h"
#include "wave_functions.hpp"
#include "simulation_context.h"
#include "beta_projectors_base.h"

namespace sirius {

/// Stores <G+k | beta> expansion
class Beta_projectors: public Beta_projectors_base<1>
{
    protected:

        /// Generate plane-wave coefficients for beta-projectors of atom types.
        void generate_pw_coefs_t(std::vector<int>& igk__)
        {
            PROFILE("sirius::Beta_projectors::generate_pw_coefs_t");
            auto& bchunk = ctx_.beta_projector_chunks();
            if (!bchunk.num_beta_t()) {
                return;
            }
            
            auto& comm = gkvec_.comm();

            auto& beta_radial_integrals = ctx_.beta_ri();

            /* compute <G+k|beta> */
            #pragma omp parallel for
            for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++) {
                int igk = igk__[igkloc];
                /* vs = {r, theta, phi} */
                auto vs = SHT::spherical_coordinates(gkvec_.gkvec_cart(igk));
                /* compute real spherical harmonics for G+k vector */
                std::vector<double> gkvec_rlm(Utils::lmmax(lmax_beta_));
                SHT::spherical_harmonics(lmax_beta_, vs[1], vs[2], &gkvec_rlm[0]);
                for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
                    auto& atom_type = ctx_.unit_cell().atom_type(iat);
                    for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
                        int l     = atom_type.indexb(xi).l;
                        int lm    = atom_type.indexb(xi).lm;
                        int idxrf = atom_type.indexb(xi).idxrf;

                        auto z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(ctx_.unit_cell().omega());
                        pw_coeffs_t_[0](igkloc, atom_type.offset_lo() + xi) = z * gkvec_rlm[lm] * beta_radial_integrals.value<int, int>(idxrf, iat, vs[0]);
                    }
                }
            }

            if (ctx_.control().print_checksum_) {
                auto c1 = pw_coeffs_t_[0].checksum();
                comm.allreduce(&c1, 1);
                if (comm.rank() == 0) {
                    DUMP("checksum(beta_pw_coeffs_t) : %18.10f %18.10f", c1.real(), c1.imag())
                }
            }

            //if (ctx_.processing_unit() == GPU) {
            //    pw_coeffs_t_[0].copy<memory_t::host, memory_t::device>();
            //}
        }
                    
    public:

        Beta_projectors(Simulation_context& ctx__,
                        Gvec const&         gkvec__,
                        std::vector<int>&   igk__)
            : Beta_projectors_base<1>(ctx__, gkvec__, igk__)
        {
            PROFILE("sirius::Beta_projectors::Beta_projectors");
            generate_pw_coefs_t(igk__);
        }

        void generate(int chunk__)
        {
            Beta_projectors_base<1>::generate(chunk__, 0);
        }

        inline Beta_projector_chunks const& beta_projector_chunks() const
        {
            return ctx_.beta_projector_chunks();
        }
};

} // namespace

#endif
