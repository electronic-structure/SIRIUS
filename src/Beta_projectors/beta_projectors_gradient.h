// Copyright (c) 2013-2017 Anton Kozhevnikov, Ilia Sivkov, Thomas Schulthess
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

/** \file beta_projectors_gradient.h
 *
 *  \brief Contains declaration and implementation of sirius::Beta_projectors_gradient class.
 */

#ifndef __BETA_PROJECTORS_GRADIENT_H__
#define __BETA_PROJECTORS_GRADIENT_H__

#include "beta_projectors_base.h"
#include "beta_projectors.h"

namespace sirius {

/// Compute gradient of beta-projectors over atomic positions \f$ d \langle {\bf G+k} | \beta \rangle / d \tau_{\alpha} \f$.
class Beta_projectors_gradient: public Beta_projectors_base<3>
{
  private:
    void generate_pw_coefs_t(Beta_projectors& beta__)
    {
        auto& bchunk = ctx_.beta_projector_chunks();

        if (!bchunk.num_beta_t()) {
            return;
        }
        
        auto& comm = gkvec_.comm();

        for (int x = 0; x < 3; x++) {
            #pragma omp parallel for
            for (int i = 0; i < bchunk.num_beta_t(); i++) {
                for (int igkloc = 0; igkloc < gkvec_.gvec_count(comm.rank()); igkloc++) {
                    int igk = gkvec_.gvec_offset(comm.rank()) + igkloc;
                    auto vgc = gkvec_.gkvec_cart(igk);
                    pw_coeffs_t_[x](igkloc, i) = double_complex(0, -vgc[x]) * beta__.pw_coeffs_t(0)(igkloc, i);
                }
            }
        }
        if (ctx_.processing_unit() == GPU) {
            for (int x = 0; x < 3; x++) {
                pw_coeffs_t_[x].copy<memory_t::host, memory_t::device>();
            }
        }
    }

  public:
    Beta_projectors_gradient(Simulation_context& ctx__,
                             Gvec const&         gkvec__,
                             Beta_projectors&    beta__)
        : Beta_projectors_base<3>(ctx__, gkvec__)
    {
        generate_pw_coefs_t(beta__);
    }
};

}

#endif // __BETA_PROJECTORS_GRADIENT_H__
