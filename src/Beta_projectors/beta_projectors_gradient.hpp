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

/** \file beta_projectors_gradient.hpp
 *
 *  \brief Contains declaration and implementation of sirius::Beta_projectors_gradient class.
 */

#ifndef __BETA_PROJECTORS_GRADIENT_HPP__
#define __BETA_PROJECTORS_GRADIENT_HPP__

#include "beta_projectors_base.hpp"
#include "beta_projectors.hpp"

namespace sirius {

/// Compute gradient of beta-projectors over atomic positions \f$ d \langle {\bf G+k} | \beta \rangle / d \tau_{\alpha} \f$.
class Beta_projectors_gradient: public Beta_projectors_base
{
  private:
    void generate_pw_coefs_t(Beta_projectors& beta__, std::vector<int> const& igk__)
    {
        if (!num_beta_t()) {
            return;
        }

        for (int x = 0; x < 3; x++) {
            #pragma omp parallel for
            for (int i = 0; i < num_beta_t(); i++) {
                for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++) {
                    int igk = igk__[igkloc];
                    auto vgc = gkvec_.gkvec_cart<index_domain_t::global>(igk);
                    pw_coeffs_t_(igkloc, i, x) = double_complex(0, -vgc[x]) * beta__.pw_coeffs_t(igkloc, i, 0);
                }
            }
        }
    }

  public:
    Beta_projectors_gradient(Simulation_context&     ctx__,
                             Gvec const&             gkvec__,
                             std::vector<int> const& igk__,
                             Beta_projectors&        beta__)
        : Beta_projectors_base(ctx__, gkvec__, igk__, 3)
    {
        generate_pw_coefs_t(beta__, igk__);
    }
};

}

#endif // __BETA_PROJECTORS_GRADIENT_HPP__
