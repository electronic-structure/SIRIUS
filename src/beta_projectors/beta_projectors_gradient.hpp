/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file beta_projectors_gradient.hpp
 *
 *  \brief Contains declaration and implementation of sirius::Beta_projectors_gradient class.
 */

#ifndef __BETA_PROJECTORS_GRADIENT_HPP__
#define __BETA_PROJECTORS_GRADIENT_HPP__

#include "beta_projectors_base.hpp"
#include "beta_projectors.hpp"

namespace sirius {

/// Compute gradient of beta-projectors over atomic positions \f$ d \langle {\bf G+k} | \beta \rangle / d \tau_{\alpha}
/// \f$.
template <typename T>
class Beta_projectors_gradient : public Beta_projectors_base<T>
{
  private:
    void
    generate_pw_coefs_t(Beta_projectors<T>& beta__)
    {
        if (!this->num_beta_t()) {
            return;
        }

        for (int x = 0; x < 3; x++) {
            #pragma omp parallel for
            for (int i = 0; i < this->num_beta_t(); i++) {
                for (int igkloc = 0; igkloc < this->num_gkvec_loc(); igkloc++) {
                    auto vgc                         = this->gkvec_.gkvec_cart(gvec_index_t::local(igkloc));
                    this->pw_coeffs_t_(igkloc, i, x) = std::complex<T>(0, -vgc[x]) * beta__.pw_coeffs_t(igkloc, i, 0);
                }
            }
        }
    }

  public:
    Beta_projectors_gradient(Simulation_context& ctx__, fft::Gvec const& gkvec__, Beta_projectors<T>& beta__)
        : Beta_projectors_base<T>(ctx__, gkvec__, 3)
    {
        generate_pw_coefs_t(beta__);
    }
};

} // namespace sirius

#endif // __BETA_PROJECTORS_GRADIENT_HPP__
