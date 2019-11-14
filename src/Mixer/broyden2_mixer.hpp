// Copyright (c) 2013-2019 Simon Frasch, Anton Kozhevnikov, Thomas Schulthess
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

/** \file broyden2_mixer.hpp
 *
 *   \brief Contains definition and implementation of sirius::Broyden2.
 */

#ifndef __BROYDEN2_MIXER_HPP__
#define __BROYDEN2_MIXER_HPP__

#include <tuple>
#include <functional>
#include <utility>
#include <vector>
#include <limits>
#include <memory>
#include <exception>
#include <cmath>
#include <numeric>

#include "SDDK/memory.hpp"
#include "Mixer/mixer.hpp"

namespace sirius {
namespace mixer {

/// Broyden mixer.
/** Second version of the Broyden mixer, which doesn't requre inversion of the Jacobian matrix.
 *  Reference paper: "Robust acceleration of self consistent field calculations for
 *  density functional theory", Baarman K, Eirola T, Havu V., J Chem Phys. 134, 134109 (2011)
 */
template <typename... FUNCS>
class Broyden2 : public Mixer<FUNCS...>
{
  public:
    Broyden2(std::size_t max_history, double beta, double beta0, double beta_scaling_factor, double linear_mix_rmse_tol)
        : Mixer<FUNCS...>(max_history)
        , beta_(beta)
        , beta0_(beta0)
        , beta_scaling_factor_(beta_scaling_factor)
        , linear_mix_rmse_tol_(linear_mix_rmse_tol)
    {
    }

    void mix_impl() override
    {
        const auto idx_step      = this->idx_hist(this->step_);
        const auto idx_next_step = this->idx_hist(this->step_ + 1);

        const auto history_size = std::min(this->step_, this->max_history_);

        // beta scaling
        if (this->step_ > this->max_history_) {
            const double rmse_avg = std::accumulate(this->rmse_history_.begin(), this->rmse_history_.end(), 0.0) /
                                    this->rmse_history_.size();
            if (this->rmse_history_[idx_step] > rmse_avg) {
                this->beta_ = std::max(beta0_, this->beta_ * beta_scaling_factor_);
            }
        }

        const double rmse = this->rmse_history_[idx_step];

        const bool normalize = false;

        if ((history_size > 1 && rmse < this->linear_mix_rmse_tol_ && this->linear_mix_rmse_tol_ > 0) ||
            (this->linear_mix_rmse_tol_ <= 0 && this->step_ > this->max_history_)) {
            sddk::mdarray<double, 2> S(history_size, history_size);
            S.zero();
            for (int j1 = 0; j1 < static_cast<int>(history_size); j1++) {
                int i1 = this->idx_hist(this->step_ - history_size + j1);
                for (int j2 = 0; j2 <= j1; j2++) {
                    int i2 = this->idx_hist(this->step_ - history_size + j2);

                    S(j2, j1) = S(j1, j2) =
                        this->template inner_product<normalize>(this->residual_history_[i1], this->residual_history_[i2]);
                }
            }

            sddk::mdarray<long double, 2> gamma_k(2 * history_size, history_size);
            gamma_k.zero();
            /* initial gamma_0 */
            for (int i = 0; i < static_cast<int>(history_size); i++) {
                gamma_k(i, i) = 0.25;
            }

            std::vector<long double> v1(history_size);
            std::vector<long double> v2(2 * history_size);

            /* update gamma_k by recursion */
            for (int k = 0; k < static_cast<int>(history_size) - 1; k++) {
                /* denominator df_k^{T} S df_k */
                long double d = S(k, k) + S(k + 1, k + 1) - S(k, k + 1) - S(k + 1, k);
                /* nominator */
                std::memset(&v1[0], 0, history_size * sizeof(long double));
                for (int j = 0; j < static_cast<int>(history_size); j++) {
                    v1[j] = S(k + 1, j) - S(k, j);
                }

                std::memset(&v2[0], 0, 2 * history_size * sizeof(long double));
                for (int j = 0; j < static_cast<int>(2 * history_size); j++) {
                    v2[j] = -(gamma_k(j, k + 1) - gamma_k(j, k));
                }
                v2[history_size + k] -= 1;
                v2[history_size + k + 1] += 1;

                for (int j1 = 0; j1 < static_cast<int>(history_size); j1++) {
                    for (int j2 = 0; j2 < static_cast<int>(2 * history_size); j2++) {
                        gamma_k(j2, j1) += v2[j2] * v1[j1] / d;
                    }
                }
            }

            std::memset(&v2[0], 0, 2 * history_size * sizeof(long double));
            for (int j = 0; j < static_cast<int>(2 * history_size); j++) {
                v2[j] = -gamma_k(j, static_cast<int>(history_size - 1));
            }
            v2[2 * history_size - 1] += 1;

            // set input to 0, to use as buffer
            this->scale(0.0, this->input_);

            /* make linear combination of vectors and residuals; this is the update vector \tilda x */
            for (int j = 0; j < static_cast<int>(history_size); j++) {
                int i1 = this->idx_hist(this->step_ - history_size + j);
                this->axpy(v2[j], this->residual_history_[i1], this->input_);
                this->axpy(v2[j + history_size], this->output_history_[i1], this->input_);
            }
        }
        this->copy(this->input_, this->output_history_[idx_next_step]);
        this->scale(beta_, this->output_history_[idx_next_step]);
        this->axpy(1.0 - beta_, this->output_history_[idx_step], this->output_history_[idx_next_step]);
    }

  private:
    double beta_;
    double beta0_;
    double beta_scaling_factor_;
    double linear_mix_rmse_tol_;
};
} // namespace mixer
} // namespace sirius

#endif // __MIXER_HPP__
