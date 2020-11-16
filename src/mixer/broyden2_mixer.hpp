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
#include "mixer/mixer.hpp"

namespace sirius {
namespace mixer {

/// Broyden mixer.
/** Second version of the Broyden mixer, which doesn't require inversion of the Jacobian matrix.
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
        , S_(max_history, max_history)
        , gamma_(max_history)
    {
    }

    void mix_impl() override
    {
        const auto idx_step      = this->idx_hist(this->step_);
        const auto idx_next_step = this->idx_hist(this->step_ + 1);

        const auto n = static_cast<int>(std::min(this->step_, this->max_history_ - 1));

        const bool normalize = false;

        for (int i = 0; i <= n; ++i) {
            int j = this->idx_hist(this->step_ - i);
            this->S_(n - i, n) = this->S_(n, n - i) = this->template inner_product<normalize>(
                this->residual_history_[j],
                this->residual_history_[idx_step]
            );
        }

        // Expand (I - Δf₁Δf₁ᵀ/Δf₁ᵀΔf₁)...(I - Δfₙ₋₁Δfₙ₋₁ᵀ/Δfₙ₋₁ᵀΔfₙ₋₁)fₙ
        // to γ₁Δf₁ + ... + γₙ₋₁Δfₙ₋₁ + fₙ
        // the denominator ΔfᵢᵀΔfᵢ is constant, so apply only at the end
        for (int i = 1; i <= n; ++i) {
            // compute -Δfᵢᵀfₙ
            // = -(fᵢ₊₁ᵀfₙ - fᵢᵀfₙ)
            this->gamma_(n - i) = this->S_(n - i, n) - this->S_(n - i + 1, n);

            for (int j = 1; j < i; ++j) {
                // compute -ΔfᵢᵀΔfⱼ
                // = -(fᵢ₊₁ - fᵢ)(fⱼ₊₁ - fⱼ) 
                // = -fᵢ₊₁ᵀfⱼ₊₁ + fᵢᵀfⱼ₊₁ + fᵢ₊₁ᵀfⱼ - fᵢᵀfⱼ 
                this->gamma_(n - i) += (-this->S_(n - i + 1, n - j + 1) + this->S_(n - i + 1, n - j) + this->S_(n - i, n - j + 1) - this->S_(n - i, n - j)) * this->gamma_(n - j);
            }

            this->gamma_(n - i) /= this->S_(n - i + 1, n - i + 1) - this->S_(n - i + 1, n - i) - this->S_(n - i, n - i + 1) + this->S_(n - i, n - i);
        }

        this->copy(this->output_history_[idx_step], this->input_);

        if (n > 0) {
            // first vec is special
            {
                int j = this->idx_hist(this->step_ - n);
                this->axpy(-this->beta_ * this->gamma_(0), this->residual_history_[j], this->input_);
                this->axpy(-this->gamma_(0), this->output_history_[j], this->input_);
            }

            for (int i = 1; i < n; ++i) {
                auto coeff = this->gamma_(n - i - 1) - this->gamma_(n - i);
                int j = this->idx_hist(this->step_ - i);
                this->axpy(this->beta_ * coeff, this->residual_history_[j], this->input_);
                this->axpy(coeff, this->output_history_[j], this->input_);
            }

            // last vec is special.
            {
                int j = this->idx_hist(this->step_);
                this->axpy(this->beta_ * (this->gamma_(n - 1) + 1), this->residual_history_[j], this->input_);
                this->axpy(this->gamma_(n - 1), this->output_history_[j], this->input_);
            }
        } else {
            // Linear mixing step.
            this->axpy(this->beta_, this->residual_history_[idx_step], this->input_);
        }

        this->copy(this->input_, this->output_history_[idx_next_step]);

        if (n == static_cast<int>(this->max_history_) - 1) {
            for (int col = 0; col < n; ++col) {
                for (int row = 0; row < n; ++row) {
                    this->S_(row, col) = this->S_(row + 1, col + 1);
                }
            }
        }
    }

  private:
    double beta_;
    double beta0_;
    double beta_scaling_factor_;
    double linear_mix_rmse_tol_;
    sddk::mdarray<double, 2> S_;
    sddk::mdarray<double, 1> gamma_;
};
} // namespace mixer
} // namespace sirius

#endif // __MIXER_HPP__
