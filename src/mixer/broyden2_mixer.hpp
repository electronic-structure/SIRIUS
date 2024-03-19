/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

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

#include "mixer/mixer.hpp"
#include "core/memory.hpp"

namespace sirius {
namespace mixer {

/// Broyden mixer.
/**
 * Quasi-Newton, limited-memory method which updates \f$ x_{n+1} = x_n - G_nf_n \f$
 * where \f$ G_n \f$ is an approximate inverse Jacobian. Broyden 2 is derived
 * by recursively taking rank-1 updates to the inverse Jacobian.
 * It requires the secant equation \f$ G_{n+1}\Delta f_n = \Delta x_n\f$ to hold for the next approximate Jacobian
 * \f$ G_{n+1}\f$, leading to the update:
 * \f[
 *     G_{n+1} := G_n + (\Delta x_n - G_n \Delta f_n)(\Delta f_n^*\Delta f_n)^{-1}\Delta f_n^*.
 * \f]
 * By induction we can show that \f$ G_n \f$ satisfies
 * \f[
 * G_n = G_1\prod_{i=1}^{n-1}\left(I - \frac{\Delta f_i \Delta f_i^*}{\Delta f_i^*\Delta f_i}\right)
 *       + \sum_{i=1}^{n-1}\frac{\Delta x_i \Delta f_i^*}{\Delta f_i^*\Delta f_i}\prod_{j=i+1}^{n-1}
 *         \left(I - \frac{\Delta f_j \Delta f_j^*}{\Delta f_j^*\Delta f_j}\right)
 * \f]
 * which shows orthogonalization with respect to the \f$ \Delta f_i \f$ vectors. A practical
 * implementation can be derived from the identity
 * \f[
 *     G_n = G_1 + \sum_{i=1}^{n-1}\left(\Delta x_i - G_1\Delta f_i\right)\gamma_{i,n}^*
 * \f]
 * where
 * \f[
 *     \gamma_{i,n}^* = \frac{\Delta f_i^*}{\Delta f_i^*\Delta f_i}\prod_{j=i+1}^{n-1}
 *     \left(I - \frac{\Delta f_j\Delta f_j^*}{\Delta f_j^*\Delta f_j}\right).
 * \f]
 * The \f$ \gamma_{i,n}\f$ vector satisfies the following recursion relation:
 * \f[
 *     \gamma_{i,n}^* = \frac{\Delta f_i^*}{\Delta f_i^*\Delta f_i}\left(I - \sum_{j=i+1}^{n-1}
 *     \Delta f_j \gamma_{j,n}^*\right)
 * \f]
 * When updating \f$ x_{n+1} = x_n - G_nf_n\f$ we only have to compute:
 * \f[
 *     \alpha_i := \gamma_i^*f_n = \frac{1}{\Delta f_i^*\Delta f_i}\left[\Delta f_i^*f_n
 *     - \sum_{j=i+1}^{n-1}\alpha_j\Delta f_i^* \Delta f_j \right]
 * \f]
 * for \f$ i \f$  from \f$ n-1\f$ down to \f$ 1 \f$ so that
 * \f[\begin{aligned}
 *     x_{n+1} &= x_n - G_nf_n \\
 *             &= x_n - G_1f_n + \sum_{i=1}^{n-1}\alpha_iG_1\Delta f_i
 *                - \sum_{i=1}^{n-1}\alpha_i\Delta x_i.
 * \end{aligned}\f]
 * In particular when \f$ G_1 = -\beta I\f$ we get the following update:
 * \f[
 *     x_{n+1} = x_n + \beta f_n - \sum_{i=1}^{n-1}\alpha_i \beta \Delta f_i - \sum_{i=1}^{n-1}\alpha_i\Delta x_i.
 * \f]
 * Finally, we store the vectors \f$ f_1, \cdots, f_n \f$ and \f$ x_1, \dots, x_n \f$ and update the Gram-matrix
 * \f$ S_{ij} = f_i^*f_j \f$ in every iteration. The \f$ \alpha_i \f$ coefficients can be easily computed from
 * \f$ S \f$.
 *
 * \note
 * This class does not do anything to improve the stability of the Gram-Schmidt procedure (clearly visible in the
 * first explicit expression for \f$ G_n \f$) and can therefore be very unstable for larger history sizes.
 */
template <typename... FUNCS>
class Broyden2 : public Mixer<FUNCS...>
{
  private:
    double beta_;
    mdarray<double, 2> S_;
    mdarray<double, 1> gamma_;

  public:
    Broyden2(std::size_t max_history, double beta)
        : Mixer<FUNCS...>(max_history)
        , beta_(beta)
        , S_({max_history, max_history})
        , gamma_({max_history})
    {
    }

    void
    mix_impl() override
    {
        const auto idx_step      = this->idx_hist(this->step_);
        const auto idx_next_step = this->idx_hist(this->step_ + 1);

        const auto n = static_cast<int>(std::min(this->step_, this->max_history_ - 1));

        const bool normalize = false;

        for (int i = 0; i <= n; ++i) {
            int j              = this->idx_hist(this->step_ - i);
            this->S_(n - i, n) = this->S_(n, n - i) = this->template inner_product<normalize>(
                    this->residual_history_[j], this->residual_history_[idx_step]);
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
                this->gamma_(n - i) += (-this->S_(n - i + 1, n - j + 1) + this->S_(n - i + 1, n - j) +
                                        this->S_(n - i, n - j + 1) - this->S_(n - i, n - j)) *
                                       this->gamma_(n - j);
            }

            this->gamma_(n - i) /= this->S_(n - i + 1, n - i + 1) - this->S_(n - i + 1, n - i) -
                                   this->S_(n - i, n - i + 1) + this->S_(n - i, n - i);
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
                int j      = this->idx_hist(this->step_ - i);
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
};

} // namespace mixer
} // namespace sirius

#endif // __MIXER_HPP__
