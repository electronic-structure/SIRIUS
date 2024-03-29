/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file anderson_stable_mixer.hpp
 *
 *   \brief Contains definition and implementation sirius::Anderson_stable.
 */

#ifndef __ANDERSON_STABLE_MIXER_HPP__
#define __ANDERSON_STABLE_MIXER_HPP__

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
#include "core/la/linalg.hpp"

namespace sirius {
namespace mixer {

/// Anderson mixer.
/**
 * Quasi-Newton limited-memory method which updates \f$ x_{n+1} = x_n - G_nf_n \f$
 * where \f$ G_n \f$ is an approximate inverse Jacobian. Anderson is derived
 * by taking the low-rank update to the inverse Jacobian
 *
 * \f[
 * G_{n+1} = (G_n + \Delta X_n - G_n \Delta F_n)(\Delta F_n^T \Delta F_n)^{-1}\Delta F_n^T
 * \f]
 *
 * such that the secant equations \f$ G_{n+1} \Delta F_n = \Delta X_n \f$ are satisfied for previous
 * iterations. Then \f$ G_n \f$ is taken \f$ -\beta I \f$. This implementation uses Gram-Schmidt
 * to orthogonalize \f$ \Delta F_n \f$ to solve the least-squares problem. If stability is
 * not an issue, use Anderson instead.
 *
 * Reference paper: Fang, Haw‐ren, and Yousef Saad. "Two classes of multisecant
 * methods for nonlinear acceleration." Numerical Linear Algebra with Applications
 * 16.3 (2009): 197-221.
 */
template <typename... FUNCS>
class Anderson_stable : public Mixer<FUNCS...>
{
  private:
    double beta_;
    mdarray<double, 2> R_;
    std::size_t history_size_;

  public:
    Anderson_stable(std::size_t max_history, double beta)
        : Mixer<FUNCS...>(max_history)
        , beta_(beta)
        , R_({max_history - 1, max_history - 1})
        , history_size_(0)
    {
        this->R_.zero();
    }

    void
    mix_impl() override
    {
        const auto idx_step      = this->idx_hist(this->step_);
        const auto idx_next_step = this->idx_hist(this->step_ + 1);
        const auto idx_step_prev = this->idx_hist(this->step_ - 1);

        const bool normalize = false;

        const auto history_size = static_cast<int>(this->history_size_);

        // TODO: beta scaling?

        // Set up the next x_{n+1} = x_n.
        // Can't use this->output_history_[idx_step + 1] directly here,
        // as it's still used when history is full.
        this->copy(this->output_history_[idx_step], this->input_);

        // + beta * f_n
        this->axpy(this->beta_, this->residual_history_[idx_step], this->input_);

        if (history_size > 0) {
            // Compute the difference residual[step] - residual[step - 1]
            // and store it in residual[step - 1], but don't destroy
            // residual[step]
            this->scale(-1.0, this->residual_history_[idx_step_prev]);
            this->axpy(1.0, this->residual_history_[idx_step], this->residual_history_[idx_step_prev]);

            // Do the same for difference x
            this->scale(-1.0, this->output_history_[idx_step_prev]);
            this->axpy(1.0, this->output_history_[idx_step], this->output_history_[idx_step_prev]);

            // orthogonalize residual_history_[step-1] w.r.t. residual_history_[1:step-2] using modified Gram-Schmidt.
            for (int i = 1; i <= history_size - 1; ++i) {
                auto j  = this->idx_hist(this->step_ - i - 1);
                auto sz = this->template inner_product<normalize>(this->residual_history_[j],
                                                                  this->residual_history_[idx_step_prev]);
                this->R_(history_size - 1 - i, history_size - 1) = sz;
                this->axpy(-sz, this->residual_history_[j], this->residual_history_[idx_step_prev]);
            }

            // repeat orthogonalization.. seems really necessary.
            for (int i = 1; i <= history_size - 1; ++i) {
                auto j  = this->idx_hist(this->step_ - i - 1);
                auto sz = this->template inner_product<normalize>(this->residual_history_[j],
                                                                  this->residual_history_[idx_step_prev]);
                this->R_(history_size - 1 - i, history_size - 1) += sz;
                this->axpy(-sz, this->residual_history_[j], this->residual_history_[idx_step_prev]);
            }

            // normalize the new residual difference vec itself
            auto nrm2 = this->template inner_product<normalize>(this->residual_history_[idx_step_prev],
                                                                this->residual_history_[idx_step_prev]);

            if (nrm2 > 0) {
                auto sz                                      = std::sqrt(nrm2);
                this->R_(history_size - 1, history_size - 1) = sz;
                this->scale(1.0 / sz, this->residual_history_[idx_step_prev]);

                // Now do the Anderson iteration bit

                // Compute h = Q' * f_n
                mdarray<double, 1> h({history_size});
                for (int i = 1; i <= history_size; ++i) {
                    auto j              = this->idx_hist(this->step_ - i);
                    h(history_size - i) = this->template inner_product<normalize>(this->residual_history_[j],
                                                                                  this->residual_history_[idx_step]);
                }

                // next compute k = R⁻¹ * h... just do that by hand for now, can dispatch to blas later.
                mdarray<double, 1> k({history_size});
                for (int i = 0; i < history_size; ++i) {
                    k[i] = h[i];
                }

                for (int j = history_size - 1; j >= 0; --j) {
                    k(j) /= this->R_(j, j);
                    for (int i = j - 1; i >= 0; --i) {
                        k(i) -= this->R_(i, j) * k(j);
                    }
                }

                // - beta * Q * h
                for (int i = 1; i <= history_size; ++i) {
                    auto j = this->idx_hist(this->step_ - i);
                    this->axpy(-this->beta_ * h(history_size - i), this->residual_history_[j], this->input_);
                }

                // - (delta X) k
                for (int i = 1; i <= history_size; ++i) {
                    auto j = this->idx_hist(this->step_ - i);
                    this->axpy(-k(history_size - i), this->output_history_[j], this->input_);
                }
            } else {
                // In the unlikely event of a breakdown when exactly
                // converged or an inner product that is broken, simply
                // reset the history size to 0 to restart the mixer.
                this->history_size_ = 0;
            }
        }

        // When the history is full, drop the first column.
        // Basically we have delta F = [q1 Q2] * [r11 R12; O R22]
        // and we apply a couple rotations to make [R12; R22] upper triangular again
        // and simultaneously apply the adjoint of the rotations to [q1 Q2].
        // afterwards we drop the first column and last row of the new R, the new
        // Q currently ends up in the first so many columns of delta F, so we have
        // to do some swapping to restore the circular buffer for residual_history_
        if (this->history_size_ == this->max_history_ - 1) {
            // Restore [R12; R22] to upper triangular
            for (int row = 1; row <= history_size - 1; ++row) {
                auto rotation = la::wrap(la::lib_t::lapack).lartg(this->R_(row - 1, row), this->R_(row, row));
                auto c        = std::get<0>(rotation);
                auto s        = std::get<1>(rotation);
                auto nrm      = std::get<2>(rotation);

                // Apply the Given's rotation to the initial column
                this->R_(row - 1, row) = nrm;
                this->R_(row, row)     = 0;

                // Apply Given's rotation to R
                for (int col = row + 1; col < history_size; ++col) {
                    auto r1 = this->R_(row - 1, col);
                    auto r2 = this->R_(row, col);

                    this->R_(row - 1, col) = c * r1 + s * r2;
                    this->R_(row, col)     = -s * r1 + c * r2;
                }

                // Apply the Given's rotation to Q (i.e. orthonormal basis for ΔF)
                int i1 = this->idx_hist(this->step_ - history_size + row - 1);
                int i2 = this->idx_hist(this->step_ - history_size + row);
                this->rotate(c, s, this->residual_history_[i1], this->residual_history_[i2]);
            }

            // Move the columns one place to the right
            for (int i = 1; i <= history_size - 1; ++i) {
                int i1 = this->idx_hist(this->step_ - i - 1);
                int i2 = this->idx_hist(this->step_ - i);
                std::swap(this->residual_history_[i2], this->residual_history_[i1]);
            }

            // Delete last row and first column of R.
            for (int col = 0; col <= history_size - 2; ++col) {
                for (int row = 0; row <= col; ++row) {
                    this->R_(row, col) = this->R_(row, col + 1);
                }
            }
        }

        this->copy(this->input_, this->output_history_[idx_next_step]);
        this->history_size_ = std::min(this->history_size_ + 1, this->max_history_ - 1);
    }
};
} // namespace mixer
} // namespace sirius

#endif // __ANDERSON_STABLE_MIXER_HPP__
