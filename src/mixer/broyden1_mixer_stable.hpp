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

/** \file broyden1_mixer.hpp
 *
 *   \brief Contains definition and implementation sirius::Broyden1.
 */

#ifndef __BROYDEN1_STABLE_MIXER_HPP__
#define __BROYDEN1_STABLE_MIXER_HPP__

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
#include "linalg/linalg.hpp"

namespace sirius {
namespace mixer {

/// Broyden mixer.
/** First version of the Broyden mixer, which requires inversion of the Jacobian matrix.
 *  Reference paper: "Robust acceleration of self consistent field calculations for
 *  density functional theory", Baarman K, Eirola T, Havu V., J Chem Phys. 134, 134109 (2011)
 */
template <typename... FUNCS>
class Broyden1Stable : public Mixer<FUNCS...>
{
  private:
    double beta_;
    sddk::mdarray<double, 2> R_;

  public:
    Broyden1Stable(std::size_t max_history, double beta)
        : Mixer<FUNCS...>(max_history)
        , beta_(beta)
        , R_(max_history - 1, max_history - 1)
    {
        this->R_.zero();
    }

    void mix_impl() override
    {
        const auto idx_step      = this->idx_hist(this->step_);
        const auto idx_next_step = this->idx_hist(this->step_ + 1);
        const auto idx_step_prev = this->idx_hist(this->step_ - 1);

        const int history_size = static_cast<int>(std::min(this->step_, this->max_history_ - 1));

        const bool normalize = false;

        // TODO: beta scaling?

        // Set up the next x_{n+1} = x_n
        this->copy(this->output_history_[idx_step], this->tmp1_);

        // + beta * f_n
        this->axpy(this->beta_, this->residual_history_[idx_step], this->tmp1_);

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
                auto j = this->idx_hist(this->step_ - i - 1);
                auto sz = this->template inner_product<normalize>(
                    this->residual_history_[j],
                    this->residual_history_[idx_step_prev]
                );
                this->R_(history_size - 1 - i, history_size - 1) = sz;
                this->axpy(-sz, this->residual_history_[j], this->residual_history_[idx_step_prev]);
            }

            // repeat orthogonalization.. seems really necessary.
            for (int i = 1; i <= history_size - 1; ++i) {
                auto j = this->idx_hist(this->step_ - i - 1);
                auto sz = this->template inner_product<normalize>(
                    this->residual_history_[j],
                    this->residual_history_[idx_step_prev]
                );
                this->R_(history_size - 1 - i, history_size - 1) += sz;
                this->axpy(-sz, this->residual_history_[j], this->residual_history_[idx_step_prev]);
            }

            // normalize the new residual difference vec itself
            auto sz = std::sqrt(this->template inner_product<normalize>(this->residual_history_[idx_step_prev], this->residual_history_[idx_step_prev]));
            this->R_(history_size - 1, history_size - 1) = sz;
            this->scale(1.0 / sz, this->residual_history_[idx_step_prev]);

            // Now do the Anderson iteration bit

            // Compute h = Q' * f_n
            sddk::mdarray<double, 1> h(history_size);
            for (int i = 1; i <= history_size; ++i) {
                auto j = this->idx_hist(this->step_ - i);
                h(history_size - i) = this->template inner_product<normalize>(this->residual_history_[j], this->residual_history_[idx_step]);
            }

            // next compute k = R⁻¹ * h... just do that by hand for now, can dispatch to blas later.
            sddk::mdarray<double, 1> k(history_size);
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
                this->axpy(-this->beta_ * h(history_size - i), this->residual_history_[j], this->tmp1_);
            }

            // - (delta X) k
            for (int i = 1; i <= history_size; ++i) {
                auto j = this->idx_hist(this->step_ - i);
                this->axpy(-k(history_size - i), this->output_history_[j], this->tmp1_);
            }
        }

        // When the history is full, drop the first column.
        // Basically we have delta F = [q1 Q2] * [r11 R12; O R22]
        // and we apply a couple rotations to make [R12; R22] upper triangular again
        // and simultaneously apply the adjoint of the rotations to [q1 Q2].
        // afterwards we drop the first column and last row of the new R, the new
        // Q currently ends up in the first so many columns of delta F, so we have
        // to do some swapping to restore the circular buffer for residual_history_
        if (history_size == this->max_history_ - 1) {
            for (int row = 1; row <= history_size - 1; ++row) {
                auto rotation = sddk::linalg(sddk::linalg_t::lapack).lartg(this->R_(row - 1, row), this->R_(row, row));
                auto c = std::get<0>(rotation);
                auto s = std::get<1>(rotation);
                auto nrm = std::get<2>(rotation);

                // Apply the Given's rotation to the initial column
                this->R_(row - 1, row) = nrm;
                this->R_(row    , row) = 0;

                // Apply Given's rotation to R
                for (int col = row + 1; col < history_size; ++col) {
                    auto r1 = this->R_(row - 1, col);
                    auto r2 = this->R_(row    , col);

                    this->R_(row - 1, col) =  c * r1 + s * r2;
                    this->R_(row    , col) = -s * r1 + c * r2;
                }

                // Apply the Given's rotation to Q (i.e. orthonormal basis for ΔF)
                int i1 = this->idx_hist(this->step_ - history_size + row - 1);
                int i2 = this->idx_hist(this->step_ - history_size + row);
                this->rotate(c, s, this->residual_history_[i1], this->residual_history_[i2]);
            }

            // Move all the guys one to the right -- maybe circular buffers aren't that great here...
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

        this->copy(this->tmp1_, this->output_history_[idx_next_step]);
    }
};
} // namespace mixer
} // namespace sirius

#endif // __MIXER_HPP__
