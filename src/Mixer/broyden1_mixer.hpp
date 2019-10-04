// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

#ifndef __BROYDEN1_MIXER_HPP__
#define __BROYDEN1_MIXER_HPP__

#include <tuple>
#include <functional>
#include <utility>
#include <vector>
#include <limits>
#include <memory>
#include <exception>
#include <cmath>
#include <numeric>

#include "SDDK/communicator.hpp"
#include "Mixer/mixer.hpp"

namespace sirius {
namespace mixer {
template <typename... FUNCS>
class Broyden1 : public Mixer<FUNCS...>
{
  public:
    Broyden1(std::size_t max_history, double beta, double beta0, double beta_scaling_factor, Communicator const& comm,
             const FunctionProperties<FUNCS>&... function_prop)
        : Mixer<FUNCS...>(max_history, comm, function_prop...)
        , beta_(beta)
        , beta0_(beta0)
        , beta_scaling_factor_(beta_scaling_factor)
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

        // set input to 0, to use as buffer
        this->scale(0.0, this->input_);

        if (history_size > 0) {
            mdarray<double, 2> S(history_size, history_size);
            S.zero();
            mdarray<double, 2> S_local(history_size, history_size);
            S_local.zero();
            for (int j1 = 0; j1 < history_size; j1++) {
                int i1 = this->idx_hist(this->step_ - j1);
                int i2 = this->idx_hist(this->step_ - j1 - 1);
                this->copy(this->residual_history_[i1], this->tmp1_);
                this->axpy(-1.0, this->residual_history_[i2], this->tmp1_);
                for (int j2 = 0; j2 <= j1; j2++) {
                    int i3 = this->idx_hist(this->step_ - j2);
                    int i4 = this->idx_hist(this->step_ - j2 - 1);
                    this->copy(this->residual_history_[i3], this->tmp2_);
                    this->axpy(-1.0, this->residual_history_[i4], this->tmp2_);

                    S(j2, j1) = S(j1, j2) = this->inner_product(false, this->tmp1_, this->tmp2_);
                    S_local(j2, j1) = S_local(j1, j2) = this->inner_product(true, this->tmp1_, this->tmp2_);
                }
            }
            this->comm_.allreduce(S.at(memory_t::host), (int)S.size());
            for (int j1 = 0; j1 < history_size; j1++) {
                for (int j2 = 0; j2 < history_size; j2++) {
                    S(j1, j2) += S_local(j1, j2);
                }
            }

            /* invert matrix */
            linalg<device_t::CPU>::syinv(history_size, S);
            /* restore lower triangular part */
            for (int j1 = 0; j1 < history_size; j1++) {
                for (int j2 = 0; j2 < j1; j2++) {
                    S(j1, j2) = S(j2, j1);
                }
            }

            mdarray<double, 1> c(history_size);
            c.zero();
            mdarray<double, 1> c_local(history_size);
            c_local.zero();
            for (int j = 0; j < history_size; j++) {
                int i1 = this->idx_hist(this->step_ - j);
                int i2 = this->idx_hist(this->step_ - j - 1);

                this->copy(this->residual_history_[i1], this->tmp1_);
                this->axpy(-1.0, this->residual_history_[i2], this->tmp1_);

                c(j)       = this->inner_product(false, this->tmp1_, this->residual_history_[idx_step]);
                c_local(j) = this->inner_product(true, this->tmp1_, this->residual_history_[idx_step]);
            }
            this->comm_.allreduce(c.at(memory_t::host), (int)c.size());
            for (int j = 0; j < history_size; j++) {
                c(j) += c_local(j);
            }

            for (int j = 0; j < history_size; j++) {
                double gamma = 0;
                for (int i = 0; i < history_size; i++) {
                    gamma += c(i) * S(i, j);
                }

                int i1 = this->idx_hist(this->step_ - j);
                int i2 = this->idx_hist(this->step_ - j - 1);

                this->copy(this->residual_history_[i1], this->tmp1_);
                this->axpy(-1.0, this->residual_history_[i2], this->tmp1_);

                this->copy(this->output_history_[i1], this->tmp2_);
                this->axpy(-1.0, this->output_history_[i2], this->tmp2_);

                this->axpy(this->beta_, this->tmp1_, this->tmp2_);
                this->axpy(-gamma, this->tmp2_, this->input_);
            }
        }
        this->copy(this->output_history_[idx_step], this->output_history_[idx_next_step]);
        this->axpy(this->beta_, this->residual_history_[idx_step], this->output_history_[idx_next_step]);
        this->axpy(1.0, this->input_, this->output_history_[idx_next_step]);
    }

  private:
    double beta_;
    double beta0_;
    double beta_scaling_factor_;
};
} // namespace mixer
} // namespace sirius

#endif // __MIXER_HPP__
