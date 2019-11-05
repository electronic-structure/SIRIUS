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

/** \file linear_mixer.hpp
 *
 *   \brief Contains definition and implementation of sirius::Linear_mixer.
 */

#ifndef __LINEAR_MIXER_HPP__
#define __LINEAR_MIXER_HPP__

#include <tuple>
#include <functional>
#include <utility>
#include <vector>
#include <limits>
#include <memory>
#include <exception>
#include <cmath>
#include <numeric>

#include "Mixer/mixer.hpp"

namespace sirius {
namespace mixer {
template <typename... FUNCS>
class Linear : public Mixer<FUNCS...>
{
  public:
    explicit Linear(double beta)
        : Mixer<FUNCS...>(2)
        , beta_(beta)
    {
    }

    void mix_impl() override
    {
        const auto idx = this->idx_hist(this->step_ + 1);

        this->copy(this->input_, this->output_history_[idx]);
        this->scale(beta_, this->output_history_[idx]);
        this->axpy(1.0 - beta_, this->output_history_[this->idx_hist(this->step_)], this->output_history_[idx]);
    }

  private:
    double beta_;
};
} // namespace mixer
} // namespace sirius

#endif // __MIXER_HPP__
