/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

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

#include "mixer/mixer.hpp"

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

    void
    mix_impl() override
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
