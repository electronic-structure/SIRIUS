// Copyright (c) 2013-2018 Anton Kozhevnikov, Ilia Sivkov, Mathieu Taillefumier, Thomas Schulthess
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

/** \file non_local_functor.hpp
 *
 *  \brief Common operation for forces and stress tensor.
 */

#ifndef __NON_LOCAL_FUNCTOR_HPP__
#define __NON_LOCAL_FUNCTOR_HPP__

#include "periodic_function.hpp"
#include "K_point/k_point.hpp"
#include "Density/augmentation_operator.hpp"
#include "Beta_projectors/beta_projectors.hpp"

namespace sirius {

template <typename T>
class Non_local_functor
{
  private:
    Simulation_context& ctx_;
    Beta_projectors_base& bp_base_;
  public:

    Non_local_functor(Simulation_context& ctx__, Beta_projectors_base& bp_base__)
        : ctx_(ctx__)
        , bp_base_(bp_base__)
    {
    }

    /// Collect summation result in an array
    void add_k_point_contribution(K_point& kpoint__, sddk::mdarray<double, 2>& collect_res__);
};
}

#endif /* __NON_LOCAL_FUNCTOR_HPP__ */
