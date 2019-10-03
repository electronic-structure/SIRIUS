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

/** \file mixer.h
 *
 *   \brief Contains definition and implementation of sirius::Mixer base class.
 */

#ifndef __MIXER_FACTORY_HPP__
#define __MIXER_FACTORY_HPP__

#include "Mixer/mixer.hpp"
#include "Mixer/broyden1_mixer.hpp"
#include "Mixer/broyden2_mixer.hpp"
#include "Mixer/linear_mixer.hpp"
#include "input.hpp"

namespace sirius {
template <typename... FUNCS>
inline std::unique_ptr<Mixer<FUNCS...>> Mixer_factory(Mixer_input mix_cfg, Communicator const& comm,
                                                      const MixerFunctionProperties<FUNCS>&... function_prop)
{
    std::unique_ptr<Mixer<FUNCS...>> mixer;

    if (mix_cfg.type_ == "linear") {
        mixer.reset(new Linear_mixer<FUNCS...>(mix_cfg.beta_, comm, function_prop...));
    } else if (mix_cfg.type_ == "broyden1") {
        mixer.reset(new Broyden1<FUNCS...>(mix_cfg.max_history_, mix_cfg.beta_, mix_cfg.beta0_,
                                           mix_cfg.beta_scaling_factor_, comm, function_prop...));
    } else if (mix_cfg.type_ == "broyden2") {
        mixer.reset(new Broyden2<FUNCS...>(mix_cfg.max_history_, mix_cfg.beta_, mix_cfg.beta0_,
                                           mix_cfg.beta_scaling_factor_, mix_cfg.linear_mix_rms_tol_, comm,
                                           function_prop...));
    } else {
        TERMINATE("wrong type of mixer");
    }
    return mixer;
}

} // namespace sirius

#endif // __MIXER_HPP__
