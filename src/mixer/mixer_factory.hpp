/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file mixer_factory.hpp
 *
 *  \brief Contains the mixer facttory for creating different types of mixers.
 */

#ifndef __MIXER_FACTORY_HPP__
#define __MIXER_FACTORY_HPP__

#include "mixer/mixer.hpp"
#include "mixer/anderson_mixer.hpp"
#include "mixer/anderson_stable_mixer.hpp"
#include "mixer/broyden2_mixer.hpp"
#include "mixer/linear_mixer.hpp"
#include "context/simulation_parameters.hpp"

namespace sirius {
namespace mixer {

/// Select and create a new mixer.
/** \param [in]  mix_cfg  Parameters for mixer selection and creation.
 *  \param [in]  comm     Communicator passed to the mixer.
 */
template <typename... FUNCS>
inline std::unique_ptr<Mixer<FUNCS...>>
Mixer_factory(config_t::mixer_t const& mix_cfg)
{
    std::unique_ptr<Mixer<FUNCS...>> mixer;

    if (mix_cfg.type() == "linear") {
        mixer.reset(new Linear<FUNCS...>(mix_cfg.beta()));
    }
    // broyden1 is a misnomer, but keep it for backward compatibility
    else if (mix_cfg.type() == "broyden1" || mix_cfg.type() == "anderson") {
        mixer.reset(new Anderson<FUNCS...>(mix_cfg.max_history(), mix_cfg.beta(), mix_cfg.beta0(),
                                           mix_cfg.beta_scaling_factor()));
    } else if (mix_cfg.type() == "anderson_stable") {
        mixer.reset(new Anderson_stable<FUNCS...>(mix_cfg.max_history(), mix_cfg.beta()));
    } else if (mix_cfg.type() == "broyden2") {
        mixer.reset(new Broyden2<FUNCS...>(mix_cfg.max_history(), mix_cfg.beta()));
    } else {
        RTE_THROW("wrong type of mixer");
    }
    return mixer;
}

} // namespace mixer
} // namespace sirius

#endif // __MIXER_HPP__
