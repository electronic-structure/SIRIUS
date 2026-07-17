/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file interstitial_functions.hpp
 *
 *  \brief LAPW interstitial functions used in the construction and application of the Hamiltonian.
 */

#ifndef __INTERSTITIAL_FUNCTIONS_HPP__
#define __INTERSTITIAL_FUNCTIONS_HPP__

#include "potential/potential.hpp"
#include "core/constants.hpp"

namespace sirius {

inline auto
interstitial_potential(Potential const& potential__, int j__)
{
    return [&potential__, j__](int ir) {
        return potential__.component(j__).rg().value(ir) * potential__.ctx().theta(ir);
    };
}

inline auto
interstitial_step_function(Simulation_context const& ctx__)
{
    return [&ctx__](int ir) {
        return ctx__.theta(ir);
    };
}

inline auto
interstitial_canonical_potential(Potential const& potential__, double sign__)
{
    return [&potential__, sign__](int ir) {
        // V +/- Bz
        return (potential__.component(0).rg().value(ir) +
                potential__.component(1).rg().value(ir) * sign__) * potential__.ctx().theta(ir);
    };
}

inline auto
interstitial_canonical_potential(Potential const& potential__, std::complex<double> sign__)
{
    return [&potential__, sign__](int ir) {
        return (potential__.component(2).rg().value(ir) + 
                potential__.component(3).rg().value(ir) * sign__) * potential__.ctx().theta(ir);
    };
}

template <int P>
inline auto
interstitial_mass(Potential const& potential__)
{
    return [&potential__](int ir) {

        double M = 1.0 - sq_alpha_half * potential__.effective_potential().rg().value(ir);

        if constexpr (P == 1) {
            return potential__.ctx().theta(ir) / M;
        } else {
            static_assert(P == 2);
            return potential__.ctx().theta(ir) / (M * M);
        }
    };
}

}

#endif
