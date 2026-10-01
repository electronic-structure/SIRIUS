/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __LAPW_XC_FIELDS_HPP__
#define __LAPW_XC_FIELDS_HPP__

#include "function3d/smooth_periodic_function.hpp"
#include "function3d/spheric_function.hpp"

namespace sirius {

/// Discrete XC covectors before contraction with the current LAPW radial basis.
struct lapw_xc_field_adjoint_t
{
    /// Already restricted to interstitial targets. Do not apply another sphere mask.
    std::vector<Smooth_periodic_function<double>> rho;
    std::vector<Smooth_periodic_function<double>> tau;
    /// Raw radial-sample covectors {rho, tau}, in atom order and per spin.
    std::vector<std::array<std::vector<Flm>, 2>> local;
    /// Raw derivatives with respect to spin-summed core rho/tau radial values.
    std::vector<std::array<std::vector<double>, 2>> core;
};

/// Contractions with total charge, magnetization and spin-resolved kinetic density.
struct lapw_xc_contractions_t
{
    double rho{0};
    double magnetic{0};
    double tau{0};
};

} // namespace sirius

#endif
