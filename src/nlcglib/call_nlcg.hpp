/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file call_nlcg.hpp
 *
 *  \brief Wrapper to invoke nlcglib direct minimization.
 */

#ifndef __CALL_NLCG_HPP__
#define __CALL_NLCG_HPP__

// #include "context/config.hpp"
#include "context/simulation_context.hpp"
#include "nlcglib/adaptor.hpp"
#include "nlcglib/ultrasoft_precond.hpp"
#include "nlcglib/overlap.hpp"
#include "nlcglib/nlcglib.hpp"
#include "hamiltonian/hamiltonian.hpp"

namespace sirius {

inline void
call_nlcg(Simulation_context& ctx, config_t::nlcg_t const& nlcg_params, Energy& energy, K_point_set& kset,
          Potential& potential)
{
    using numeric_t = std::complex<double>;

    double temp  = nlcg_params.T();
    double tol   = nlcg_params.tol();
    double kappa = nlcg_params.kappa();
    double tau   = nlcg_params.tau();
    int maxiter  = nlcg_params.maxiter();
    int restart  = nlcg_params.restart();
    auto nlcg_pu = ctx.processing_unit();
    if (nlcg_params.processing_unit() != "") {
        nlcg_pu = get_device_t(nlcg_params.processing_unit());
    }

    std::string smear = ctx.cfg().parameters().smearing();

    nlcglib::smearing_type smearing;
    if (smear.compare("fermi_dirac") == 0) {
        smearing = nlcglib::smearing_type::FERMI_DIRAC;
    } else if (smear.compare("gaussian_spline") == 0) {
        smearing = nlcglib::smearing_type::GAUSSIAN_SPLINE;
    } else if (smear.compare("gaussian") == 0) {
        smearing = nlcglib::smearing_type::GAUSS;
    } else if (smear.compare("methfessel_paxton") == 0) {
        smearing = nlcglib::smearing_type::METHFESSEL_PAXTON;
    } else if (smear.compare("cold") == 0) {
        smearing = nlcglib::smearing_type::COLD;
    } else {
        RTE_THROW("invalid smearing type given");
    }

    Hamiltonian0<double> H0(potential, false /* precompute laplw */);

    sirius::UltrasoftPrecond us_precond(kset, ctx, H0.Q());
    sirius::Overlap_operators<sirius::S_k<numeric_t>> S(kset, ctx, H0.Q());

    // ultrasoft pp
    switch (nlcg_pu) {
        case device_t::CPU: {
            nlcglib::nlcg_us_cpu(energy, us_precond, S, smearing, temp, tol, kappa, tau, maxiter, restart);
            break;
        }
        case device_t::GPU: {
            nlcglib::nlcg_us_device(energy, us_precond, S, smearing, temp, tol, kappa, tau, maxiter, restart);
            break;
        }
    }
}

} // namespace sirius

#endif /* __CALL_NLCG_HPP__ */
