// Copyright (c) 2023 Simon Pintarelli, Anton Kozhevnikov, Thomas Schulthess
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
call_nlcg(Simulation_context& ctx, const config_t::nlcg_t& nlcg_params, Energy& energy, K_point_set& kset,
          Potential& potential)
{
    using numeric_t = std::complex<double>;

    double temp  = nlcg_params.T();
    double tol   = nlcg_params.tol();
    double kappa = nlcg_params.kappa();
    double tau   = nlcg_params.tau();
    int maxiter  = nlcg_params.maxiter();
    int restart  = nlcg_params.restart();
    auto nlcg_pu = sddk::get_device_t(nlcg_params.processing_unit());

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
        case sddk::device_t::CPU: {
            nlcglib::nlcg_us_cpu(energy, us_precond, S, smearing, temp, tol, kappa, tau, maxiter, restart);
            break;
        }
        case sddk::device_t::GPU: {
            nlcglib::nlcg_us_device(energy, us_precond, S, smearing, temp, tol, kappa, tau, maxiter, restart);
            break;
        }
    }
}

} // namespace sirius

#endif /* __CALL_NLCG_HPP__ */
