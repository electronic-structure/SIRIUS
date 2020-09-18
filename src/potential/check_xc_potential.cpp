// Copyright (c) 2013-2020 Anton Kozhevnikov, Thomas Schulthess
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

/** \file check_xc_potential.cpp
 *
 *  \brief Check XC potential by doing numerical functional derivative.
 */

#include "potential/potential.hpp"
#include "dft/energy.hpp"

namespace sirius {

void check_xc_potential(Density const& rho__)
{
    Potential p0(const_cast<Simulation_context&>(rho__.ctx()));
    p0.generate(rho__);

    double evxc{0}, ebxc{0};
    if (rho__.ctx().full_potential()) {
    } else {
        evxc = p0.energy_vxc(rho__) + p0.energy_vxc_core(rho__);
        ebxc = energy_bxc(rho__, p0);
    }
    std::printf("<vxc|rho>        : %18.12f\n", evxc);
    std::printf("<bxc|mag>        : %18.12f\n", ebxc);

    double eps{0.1};
    double best_result{1e10};
    double best_eps{0};
    for (int i = 0; i < 10; i++) {
        Potential p1(const_cast<Simulation_context&>(rho__.ctx()));
        /* compute Exc, Vxc at  rho + delta * rho = (1+delta)rho */
        p1.add_delta_rho_xc(eps);
        p1.generate(rho__);

        double deriv_mag{0};

        if (rho__.ctx().num_mag_dims() > 0) {
            Potential p2(const_cast<Simulation_context&>(rho__.ctx()));
            /* compute Exc, Vxc at mag + delta * mag = (1+delta)mag */
            p2.add_delta_mag_xc(eps);
            p2.generate(rho__);

            deriv_mag = (p2.energy_exc(rho__) - p0.energy_exc(rho__)) / eps;
        }

        double deriv_rho = (p1.energy_exc(rho__) - p0.energy_exc(rho__)) / eps;

        std::printf("eps: %18.12f, drho: %18.12f, dmag: %18.12f, dE/dmag: %18.12f\n", eps, std::abs(evxc - deriv_rho),
                    std::abs(ebxc - deriv_mag), deriv_mag);

        if (std::abs(evxc - deriv_rho) < best_result) {
            best_result = std::abs(evxc - deriv_rho);
            best_eps = eps;
        }

        eps /= 10;
    }
    std::printf("best result : %18.12f for epsilon %18.12f\n", best_result, best_eps);
}

}
