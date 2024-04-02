/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file check_xc_potential.cpp
 *
 *  \brief Check XC potential by doing numerical functional derivative.
 */

#include "potential/potential.hpp"
#include "dft/energy.hpp"

namespace sirius {

void
check_xc_potential(Density const& rho__)
{
    Potential p0(const_cast<Simulation_context&>(rho__.ctx()));
    p0.generate(rho__, rho__.ctx().use_symmetry(), true);

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
        p1.generate(rho__, rho__.ctx().use_symmetry(), true);

        double deriv_mag{0};

        if (rho__.ctx().num_mag_dims() > 0) {
            Potential p2(const_cast<Simulation_context&>(rho__.ctx()));
            /* compute Exc, Vxc at mag + delta * mag = (1+delta)mag */
            p2.add_delta_mag_xc(eps);
            p2.generate(rho__, rho__.ctx().use_symmetry(), true);

            deriv_mag = (p2.energy_exc(rho__) - p0.energy_exc(rho__)) / eps;
        }

        double deriv_rho = (p1.energy_exc(rho__) - p0.energy_exc(rho__)) / eps;

        std::printf("eps: %18.12f, drho: %18.12f, dmag: %18.12f\n", eps, std::abs(evxc - deriv_rho),
                    std::abs(ebxc - deriv_mag));

        if (std::abs(evxc - deriv_rho) + std::abs(ebxc - deriv_mag) < best_result) {
            best_result = std::abs(evxc - deriv_rho) + std::abs(ebxc - deriv_mag);
            best_eps    = eps;
        }

        eps /= 10;
    }
    std::printf("best total result : %18.12f for epsilon %18.12f\n", best_result, best_eps);
}

} // namespace sirius
