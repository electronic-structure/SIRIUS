/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file newton_minimization.cpp
 *
 *  \brief Contains  implementation of sirius::newton_minimization_chemical_potential.
 */

#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include "core/expected.hpp"

namespace sirius {

namespace local {
struct newton_minimization_result
{
    double mu;   // chemical potential
    int iter{0}; // newton information
    double ne_diff;
    std::vector<double> ys; // newton history
};
}  // local
/**
 *  Newton minimization to determine the chemical potential.
 *
 *  \param  N       number of electrons as a function of \f$\mu\f$
 *  \param  dN      \f$\partial_\mu N(\mu)\f$
 *  \param  ddN     \f$\partial^2_\mu N(\mu)\f$
 *  \param  mu0     initial guess
 *  \param  ne      target number of electrons
 *  \param  tol     tolerance
 *  \param  maxstep max number of Newton iterations
 */
template <class Nt, class DNt, class D2Nt>
util::expected<local::newton_minimization_result, std::string>
newton_minimization_chemical_potential(Nt&& N, DNt&& dN, D2Nt&& ddN, double mu0, double ne, double tol,
                                       int maxstep = 1000)
{
    // Newton finds the minimum, not necessarily N(mu) == ne, tolerate up to `tol_ne` difference in number of electrons
    // if |N(mu_0) -ne| > tol_ne an error is thrown.
    local::newton_minimization_result res;
    const double tol_ne = 1e-10;

    double mu = mu0;
    double alpha{1.0}; // Newton damping
    int iter{0};

    if (std::abs(N(mu) - ne) < tol) {
        res.mu   = mu;
        res.iter = iter;
        res.ys   = {};
        return res;
    }

    while (true) {
        // compute
        double Nf   = N(mu);
        double dNf  = dN(mu);
        double ddNf = ddN(mu);
        /* minimize (N(mu) - ne)^2  */
        // double F = (Nf - ne) * (Nf - ne);
        double dF   = 2 * (Nf - ne) * dNf;
        double ddF  = 2 * dNf * dNf + 2 * (Nf - ne) * ddNf;
        double step = alpha * dF / std::abs(ddF);
        mu          = mu - step;

        res.ys.push_back(mu);

        if (std::abs(ddF) < 1e-30) {
            std::stringstream s;
            s << "Newton minimization (Fermi energy) failed because 2nd derivative too close to zero!"
              << std::setprecision(8) << std::abs(Nf - ne) << "\n";
            return util::unexpected(s.str());
        }

        double ne_diff = std::abs(Nf - ne);
        if (std::abs(step) < tol || ne_diff < tol)
        {
            if (ne_diff > tol_ne) {
                std::stringstream s;
                s << "Newton minimization (Fermi energy) got stuck in a local minimum. Fallback to bisection search."
                  << "\n";
                return util::unexpected(s.str());
            }

            res.iter = iter;
            res.mu   = mu;
            res.ne_diff = ne_diff;
            return res;
        }

        iter++;
        if (iter > maxstep) {
            std::stringstream s;
            s << "Newton minimization (chemical potential) failed after " << maxstep << " steps!" << std::endl
              << "target number of electrons : " << ne << std::endl
              << "initial guess for chemical potential : " << mu0 << std::endl
              << "current value of chemical potential : " << mu;
            return util::unexpected(s.str());
        }
    }
}



}  // sirius
