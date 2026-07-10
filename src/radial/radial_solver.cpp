/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file radial_solver.cpp
 *
 *  \brief Contains partial implementation of sirius::Radial_solver class.
 */

#include "radial_solver.hpp"

namespace sirius {

void
Enu_finder::find_enu(relativity_t rel__, enu_search_t& enu__)
{
    int np = num_points();

    Spline<double> chi_p(radial_grid());
    Spline<double> chi_q(radial_grid());

    std::vector<double> p(np);
    std::vector<double> q(np);
    std::vector<double> dpdr(np);
    std::vector<double> dqdr(np);

    double constexpr enu_tol{1e-9};

    std::stringstream sinfo;

    auto compute_etop = [&]() -> int {
        /* We want to find enu such that the wave-function at the muffin-tin boundary is zero
         * and the number of nodes inside muffin-tin is equal to n-l-1. This will be the top
         * of the band. */
        int s{1};
        int sp;
        double denu{1e-5};
        double e1;
        sinfo << "find_enu(): find top enery" << std::endl
              << "  n         : " << n_ << ", l : " << l_ << std::endl
              << "  enu_start : " << enu__.etop << std::endl;
        try {
            /* 1st pass: estimate upper and lower boundaries of the etop */
            e1 = integrate_forward_until(rel__, enu__.etop, l_, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                                         [&s, &sp, &denu, this](int iter, int nn, double& enu) {
                                             sp = s;
                                             s  = (nn > (n_ - l_ - 1)) ? -1 : 1;
                                             if (s != sp && iter > 0) {
                                                 return true;
                                             }
                                             denu = std::min(0.5, denu * 2);
                                             enu += s * denu;
                                             return false;
                                         });
        } catch (std::exception const& e) {
            sinfo << e.what() << std::endl << "denu : " << denu;
            return 1;
        }

        /* only with this condition we need to refine top energy by bisection;
         * otherwise etop stays untouched */
        if (std::abs(e1 - enu__.etop_first_pass) > enu_tol) {
            enu__.etop_first_pass = e1;

            double e2 = e1 - sp * denu;

            /* e1 is bottom, e2 is top energy */
            if (e1 > e2) {
                std::swap(e1, e2);
            }

            sinfo << "find_enu(): refine top energy" << std::endl
                  << "  e1 : " << e1 << ", e2 : " << e2 << std::endl
                  << "  enu_start : " << (e1 + e2) / 2 << std::endl;

            try {
                /* 2nd pass: refine by bisection */
                enu__.etop = integrate_forward_until(rel__, (e1 + e2) / 2, l_, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                                                [&e1, &e2, enu_tol, this](int iter, int nn, double& enu) {
                                                    if (nn > (n_ - l_ - 1)) {
                                                        e2 = enu;
                                                    } else {
                                                        e1 = enu;
                                                    }
                                                    enu = (e1 + e2) / 2.0;
                                                    return std::abs(e1 - e2) < enu_tol;
                                                });
            } catch (std::exception const& e) {
                sinfo << e.what() << std::endl;
                return 1;
            }
        }
        return 0;
    };

    /* start by computing top of linearization energy */
    if (compute_etop() != 0) {
        sinfo << "find_enu(): top of the linearization energy interval is not found";
        RTE_THROW(sinfo);
    }

    auto surface_deriv = [this, &dpdr, &p]() {
        if (true) {
            /* return  p'(R) */
            return dpdr.back();
        } else {
            /* return R*u'(R) */
            return dpdr.back() - p.back() / radial_grid_.last();
        }
    };

    /* current surface derivative */
    double sd = surface_deriv();
    /* try several steps first */
    double denu{1e-5};
    for (auto de: {1e-4, 1e-3, 1e-2, 1e-1, 1.0}) {
        int num_nodes;
        integrate_forward_until(rel__, enu__.etop - de, l_, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                         [&num_nodes](int iter, int nn, double& enu) {
                                num_nodes = nn;
                                return true;
                            });
        if (surface_deriv() * sd < 0 || num_nodes != (n_ - l_ - 1)) {
            break;
        }
        denu = de;
    }

    /* Now we go down in energy and search for enu such that the wave-function derivative is zero
     * at the muffin-tin boundary. This will be the bottom of the band. Here we look at a sign change
     * of the derivative. */
    try {
        sinfo << "find_enu(): find bottom energy" << std::endl << "  enu_start : " << enu__.etop << std::endl;
        auto sd = surface_deriv();
        auto e1 = integrate_forward_until(rel__, enu__.etop, l_, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                                            [this, &denu, sd, &surface_deriv, &p](int iter, int nn, double& enu) {
                                                if (surface_deriv() * sd < 0) {
                                                    return true;
                                                }
                                                /* do not allow step in energy to grow too much */
                                                //denu = std::min(0.1, denu * 1.2);
                                                denu *= 1.1;
                                                enu -= denu;
                                                return false;
                                            });

        /* refine bottom energy */
        auto e2 = e1 + denu;
        /* simple estimation of the bottom energy */
        if (enu__.auto_enu == 3) {
            enu__.ebot =  (e1 + e2) * 0.5;
            enu__.enu = (enu__.ebot + enu__.etop) / 2.0;
        } else {
            sinfo << "find_enu(): refine bottom energy" << std::endl << "  enu_start : " << (e1 + e2) / 2 << std::endl;
            enu__.ebot = integrate_forward_until(rel__, (e1 + e2) / 2, l_, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                                            [this, &e1, &e2, sd, &surface_deriv](int iter, int nn, double& enu) {
                                                if (surface_deriv() * sd > 0) {
                                                    e2 = enu;
                                                } else {
                                                    e1 = enu;
                                                }
                                                enu = (e1 + e2) / 2.0;
                                                return std::abs(surface_deriv()) < enu_tol;
                                            });
            switch (enu__.auto_enu) {
                case 1: {
                    enu__.enu = (enu__.ebot + enu__.etop) / 2.0;
                    break;
                }
                case 2: {
                    enu__.enu = enu__.ebot;
                    break;
                }
                default: {
                    RTE_THROW("wrong type of auto_enu");
                }
            }
        }
    } catch (std::exception const& e) {
        sinfo << e.what() << std::endl << "denu : " << denu << std::endl;
    }
}

}; // namespace sirius
