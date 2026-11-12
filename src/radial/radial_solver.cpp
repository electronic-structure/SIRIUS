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
#include "core/json.hpp"

namespace sirius {

enu_search_t
find_enu(relativity_t rel__, int zn__, int n__, int l__, Radial_grid<double> const& radial_grid__,
         std::vector<double> const& v__, enu_search_t const& enu__)
{
    Radial_solver solver(zn__, v__, radial_grid__);

    enu_search_t enu_out = enu__;

    int np = solver.num_points();

    Spline<double> chi_p(radial_grid__);
    Spline<double> chi_q(radial_grid__);

    std::vector<double> p(np);
    std::vector<double> q(np);
    std::vector<double> dpdr(np);
    std::vector<double> dqdr(np);

    double constexpr enu_tol{1e-9};

    std::stringstream sinfo;
    sinfo << "find_enu()" << std::endl
          << "  rel : " << get_relativity(rel__) << std::endl
          << "  zn  : " << zn__ << std::endl
          << "  n   : " << n__ << std::endl
          << "  l   : " << l__ << std::endl
          << "  auto_enu        : " << enu__.auto_enu << std::endl
          << "  etop            : " << enu__.etop << std::endl
          << "  etop_first_pass : " << enu__.etop_first_pass << std::endl
          << "  ebot            : " << enu__.ebot << std::endl;

    auto compute_etop = [&]() -> int {
        /* We want to find enu such that the wave-function at the muffin-tin boundary is zero
         * and the number of nodes inside muffin-tin is equal to n-l-1. This will be the top
         * of the band. */
        int s{1};
        int sp;
        double denu{1e-6};
        double e1;
        sinfo << "  step : estimate top of the energy" << std::endl;
        try {
            /* 1st pass: estimate upper and lower boundaries of the etop */
            e1 = solver.integrate_forward_until(rel__, enu__.etop, l__, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                                                [&s, &sp, &denu, n__, l__](int iter, int nn, double& enu) {
                                                    sp = s;
                                                    s  = (nn > (n__ - l__ - 1)) ? -1 : 1;
                                                    if (s != sp && iter > 0) {
                                                        return true;
                                                    }
                                                    denu = std::min(0.5, denu * 2);
                                                    enu += s * denu;
                                                    return false;
                                                });
        } catch (std::exception const& e) {
            sinfo << "  denu  : " << denu << std::endl;
            sinfo << "  error : " << e.what() << std::endl;
            return 1;
        }

        if (std::abs(e1 - enu__.etop_first_pass) > enu_tol) {
            sinfo << "  step : refine top of the energy" << std::endl;

            enu_out.etop_first_pass = e1;

            double e2 = e1 - sp * denu;

            /* e1 is bottom, e2 is top energy */
            if (e1 > e2) {
                std::swap(e1, e2);
            }

            sinfo << "  e1 : " << e1 << std::endl << "  e2 : " << e2 << std::endl;

            try {
                /* 2nd pass: refine by bisection */
                enu_out.etop = solver.integrate_forward_until(
                        rel__, (e1 + e2) / 2, l__, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                        [&e1, &e2, enu_tol, n__, l__](int iter, int nn, double& enu) {
                            if (nn > (n__ - l__ - 1)) {
                                e2 = enu;
                            } else {
                                e1 = enu;
                            }
                            enu = (e1 + e2) / 2.0;
                            return std::abs(e1 - e2) < enu_tol;
                        });
            } catch (std::exception const& e) {
                sinfo << " last e1 : " << e1 << std::endl
                      << " last e2 : " << e2 << std::endl
                      << " error   : " << e.what() << std::endl;
                return 1;
            }
        } else {
            enu_out.etop = enu__.etop;
        }

        return 0;
    };

    /* start by computing top of linearization energy */
    if (compute_etop() != 0) {
        sinfo << "  error : top of the linearization energy interval is not found";
        RTE_THROW(sinfo);
    }

    /* check for quick exit */
    if (enu_out.etop == enu__.etop) {
        enu_out = enu__;
        return enu_out;
    }

    auto surface_deriv = [&dpdr, &p, &radial_grid__]() {
        if (true) {
            /* return  p'(R) */
            return dpdr.back();
        } else {
            /* return R*u'(R) */
            return dpdr.back() - p.back() / radial_grid__.last();
        }
    };

    /* current surface derivative */
    double sd = surface_deriv();
    /* try several steps first */
    double denu{1e-5};
    //for (auto de: {1e-4, 1e-3, 1e-2, 1e-1}) {
    //    int num_nodes;
    //    solver.integrate_forward_until(rel__, enu_out.etop - de, l__, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
    //                     [&num_nodes](int iter, int nn, double& enu) {
    //                            num_nodes = nn;
    //                            return true;
    //                        });
    //    if (surface_deriv() * sd < 0 || num_nodes != (n__ - l__ - 1)) {
    //        break;
    //    }
    //    denu = de;
    //}

    /* Now we go down in energy and search for enu such that the wave-function derivative is zero
     * at the muffin-tin boundary. This will be the bottom of the band. Here we look at a sign change
     * of the derivative. */
    try {
        sinfo << "  step : estimate bottom of the energy" << std::endl;
        auto sd = surface_deriv();
        auto e1 = solver.integrate_forward_until(rel__, enu_out.etop, l__, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                                                 [&denu, sd, &surface_deriv, &p](int iter, int nn, double& enu) {
                                                     if (surface_deriv() * sd < 0) {
                                                         return true;
                                                     }
                                                     /* do not allow step in energy to grow too much */
                                                     //denu = std::min(0.1, denu * 1.1);
                                                     denu *= 1.1;
                                                     enu -= denu;
                                                     return false;
                                                 });

        /* refine bottom energy */
        auto e2 = e1 + denu;
        /* simple estimation of the bottom energy */
        if (enu__.auto_enu == 3) {
            enu_out.ebot = (e1 + e2) * 0.5;
            enu_out.enu  = (enu_out.ebot + enu_out.etop) / 2.0;
        } else {
            sinfo << "  step : refine bottom of the energy";
            sinfo << "  e1 : " << e1 << std::endl << "  e2 : " << e2 << std::endl;
            enu_out.ebot =
                    solver.integrate_forward_until(rel__, (e1 + e2) / 2, l__, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                                                   [&e1, &e2, sd, &surface_deriv](int iter, int nn, double& enu) {
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
                    enu_out.enu = (enu_out.ebot + enu_out.etop) / 2.0;
                    break;
                }
                case 2: {
                    enu_out.enu = enu_out.ebot;
                    break;
                }
                default: {
                    RTE_THROW("wrong type of auto_enu");
                }
            }
        }
    } catch (std::exception const& e) {
        sinfo << "  last p(R)  : " << p.back() << std::endl
              << "  last p'(R) : " << dpdr.back() << std::endl
              << "  error      : " << e.what() << std::endl;
        RTE_THROW(sinfo);
    }

    return enu_out;
}

enu_search_t
find_enu_v2(relativity_t rel__, int zn__, int n__, int l__, Radial_grid<double> const& radial_grid__,
            std::vector<double> const& v__, enu_search_t const& enu__)
{
    Radial_solver solver(zn__, v__, radial_grid__);

    enu_search_t enu_out = enu__;

    int np = solver.num_points();

    Spline<double> chi_p(radial_grid__);
    Spline<double> chi_q(radial_grid__);

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
              << "  n         : " << n__ << ", l : " << l__ << std::endl
              << "  enu_start : " << enu__.etop << std::endl;
        try {
            /* 1st pass: estimate upper and lower boundaries of the etop */
            e1 = solver.integrate_forward_until(rel__, enu__.etop, l__, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                                                [&s, &sp, &denu, n__, l__](int iter, int nn, double& enu) {
                                                    sp = s;
                                                    s  = (nn > (n__ - l__ - 1)) ? -1 : 1;
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
            enu_out.etop_first_pass = e1;

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
                enu_out.etop = solver.integrate_forward_until(
                        rel__, (e1 + e2) / 2, l__, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                        [&e1, &e2, enu_tol, n__, l__](int iter, int nn, double& enu) {
                            if (nn > (n__ - l__ - 1)) {
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

    auto surface_deriv = [&dpdr, &p, &radial_grid__]() {
        if (true) {
            /* return  p'(R) */
            return dpdr.back();
        } else {
            /* return R*u'(R) */
            return dpdr.back() - p.back() / radial_grid__.last();
        }
    };

    /* current surface derivative */
    double sd = surface_deriv();
    /* try several steps first */
    double denu{1e-5};
    for (auto de : {1e-4, 1e-3, 1e-2, 1e-1, 1.0}) {
        int num_nodes;
        solver.integrate_forward_until(rel__, enu_out.etop - de, l__, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                                       [&num_nodes](int iter, int nn, double& enu) {
                                           num_nodes = nn;
                                           return true;
                                       });
        if (surface_deriv() * sd < 0 || num_nodes != (n__ - l__ - 1)) {
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
        auto e1 = solver.integrate_forward_until(rel__, enu_out.etop, l__, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                                                 [&denu, sd, &surface_deriv, &p](int iter, int nn, double& enu) {
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
            enu_out.ebot = (e1 + e2) * 0.5;
            enu_out.enu  = (enu_out.ebot + enu_out.etop) / 2.0;
        } else {
            sinfo << "find_enu(): refine bottom energy" << std::endl << "  enu_start : " << (e1 + e2) / 2 << std::endl;
            enu_out.ebot =
                    solver.integrate_forward_until(rel__, (e1 + e2) / 2, l__, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                                                   [&e1, &e2, sd, &surface_deriv](int iter, int nn, double& enu) {
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
                    enu_out.enu = (enu_out.ebot + enu_out.etop) / 2.0;
                    break;
                }
                case 2: {
                    enu_out.enu = enu_out.ebot;
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

    return enu_out;
}

void
Bound_state::solve(relativity_t rel__, double enu_start__, double alpha0__, double alpha1__)
{
    int np = num_points();

    Spline<double> chi_p(radial_grid());
    Spline<double> chi_q(radial_grid());

    std::vector<double> p(np);
    std::vector<double> q(np);
    std::vector<double> dqdr(np);
    std::vector<double> rdudr(np);
    dpdr_ = std::vector<double>(np);

    int s{1};
    int sp;
    double denu = enu_tolerance_;

    /* 1st pass: estimate upper and lower boundaries */
    enu_ = integrate_forward_until(rel__, enu_start__, l_, k_, chi_p, chi_q, p, dpdr_, q, dqdr, true,
                                   [&s, &sp, &denu, alpha0__, alpha1__, this](int iter, int nn, double& enu) {
                                       sp = s;
                                       s  = (nn > (n_ - l_ - 1)) ? -1 : 1;
                                       if (s != sp && iter > 3) {
                                           return true;
                                       }
                                       denu = (s != sp) ? denu * alpha0__ : denu * alpha1__;
                                       /* do not allow big step in energy */
                                       denu = std::min(denu, 10.0);
                                       enu += s * denu;
                                       return false;
                                   });

    double e1 = enu_;
    double e2 = enu_ - sp * denu;

    std::stringstream sout;

    /* e1 is bottom, e2 is top energy */
    if (e1 > e2) {
        std::swap(e1, e2);
    }

    /* 2nd pass: refine by bisection */
    enu_ = integrate_forward_until(rel__, (e1 + e2) / 2, l_, k_, chi_p, chi_q, p, dpdr_, q, dqdr, true,
                                   [&e1, &e2, this](int iter, int nn, double& enu) {
                                       if (nn > (n_ - l_ - 1)) {
                                           e2 = enu;
                                       } else {
                                           e1 = enu;
                                       }
                                       enu = (e1 + e2) / 2.0;
                                       return std::abs(e1 - e2) < enu_tolerance_;
                                   });

    /* final choice for enu: bottom enery of the refined interval */
    /* radial functions are also recomputed for the final time */
    enu_ = integrate_forward_until(rel__, e1, l_, k_, chi_p, chi_q, p, dpdr_, q, dqdr, true,
                                   [](int iter, int nn, double& enu) { return true; });

    sout << "final value for enu = " << enu_ << std::endl;

    /* compute r * u'(r) */
    for (int i = 0; i < num_points(); i++) {
        rdudr[i] = dpdr_[i] - p[i] / radial_grid(i);
    }

    for (int i = 0; i < np; i++) {
        p_(i)     = p[i];
        q_(i)     = q[i];
        u_(i)     = p[i] * radial_grid_.x_inv(i);
        rdudr_(i) = rdudr[i];
    }
    p_.interpolate();
    q_.interpolate();
    u_.interpolate();
    rdudr_.interpolate();

    /* p is not divided by r, so we integrate with r^0 prefactor */
    double norm = inner(p_, p_, 0);
    if (rel__ == relativity_t::dirac) {
        norm += inner(q_, q_, 0);
    }

    norm = 1.0 / std::sqrt(norm);
    p_.scale(norm);
    q_.scale(norm);
    u_.scale(norm);
    rdudr_.scale(norm);

    /* count number of nodes of the function */
    int nn{0};
    std::vector<double> xn;
    for (int i = 0; i < np - 1; i++) {
        if (p_(i) * p_(i + 1) < 0.0) {
            nn++;
            xn.push_back(radial_grid(i));
        }
    }

    if (nn != (n_ - l_ - 1)) {
        nlohmann::json dict;
        dict["x"]  = radial_grid().values();
        dict["p"]  = p_.values();
        dict["ve"] = this->ve_.values();
        dict["z"]  = this->zn_;
        write_json_to_file(dict, "radial_solver_zn" + std::to_string(this->zn_) + "_n" + std::to_string(this->n_) +
                                         "_l" + std::to_string(this->l_) + "_k" + std::to_string(this->k_) + ".json");

        sout << "n = " << n_ << std::endl
             << "l = " << l_ << std::endl
             << "k = " << k_ << std::endl
             << "wrong number of nodes : " << nn << " instead of " << (n_ - l_ - 1) << std::endl;
        sout << "node location : ";
        for (auto e : xn) {
            sout << e << " ";
        }
        sout << std::endl << "last value of ve : " << this->ve_(np - 1);
        RTE_THROW(sout);
    }

    for (int i = 0; i < np - 1; i++) {
        rho_(i) += std::pow(u_(i), 2);
        if (rel__ == relativity_t::dirac) {
            rho_(i) += std::pow(q_(i) * radial_grid_.x_inv(i), 2);
        }
    }
}

}; // namespace sirius
