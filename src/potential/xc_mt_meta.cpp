/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "xc_mt_meta.hpp"

namespace sirius {

spheric_xc_derivatives_t
xc_mt_meta(SHT const& sht__, std::vector<XC_functional> const& functionals__, std::vector<Flm> const& rho__,
           std::vector<Flm> const& tau__)
{
    int ns = static_cast<int>(rho__.size());
    if ((ns != 1 && ns != 2) || tau__.size() != rho__.size()) {
        RTE_THROW("spherical meta-GGA requires one or two matching spin channels");
    }
    auto const& grid = rho__[0].radial_grid();
    int nr = grid.num_points(), lmmax = rho__[0].angular_domain_size(), nt = sht__.num_points();
    int np = nr * nt, ng = ns == 1 ? 1 : 3;
    if (nr < 4 || grid.first() <= 0 || lmmax > sht__.lmmax()) {
        RTE_THROW("spherical meta-GGA: invalid radial or angular quadrature");
    }
    for (int s = 0; s < ns; s++) {
        if (rho__[s].angular_domain_size() != lmmax || tau__[s].angular_domain_size() != lmmax ||
            rho__[s].radial_grid().hash() != grid.hash() || tau__[s].radial_grid().hash() != grid.hash()) {
            RTE_THROW("spherical meta-GGA: incompatible fields");
        }
    }
    // The energy integrates the spline of r^2 times its angular integral, as inner() does.
    Spline<double> prototype(grid);
    mdarray<double, 2> integral({nr, 4});
    integral.zero();
    for (int ir = 0; ir < nr - 1; ir++) {
        double h = grid[ir + 1] - grid[ir];
        for (int j = 0; j < 4; j++) {
            integral(ir, j) = std::pow(h, j + 1) / (j + 1);
        }
    }
    auto radial_weights = prototype.interpolation_adjoint(integral);
    std::vector<double> weights(np), n(np * ns), t(np * ns), sigma(np * ng);
    mdarray<double, 3> angular_gradient({lmmax, 3, nt});
    for (int p = 0; p < nt; p++) {
        mdarray<double, 2> g({lmmax, 3}, &angular_gradient(0, 0, p));
        auto direction = sht__.coord(p);
        sf::dRlm_dr(sf::lmax(lmmax), direction, g);
        for (int ir = 0; ir < nr; ir++) {
            weights[ir * nt + p] = fourpi * sht__.weight(p) * radial_weights[ir] * grid[ir] * grid[ir];
        }
    }
    mdarray<double, 3> gradients({np, 3, ns});
    gradients.zero();
    for (int s = 0; s < ns; s++) {
        for (int lm = 0; lm < lmmax; lm++) {
            auto spline = rho__[s].component(lm);
            for (int ir = 0; ir < nr; ir++) {
                int segment  = std::min(ir, nr - 2);
                double h     = grid[ir] - grid[segment];
                double slope = spline.coeffs()(segment, 1) + 2 * h * spline.coeffs()(segment, 2) +
                               3 * h * h * spline.coeffs()(segment, 3);
                for (int p = 0; p < nt; p++) {
                    int point = ir * nt + p;
                    double y  = sht__.rlm_backward(lm, p);
                    n[point * ns + s] += y * rho__[s](lm, ir);
                    t[point * ns + s] += y * tau__[s](lm, ir);
                    for (int x = 0; x < 3; x++) {
                        gradients(point, x, s) += slope * sht__.coord(x, p) * y +
                                                  rho__[s](lm, ir) * angular_gradient(lm, x, p) / grid[ir];
                    }
                }
            }
        }
    }
    for (int p = 0; p < np; p++) {
        for (int s = 0; s < ns; s++) {
            if (!std::isfinite(n[p * ns + s]) || !std::isfinite(t[p * ns + s]) || n[p * ns + s] < 0 ||
                t[p * ns + s] < 0) {
                RTE_THROW("spherical meta-GGA requires finite nonnegative spin density and tau");
            }
        }
        for (int x = 0; x < 3; x++) {
            sigma[p * ng] += std::pow(gradients(p, x, 0), 2);
            if (ns == 2) {
                sigma[p * ng + 1] += gradients(p, x, 0) * gradients(p, x, 1);
                sigma[p * ng + 2] += std::pow(gradients(p, x, 1), 2);
            }
        }
    }
    spheric_xc_derivatives_t result;
    std::vector<double> vrho(np * ns), vtau(np * ns), vsigma(np * ng), exc(np);
    std::vector<double> drho(np * ns), dtau(np * ns), dsigma(np * ng);
    for (auto const& functional : functionals__) {
        std::fill(vtau.begin(), vtau.end(), 0);
        std::fill(vsigma.begin(), vsigma.end(), 0);
        if (functional.is_meta_gga()) {
            functional.get_meta(np, n.data(), sigma.data(), t.data(), vrho.data(), vsigma.data(), vtau.data(),
                                exc.data());
        } else if (functional.is_gga()) {
            xc_gga_exc_vxc(functional.handler(), np, n.data(), sigma.data(), exc.data(), vrho.data(), vsigma.data());
        } else if (functional.is_lda()) {
            xc_lda_exc_vxc(functional.handler(), np, n.data(), exc.data(), vrho.data());
        } else {
            RTE_THROW("spherical meta-GGA supports only LDA/GGA/positive-tau meta-GGA terms");
        }
        for (int p = 0; p < np; p++) {
            double weight = functional.weight() * weights[p];
            for (int s = 0; s < ns; s++) {
                result.energy += weight * n[p * ns + s] * exc[p];
                drho[p * ns + s] += weight * vrho[p * ns + s];
                dtau[p * ns + s] += weight * vtau[p * ns + s];
            }
            for (int g = 0; g < ng; g++) {
                dsigma[p * ng + g] += weight * vsigma[p * ng + g];
            }
        }
    }
    for (int s = 0; s < ns; s++) {
        result.fields[0].emplace_back(lmmax, grid);
        result.fields[1].emplace_back(lmmax, grid);
        result.fields[0].back().zero();
        result.fields[1].back().zero();
        for (int lm = 0; lm < lmmax; lm++) {
            mdarray<double, 2> coefficients({nr, 4});
            coefficients.zero();
            for (int ir = 0; ir < nr; ir++) {
                double slope_covector{0};
                for (int p = 0; p < nt; p++) {
                    int point = ir * nt + p;
                    double y  = sht__.rlm_backward(lm, p);
                    result.fields[0][s](lm, ir) += y * drho[point * ns + s];
                    result.fields[1][s](lm, ir) += y * dtau[point * ns + s];
                    for (int x = 0; x < 3; x++) {
                        double flux = 2 * dsigma[point * ng + 2 * s] * gradients(point, x, s);
                        if (ns == 2) {
                            flux += dsigma[point * ng + 1] * gradients(point, x, 1 - s);
                        }
                        result.fields[0][s](lm, ir) += flux * angular_gradient(lm, x, p) / grid[ir];
                        slope_covector += flux * sht__.coord(x, p) * y;
                    }
                }
                int segment = std::min(ir, nr - 2);
                double h    = grid[ir] - grid[segment];
                coefficients(segment, 1) += slope_covector;
                coefficients(segment, 2) += 2 * h * slope_covector;
                coefficients(segment, 3) += 3 * h * h * slope_covector;
            }
            auto derivative = prototype.interpolation_adjoint(coefficients);
            for (int ir = 0; ir < nr; ir++) {
                result.fields[0][s](lm, ir) += derivative[ir];
            }
        }
        for (int field = 0; field < 2; field++) {
            for (size_t i = 0; i < result.fields[field][s].size(); i++) {
                if (!std::isfinite(result.fields[field][s][i])) {
                    RTE_THROW("spherical meta-GGA returned a nonfinite field derivative");
                }
            }
        }
    }
    if (!std::isfinite(result.energy)) {
        RTE_THROW("spherical meta-GGA returned a nonfinite energy");
    }
    return result;
}

} // namespace sirius
