/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __SPHERIC_FUNCTION_INTERPOLATOR_HPP__
#define __SPHERIC_FUNCTION_INTERPOLATOR_HPP__

#include "function3d/spheric_function.hpp"

namespace sirius {

enum class spheric_inner_extension_t
{
    none,
    charge,
    kinetic
};

/// Sample an atom-local real-harmonic field and return derivatives to its radial samples.
/** Coordinates are Cartesian displacements from the source atom, in bohr. The
 *  caller supplies the displacements for every contributing periodic image.
 *  Values and gradients use the same not-a-knot spline, without independent
 *  interpolation of precomputed gradients. An optional inner continuation uses
 *  r^p times a quadratic, matching value and the first two radial derivatives
 *  at the first sample. Charge multipoles use p=l. Positive kinetic density can
 *  contain r^(l-1) terms from cusp/angular-gradient products (in particular a
 *  nonvanishing s/p dipole), so it uses p=max(l-1,0). This is a specified finite-
 *  grid continuation, not a substitute for setup/radial convergence. Exact
 *  nuclei remain excluded because cusp gradients need not have a unique limit.
 *  Outside-support truncation belongs to the reconstruction.
 */
class Spheric_function_interpolator
{
  private:
    Radial_grid<double> const& grid_;
    int lmax_;
    std::vector<Spline<double>> radial_;
    spheric_inner_extension_t inner_;
    std::vector<int> inner_power_;

    struct gradient_coupling_t
    {
        int from, to, angular;
        r3::vector<double> weight;
    };
    std::vector<gradient_coupling_t> gradient_couplings_;

    // Transform the spherical-gradient couplings used by gradient(Flm) to real
    // harmonics and Cartesian components. Retain the extra l shell for a Hessian.
    void
    build_gradient_couplings()
    {
        double isqrt2 = 1 / std::sqrt(2.0);
        for (int l = 0; l <= lmax_ + 1; l++) {
            for (int m = -l; m <= l; m++) {
                for (int lp : {l - 1, l + 1}) {
                    if (lp < 0) {
                        continue;
                    }
                    std::vector<r3::vector<double>> weights(2 * lp + 1);
                    for (int k = -std::abs(m); k <= std::abs(m); k += std::max(1, 2 * std::abs(m))) {
                        auto input = SHT::ylm_dot_rlm(l, k, m);
                        for (int mu = -1; mu <= 1; mu++) {
                            int kp = k + mu;
                            if (std::abs(kp) > lp) {
                                continue;
                            }
                            double d = lp > l ? std::sqrt(double(l + 1) / (2 * l + 3))
                                              : -std::sqrt(double(l) / (2 * l - 1));
                            d *= SHT::clebsch_gordan(l, 1, lp, k, mu, kp);
                            std::array<std::complex<double>, 3> cart{};
                            if (mu == 0) {
                                cart[2] = 1;
                            } else {
                                cart[0] = -mu * isqrt2;
                                cart[1] = std::complex<double>(0, isqrt2);
                            }
                            for (int mp = -std::abs(kp); mp <= std::abs(kp); mp += std::max(1, 2 * std::abs(kp))) {
                                auto factor = d * input * SHT::rlm_dot_ylm(lp, mp, kp);
                                for (int x = 0; x < 3; x++) {
                                    weights[mp + lp][x] += std::real(factor * cart[x]);
                                }
                            }
                        }
                    }
                    for (int mp = -lp; mp <= lp; mp++) {
                        auto w = weights[mp + lp];
                        if (w[0] != 0 || w[1] != 0 || w[2] != 0) {
                            gradient_couplings_.push_back({sf::lm(l, m), sf::lm(lp, mp), lp > l ? -l : l + 1, w});
                        }
                    }
                }
            }
        }
    }

    struct point_t
    {
        int segment;
        double dr;
        double radius;
        r3::vector<double> direction;
        std::vector<double> harmonics;
        mdarray<double, 2> gradient;
    };

    point_t
    point(r3::vector<double> position__) const
    {
        for (int x = 0; x < 3; x++) {
            if (!std::isfinite(position__[x])) {
                RTE_THROW("spherical interpolation: nonfinite coordinate");
            }
        }
        auto spherical = r3::spherical_coordinates(position__);
        double r       = spherical[0];
        if ((r < grid_.first() && inner_ == spheric_inner_extension_t::none) || r > grid_.last() || r < 1e-12) {
            RTE_THROW("spherical interpolation: point outside radial support or at the nucleus");
        }
        int segment = r < grid_.first() ? 0 : (r == grid_.last() ? grid_.num_points() - 2 : grid_.index_of(r));
        if (segment < 0) {
            RTE_THROW("spherical interpolation: radial segment not found");
        }
        int n = sf::lmmax(lmax_);
        point_t p{segment,
                  r - grid_[segment],
                  r,
                  (1.0 / r) * position__,
                  std::vector<double>(n),
                  mdarray<double, 2>({n, 3})};
        sf::spherical_harmonics(lmax_, position__, p.harmonics.data());
        sf::dRlm_dr(lmax_, position__, p.gradient);
        return p;
    }

    // Weights on spline coefficients, shared by value/gradient evaluation and its exact transpose.
    std::array<std::array<double, 4>, 3>
    radial_weights(point_t const& p__, int lm__) const
    {
        double d = p__.dr;
        if (p__.radius >= grid_.first()) {
            return {{{1, d, d * d, d * d * d}, {0, 1, 2 * d, 3 * d * d}, {0, 0, 2, 6 * d}}};
        }
        double r0 = grid_.first();
        double u = p__.radius / r0, v = u - 1;
        int p     = inner_power_[lm__];
        double up = std::pow(u, p), dup = p == 0 ? 0 : p * std::pow(u, p - 1) / r0;
        double ddup = p < 2 ? 0 : p * (p - 1) * std::pow(u, p - 2) / (r0 * r0);
        std::array<double, 4> a{1 - p * v + 0.5 * p * (p + 1) * v * v, r0 * (v - p * v * v), r0 * r0 * v * v, 0};
        std::array<double, 4> da{(-p + p * (p + 1) * v) / r0, 1 - 2 * p * v, 2 * r0 * v, 0};
        std::array<double, 4> dda{p * (p + 1) / (r0 * r0), -2.0 * p / r0, 2, 0};
        std::array<std::array<double, 4>, 3> result{};
        for (int j = 0; j < 4; j++) {
            result[0][j] = up * a[j];
            result[1][j] = dup * a[j] + up * da[j];
            result[2][j] = ddup * a[j] + 2 * dup * da[j] + up * dda[j];
        }
        return result;
    }

  public:
    explicit Spheric_function_interpolator(Flm const& field__,
                                           spheric_inner_extension_t inner__ = spheric_inner_extension_t::none)
        : grid_(field__.radial_grid())
        , lmax_(sf::lmax(field__.angular_domain_size()))
        , inner_(inner__)
        , inner_power_(sf::l_by_lm(lmax_))
    {
        if (grid_.num_points() < 4 || grid_.first() < 0 || sf::lmmax(lmax_) != field__.angular_domain_size()) {
            RTE_THROW("spherical interpolation: invalid grid or harmonic extent");
        }
        for (int lm = 0; lm < field__.angular_domain_size(); lm++) {
            radial_.push_back(field__.component(lm));
            if (inner_ == spheric_inner_extension_t::kinetic) {
                inner_power_[lm] = std::max(0, inner_power_[lm] - 1);
            }
        }
        build_gradient_couplings();
    }

    /// Return {field, d(field)/dx, d(field)/dy, d(field)/dz}.
    std::array<double, 4>
    value_gradient(r3::vector<double> const& position__) const
    {
        auto p = point(position__);
        std::array<double, 4> result{};
        for (int lm = 0; lm < static_cast<int>(radial_.size()); lm++) {
            auto weights = radial_weights(p, lm);
            double f{0}, df{0};
            for (int j = 0; j < 4; j++) {
                f += weights[0][j] * radial_[lm].coeffs()(p.segment, j);
                df += weights[1][j] * radial_[lm].coeffs()(p.segment, j);
            }
            result[0] += f * p.harmonics[lm];
            for (int x = 0; x < 3; x++) {
                result[x + 1] += df * p.direction[x] * p.harmonics[lm] + f * p.gradient(lm, x);
            }
        }
        return result;
    }

    /// Cartesian position derivative of a raw value/gradient energy covector.
    /** Radial samples are held fixed. The gradient response uses the Hessian
     *  of the same spline and inner continuation, without interpolating a
     *  separately sampled derivative. Atom translations give the negative
     *  displacement derivative; periodic-image cell terms belong to the caller.
     */
    r3::vector<double>
    position_adjoint(r3::vector<double> const& position__, std::array<double, 4> const& covector__) const
    {
        for (double d : covector__) {
            if (!std::isfinite(d)) {
                RTE_THROW("spherical geometry adjoint: nonfinite covector");
            }
        }
        if (covector__[1] == 0 && covector__[2] == 0 && covector__[3] == 0) {
            auto v = value_gradient(position__);
            return covector__[0] * r3::vector<double>({v[1], v[2], v[3]});
        }
        auto p = point(position__);
        int n  = static_cast<int>(radial_.size());
        std::vector<std::array<double, 3>> radial(n);
        for (int lm = 0; lm < n; lm++) {
            auto weights = radial_weights(p, lm);
            for (int a = 0; a < 3; a++) {
                for (int j = 0; j < 4; j++) {
                    radial[lm][a] += weights[a][j] * radial_[lm].coeffs()(p.segment, j);
                }
            }
        }
        std::vector<r3::vector<double>> grad(sf::lmmax(lmax_ + 1)), dgrad(grad.size());
        for (auto const& c : gradient_couplings_) {
            if (c.from < n) {
                auto f = radial[c.from];
                grad[c.to] += (f[1] + c.angular * f[0] / p.radius) * c.weight;
                dgrad[c.to] += (f[2] + c.angular * (f[1] - f[0] / p.radius) / p.radius) * c.weight;
            }
        }
        std::vector<double> harmonics(sf::lmmax(lmax_ + 2));
        sf::spherical_harmonics(lmax_ + 2, position__, harmonics.data());
        r3::vector<double> result, dg({covector__[1], covector__[2], covector__[3]});
        for (size_t lm = 0; lm < grad.size(); lm++) {
            result += (covector__[0] * harmonics[lm]) * grad[lm];
        }
        for (auto const& c : gradient_couplings_) {
            double value = r3::dot(dg, dgrad[c.from] + (c.angular / p.radius) * grad[c.from]);
            result += (value * harmonics[c.to]) * c.weight;
        }
        return result;
    }

    /// Euclidean transpose of the combined value/Cartesian-gradient sampling.
    /** dvalues__ contains derivatives of the complete energy, including any
     *  integration/partition weights already applied by the evaluator. The
     *  returned entries are dE/d(field_lm(r_i)), not a radial potential to be
     *  integrated again. Duplicate points/images add, and empty point lists
     *  return zero. The original radial grid must outlive this object/results.
     */
    Flm
    adjoint(std::vector<r3::vector<double>> const& positions__,
            std::vector<std::array<double, 4>> const& dvalues__) const
    {
        if (positions__.size() != dvalues__.size()) {
            RTE_THROW("spherical interpolation adjoint: incompatible point derivatives");
        }
        int n  = static_cast<int>(radial_.size());
        int nr = grid_.num_points();
        mdarray<double, 3> coeffs({nr, 4, n});
        coeffs.zero();
        for (size_t i = 0; i < positions__.size(); i++) {
            auto p = point(positions__[i]);
            for (int lm = 0; lm < n; lm++) {
                double f = dvalues__[i][0] * p.harmonics[lm];
                double df{0};
                for (int x = 0; x < 3; x++) {
                    f += dvalues__[i][x + 1] * p.gradient(lm, x);
                    df += dvalues__[i][x + 1] * p.direction[x] * p.harmonics[lm];
                }
                auto weights = radial_weights(p, lm);
                for (int j = 0; j < 4; j++) {
                    coeffs(p.segment, j, lm) += weights[0][j] * f + weights[1][j] * df;
                }
            }
        }
        Flm result(n, grid_);
        for (int lm = 0; lm < n; lm++) {
            mdarray<double, 2> component({nr, 4}, &coeffs(0, 0, lm));
            auto values = radial_[lm].interpolation_adjoint(component);
            for (int ir = 0; ir < nr; ir++) {
                result(lm, ir) = values[ir];
            }
        }
        return result;
    }
};

} // namespace sirius

#endif
