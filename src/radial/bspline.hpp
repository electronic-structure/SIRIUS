/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file bspline.hpp
 *
 *  \brief Implementation of B-Spline
 */

#ifndef __BSPLINE_HPP__
#define __BSPLINE_HPP__

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include "radial_grid.hpp"

namespace sirius {

inline auto
make_interp_knots(Radial_grid<double> const& grid__, int order__, int num_points__)
{
    RTE_ASSERT(num_points__ > order__);

    std::vector<double> knots(num_points__ + order__);

    double x0 = grid__.first();
    double x1 = grid__.last();

    for (int i = 0; i < order__; i++) {
        knots[i] = x0;
    }

    int num_inner = num_points__ - order__;
    Radial_grid_lin<double> inner_grid(num_inner + 2, x0, x1);
    for (int i = 0; i < num_inner; i++) {
        knots[order__ + i] = inner_grid.x(i + 1);
    }

    for (int i = num_points__; i < num_points__ + order__; i++) {
        knots[i] = x1;
    }

    return knots;
}

inline void
gauss_legendre_rule(int n__, std::vector<double>& x__, std::vector<double>& w__)
{
    if (n__ <= 0) {
        RTE_THROW("gauss_legendre_rule: n must be positive");
    }

    x__.resize(n__);
    w__.resize(n__);

    gsl_integration_glfixed_table* table = gsl_integration_glfixed_table_alloc(static_cast<size_t>(n__));

    if (table == nullptr) {
        RTE_THROW("gauss_legendre_rule: failed to allocate GSL table");
    }

    for (int i = 0; i < n__; i++) {
        double xi;
        double wi;
        int status = gsl_integration_glfixed_point(-1.0, 1.0, static_cast<size_t>(i), &xi, &wi, table);

        if (status != GSL_SUCCESS) {
            gsl_integration_glfixed_table_free(table);
            RTE_THROW("gauss_legendre_rule: gsl_integration_glfixed_point failed");
        }

        x__[i] = xi;
        w__[i] = wi;
    }
    gsl_integration_glfixed_table_free(table);
}

struct bspline_basis_pair
{
    int i;
    int j;
    std::vector<double> w;
    std::vector<double> r;
    std::vector<double> Bi;
    std::vector<double> dBi;
    std::vector<double> Bj;
    std::vector<double> dBj;
};

template <int order>
/// B-spline basis of fixed order on a given knot sequence.
/** The order of a B-spline is one plus its polynomial degree. Basis function \f$ B_{i,k}(x) \f$ of order
 *  \f$ k \f$ has support on \f$ [t_i, t_{i+k}] \f$, where \f$ t_i \f$ are knots. The implementation uses the
 *  Cox-de Boor recursion formula and assumes a nondecreasing knot sequence. */
class bspline_basis
{
  private:
    /// Knot sequence.
    std::vector<double> knots_;

    std::vector<bspline_basis_pair> basis_pairs_;

    /// Evaluate a B-spline basis function of a given recursive order.
    /** \param [in] i  Index of the B-spline.
     *  \param [in] k  Recursive order.
     *  \param [in] x  Point at which the B-spline is evaluated.
     *  \return Value of \f$ B_{i,k}(x) \f$.
     */
    double
    value(int i__, int k__, double x__) const
    {
        if (k__ == 1) {
            return (knots_[i__] <= x__ && x__ < knots_[i__ + 1]) ? 1.0 : 0.0;
        }

        double v{0};

        double d0 = knots_[i__ + k__ - 1] - knots_[i__];
        if (d0 > 0) {
            v += (x__ - knots_[i__]) / d0 * value(i__, k__ - 1, x__);
        }

        double d1 = knots_[i__ + k__] - knots_[i__ + 1];
        if (d1 > 0) {
            v += (knots_[i__ + k__] - x__) / d1 * value(i__ + 1, k__ - 1, x__);
        }

        return v;
    }

    /// Evaluate the first derivative of a B-spline basis function.
    /** \param [in] i  Index of the B-spline.
     *  \param [in] k  Recursive order.
     *  \param [in] x  Point at which the derivative is evaluated.
     *  \return Value of \f$ dB_{i,k}(x) / dx \f$.
     */
    double
    deriv(int i__, int k__, double x__) const
    {
        if (k__ == 1) {
            return 0.0;
        }

        double v{0};

        double d0 = knots_[i__ + k__ - 1] - knots_[i__];
        if (d0 > 0) {
            v += double(k__ - 1) / d0 * value(i__, k__ - 1, x__);
        }

        double d1 = knots_[i__ + k__] - knots_[i__ + 1];
        if (d1 > 0) {
            v -= double(k__ - 1) / d1 * value(i__ + 1, k__ - 1, x__);
        }

        return v;
    }

  public:
    /// Constructor.
    bspline_basis()
    {
    }

    /// Constructor.
    bspline_basis(Radial_grid<double> const& rgrid__, int num_inner_points__, int nq__)
        : bspline_basis(make_interp_knots(rgrid__, order, num_inner_points__), nq__)
    {
    }

    /// Construct an evaluation-only basis on a prescribed knot sequence.
    explicit bspline_basis(std::vector<double> knots__)
        : knots_(std::move(knots__))
    {
        if (knots_.size() < 2 * order ||
            !std::all_of(knots_.begin(), knots_.end(), [](double x) { return std::isfinite(x); }) ||
            !std::is_sorted(knots_.begin(), knots_.end()) || knots_.front() == knots_.back()) {
            RTE_THROW("bspline_basis: invalid knot sequence");
        }
    }

    /// Construct on a prescribed nondecreasing knot sequence, including repeated end knots.
    bspline_basis(std::vector<double> knots__, int nq__, bool include_boundary__ = false, double origin_power__ = 1)
        : bspline_basis(std::move(knots__))
    {
        if (!std::isfinite(origin_power__) || origin_power__ < 1) {
            RTE_THROW("bspline_basis: invalid origin power");
        }
        std::vector<double> xg;
        std::vector<double> wg;
        gauss_legendre_rule(nq__, xg, wg);

        int n = this->size() - (include_boundary__ ? 0 : 1);
        struct interval_t
        {
            int first{0};
            std::vector<double> r, w;
            std::vector<std::vector<double>> values, derivatives;
        };
        std::vector<interval_t> intervals(knots_.size() - 1);
        // Each nonzero overlap of two knot intervals is the same global interval.
        for (int q = 0; q < static_cast<int>(intervals.size()); q++) {
            double a = knot(q), b = knot(q + 1);
            if (b <= a) {
                continue;
            }
            auto& interval = intervals[q];
            interval.first = std::max(0, q - order + 1);
            int count      = std::max(0, std::min(n, q + 1) - interval.first);
            interval.values.resize(count);
            interval.derivatives.resize(count);
            for (int iq = 0; iq < nq__; iq++) {
                double r = 0.5 * ((b - a) * xg[iq] + (b + a));
                double w = 0.5 * (b - a) * wg[iq];
                if (a == 0 && origin_power__ != 1) {
                    // r=b*t^p resolves integrable fractional powers at the nucleus.
                    double t = 0.5 * (xg[iq] + 1);
                    r        = b * std::pow(t, origin_power__);
                    w        = 0.5 * b * origin_power__ * std::pow(t, origin_power__ - 1) * wg[iq];
                }
                interval.r.push_back(r);
                interval.w.push_back(w);
                for (int j = 0; j < count; j++) {
                    interval.values[j].push_back((*this)(interval.first + j, r));
                    interval.derivatives[j].push_back(deriv(interval.first + j, r));
                }
            }
        }
        for (int i = 0; i < n; i++) {
            for (int ik = 0; ik < order; ik++) {
                auto const& interval = intervals[i + ik];
                for (int j = 0; j < static_cast<int>(interval.values.size()); j++) {
                    int offset = i - interval.first;
                    basis_pairs_.push_back({i, interval.first + j, interval.w, interval.r, interval.values[offset],
                                            interval.derivatives[offset], interval.values[j], interval.derivatives[j]});
                }
            }
        }
    }

    /// Compile-time order of the B-spline basis.
    static constexpr int order_value{order};

    /// Number of basis functions.
    int
    size() const
    {
        return static_cast<int>(knots_.size()) - order;
    }

    /// Evaluate a B-spline basis function.
    /** \param [in] i  Index of the B-spline.
     *  \param [in] x  Point at which the B-spline is evaluated.
     *  \return Value of \f$ B_{i,k}(x) \f$.
     *
     *  The last point of the domain is treated explicitly because the recursive definition uses half-open knot
     *  intervals. */
    double
    operator()(int i__, double x__) const
    {
        if (x__ == knots_.back()) {
            return (i__ == size() - 1) ? 1.0 : 0.0;
        }

        return value(i__, order, x__);
    }

    /// Evaluate the first derivative of a B-spline basis function.
    /** \param [in] i  Index of the B-spline.
     *  \param [in] x  Point at which the derivative is evaluated.
     *  \return Value of \f$ dB_{i,k}(x) / dx \f$.
     *
     *  At the right boundary the derivative is evaluated as the left-sided limit. */
    double
    deriv(int i__, double x__) const
    {
        if (x__ == knots_.back()) {
            x__ = std::nextafter(x__, knots_.front());
        }
        return deriv(i__, order, x__);
    }

    /// Number of knots.
    int
    num_knots() const
    {
        return static_cast<int>(knots_.size());
    }

    /// Return a knot value.
    /** \param [in] i  Index of the knot. */
    double
    knot(int i__) const
    {
        return knots_[i__];
    }

    auto const&
    basis_pairs() const
    {
        return basis_pairs_;
    }
};

} // namespace sirius

#endif
