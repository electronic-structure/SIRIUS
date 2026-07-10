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

    gsl_integration_glfixed_table* table =
        gsl_integration_glfixed_table_alloc(static_cast<size_t>(n__));

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
        : knots_(make_interp_knots(rgrid__, order, num_inner_points__))
    {
        std::vector<double> xg;
        std::vector<double> wg;
        gauss_legendre_rule(nq__, xg, wg);

        int n = this->size() - 1;

        for (int i = 0; i < n; i++) {
            for (int ik = 0; ik < order; ik++) {
                double ai = this->knot(i + ik);
                double bi = this->knot(i + ik + 1);
                if (bi <= ai) {
                    continue;
                }

                for (int j = 0; j < n; j++) {
                    for (int jk = 0; jk < order; jk++) {
                        double a = std::max(ai, this->knot(j + jk));
                        double b = std::min(bi, this->knot(j + jk + 1));
                        if (b <= a) {
                            continue;
                        }

                        bspline_basis_pair p;
                        p.i = i;
                        p.j = j;
                        for (int iq = 0; iq < nq__; iq++) {
                            double r = 0.5 * ((b - a) * xg[iq] + (b + a));
                            double w = 0.5 * (b - a) * wg[iq];

                            double Bi  = this->operator()(i, r);
                            double dBi = this->deriv(i, r);
                            double Bj  = this->operator()(j, r);
                            double dBj = this->deriv(j, r);

                            p.r.push_back(r);
                            p.w.push_back(w);
                            p.Bi.push_back(Bi);
                            p.dBi.push_back(dBi);
                            p.Bj.push_back(Bj);
                            p.dBj.push_back(dBj);
                        }
                        basis_pairs_.push_back(p);
                    }
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

} // namespace

#endif
