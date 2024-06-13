/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file radial_solver.hpp
 *
 *  \brief Contains declaration and partial implementation of sirius::Radial_solver class.
 */

#ifndef __RADIAL_SOLVER_HPP__
#define __RADIAL_SOLVER_HPP__

#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>
#include <tuple>
#include "spline.hpp"
#include "core/constants.hpp"
#include "core/typedefs.hpp"
#include "core/rte/rte.hpp"
#include "core/math_tools.hpp"

namespace sirius {

/// Result of the Radial_solver::solve() method.
struct radial_solver_result_t
{
    /// Number of nodes of the radial function.
    int num_nodes;
    /// p(r) = u(r)*r and u(r) is the radial part of atomic wave-function psi_{nlm}(r) = u_{nl}(r) * Y_{lm}(r)
    std::vector<double> p;
    /// Stores r * u'(r)
    std::vector<double> rdudr;
    /// Stores surface 0-, 1- and 2-order derivatives u(R), u'(R) and u"(R)
    std::array<double, 3> uderiv;
};

namespace radial_solver_local {

double const rest_energy = std::pow(speed_of_light, 2);

double const alpha = 1.0 / speed_of_light;

double const sq_alpha_half = 0.5 / rest_energy;

template <relativity_t rel>
inline double
rel_mass(double enu__, double v__)
{
    switch (rel) {
        case relativity_t::none: {
            return 1.0;
        }
        case relativity_t::koelling_harmon: {
            return 1.0 + sq_alpha_half * (enu__ - v__);
        }
        case relativity_t::zora: {
            return 1.0 - sq_alpha_half * v__;
        }
        case relativity_t::iora: {
            double m0 = 1.0 - sq_alpha_half * v__;
            return m0 / (1 - sq_alpha_half * enu__ / m0);
        }
        default: {
            return 1.0;
        }
    }
};

template <relativity_t rel>
inline double
rel_mass_deriv(double enu__, double v__, double v_deriv__)
{
    switch (rel) {
        case relativity_t::none: {
            return 0.0;
        }
        case relativity_t::koelling_harmon:
        case relativity_t::zora: {
            return -sq_alpha_half * v_deriv__;
        }
        case relativity_t::iora: {
            double K = 1.0 - sq_alpha_half * enu__ / (1.0 - sq_alpha_half * v__);
            return (-sq_alpha_half * v_deriv__ / K +
                    std::pow(sq_alpha_half / K, 2) * enu__ * v_deriv__ / (1 - sq_alpha_half * v__));
        }
        default: {
            return 1.0;
        }
    }
};

} // namespace radial_solver_local

/// Finds a solution to radial Schrodinger, Koelling-Harmon or Dirac equation.
/** Non-relativistic radial Schrodinger equation:
 *  \f[
 *    -\frac{1}{2}p''(r) + V_{eff}(r)p(r) = Ep(r)
 *  \f]
 *  where \f$ V_{eff}(r) = V(r) + \frac{\ell (\ell+1)}{2r^2} \f$, \f$ p(r) = u(r)r \f$ and \f$ u(r) \f$ is the
 *  radial wave-function. Energy derivatives of radial solutions obey the slightly different equation:
 *  \f[
 *    -\frac{1}{2}\dot{p}''(r) + V_{eff}(r)\dot{p}(r) = E\dot{p}(r) + p(r)
 *  \f]
 *  \f[
 *    -\frac{1}{2}\ddot{p}''(r) + V_{eff}(r)\ddot{p}(r) = E\ddot{p}(r) + 2\dot{p}(r)
 *  \f]
 *  and so on. We can generalize the radial Schrodinger equation like this:
 *  \f[
 *    -\frac{1}{2}p''(r) + \big(V_{eff}(r) - E\big) p(r) = \chi(r)
 *  \f]
 *  where now \f$ p(r) \f$ represents m-th energy derivative of the radial solution and
 *  \f$ \chi(r) = m \frac{\partial^{m-1} p(r)} {\partial^{m-1}E} \f$
 *
 *  Let's now decouple second-order differential equation into a system of two first-order equations. From
 *  \f$ p(r) = u(r)r \f$ we have
 *  \f[
 *    p'(r) = u'(r)r + u''(r) = 2q(r) + \frac{p(r)}{r}
 *  \f]
 *  where we have introduced a new variable \f$ q(r) = \frac{u'(r) r}{2} \f$. Differentiating \f$ p'(r) \f$ again
 *  we arrive to the following equation for \f$ q'(r) \f$:
 *  \f[
 *    p''(r) = 2q'(r) + \frac{p'(r)}{r} - \frac{p(r)}{r^2} = 2q'(r) + \frac{2q(r)}{r} + \frac{p(r)}{r^2} -
 *     \frac{p(r)}{r^2} \f] \f[ q'(r) = \frac{1}{2}p''(r) - \frac{q(r)}{r} = \big(V_{eff}(r) - E\big) p(r) -
 *     \frac{q(r)}{r} \chi(r)
 *  \f]
 *  Final expression for a linear system of differential equations for m-th energy derivative is:
 *  \f{eqnarray*}{
 *    p'(r) &=& 2q(r) + \frac{p(r)}{r} \\
 *    q'(r) &=& \big(V_{eff}(r) - E\big) p(r) - \frac{q(r)}{r} - \chi(r)
 *  \f}
 *  Scalar-relativistic Koelling-Harmon equations look similar. Second-order ODE
 *  \f[
 *    p''(r) - \frac{\ell(\ell +1)}{r^2}p(r) = 2 M \big(V - E\big)p(r) + \frac{M'}{M}\big(p'(r) - \frac{p(r)}{r}\big)
 *  \f]
 *  decouples into a system of 1st order ODE:
 *  \f{eqnarray*}{
 *    p'(r) &=& 2Mq(r) + \frac{p(r)}{r} \\
 *    q'(r) &=& \big(V(r) - E + \frac{\ell(\ell+1)}{2Mr^2}\big) p(r) - \frac{q(r)}{r}
 *  \f}
 *  where \f$ M = 1 + \frac{\alpha^2}{2}\big(E-V(r)\big) \f$. Because M depends on energy, the m-th energy
 *  derivatives of the scalar-relativistic solution take a slightly different form. For m=1 we have:
 *  \f{eqnarray*}{
 *    \dot{p}'(r) &=& 2M\dot{q}(r) + \frac{\dot{p}(r)}{r} + \alpha^2 q(r) \\
 *    \dot{q}'(r) &=& \big(V(r) - E + \frac{\ell(\ell+1)}{2Mr^2}\big) \dot{p}(r) - \frac{\dot{q}(r)}{r} -
 *     \big(1 + \frac{\ell(\ell+1)\alpha^2}{4M^2r^2}\big)p(r)
 *  \f}
 *  For m=2:
 *  \f{eqnarray*}{
 *    \ddot{p}'(r) &=& 2M\ddot{q}(r) + \frac{\ddot{p}(r)}{r} + 2 \alpha^2 \dot{q}(r) \\
 *    \ddot{q}'(r) &=& \big(V(r) - E + \frac{\ell(\ell+1)}{2Mr^2}\big) \ddot{p}(r) - \frac{\ddot{q}(r)}{r} -
 *     2 \big(1 + \frac{\ell(\ell+1)\alpha^2}{4M^2r^2}\big)\dot{p}(r) + \frac{\ell(\ell+1)\alpha^4}{4M^3r^2} p(r)
 *  \f}
 *
 *  Derivation of IORA.
 *
 *  First thing to do is to expand the \f$ 1/M \f$ to the first order in \f$ E \f$:
 *  \f[
 *    \frac{1}{M} = \frac{1}{1-\frac{\alpha^2}{2} V(r)} -
 *     \frac{\alpha^2 E}{2 (1-\frac{\alpha^2}{2} V(r))^2} + O(E^2) =
 *     \frac{1}{M_0} - \frac{\alpha^2}{2} \frac{1}{M_0^2} E
 *  \f]
 *  From this expansion we can derive the expression for \f$ M \f$:
 *  \f[
 *    M = \Big(\frac{1}{M}\Big)^{-1} = \frac{M_0}{1 - \frac{\alpha^2}{2}\frac{E}{M_0}}
 *  \f]
 *  Now we insert this expressions into the scalar-relativistic equations:
 *  \f[
 *    \left\{
 *    \begin{array}{ll}
 *    \displaystyle p'(r) = 2 \frac{M_0}{1 - \frac{\alpha^2}{2}\frac{E}{M_0}} q(r) + \frac{p(r)}{r} \\
 *    \displaystyle q'(r) = \Big(V(r) - E + \frac{\ell(\ell+1)}{2r^2} \big(\frac{1}{M_0} - \frac{\alpha^2}{2}
 *                           \frac{E}{M_0^2} \big) \Big) p(r) - \frac{q(r)}{r}
 *                        = \Big(V(r) - E + \frac{\ell(\ell+1)}{2 M_0 r^2} -
 *                           \frac{\alpha^2}{2} \frac{\ell(\ell+1) E}{2 M_0^2 r^2} \Big) p(r) -
 *                           \frac{q(r)}{r} \end{array} \right.
 *  \f]
 *  The first energy derivatives are then:
 *  \f[
 *    \left\{
 *    \begin{array}{ll}
 *    \displaystyle \dot{p}'(r) = 2 \frac{M_0}{1 - \frac{\alpha^2}{2}\frac{E}{M_0}} \dot{q}(r) +
 *                                \frac{\dot{p}(r)}{r} + \frac{\alpha^2}{(1 - \frac{\alpha^2 E}{2 M_0})^2} q(r)  \\
 *    \displaystyle \dot{q}'(r) = \Big(V(r) - E + \frac{\ell(\ell+1)}{2 M_0 r^2} - \frac{\alpha^2}{2}
 *                                \frac{\ell(\ell+1)}{2 M_0^2 r^2} E \Big) \dot{p}(r) -
 *                                \frac{\dot{q}(r)}{r} -
 *                                \Big(1 + \frac{\alpha^2}{2} \frac{\ell(\ell+1)}{2 M_0^2 r^2} \Big) p(r)
 *    \end{array}
 *    \right.
 *  \f]
 *  The second energy derivatives are:
 *  \f[
 *    \left\{
 *    \begin{array}{ll}
 *    \displaystyle \ddot{p}'(r) = 2 \frac{M_0}{1 - \frac{\alpha^2}{2}\frac{E}{M_0}} \ddot{q}(r) +
 *                                 \frac{\ddot{p}(r)}{r} +
 *                                 \frac{2 \alpha^2}{(1 - \frac{\alpha^2 E}{2 M_0})^2} \dot{q}(r) +
 *                                 \frac{\alpha^4}{2 M_0 (1 - \frac{\alpha^2 E}{2 M_0})^3} q(r) \\
 *    \displaystyle \ddot{q}'(r) = \Big(V(r) - E + \frac{\ell(\ell+1)}{2 M_0 r^2} - \frac{\alpha^2}{2}
 *                                 \frac{\ell(\ell+1)}{2 M_0^2 r^2} E \Big) \ddot{p}(r) -
 *                                  \frac{\ddot{q}(r)}{r} -
 *                                  2 \Big(1 + \frac{\alpha^2}{2} \frac{\ell(\ell+1)}{2 M_0^2 r^2} \Big) \dot{p}(r)
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  \section ederiv Full derivation of energy derivatives.
 *
 *  Schroedinger equation with \f$ M = 1 \f$. Energy derivatives of p:
 *  \f[
 *    \begin{array}{ll}
 *      \displaystyle \frac{\partial }{\partial \epsilon ^1}\left(2 M(\epsilon ) q(\epsilon )+
 *      \frac{p(\epsilon )}{r}\right) = \frac{p'(\epsilon )}{r}+2 q'(\epsilon ) \\
 *    \displaystyle \frac{\partial ^2\left(2 M(\epsilon ) q(\epsilon )+
 *    \frac{p(\epsilon )}{r}\right)}{\partial \epsilon ^2} = \frac{p''(\epsilon )}{r}+2 q''(\epsilon ) \\
 *    \displaystyle \frac{\partial ^3\left(2 M(\epsilon ) q(\epsilon )+
 *     \frac{p(\epsilon )}{r}\right)}{\partial \epsilon ^3} = \frac{p^{(3)}(\epsilon )}{r}+2 q^{(3)}(\epsilon )
 *    \end{array}
 *  \f]
 *  Energy derivatives of q:
 *  \f[
 *    \begin{array}{ll}
 *    \displaystyle \frac{\partial }{\partial \epsilon ^1}\left(\left(V-\epsilon +\frac{b}{r^2 M(\epsilon )}\right)
 *      p(\epsilon )-\frac{q(\epsilon )}{r}\right) = p'(\epsilon ) \left(\frac{b}{r^2}+V-\epsilon \right)-
 *    p(\epsilon )-\frac{q'(\epsilon )}{r} \\
 *    \displaystyle \frac{\partial ^2}{\partial \epsilon ^2}\left(\left(V-\epsilon +\frac{b}{r^2 M(\epsilon )}\right)
 *      p(\epsilon )-\frac{q(\epsilon )}{r}\right) = p''(\epsilon ) \left(\frac{b}{r^2}+V-\epsilon \right)-
 *      2 p'(\epsilon )-\frac{q''(\epsilon )}{r} \\
 *    \displaystyle \frac{\partial ^3}{\partial \epsilon ^3}\left(\left(V-\epsilon +\frac{b}{r^2 M(\epsilon )}\right)
 *      p(\epsilon )-\frac{q(\epsilon )}{r}\right) = p^{(3)}(\epsilon ) \left(\frac{b}{r^2}+V-\epsilon \right)-
 *       3 p''(\epsilon )-\frac{q^{(3)}(\epsilon )}{r}
 *    \end{array}
 *  \f]
 *  Here and below \f$ a = \alpha^2/2 \f$, \f$ b = \ell(\ell + 1) / 2 \f$.
 *
 *  Koelling-Harmon equation with \f$ M = 1 + \alpha^2 (\epsilon - V) / 2 \f$. Energy derivatives of p:
 *  \f[
 *    \begin{array}{ll}
 *     \displaystyle \frac{\partial }{\partial \epsilon ^1}\left(2 M(\epsilon ) q(\epsilon )+
 *      \frac{p(\epsilon )}{r}\right) = 2 a q(\epsilon )+2 M q'(\epsilon )+\frac{p'(\epsilon )}{r} \\
 *     \displaystyle \frac{\partial ^2}{\partial \epsilon ^2}\left(2 M(\epsilon ) q(\epsilon )+
 *       \frac{p(\epsilon )}{r}\right) = 4 a q'(\epsilon )+2 M q''(\epsilon )+\frac{p''(\epsilon )}{r} \\
 *     \displaystyle \frac{\partial ^3}{\partial \epsilon ^3}\left(2 M(\epsilon ) q(\epsilon )+
 *       \frac{p(\epsilon )}{r}\right) = 6 a q''(\epsilon )+2 M q^{(3)}(\epsilon )+\frac{p^{(3)}(\epsilon )}{r}
 *    \end{array}
 *  \f]
 *
 *  Energy derivatives of q:
 *  \f[
 *    \begin{array}{ll}
 *    \displaystyle \frac{\partial }{\partial \epsilon ^1}\left(\left(V-\epsilon +
 *      \frac{b}{r^2 M(\epsilon )}\right) p(\epsilon )-\frac{q(\epsilon )}{r}\right) =
 *      p(\epsilon ) \left(-\frac{a b}{M^2 r^2}-1\right)+p'(\epsilon ) \left(\frac{b}{M r^2}+V-\epsilon \right)-
 *      \frac{q'(\epsilon )}{r} \\
 *    \displaystyle \frac{\partial ^2}{\partial \epsilon ^2}\left(\left(V-\epsilon +
 *      \frac{b}{r^2 M(\epsilon )}\right) p(\epsilon )-\frac{q(\epsilon )}{r}\right) =
 *      \frac{2 a^2 b p(\epsilon )}{M^3 r^2}+2 p'(\epsilon ) \left(-\frac{a b}{M^2 r^2}-1\right)+
 *      p''(\epsilon ) \left(\frac{b}{M r^2}+V-\epsilon \right)-\frac{q''(\epsilon )}{r} \\
 *    \displaystyle \frac{\partial ^3}{\partial \epsilon ^3}\left(\left(V-\epsilon +\frac{b}{r^2 M(\epsilon )}\right)
 *      p(\epsilon )-\frac{q(\epsilon )}{r}\right) = -\frac{6 a^3 b p(\epsilon )}{M^4 r^2}+
 *      \frac{6 a^2 b p'(\epsilon )}{M^3 r^2}+3 p''(\epsilon ) \left(-\frac{a b}{M^2 r^2}-1\right)+
 *      p^{(3)}(\epsilon ) \left(\frac{b}{M r^2}+V-\epsilon \right)-\frac{q^{(3)}(\epsilon )}{r}
 *    \end{array}
 *  \f]
 *
 *  ZORA equation with \f$ M = 1 - \alpha^2 V / 2 \f$. Energy derivatives of p:
 *  \f[
 *    \begin{array}{ll}
 *      \displaystyle \frac{\partial }{\partial \epsilon ^1}\left(2 M(\epsilon ) q(\epsilon )+
 *       \frac{p(\epsilon )}{r}\right) = 2 M q'(\epsilon )+\frac{p'(\epsilon )}{r} \\
 *      \displaystyle \frac{\partial ^2}{\partial \epsilon ^2}\left(2 M(\epsilon ) q(\epsilon )+
 *        \frac{p(\epsilon )}{r}\right) = 2 M q''(\epsilon )+\frac{p''(\epsilon )}{r} \\
 *      \displaystyle \frac{\partial ^3}{\partial \epsilon ^3}\left(2 M(\epsilon ) q(\epsilon )+
 *       \frac{p(\epsilon )}{r}\right) = 2 M q^{(3)}(\epsilon )+\frac{p^{(3)}(\epsilon )}{r}
 *    \end{array}
 *  \f]
 *  Energy derivatives of q:
 *  \f[
 *    \begin{array}{ll}
 *      \displaystyle \frac{\partial }{\partial \epsilon ^1}\left(\left(V-\epsilon +\frac{b}{r^2 M(\epsilon )}\right)
 *        p(\epsilon )-\frac{q(\epsilon )}{r}\right) =
 *        p'(\epsilon ) \left(\frac{b}{M r^2}+V-\epsilon \right)-p(\epsilon )-\frac{q'(\epsilon )}{r} \\
 *      \displaystyle \frac{\partial ^2}{\partial \epsilon ^2}\left(\left(V-\epsilon +\frac{b}{r^2 M(\epsilon )}\right)
 *        p(\epsilon )-\frac{q(\epsilon )}{r}\right) =
 *        p''(\epsilon ) \left(\frac{b}{M r^2}+V-\epsilon \right)-2 p'(\epsilon )-\frac{q''(\epsilon )}{r}\\
 *      \displaystyle \frac{\partial ^3}{\partial \epsilon ^3}\left(\left(V-\epsilon +\frac{b}{r^2 M(\epsilon )}\right)
 *        p(\epsilon )-\frac{q(\epsilon )}{r}\right) =
 *        p^{(3)}(\epsilon ) \left(\frac{b}{M r^2}+V-\epsilon \right)-3 p''(\epsilon )-\frac{q^{(3)}(\epsilon )}{r}
 *    \end{array}
 *  \f]
 *
 *  IORA equation with \f$ M = M_0/(1 - \alpha^2 \epsilon / 2 / M_0) \f$. \f$ M_0 \f$ is ZORA electron mass.
 *  Energy derivatives of p:
 *  \f[
 *    \begin{array}{ll}
 *    \displaystyle \frac{\partial }{\partial \epsilon ^1}\left(2 M(\epsilon ) q(\epsilon )+
 *    \frac{p(\epsilon )}{r}\right) = \frac{2 a q(\epsilon )}{U^2}+2 M q'(\epsilon )+\frac{p'(\epsilon )}{r} \\
 *    \displaystyle \frac{\partial ^2}{\partial \epsilon ^2}\left(2 M(\epsilon ) q(\epsilon )+
 *    \frac{p(\epsilon )}{r}\right) = \frac{4 a^2 q(\epsilon )}{M_0 U^3}+
 *    \frac{4 a q'(\epsilon )}{U^2}+2 M q''(\epsilon )+\frac{p''(\epsilon )}{r} \\
 *    \displaystyle \frac{\partial ^3}{\partial \epsilon ^3}\left(2 M(\epsilon )
 *    q(\epsilon )+\frac{p(\epsilon )}{r}\right) = \frac{12 a^3 q(\epsilon )}{M_0^2 U^4}+
 *    \frac{12 a^2 q'(\epsilon )}{M_0 U^3}+\frac{6 a q''(\epsilon )}{U^2}+2 M q^{(3)}(\epsilon )+
 *    \frac{p^{(3)}(\epsilon )}{r}
 *    \end{array}
 *  \f]
 *  Energy derivatives of q:
 *  \f[
 *    \begin{array}{ll}
 *    \displaystyle \frac{\partial }{\partial \epsilon ^1}\left(\left(V-\epsilon +\frac{b}{r^2 M(\epsilon )}\right)
 *     p(\epsilon )-\frac{q(\epsilon )}{r}\right) = p(\epsilon ) \left(-\frac{a b}{\text{M0}^2 r^2}-1\right)+
 *     p'(\epsilon ) \left(\frac{b}{M r^2}+V-\epsilon \right)-\frac{q'(\epsilon )}{r} \\
 *    \displaystyle \frac{\partial ^2}{\partial \epsilon ^2}\left(\left(V-\epsilon +\frac{b}{r^2 M(\epsilon )}\right)
 *    p(\epsilon )-\frac{q(\epsilon )}{r}\right) = 2 p'(\epsilon ) \left(-\frac{a b}{\text{M0}^2 r^2}-1\right)+
 *     p''(\epsilon ) \left(\frac{b}{M r^2}+V-\epsilon \right)-\frac{q''(\epsilon )}{r} \\
 *    \displaystyle \frac{\partial ^3}{\partial \epsilon ^3}\left(\left(V-\epsilon +\frac{b}{r^2 M(\epsilon )}\right)
 *     p(\epsilon )-\frac{q(\epsilon )}{r}\right) = 3 p''(\epsilon ) \left(-\frac{a b}{\text{M0}^2 r^2}-1\right)+
 *      p^{(3)}(\epsilon ) \left(\frac{b}{M r^2}+V-\epsilon \right)-\frac{q^{(3)}(\epsilon )}{r}
 *    \end{array}
 *  \f]
 */
class Radial_solver
{
  protected:
    /// Positive charge of the nucleus.
    int zn_;

    /// Radial grid.
    Radial_grid<double> const& radial_grid_;

    /// Electronic part of potential.
    Spline<double> ve_;

    double epsabs_{1e-3};
    double epsrel_{1e-3};

    /// Integrate system of two first-order differential equations forward starting from the origin.
    template <relativity_t rel>
    int
    integrate_forward_gsl(double enu__, int l__, int k__, Spline<double>& chi_p__, Spline<double>& chi_q__,
                          std::vector<double>& p__, std::vector<double>& dpdr__, std::vector<double>& q__,
                          std::vector<double>& dqdr__, bool bound_state__) const
    {
        struct params
        {
            int ir;
            double x0;
            double enu;
            int l;
            double kappa;
            int zn;
            Spline<double> const* ve;
            Spline<double> const* chi_p;
            Spline<double> const* chi_q;
        };

        auto func = [](double x, double const* y, double* f, void* params__) -> int {
            using namespace radial_solver_local;

            params* p = static_cast<params*>(params__);
            double dx = x - p->x0;
            /* reconstruct V(x) = -z/x + Vel(x) */
            double V     = p->ve->operator()(p->ir, dx) - p->zn / x;
            double chi_p = p->chi_p->operator()(p->ir, dx);
            double chi_q = p->chi_q->operator()(p->ir, dx);
            /* system of two coupled equations
             * -------------------------------
             *
             *   Schroedinger radial equation case:
             *     p'(x) = 2 * q(x) + p(x) / x
             *     q'(x) = (V(x) - enu + l * (l + 1) / 2 / x^2) * p(x) - q(x) / x + chi_q(x)
             *
             *   Scalar-relativistic (Koelling-Harmon) case:
             *     p'(x) = 2 * M(x) * q(x) + p(x) / x + chi_p(x)
             *     q'(x) = (V(x) - enu + l * (l + 1) / 2 / M(x) / x^2) * p(x) - q(x) / x + chi_q(x)
             *
             *     where M(x) = 1 + 0.5 * alpha^2 * (enu - V(x))
             *     with M = 1 (alpha -> 0) and chi_p(x) = 0 we obtain "classical" Schroedinger radial equation
             *
             *   ZORA case:
             *     same as Koelling-Harmon but with M(x) = 1 - 0.5 * alpha^2 * V(x)
             *
             *   Dirac radial equation case:
             *     p'(x) = alpha * (enu - V(x) + 2 * E0) * q(x) - p(x) * kappa / x
             *     q'(x) = -alpha * (enu - V(x)) * p(x) + q(x) * kappa / x
             */
            switch (rel) {
                case relativity_t::none:
                case relativity_t::koelling_harmon:
                case relativity_t::zora:
                case relativity_t::iora: {
                    double ll_half = p->l * (p->l + 1) / 2.0;
                    double M       = rel_mass<rel>(p->enu, V);
                    /* p'(x) = 2 * M(x) * q(x) + p(x) / x + chi_p(x) */
                    f[0] = 2 * M * y[1] + y[0] / x + chi_p;
                    /* q'(x) = (V(x) - enu + l * (l + 1) / 2 / M(x) / x^2) * p(x) - q(x) / x + chi_q(x) */
                    f[1] = (V - p->enu + ll_half / (M * std::pow(x, 2))) * y[0] - y[1] / x + chi_q;
                    break;
                }
                case relativity_t::dirac: {
                    f[0] = alpha * (p->enu - V + 2 * rest_energy) * y[1] - y[0] * p->kappa / x;
                    f[1] = -alpha * (p->enu - V) * y[0] + y[1] * p->kappa / x;
                    break;
                }
            }
            return GSL_SUCCESS;
        };

        auto jac = [](double x, double const* y, double* dfdy, double* dfdx, void* params__) -> int {
            using namespace radial_solver_local;

            params* p      = static_cast<params*>(params__);
            double dx      = x - p->x0;
            double ll_half = p->l * (p->l + 1) / 2.0;
            double V       = p->ve->operator()(p->ir, dx) - p->zn / x;
            double M       = rel_mass<rel>(p->enu, V);

            double ve_deriv    = p->ve->deriv(1, p->ir, dx);
            double chi_p_deriv = p->chi_p->deriv(1, p->ir, dx);
            double chi_q_deriv = p->chi_q->deriv(1, p->ir, dx);

            auto dfdy_mat = gsl_matrix_view_array(dfdy, 2, 2);
            auto m        = &dfdy_mat.matrix;

            double xinv  = std::pow(x, -1);
            double x2inv = std::pow(x, -2);
            double x3inv = std::pow(x, -3);

            double V_deriv = p->zn * x2inv + ve_deriv;
            double M_deriv = rel_mass_deriv<rel>(p->enu, V, V_deriv);
            /* derivatives of right hand side of coupled radial equations
             * ----------------------------------------------------------
             *
             *   Schroedinger radial equation case:
             *     dF_0 / dp = 1 / x
             *     dF_0 / dq = 2
             *     dF_1 / dp = V(x) - enu + l * (l + 1) / 2 / x^2
             *     dF_1 / dq = -1 / x
             *
             *     dF_0 / dx = -p(x) / x^2
             *     dF_1 / dx = (V'(x) - 2 * l * (l + 1) / 2 / x^3) * p(x) + q(x) / x^2 + chi_q'(x)
             *
             *   Scalar-relativistic (Koelling-Harmon) case:
             *     dF_0 / dp = 1 / x
             *     dF_0 / dq = 2 * M(x)
             *     dF_1 / dp = V - enu + l * (l + 1) / 2 / M / x^2
             *     dF_1 / dq = -1 / x
             *
             *     dF_0 / dx = 2 * M'(x) * q(x) - p(x) / x / x + chi_p'(x)
             *     dF_1 / dx = (V' + (l * (l + 1) / 2 / M(x) / x^2)') * p(x) + q(x) / x^2 + chi_q'(x)
             *
             *     where:
             *       M'(x) = alpha^2 * (-zn / x^2 - Vel'(x)) / 2
             *
             *       (l * (l + 1) / 2 / M(x) / x^2)' = (l * (l + 1) / 2 / x^2)' * (1 / M(x)) +
             *                l * (l + 1) / 2 / x^2 * (-M'(x) / M(x)^2)
             *
             *   ZORA case:
             *     same as scalar-relativistic with M(x) = 1 - 0.5 * alpha^2 * V(x)
             */
            switch (rel) {
                case relativity_t::none:
                case relativity_t::koelling_harmon:
                case relativity_t::zora:
                case relativity_t::iora: {
                    gsl_matrix_set(m, 0, 0, xinv);
                    gsl_matrix_set(m, 0, 1, 2.0 * M);
                    gsl_matrix_set(m, 1, 0, V - p->enu + ll_half * x2inv / M);
                    gsl_matrix_set(m, 1, 1, -xinv);

                    dfdx[0] = 2 * M_deriv * y[1] - y[0] * x2inv + chi_p_deriv;
                    dfdx[1] = (V_deriv - ll_half * M_deriv * x2inv / std::pow(M, 2) - 2 * ll_half * x3inv / M) * y[0] +
                              y[1] * x2inv + chi_q_deriv;
                    break;
                }
                case relativity_t::dirac: {
                    gsl_matrix_set(m, 0, 0, -p->kappa * xinv);
                    gsl_matrix_set(m, 0, 1, alpha * (p->enu - V + 2 * rest_energy));
                    gsl_matrix_set(m, 1, 0, -alpha * (p->enu - V));
                    gsl_matrix_set(m, 1, 1, p->kappa * xinv);

                    dfdx[0] = alpha * y[1] * (-ve_deriv - p->zn * x2inv) + y[0] * p->kappa * x2inv;
                    dfdx[1] = -alpha * y[0] * (-ve_deriv - p->zn * x2inv) - y[1] * p->kappa * x2inv;

                    break;
                }
            }
            return GSL_SUCCESS;
        };

        p__ = std::vector<double>(radial_grid_.num_points());
        q__ = std::vector<double>(radial_grid_.num_points());

        int last_point{radial_grid_.num_points() - 1};

        auto integrate_forward = [&](int last_point) -> std::vector<int> {
            params p;
            p.enu   = enu__;
            p.l     = l__;
            p.kappa = 0;
            p.zn    = zn_;
            p.ve    = &ve_;
            p.chi_p = &chi_p__;
            p.chi_q = &chi_q__;
            if (rel == relativity_t::dirac) {
                if (k__ == l__) {
                    p.kappa = k__;
                } else if (k__ == l__ + 1) {
                    p.kappa = -k__;
                } else {
                    std::stringstream s;
                    s << "wrong k : " << k__ << " for l = " << l__;
                    RTE_THROW(s);
                }
            }

            /* initial conditions */
            double x = radial_grid_[0];
            double y[2];
            switch (rel) {
                case relativity_t::none: {
                    if (l__ == 0) {
                        y[0] = 2 * x * zn_;
                        y[1] = -std::pow(zn_, 2) * x;
                    } else {
                        y[0] = std::pow(x, l__ + 1) / (2 * l__ + 1);
                        y[1] = std::pow(x, l__) / 4;
                    }
                    break;
                }
                case relativity_t::koelling_harmon:
                case relativity_t::zora:
                case relativity_t::iora: {
                    double a = l__ * (l__ + 1) + 1 - std::pow(radial_solver_local::alpha * zn_, 2);
                    double b = 0.5 * (1 + std::sqrt(1 + 4 * a));
                    y[0]     = std::pow(x, b);
                    y[1]     = (x * b * std::pow(x, b - 1) - y[0]) / zn_ / std::pow(radial_solver_local::alpha, 2);
                    break;
                }
                case relativity_t::dirac: {
                    double b = std::sqrt(std::pow(p.kappa, 2) - std::pow(zn_ / speed_of_light, 2));
                    y[0]     = std::pow(x, b);
                    y[1]     = std::pow(x, b - 1) * (b + p.kappa) * x / radial_solver_local::alpha / zn_;
                    break;
                }
            }

            p__[0] = y[0];
            q__[0] = y[1];

            gsl_odeiv2_system sys = {func, jac, 2, &p};

            const gsl_odeiv2_step_type* T = gsl_odeiv2_step_rk8pd;
            gsl_odeiv2_step* s            = gsl_odeiv2_step_alloc(T, 2);
            gsl_odeiv2_control* c         = gsl_odeiv2_control_y_new(epsabs_, epsrel_);
            gsl_odeiv2_evolve* e          = gsl_odeiv2_evolve_alloc(2);

            std::vector<int> ridx;

            for (p.ir = 0; p.ir < last_point; p.ir++) {
                double x0 = radial_grid_[p.ir];
                double x1 = radial_grid_[p.ir + 1];
                double h  = radial_grid_.dx(p.ir);
                p.x0      = x0;
                while (x0 < x1) {
                    int status = gsl_odeiv2_evolve_apply(e, c, s, &sys, &x0, x1, &h, y);
                    if (status != GSL_SUCCESS) {
                        std::stringstream s;
                        s << "error in gsl_odeiv2_driver_apply()" << std::endl
                          << "  enu = " << enu__ << std::endl
                          << "  ir = " << p.ir << std::endl
                          << "  x0 = " << x0 << std::endl
                          << "  x1 = " << x1 << std::endl
                          << "  h = " << h << std::endl
                          << "  l = " << l__ << std::endl
                          << "  z = " << zn_;
                        RTE_THROW(s);
                    }
                }

                /* we pass through a node; reset a counter where wave-function was renormalized */
                if (y[0] * p__[p.ir] < 0) {
                    ridx.clear();
                }

                p__[p.ir + 1] = y[0];
                q__[p.ir + 1] = y[1];

                double const max_val{1e6};

                if (std::abs(y[0]) > max_val) {
                    ridx.push_back(p.ir + 1);
                    for (int j = 0; j <= p.ir + 1; j++) {
                        p__[j] /= max_val;
                        q__[j] /= max_val;
                    }
                    y[0] /= max_val;
                    y[1] /= max_val;

                    chi_p__.scale(1.0 / max_val);
                    chi_q__.scale(1.0 / max_val);
                }
            }
            gsl_odeiv2_evolve_free(e);
            gsl_odeiv2_control_free(c);
            gsl_odeiv2_step_free(s);

            return ridx;
        };

        if (!bound_state__) {
            /* just integrate forward */
            integrate_forward(last_point);
        } else {
            auto ridx = integrate_forward(last_point);
            if (!ridx.empty()) {
                /* take fisrt point where wave-functions was renormalized */
                last_point = ridx.front();
                integrate_forward(last_point);
            }

            /* starting from the last point and go backward to find the mimimum value or a node */
            for (int j = last_point; j >= 1; j--) {
                if ((p__[j] * p__[j - 1] < 0) ||
                    ((std::abs(p__[j]) < std::abs(p__[j - 1])) && (p__[j] * p__[j - 1] > 0))) {
                    last_point = j;
                    break;
                }
            }
            integrate_forward(last_point);
            for (int i = last_point + 1; i < radial_grid_.num_points(); i++) {
                p__[i] = 0.0;
                q__[i] = 0.0;
            }
        }
        /* get number of nodes */
        int nn{0};
        for (int i = 0; i < last_point; i++) {
            if (p__[i] * p__[i + 1] < 0.0) {
                nn++;
            }
        }

        double ll_half = l__ * (l__ + 1) / 2.0;

        for (int i = 0; i < radial_grid_.num_points(); i++) {
            double V = ve_(i) - zn_ * radial_grid_.x_inv(i);
            double M = radial_solver_local::rel_mass<rel>(enu__, V);

            switch (rel) {
                case relativity_t::none:
                case relativity_t::koelling_harmon:
                case relativity_t::zora:
                case relativity_t::iora: {
                    /* p' = 2Mq + \frac{p}{r} + chi_p */
                    dpdr__[i] = 2 * M * q__[i] + p__[i] * radial_grid_.x_inv(i) + chi_p__(i);
                    /* q' = (V - enu + \frac{\ell(\ell + 1)}{2 M x^2}) p - \frac{p}{r} + chi_q */
                    dqdr__[i] = (V - enu__ + ll_half / std::pow(radial_grid_.x(i), 2) / M) * p__[i] -
                                q__[i] * radial_grid_.x_inv(i) + chi_q__(i);
                    break;
                }
                case relativity_t::dirac: {
                    /* Dirac equation is only solved for core states and p' and q' are not needed */
                    break;
                }
            }
        }

        /* normalize */
        Spline<double> s(radial_grid_);
        for (int i = 0; i < radial_grid_.num_points(); i++) {
            s(i) = std::pow(p__[i], 2);
        }
        auto norm = 1.0 / std::sqrt(s.interpolate().integrate(0));
        for (int i = 0; i < radial_grid_.num_points(); i++) {
            p__[i] *= norm;
            q__[i] *= norm;
            dpdr__[i] *= norm;
            dqdr__[i] *= norm;
        }

        return nn;
    }

    inline double
    integrate_forward_until(relativity_t rel__, double enu__, int l__, int k__, Spline<double>& chi_p__,
                            Spline<double>& chi_q__, std::vector<double>& p__, std::vector<double>& dpdr__,
                            std::vector<double>& q__, std::vector<double>& dqdr__, bool bound_state__,
                            std::function<bool(int, int, double&)> condition__)
    {

        auto integrate_forward = [&](double enu__) -> int {
            int nn{0};
            switch (rel__) {
                case relativity_t::none: {
                    nn = integrate_forward_gsl<relativity_t::none>(enu__, l__, k__, chi_p__, chi_q__, p__, dpdr__, q__,
                                                                   dqdr__, bound_state__);
                    break;
                }
                case relativity_t::koelling_harmon: {
                    nn = integrate_forward_gsl<relativity_t::koelling_harmon>(enu__, l__, k__, chi_p__, chi_q__, p__,
                                                                              dpdr__, q__, dqdr__, bound_state__);
                    break;
                }
                case relativity_t::zora: {
                    nn = integrate_forward_gsl<relativity_t::zora>(enu__, l__, k__, chi_p__, chi_q__, p__, dpdr__, q__,
                                                                   dqdr__, bound_state__);
                    break;
                }
                case relativity_t::iora: {
                    nn = integrate_forward_gsl<relativity_t::iora>(enu__, l__, k__, chi_p__, chi_q__, p__, dpdr__, q__,
                                                                   dqdr__, bound_state__);
                    break;
                }
                case relativity_t::dirac: {
                    nn = integrate_forward_gsl<relativity_t::dirac>(enu__, l__, k__, chi_p__, chi_q__, p__, dpdr__, q__,
                                                                    dqdr__, bound_state__);
                    break;
                }
                default: {
                    RTE_THROW("unsupported relativity type");
                }
            }
            return nn;
        };

        for (int i = 0; i < 1000; i++) {
            int nn = integrate_forward(enu__);
            if (condition__(i, nn, enu__)) {
                return enu__;
            }
        }
        std::stringstream s;
        s << "integrate_forward_until(): condition is not achieved in 1000 iterations" << std::endl
          << "  curent value of enu: " << enu__;
        RTE_THROW(s);
        return 0.0;
    }

  public:
    Radial_solver(int zn__, std::vector<double> const& v__, Radial_grid<double> const& radial_grid__,
                  double epsabs__ = 1e-3, double epsrel__ = 1e-3)
        : zn_(zn__)
        , radial_grid_(radial_grid__)
        , epsabs_{epsabs__}
        , epsrel_{epsrel__}
    {
        ve_ = Spline<double>(radial_grid__);

        for (int i = 0; i < num_points(); i++) {
            ve_(i) = v__[i] + zn_ * radial_grid_.x_inv(i);
        }
        ve_.interpolate();
    }

    /// Integrates the radial equation for a given energy and finds the m-th energy derivative of the radial solution.
    /** \param [in] rel Type of relativity
     *  \param [in] dme Order of energy derivative.
     *  \param [in] l Oribtal quantum number.
     *  \param [in] enu Integration energy.
     *
     *  Returns number of nodes of the radial function, \f$ p(r) = ru(r) \f$, \f$ ru'(r)\f$, \f$ u(R) \f$,
     *  \f$ u'(R) \f$ and \f$ u''(R) \f$.
     *
     *  Surface derivatives:
     *
     *  From
     *  \f[
     *    u(r) = \frac{p(r)}{r}
     *  \f]
     *  we have
     *  \f[
     *    u'(r) = \frac{p'(r)}{r} - \frac{p(r)}{r^2} = \frac{1}{r}\big(p'(r) - \frac{p(r)}{r}\big)
     *  \f]
     *
     *  \f[
     *    u''(r) = \frac{p''(r)}{r} - \frac{p'(r)}{r^2} - \frac{p'(r)}{r^2} + 2 \frac{p(r)}{r^3} =
     *      \frac{1}{r}\big(p''(r) - 2 \frac{p'(r)}{r} + 2 \frac{p(r)}{r^2}\big)
     *  \f]
     */
    auto
    solve(relativity_t rel__, int dme__, int l__, double enu__) const
    {
        int nr = num_points();
        std::vector<std::vector<double>> p;
        std::vector<std::vector<double>> q;
        std::vector<std::vector<double>> dpdr;
        std::vector<std::vector<double>> dqdr;

        Spline<double> chi_p(radial_grid_);
        Spline<double> chi_q(radial_grid_);

        int nn{0};

        using namespace radial_solver_local;

        for (int j = 0; j <= dme__; j++) {
            p.push_back(std::vector<double>(nr));
            q.push_back(std::vector<double>(nr));
            dpdr.push_back(std::vector<double>(nr));
            dqdr.push_back(std::vector<double>(nr));

            if (j) {
                if (rel__ == relativity_t::none || rel__ == relativity_t::zora) {
                    for (int i = 0; i < nr; i++) {
                        chi_q(i) = -j * p[j - 1][i];
                    }
                } else if (rel__ == relativity_t::koelling_harmon) {
                    for (int i = 0; i < nr; i++) {
                        chi_p(i) = j * 2 * sq_alpha_half * q[j - 1][i];
                    }
                    double ll_half = l__ * (l__ + 1) / 2.0;
                    if (j == 1) {
                        for (int i = 0; i < nr; i++) {
                            double x = radial_grid_[i];
                            double V = ve_(i) - zn_ * radial_grid_.x_inv(i);
                            double M = rel_mass<relativity_t::koelling_harmon>(enu__, V);
                            double c = sq_alpha_half * ll_half / std::pow(x * M, 2);
                            chi_q(i) = -p[j - 1][i] * (1 + c);
                        }
                    } else if (j == 2) {
                        for (int i = 0; i < nr; i++) {
                            double x = radial_grid_[i];
                            double V = ve_(i) - zn_ * radial_grid_.x_inv(i);
                            double M = rel_mass<relativity_t::koelling_harmon>(enu__, V);
                            double c = sq_alpha_half * ll_half / std::pow(x * M, 2);
                            chi_q(i) = -2 * p[j - 1][i] * (1 + c) + 2 * p[j - 2][i] * sq_alpha_half * c / M;
                        }
                    } else if (j == 3) {
                        for (int i = 0; i < nr; i++) {
                            double x = radial_grid_[i];
                            double V = ve_(i) - zn_ * radial_grid_.x_inv(i);
                            double M = rel_mass<relativity_t::koelling_harmon>(enu__, V);
                            double c = sq_alpha_half * ll_half / std::pow(x * M, 2);
                            chi_q(i) = -3 * p[j - 1][i] * (1 + c) + 6 * p[j - 2][i] * sq_alpha_half * c / M -
                                       6 * p[j - 3][i] * std::pow(sq_alpha_half / M, 2) * c;
                        }
                    } else {
                        std::stringstream s;
                        s << "energy derivative of the order " << j
                          << " is not implemented for Koelling-Harmon radial solver";
                        RTE_THROW(s);
                    }
                } else if (rel__ == relativity_t::iora) {
                    double ll_half = l__ * (l__ + 1) / 2.0;
                    for (int i = 0; i < nr; i++) {
                        double V  = ve_(i) - zn_ * radial_grid_.x_inv(i);
                        double M0 = rel_mass<relativity_t::zora>(enu__, V);
                        double x  = radial_grid_[i];
                        chi_q(i)  = -j * p[j - 1][i] * (1 + sq_alpha_half * ll_half / std::pow(M0 * x, 2));
                    }

                    if (j == 1) {
                        for (int i = 0; i < nr; i++) {
                            double V  = ve_(i) - zn_ * radial_grid_.x_inv(i);
                            double M0 = rel_mass<relativity_t::zora>(enu__, V);
                            double U  = (1 - sq_alpha_half * enu__ / M0);
                            chi_p(i)  = q[j - 1][i] * 2 * sq_alpha_half * std::pow(U, -2);
                        }
                    } else if (j == 2) {
                        for (int i = 0; i < nr; i++) {
                            double V  = ve_(i) - zn_ * radial_grid_.x_inv(i);
                            double M0 = rel_mass<relativity_t::zora>(enu__, V);
                            double U  = (1 - sq_alpha_half * enu__ / M0);
                            chi_p(i)  = q[j - 1][i] * 4 * sq_alpha_half * std::pow(U, -2) +
                                       q[j - 2][i] * 4 * std::pow(sq_alpha_half, 2) * std::pow(U, -3) / M0;
                        }
                    } else if (j == 3) {
                        for (int i = 0; i < nr; i++) {
                            double V  = ve_(i) - zn_ * radial_grid_.x_inv(i);
                            double M0 = rel_mass<relativity_t::zora>(enu__, V);
                            double U  = (1 - sq_alpha_half * enu__ / M0);
                            chi_p(i)  = q[j - 1][i] * 6 * sq_alpha_half * std::pow(U, -2) +
                                       q[j - 2][i] * 12 * std::pow(sq_alpha_half, 2) * std::pow(U, -2) / (M0 * U) +
                                       q[j - 3][i] * 12 * std::pow(sq_alpha_half, 3) * std::pow(M0 * U, -2) *
                                               std::pow(U, -2);
                        }
                    } else {
                        std::stringstream s;
                        s << "energy derivative of the order " << j << " is not implemented for IORA radial solver";
                        RTE_THROW(s);
                    }
                } else {
                    RTE_THROW("unsupported relativity type");
                }
                chi_p.interpolate();
                chi_q.interpolate();
            }

            switch (rel__) {
                case relativity_t::none: {
                    nn = integrate_forward_gsl<relativity_t::none>(enu__, l__, 0, chi_p, chi_q, p[j], dpdr[j], q[j],
                                                                   dqdr[j], false);
                    break;
                }
                case relativity_t::koelling_harmon: {
                    nn = integrate_forward_gsl<relativity_t::koelling_harmon>(enu__, l__, 0, chi_p, chi_q, p[j],
                                                                              dpdr[j], q[j], dqdr[j], false);
                    break;
                }
                case relativity_t::zora: {
                    nn = integrate_forward_gsl<relativity_t::zora>(enu__, l__, 0, chi_p, chi_q, p[j], dpdr[j], q[j],
                                                                   dqdr[j], false);
                    break;
                }
                case relativity_t::iora: {
                    nn = integrate_forward_gsl<relativity_t::iora>(enu__, l__, 0, chi_p, chi_q, p[j], dpdr[j], q[j],
                                                                   dqdr[j], false);
                    break;
                }
                default: {
                    RTE_THROW("unsupported relativity type");
                }
            }
        }
        /* save the results */
        std::vector<double> rdudr(nr);
        for (int i = 0; i < nr; i++) {
            rdudr[i] = dpdr.back()[i] - p.back()[i] * radial_grid_.x_inv(i);
        }

        double R = radial_grid_.last();

        /* 1st radial derivative of u(r) */
        std::array<double, 3> uderiv;
        /* p(r) = u(r)*r -> u(r) = p(r) / r */
        uderiv[0] = p.back().back() / R;
        uderiv[1] = (dpdr.back().back() - p.back().back() / R) / R;
        /* 2nd radial derivative of u(r) */
        Spline<double> sdpdr(radial_grid_, dpdr.back());
        sdpdr.interpolate();
        uderiv[2] = (sdpdr.deriv(1, nr - 1) - 2 * dpdr.back().back() / R + 2 * p.back().back() / std::pow(R, 2)) / R;

        return radial_solver_result_t{nn, p.back(), rdudr, uderiv};
    }

    inline int
    num_points() const
    {
        return radial_grid_.num_points();
    }

    inline int
    zn() const
    {
        return zn_;
    }

    inline double
    radial_grid(int i__) const
    {
        return radial_grid_[i__];
    }

    inline Radial_grid<double> const&
    radial_grid() const
    {
        return radial_grid_;
    }
};

class Bound_state : public Radial_solver
{
  private:
    int n_;

    int l_;

    int k_;

    /// Tolerance of bound state energy.
    double enu_tolerance_;

    double enu_;

    Spline<double> p_;

    Spline<double> q_;

    Spline<double> u_;

    Spline<double> rdudr_;

    Spline<double> rho_;

    std::vector<double> dpdr_;

    inline void
    solve(relativity_t rel__, double enu_start__, double alpha0__, double alpha1__)
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
                                           if (s != sp && iter > 1) {
                                               return true;
                                           }
                                           denu = (s != sp) ? denu * alpha0__ : denu * alpha1__;
                                           enu += s * denu;
                                           return false;
                                       });

        double e1 = enu_;
        double e2 = enu_ - sp * denu;

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
        enu_ = integrate_forward_until(rel__, e1, l_, k_, chi_p, chi_q, p, dpdr_, q, dqdr, true,
                                       [](int iter, int nn, double& enu) { return true; });

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
        for (int i = 0; i < np - 1; i++) {
            if (p_(i) * p_(i + 1) < 0.0) {
                nn++;
            }
        }

        if (nn != (n_ - l_ - 1)) {
            FILE* fout = fopen("p.dat", "w");
            for (int ir = 0; ir < np; ir++) {
                double x = radial_grid(ir);
                fprintf(fout, "%12.6f %16.8f\n", x, p_(ir));
            }
            fclose(fout);

            std::stringstream s;
            s << "n = " << n_ << std::endl
              << "l = " << l_ << std::endl
              << "enu = " << enu_ << std::endl
              << "wrong number of nodes : " << nn << " instead of " << (n_ - l_ - 1);
            RTE_THROW(s);
        }

        for (int i = 0; i < np - 1; i++) {
            rho_(i) += std::pow(u_(i), 2);
            if (rel__ == relativity_t::dirac) {
                rho_(i) += std::pow(q_(i) * radial_grid_.x_inv(i), 2);
            }
        }
    }

  public:
    Bound_state(relativity_t rel__, int zn__, int n__, int l__, int k__, Radial_grid<double> const& radial_grid__,
                std::vector<double> const& v__, double enu_start__, double alpha0__ = 0.5, double alpha1__ = 10.0,
                double epsabs__ = 1e-3, double epsrel__ = 1e-3)
        : Radial_solver(zn__, v__, radial_grid__, epsabs__, epsrel__)
        , n_(n__)
        , l_(l__)
        , k_(k__)
        , enu_tolerance_(1e-12)
        , p_(radial_grid__)
        , q_(radial_grid__)
        , u_(radial_grid__)
        , rdudr_(radial_grid__)
        , rho_(radial_grid__)
    {
        solve(rel__, enu_start__, alpha0__, alpha1__);
    }

    inline double
    enu() const
    {
        return enu_;
    }

    /// Return charge density, corresponding to a radial solution.
    Spline<double> const&
    rho() const
    {
        return rho_;
    }

    /// Return radial function.
    Spline<double> const&
    u() const
    {
        return u_;
    }

    /// Return radial function multiplied by x.
    Spline<double> const&
    p() const
    {
        return p_;
    }

    std::vector<double> const&
    dpdr() const
    {
        return dpdr_;
    }
};

class Enu_finder : public Radial_solver
{
  private:
    int n_;

    int l_;

    double enu_;

    double etop_;
    double ebot_;

    void
    find_enu(relativity_t rel__, double enu_start__, int auto_enu__)
    {
        int np = num_points();

        Spline<double> chi_p(radial_grid());
        Spline<double> chi_q(radial_grid());

        std::vector<double> p(np);
        std::vector<double> q(np);
        std::vector<double> dpdr(np);
        std::vector<double> dqdr(np);

        /* We want to find enu such that the wave-function at the muffin-tin boundary is zero
         * and the number of nodes inside muffin-tin is equal to n-l-1. This will be the top
         * of the band. */
        int s{1};
        int sp;
        double denu{1e-8};
        double e0;
        /* 1st pass: estimate upper and lower boundaries of the etop*/
        e0 = integrate_forward_until(rel__, enu_start__, l_, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                                     [&s, &sp, &denu, this](int iter, int nn, double& enu) {
                                         sp = s;
                                         s  = (nn > (n_ - l_ - 1)) ? -1 : 1;
                                         if (s != sp && iter > 0) {
                                             return true;
                                         }
                                         denu *= 10;
                                         enu += s * denu;
                                         return false;
                                     });

        double e1 = e0;
        double e2 = e0 - sp * denu;

        /* e1 is bottom, e2 is top energy */
        if (e1 > e2) {
            std::swap(e1, e2);
        }

        /* 2nd pass: refine by bisection */
        etop_ = integrate_forward_until(rel__, (e1 + e2) / 2, l_, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                                        [&e1, &e2, this](int iter, int nn, double& enu) {
                                            if (nn > (n_ - l_ - 1)) {
                                                e2 = enu;
                                            } else {
                                                e1 = enu;
                                            }
                                            enu = (e1 + e2) / 2.0;
                                            return std::abs(e1 - e2) < 1e-9;
                                        });

        auto surface_deriv = [this, &dpdr, &p]() {
            if (true) {
                /* return  p'(R) */
                return dpdr.back();
            } else {
                /* return R*u'(R) */
                return dpdr.back() - p.back() / radial_grid_.last();
            }
        };

        double sd = surface_deriv();

        /* Now we go down in energy and search for enu such that the wave-function derivative is zero
         * at the muffin-tin boundary. This will be the bottom of the band. Here we look at a sign change
         * of the derivative. */
        denu = 1e-8;
        e0   = integrate_forward_until(rel__, etop_, l_, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                                       [&denu, sd, &surface_deriv, this](int iter, int nn, double& enu) {
                                         if (surface_deriv() * sd <= 0 || denu > 20) {
                                             return true;
                                         }
                                         denu *= 2;
                                         enu -= denu;
                                         return false;
                                     });

        /* refine bottom energy */
        e1    = e0;
        e2    = e0 + denu;
        ebot_ = integrate_forward_until(rel__, (e1 + e2) / 2, l_, 0, chi_p, chi_q, p, dpdr, q, dqdr, false,
                                        [&e1, &e2, sd, &surface_deriv, this](int iter, int nn, double& enu) {
                                            if (surface_deriv() * sd > 0) {
                                                e2 = enu;
                                            } else {
                                                e1 = enu;
                                            }
                                            enu = (e1 + e2) / 2.0;
                                            return std::abs(surface_deriv()) < 1e-8;
                                        });

        switch (auto_enu__) {
            case 1: {
                enu_ = (ebot_ + etop_) / 2.0;
                break;
            }
            case 2: {
                enu_ = ebot_;
                break;
            }
            default: {
                RTE_THROW("wrong type of auto_enu");
            }
        }
    }

  public:
    /// Constructor
    Enu_finder(relativity_t rel__, int zn__, int n__, int l__, Radial_grid<double> const& radial_grid__,
               std::vector<double> const& v__, double enu_start__, int auto_enu__)
        : Radial_solver(zn__, v__, radial_grid__)
        , n_(n__)
        , l_(l__)
    {
        if (l_ >= n_) {
            RTE_THROW("wrong orbital quantum number");
        }
        find_enu(rel__, enu_start__, auto_enu__);
    }

    inline double
    enu() const
    {
        return enu_;
    }

    inline double
    ebot() const
    {
        return ebot_;
    }

    inline double
    etop() const
    {
        return etop_;
    }
};

}; // namespace sirius

#endif // __RADIAL_SOLVER_HPP__
