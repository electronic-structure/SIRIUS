// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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

/** \file radial_solver.hpp
 *
 *  \brief Contains declaration and partial implementation of sirius::Radial_solver class.
 */

#ifndef __RADIAL_SOLVER_HPP__
#define __RADIAL_SOLVER_HPP__

#include <tuple>
#include "spline.hpp"
#include "core/constants.hpp"
#include "core/typedefs.hpp"
#include "core/rte/rte.hpp"

namespace sirius {

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
 *  Let's now decouple second-order differential equation into a system of two first-order euquations. From
 *  \f$ p(r) = u(r)r \f$ we have
 *  \f[
 *    p'(r) = u'(r)r + u''(r) = 2q(r) + \frac{p(r)}{r}
 *  \f]
 *  where we have introduced a new variable \f$ q(r) = \frac{u'(r) r}{2} \f$. Differentiating \f$ p'(r) \f$ again
 *  we arrive to the following equation for \f$ q'(r) \f$:
 *  \f[
 *    p''(r) = 2q'(r) + \frac{p'(r)}{r} - \frac{p(r)}{r^2} = 2q'(r) + \frac{2q(r)}{r} + \frac{p(r)}{r^2} -
 * \frac{p(r)}{r^2} \f] \f[ q'(r) = \frac{1}{2}p''(r) - \frac{q(r)}{r} = \big(V_{eff}(r) - E\big) p(r) - \frac{q(r)}{r}
 * - \chi(r) \f] Final expression for a linear system of differential equations for m-th energy derivative is:
 *  \f{eqnarray*}{
 *    p'(r) &=& 2q(r) + \frac{p(r)}{r} \\
 *    q'(r) &=& \big(V_{eff}(r) - E\big) p(r) - \frac{q(r)}{r} - \chi(r)
 *  \f}
 *  Scalar-relativistic equations look similar. For m = 0 (no energy derivative) we have:
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
 *    \ddot{q}'(r) &=& \big(V(r) - E + \frac{\ell(\ell+1)}{2Mr^2}\big) \ddot{p}(r) - \frac{\ddot{q}(r)}{r} - 2 \big(1 +
 * \frac{\ell(\ell+1)\alpha^2}{4M^2r^2}\big)\dot{p}(r)
 *      + \frac{\ell(\ell+1)\alpha^4}{4M^3r^2} p(r)
 *  \f}
 *
 *  Derivation of IORA.
 *
 *  First thing to do is to expand the \f$ 1/M \f$ to the first order in \f$ E \f$:
 *  \f[
 *    \frac{1}{M} = \frac{1}{1-\frac{\alpha^2}{2} V(r)} - \frac{\alpha^2 E}{2 (1-\frac{\alpha^2}{2} V(r))^2} + O(E^2) =
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
 *      \frac{E}{M_0^2} \big) \Big) p(r) - \frac{q(r)}{r}
 *       = \Big(V(r) - E + \frac{\ell(\ell+1)}{2 M_0 r^2} - \frac{\alpha^2}{2} \frac{\ell(\ell+1) E}{2 M_0^2 r^2} \Big)
 *   p(r) - \frac{q(r)}{r} \end{array} \right. \f] The first energy derivatives are then: \f[ \left\{ \begin{array}{ll}
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

    /// Integrate system of two first-order differential equations forward starting from the origin.
    /** Use Runge-Kutta 4th order method */
    template <relativity_t rel, bool prevent_overflow>
    int
    integrate_forward_rk4(double enu__, int l__, int k__, Spline<double> const& chi_p__, Spline<double> const& chi_q__,
                          std::vector<double>& p__, std::vector<double>& dpdr__, std::vector<double>& q__,
                          std::vector<double>& dqdr__) const
    {
        /* number of mesh points */
        int nr = num_points();

        double rest_energy = std::pow(speed_of_light, 2);

        double alpha = 1.0 / speed_of_light;

        double sq_alpha_half = 0.5 / rest_energy;

        if (rel == relativity_t::none) {
            sq_alpha_half = 0;
        }

        double ll_half = l__ * (l__ + 1) / 2.0;

        double kappa{0};
        if (rel == relativity_t::dirac) {
            if (k__ == l__) {
                kappa = k__;
            } else if (k__ == l__ + 1) {
                kappa = -k__;
            } else {
                std::stringstream s;
                s << "wrong k : " << k__ << " for l = " << l__;
                RTE_THROW(s);
            }
        }

        auto rel_mass = [sq_alpha_half](double enu__, double v__) -> double {
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

        /* try to find classical turning point */
        int idx_ctp{-1};
        for (int ir = 0; ir < nr; ir++) {
            if (ve_(ir) - zn_ * radial_grid_.x_inv(ir) > enu__) {
                idx_ctp = ir;
                break;
            }
        }
        /* if we didn't fint the classical turning point, take half of the grid */
        // int idx_tail{-1};
        if (idx_ctp == -1) {
            double rmid = radial_grid_.last() / 2;
            for (int ir = 0; ir < nr; ir++) {
                if (radial_grid_[ir] > rmid) {
                    idx_ctp = ir;
                    break;
                }
            }
        }

        /* here and below var0 means var(x), var2 means var(x+h) and var1 means var(x+h/2) */
        double x2     = radial_grid_[0];
        double xinv2  = radial_grid_.x_inv(0);
        double v2     = ve_(0) - zn_ / x2;
        double M2     = rel_mass(enu__, v2);
        double chi_p2 = chi_p__(0);
        double chi_q2 = chi_q__(0);

        /* r->0 asymptotics */
        if (rel != relativity_t::dirac) {
            if (l__ == 0) {
                p__[0] = 2 * zn_ * x2;
                q__[0] = -std::pow(zn_, 2) * x2;
            } else {
                p__[0] = std::pow(x2, l__ + 1);
                q__[0] = std::pow(x2, l__) * l__ / 2;
            }
        } else {
            double b = std::sqrt(std::pow(kappa, 2) - std::pow(zn_ / speed_of_light, 2));
            p__[0]   = std::pow(x2, b);
            q__[0]   = std::pow(x2, b) * speed_of_light * (b + kappa) / zn_;
        }

        // p__[0] = std::pow(radial_grid_[0], l_ + 1);
        // if (l_ == 0)
        //{
        //    q__[0] = -zn_ * radial_grid_[0] / M2 / 2;
        //}
        // else
        //{
        //    q__[0] = std::pow(radial_grid_[0], l_) * l_ / M2 / 2;
        //}

        double p2 = p__[0];
        double q2 = q__[0];

        double pk[4];
        double qk[4];

        int last{0};

        for (int i = 0; i < nr - 1; i++) {
            /* copy previous values */
            double x0     = x2;
            double xinv0  = xinv2;
            double M0     = M2;
            double chi_p0 = chi_p2;
            double chi_q0 = chi_q2;
            double v0     = v2;
            double p0     = p2;
            double q0     = q2;
            /* radial grid step */
            double h = radial_grid_.dx(i);
            /* mid-point */
            double h_half = h / 2;
            double x1     = x0 + h_half;
            double xinv1  = 1.0 / x1;
            double chi_p1 = chi_p__(i, h_half);
            double chi_q1 = chi_q__(i, h_half);
            double v1     = ve_(i, h_half) - zn_ * xinv1;
            double M1     = rel_mass(enu__, v1);

            /* next point */
            x2     = radial_grid_[i + 1];
            xinv2  = radial_grid_.x_inv(i + 1);
            chi_p2 = chi_p__(i + 1);
            chi_q2 = chi_q__(i + 1);
            v2     = ve_(i + 1) - zn_ * xinv2;
            M2     = rel_mass(enu__, v2);

            if (rel == relativity_t::none || rel == relativity_t::koelling_harmon || rel == relativity_t::zora) {
                /* k0 = F(Y(x), x) */
                pk[0] = 2 * M0 * q0 + p0 * xinv0 + chi_p0;
                qk[0] = (v0 - enu__ + ll_half / M0 / std::pow(x0, 2)) * p0 - q0 * xinv0 + chi_q0;

                /* k1 = F(Y(x) + k0 * h/2, x + h/2) */
                pk[1] = 2 * M1 * (q0 + qk[0] * h_half) + (p0 + pk[0] * h_half) * xinv1 + chi_p1;
                qk[1] = (v1 - enu__ + ll_half / M1 / std::pow(x1, 2)) * (p0 + pk[0] * h_half) -
                        (q0 + qk[0] * h_half) * xinv1 + chi_q1;

                /* k2 = F(Y(x) + k1 * h/2, x + h/2) */
                pk[2] = 2 * M1 * (q0 + qk[1] * h_half) + (p0 + pk[1] * h_half) * xinv1 + chi_p1;
                qk[2] = (v1 - enu__ + ll_half / M1 / std::pow(x1, 2)) * (p0 + pk[1] * h_half) -
                        (q0 + qk[1] * h_half) * xinv1 + chi_q1;

                /* k3 = F(Y(x) + k2 * h, x + h) */
                pk[3] = 2 * M2 * (q0 + qk[2] * h) + (p0 + pk[2] * h) * xinv2 + chi_p2;
                qk[3] = (v2 - enu__ + ll_half / M2 / std::pow(x2, 2)) * (p0 + pk[2] * h) - (q0 + qk[2] * h) * xinv2 +
                        chi_q2;
            }
            if (rel == relativity_t::iora) {
                double m0 = 1 - sq_alpha_half * v0;
                double m1 = 1 - sq_alpha_half * v1;
                double m2 = 1 - sq_alpha_half * v2;

                double a0 = ll_half / m0 / std::pow(x0, 2);
                double a1 = ll_half / m1 / std::pow(x1, 2);
                double a2 = ll_half / m2 / std::pow(x2, 2);

                /* k0 = F(Y(x), x) */
                pk[0] = 2 * M0 * q0 + p0 * xinv0 + chi_p0;
                qk[0] = (v0 - enu__ + a0 - sq_alpha_half * a0 * enu__ / m0) * p0 - q0 * xinv0 + chi_q0;

                /* k1 = F(Y(x) + k0 * h/2, x + h/2) */
                pk[1] = 2 * M1 * (q0 + qk[0] * h_half) + (p0 + pk[0] * h_half) * xinv1 + chi_p1;
                qk[1] = (v1 - enu__ + a1 - sq_alpha_half * a1 * enu__ / m1) * (p0 + pk[0] * h_half) -
                        (q0 + qk[0] * h_half) * xinv1 + chi_q1;

                /* k2 = F(Y(x) + k1 * h/2, x + h/2) */
                pk[2] = 2 * M1 * (q0 + qk[1] * h_half) + (p0 + pk[1] * h_half) * xinv1 + chi_p1;
                qk[2] = (v1 - enu__ + a1 - sq_alpha_half * a1 * enu__ / m1) * (p0 + pk[1] * h_half) -
                        (q0 + qk[1] * h_half) * xinv1 + chi_q1;

                /* k3 = F(Y(x) + k2 * h, x + h) */
                pk[3] = 2 * M2 * (q0 + qk[2] * h) + (p0 + pk[2] * h) * xinv2 + chi_p2;
                qk[3] = (v2 - enu__ + a2 - sq_alpha_half * a2 * enu__ / m2) * (p0 + pk[2] * h) -
                        (q0 + qk[2] * h) * xinv2 + chi_q2;
            }
            if (rel == relativity_t::dirac) {
                /* k0 = F(Y(x), x) */
                pk[0] = alpha * (2 * rest_energy + enu__ - v0) * q0 - kappa * p0 * xinv0;
                qk[0] = alpha * (v0 - enu__) * p0 + kappa * q0 * xinv0;

                /* k1 = F(Y(x) + k0 * h/2, x + h/2) */
                pk[1] = alpha * (2 * rest_energy + enu__ - v1) * (q0 + qk[0] * h_half) -
                        kappa * (p0 + pk[0] * h_half) * xinv1;
                qk[1] = alpha * (v1 - enu__) * (p0 + pk[0] * h_half) + kappa * (q0 + qk[0] * h_half) * xinv1;

                /* k2 = F(Y(x) + k1 * h/2, x + h/2) */
                pk[2] = alpha * (2 * rest_energy + enu__ - v1) * (q0 + qk[1] * h_half) -
                        kappa * (p0 + pk[1] * h_half) * xinv1;
                qk[2] = alpha * (v1 - enu__) * (p0 + pk[1] * h_half) + kappa * (q0 + qk[1] * h_half) * xinv1;

                /* k3 = F(Y(x) + k2 * h, x + h) */
                pk[3] = alpha * (2 * rest_energy + enu__ - v2) * (q0 + qk[2] * h) - kappa * (p0 + pk[2] * h) * xinv2;
                qk[3] = alpha * (v2 - enu__) * (p0 + pk[2] * h) + kappa * (q0 + qk[2] * h) * xinv2;
            }
            /* Y(x + h) = Y(x) + h * (k0 + 2 * k1 + 2 * k2 + k3) / 6 */
            p2 = p0 + (pk[0] + 2 * (pk[1] + pk[2]) + pk[3]) * h / 6.0;
            q2 = q0 + (qk[0] + 2 * (qk[1] + qk[2]) + qk[3]) * h / 6.0;

            /* check overflow */
            // if (std::abs(p2) > 1e10 || std::abs(q2) > 1e10) {
            if (std::abs(p2) > 1e4) {
                /* if we didn't expect the overflow and it happened, or it happened before the
                 * classical turning point, it's bad */
                if (!prevent_overflow || (prevent_overflow && i < idx_ctp)) {
                    std::stringstream s;
                    if (!prevent_overflow) {
                        s << "unexpected overflow ";
                    } else {
                        s << "overflow before the classical turning point ";
                    }
                    s << "for atom type with zn = " << zn_ << ", l = " << l__ << ", enu = " << enu__ << ", ir = " << i
                      << ", idx_ctp: " << idx_ctp;
                    for (int j = 0; j <= i; j++) {
                        p__[j] /= 1e4;
                        q__[j] /= 1e4;
                    }
                    p2 /= 1e4;
                    q2 /= 1e4;

                    // WARNING(s);
                } else { /* if overflow happened after the classical turning point, it's ok */
                    last = i;
                    break;
                }
            }

            p__[i + 1] = p2;
            q__[i + 1] = q2;
        }

        if (prevent_overflow && last) {
            /* find the minimum value of the "tail" */
            double pmax = std::abs(p__[last]);
            /* go towards the origin */
            for (int j = last; j >= 0; j--) {
                if (std::abs(p__[j]) < pmax) {
                    pmax = std::abs(p__[j]);
                } else {
                    /* we may go through zero here and miss one node,
                     * so stay on the safe side with one extra point */
                    last = j + 1;
                    break;
                }
            }
            /* zero the tail */
            for (int j = last; j < nr; j++) {
                p__[j] = 0;
                q__[j] = 0;
            }
        }

        /* get number of nodes */
        int nn{0};
        for (int i = 0; i < nr - 1; i++) {
            if (p__[i] * p__[i + 1] < 0.0) {
                nn++;
            }
        }

        /* normalize solution */
        if (false) {
            Spline<double> s(radial_grid_);
            for (int i = 0; i < nr; i++) {
                s(i) = std::pow(p__[i], 2);
            }
            auto norm = 1.0 / std::sqrt(s.interpolate().integrate(0));
            for (int i = 0; i < nr; i++) {
                p__[i] *= norm;
                q__[i] *= norm;
            }
        }

        for (int i = 0; i < nr; i++) {
            if (rel == relativity_t::none || rel == relativity_t::koelling_harmon || rel == relativity_t::zora) {
                double V  = ve_(i) - zn_ * radial_grid_.x_inv(i);
                double M  = rel_mass(enu__, V);
                double v1 = ll_half / M / std::pow(radial_grid_[i], 2);

                /* P' = 2MQ + \frac{P}{r} */
                dpdr__[i] = 2 * M * q__[i] + p__[i] * radial_grid_.x_inv(i) + chi_p__(i);

                /* Q' = (V - E + \frac{\ell(\ell + 1)}{2 M r^2}) P - \frac{Q}{r} */
                dqdr__[i] = (V - enu__ + v1) * p__[i] - q__[i] * radial_grid_.x_inv(i) + chi_q__(i);
            }
            if (rel == relativity_t::iora) {
                double V  = ve_(i) - zn_ * radial_grid_.x_inv(i);
                double M  = rel_mass(enu__, V);
                double M0 = 1 - sq_alpha_half * V;
                double v1 = ll_half / M0 / std::pow(radial_grid_[i], 2);
                double v2 = sq_alpha_half * ll_half * enu__ / std::pow(radial_grid_[i] * M0, 2);

                /* P' = 2MQ + \frac{P}{r} */
                dpdr__[i] = 2 * M * q__[i] + p__[i] * radial_grid_.x_inv(i) + chi_p__(i);

                /* Q' = (V - E + \frac{\ell(\ell + 1)}{2 M r^2}) P - \frac{Q}{r} */
                dqdr__[i] = (V - enu__ + v1 - v2) * p__[i] - q__[i] * radial_grid_.x_inv(i) + chi_q__(i);
            }
            if (rel == relativity_t::dirac) {
                double V = ve_(i) - zn_ * radial_grid_.x_inv(i);
                /* P' = ... */
                dpdr__[i] = alpha * (2 * rest_energy + enu__ - V) * q__[i] - kappa * p__[i] * radial_grid_.x_inv(i);
                /* Q' = ... */
                dqdr__[i] = alpha * (V - enu__) * p__[i] + kappa * q__[i] * radial_grid_.x_inv(i);
            }
        }

        return nn;
    }

    //== inline double extrapolate_to_zero(int istep, double y, double* x, double* work) const
    //== {
    //==     double dy = y;
    //==     double result = y;
    //==
    //==     if (istep == 0)
    //==     {
    //==         work[0] = y;
    //==     }
    //==     else
    //==     {
    //==         double c = y;
    //==         for (int k = 1; k < istep; k++)
    //==         {
    //==             double delta = 1.0 / (x[istep - k] - x[istep]);
    //==             double f1 = x[istep] * delta;
    //==             double f2 = x[istep - k] * delta;
    //==
    //==             double q = work[k];
    //==             work[k] = dy;
    //==             delta = c - q;
    //==             dy = f1 * delta;
    //==             c = f2 * delta;
    //==             result += dy;
    //==         }
    //==     }
    //==     work[istep] = dy;
    //==     return result;
    //== }

    //== /// Integrate system of two first-order differential equations forward starting from the origin.
    //== /** Use Bulirsch-Stoer technique with Gragg's modified midpoint method */
    //== template <bool check_overflow>
    //== int integrate_forward_gbs(double enu__,
    //==                           Spline<double> const& ve__,
    //==                           Spline<double> const& mp__,
    //==                           std::vector<double>& p__,
    //==                           std::vector<double>& dpdr__,
    //==                           std::vector<double>& q__,
    //==                           std::vector<double>& dqdr__) const
    //== {
    //==     /* number of mesh points */
    //==     int nr = num_points();
    //==
    //==     double alpha2 = 0.5 * std::pow((1 / speed_of_light), 2);
    //==     if (!relativistic_) alpha2 = 0.0;

    //==     double enu0 = 0.0;
    //==     if (relativistic_) enu0 = enu__;

    //==     double ll2 = 0.5 * l_ * (l_ + 1);

    //==     double x2 = radial_grid_[0];
    //==     double v2 = ve__[0] - zn_ / x2;
    //==     double M2 = 1 - (v2 - enu0) * alpha2;

    //==     p__[0] = std::pow(radial_grid_[0], l_ + 1);
    //==     if (l_ == 0)
    //==     {
    //==         q__[0] = -zn_ * radial_grid_[0] / M2 / 2;
    //==     }
    //==     else
    //==     {
    //==         q__[0] = std::pow(radial_grid_[0], l_) * l_ / M2 / 2;
    //==     }
    //==
    //==     double step_size2[20];
    //==     double work_p[20];
    //==     double work_q[20];

    //==     int last = 0;

    //==     for (int ir = 0; ir < nr - 1; ir++)
    //==     {
    //==         double H = radial_grid_.dx(ir);
    //==         double p_est, q_est, p_old, q_old;
    //==         for (int j = 0; j < 12; j++)
    //==         {
    //==             int num_steps = 2 * (j + 1);
    //==             double h = H / num_steps;
    //==             double h2 = h + h;
    //==             step_size2[j] = std::pow(h, 2);
    //==             double x0 = radial_grid_[ir];
    //==             double x0inv = radial_grid_.x_inv(ir);
    //==             double x1inv = radial_grid_.x_inv(ir + 1);

    //==             p_old = p__[ir + 1];
    //==             q_old = q__[ir + 1];

    //==             double p0 = p__[ir];
    //==             double q0 = q__[ir];
    //==             double p1 = p0 + h * (2 * q0 + p0 * x0inv);
    //==             double q1 = q0 + h * ((ve__[ir] + x0inv * (ll2 * x0inv - zn_) - enu__) * p0 - q0 * x0inv -
    // mp__[ir]);

    //==             for (int step = 1; step < num_steps; step++)
    //==             {
    //==                 double x = x0 + h * step;
    //==                 double xinv = 1.0 / x;
    //==                 double p2 = p0 + h2 * (2 * q1 + p1 * xinv);
    //==                 double q2 = q0 + h2 * ((ve__(ir, h * step) + xinv * (ll2 * xinv - zn_) - enu__) * p1 -
    //==                                        q1 * xinv - mp__(ir, h * step));
    //==                 p0 = p1;
    //==                 p1 = p2;
    //==                 q0 = q1;
    //==                 q1 = q2;
    //==             }
    //==             p_est = 0.5 * (p0 + p1 + h * (2 * q1 + p1 * x1inv));
    //==             q_est = 0.5 * (q0 + q1 + h * ((ve__[ir + 1] + x1inv * (ll2 * x1inv - zn_) - enu__) * p1 -
    //==                                            q1 * x1inv - mp__[ir + 1]));

    //==             p__[ir + 1] = extrapolate_to_zero(j, p_est, step_size2, work_p);
    //==             q__[ir + 1] = extrapolate_to_zero(j, q_est, step_size2, work_q);
    //==
    //==             if (j > 1)
    //==             {
    //==                 if (std::abs(p__[ir + 1] - p_old) < 1e-12 && std::abs(q__[ir + 1] - q_old) < 1e-12) break;
    //==             }
    //==         }

    //==         /* don't allow overflow */
    //==         if (check_overflow && std::abs(p__[ir + 1]) > 1e10)
    //==         {
    //==             last = ir;
    //==             break;
    //==         }

    //==         if (!check_overflow && std::abs(p__[ir + 1]) > 1e10)
    //==         {
    //==             TERMINATE("overflow");
    //==         }
    //==     }

    //==     if (check_overflow && last)
    //==     {
    //==         /* find the minimum value of the "tail" */
    //==         double pmax = std::abs(p__[last]);
    //==         for (int j = last - 1; j >= 0; j++)
    //==         {
    //==             if (std::abs(p__[j]) < pmax)
    //==             {
    //==                 pmax = std::abs(p__[j]);
    //==             }
    //==             else
    //==             {
    //==                 /* we may go through zero here and miss one node,
    //==                  * so stay on the safe side with one extra point */
    //==                 last = j + 1;
    //==                 break;
    //==             }
    //==         }
    //==         for (int j = last; j < nr; j++)
    //==         {
    //==             p__[j] = 0;
    //==             q__[j] = 0;
    //==         }
    //==     }
    //==
    //==     /* get number of nodes */
    //==     int nn = 0;
    //==     for (int i = 0; i < nr - 1; i++) if (p__[i] * p__[i + 1] < 0.0) nn++;

    //==     for (int i = 0; i < nr; i++)
    //==     {
    //==         double V = ve__[i] - zn_ * radial_grid_.x_inv(i);
    //==         double M = 1.0 - (V - enu0) * alpha2;

    //==         /* P' = 2MQ + \frac{P}{r} */
    //==         dpdr__[i] = 2 * M * q__[i] + p__[i] * radial_grid_.x_inv(i);

    //==         /* Q' = (V - E + \frac{\ell(\ell + 1)}{2 M r^2}) P - \frac{Q}{r} */
    //==         dqdr__[i] = (V - enu__ + double(l_ * (l_ + 1)) / (2 * M * std::pow(radial_grid_[i], 2))) * p__[i] -
    //==                     q__[i] * radial_grid_.x_inv(i) - mp__[i];
    //==     }

    //==     return nn;
    //== }

  public:
    Radial_solver(int zn__, std::vector<double> const& v__, Radial_grid<double> const& radial_grid__)
        : zn_(zn__)
        , radial_grid_(radial_grid__)
    {
        ve_ = Spline<double>(radial_grid__);

        for (int i = 0; i < num_points(); i++) {
            ve_(i) = v__[i] + zn_ * radial_grid_.x_inv(i);
        }
        ve_.interpolate();
    }

    std::tuple<int, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
    solve(relativity_t rel__, int dme__, int l__, int k__, double enu__) const
    {
        int nr = num_points();
        std::vector<std::vector<double>> p;
        std::vector<std::vector<double>> q;
        std::vector<std::vector<double>> dpdr;
        std::vector<std::vector<double>> dqdr;

        Spline<double> chi_p(radial_grid_);
        Spline<double> chi_q(radial_grid_);

        int nn{0};

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
                    double sq_alpha = std::pow(speed_of_light, -2);
                    double ll_half  = l__ * (l__ + 1) / 2.0;
                    if (j == 1) {
                        for (int i = 0; i < nr; i++) {
                            double x = radial_grid_[i];
                            double V = ve_(i) - zn_ * radial_grid_.x_inv(i);
                            double M = 1 + 0.5 * sq_alpha * (enu__ - V);
                            chi_p(i) = sq_alpha * q[j - 1][i];
                            chi_q(i) = -p[j - 1][i] * (1 + 0.5 * sq_alpha * ll_half / std::pow(M * x, 2));
                        }
                    } else if (j == 2) {
                        for (int i = 0; i < nr; i++) {
                            double x = radial_grid_[i];
                            double V = ve_(i) - zn_ * radial_grid_.x_inv(i);
                            double M = 1 + 0.5 * sq_alpha * (enu__ - V);
                            chi_p(i) = 2 * sq_alpha * q[j - 1][i];
                            chi_q(i) = -2 * p[j - 1][i] * (1 + 0.5 * sq_alpha * ll_half / std::pow(M * x, 2)) +
                                       p[j - 2][i] * (0.5 * ll_half * std::pow(sq_alpha, 2) / std::pow(M * x, 2) / M);
                        }
                    } else {
                        std::stringstream s;
                        s << "energy derivative of the order " << j
                          << " is not implemented for Koelling-Harmon radial solver";
                        RTE_THROW(s);
                    }
                } else if (rel__ == relativity_t::iora) {
                    double sq_alpha = std::pow(speed_of_light, -2);
                    double ll_half  = l__ * (l__ + 1) / 2.0;

                    if (j == 1) {
                        for (int i = 0; i < nr; i++) {
                            double V = ve_(i) - zn_ * radial_grid_.x_inv(i);
                            double M = 1 - 0.5 * sq_alpha * V;
                            double x = radial_grid_[i];
                            chi_p(i) = q[j - 1][i] * sq_alpha / std::pow(1 - sq_alpha * enu__ / 2 / M, 2);
                            chi_q(i) = -p[j - 1][i] * (1 + 0.5 * sq_alpha * ll_half / std::pow(M * x, 2));
                        }
                    } else if (j == 2) {
                        for (int i = 0; i < nr; i++) {
                            double V = ve_(i) - zn_ * radial_grid_.x_inv(i);
                            double M = 1 - 0.5 * sq_alpha * V;
                            double x = radial_grid_[i];
                            chi_p(i) = q[j - 1][i] * 2 * sq_alpha / std::pow(1 - sq_alpha * enu__ / 2 / M, 2) +
                                       q[j - 2][i] * std::pow(sq_alpha, 2) / 2 / M /
                                               std::pow(1 - sq_alpha * enu__ / 2 / M, 3);
                            chi_q(i) = -p[j - 1][i] * 2 * (1 + 0.5 * sq_alpha * ll_half / std::pow(M * x, 2));
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
                    nn = integrate_forward_rk4<relativity_t::none, false>(enu__, l__, 0, chi_p, chi_q, p[j], dpdr[j],
                                                                          q[j], dqdr[j]);
                    break;
                }
                case relativity_t::koelling_harmon: {
                    nn = integrate_forward_rk4<relativity_t::koelling_harmon, false>(enu__, l__, 0, chi_p, chi_q, p[j],
                                                                                     dpdr[j], q[j], dqdr[j]);
                    break;
                }
                case relativity_t::zora: {
                    nn = integrate_forward_rk4<relativity_t::zora, false>(enu__, l__, 0, chi_p, chi_q, p[j], dpdr[j],
                                                                          q[j], dqdr[j]);
                    break;
                }
                case relativity_t::iora: {
                    nn = integrate_forward_rk4<relativity_t::iora, false>(enu__, l__, 0, chi_p, chi_q, p[j], dpdr[j],
                                                                          q[j], dqdr[j]);
                    break;
                }
                default: {
                    RTE_THROW("unsupported relativity type");
                }
            }
        }

        return std::make_tuple(nn, p.back(), dpdr.back(), q.back(), dqdr.back());
    }

    /// Integrates the radial equation for a given energy and finds the m-th energy derivative of the radial solution.
    /** \param [in] rel Type of relativity
     *  \param [in] dme Order of energy derivative.
     *  \param [in] l Oribtal quantum number.
     *  \param [in] enu Integration energy.
     *  \param [out] p \f$ p(r) = ru(r) \f$ radial function.
     *  \param [out] rdudr \f$ ru'(r) \f$.
     *  \param [out] uderiv \f$ u'(R) \f$ and \f$ u''(R) \f$.
     *
     *  Returns \f$ p(r) = ru(r) \f$, \f$ ru'(r)\f$, \f$ u'(R) \f$ and \f$ u''(R) \f$.
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
    int
    solve(relativity_t rel__, int dme__, int l__, double enu__, std::vector<double>& p__, std::vector<double>& rdudr__,
          std::array<double, 2>& uderiv__) const
    {
        auto result = solve(rel__, dme__, l__, 0, enu__);
        int nr      = num_points();

        auto p0 = std::get<1>(result);
        auto p1 = std::get<2>(result);
        auto q0 = std::get<3>(result);
        auto q1 = std::get<4>(result);

        /* save the results */
        p__.resize(nr);
        rdudr__.resize(nr);
        for (int i = 0; i < radial_grid_.num_points(); i++) {
            p__[i]     = p0[i];
            rdudr__[i] = p1[i] - p0[i] * radial_grid_.x_inv(i);
        }

        double R = radial_grid_.last();

        /* 1st radial derivative of u(r) */
        uderiv__[0] = (p1.back() - p0.back() / R) / R;
        /* 2nd radial derivative of u(r) */
        Spline<double> sdpdr(radial_grid_, p1);
        sdpdr.interpolate();
        uderiv__[1] = (sdpdr.deriv(1, nr - 1) - 2 * p1.back() / R + 2 * p0.back() / std::pow(R, 2)) / R;

        return std::get<0>(result);
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

    void
    solve(relativity_t rel__, double enu_start__)
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
        enu_ = enu_start__;
        double denu{0.1};

        /* search for the bound state */
        for (int iter = 0; iter < 1000; iter++) {
            int nn{0};

            switch (rel__) {
                case relativity_t::none: {
                    nn = integrate_forward_rk4<relativity_t::none, true>(enu_, l_, k_, chi_p, chi_q, p, dpdr_, q, dqdr);
                    break;
                }
                case relativity_t::koelling_harmon: {
                    nn = integrate_forward_rk4<relativity_t::koelling_harmon, true>(enu_, l_, k_, chi_p, chi_q, p,
                                                                                    dpdr_, q, dqdr);
                    break;
                }
                case relativity_t::zora: {
                    nn = integrate_forward_rk4<relativity_t::zora, true>(enu_, l_, k_, chi_p, chi_q, p, dpdr_, q, dqdr);
                    break;
                }
                case relativity_t::dirac: {
                    nn = integrate_forward_rk4<relativity_t::dirac, true>(enu_, l_, k_, chi_p, chi_q, p, dpdr_, q,
                                                                          dqdr);
                    break;
                }
                default: {
                    RTE_THROW("unsupported relativity type");
                }
            }

            sp   = s;
            s    = (nn > (n_ - l_ - 1)) ? -1 : 1;
            denu = s * std::abs(denu);
            denu = (s != sp) ? denu * 0.5 : denu * 1.25;
            enu_ += denu;

            if (std::abs(denu) < enu_tolerance_ && iter > 4) {
                break;
            }
        }

        if (std::abs(denu) >= enu_tolerance_) {
            std::stringstream s;
            s << "enu is not converged for n = " << n_ << " and l = " << l_ << std::endl
              << "enu = " << enu_ << ", denu = " << denu;
            RTE_THROW(s);
        }

        /* compute r * u'(r) */
        for (int i = 0; i < num_points(); i++) {
            rdudr[i] = dpdr_[i] - p[i] / radial_grid(i);
        }

        /* search for the turning point */
        int idxtp = np - 1;
        for (int i = 0; i < np; i++) {
            if (ve_(i) - zn_ * radial_grid_.x_inv(i) > enu_) {
                idxtp = i;
                break;
            }
        }

        /* zero the tail of the wave-function */
        double t1 = 1e100;
        for (int i = idxtp; i < np; i++) {
            if (std::abs(p[i]) < t1 && p[i - 1] * p[i] > 0) {
                t1 = std::abs(p[i]);
            } else {
                t1       = 0.0;
                p[i]     = 0.0;
                q[i]     = 0.0;
                rdudr[i] = 0.0;
            }
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
                std::vector<double> const& v__, double enu_start__)
        : Radial_solver(zn__, v__, radial_grid__)
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
        solve(rel__, enu_start__);
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
    find_enu(relativity_t rel__, double enu_start__)
    {
        int np = num_points();

        Spline<double> chi_p(radial_grid());
        Spline<double> chi_q(radial_grid());

        std::vector<double> p(np);
        std::vector<double> q(np);
        std::vector<double> dpdr(np);
        std::vector<double> dqdr(np);

        double enu = enu_start__;
        double de  = 0.001;
        bool found = false;
        int nndp   = 0;

        /* We want to find enu such that the wave-function at the muffin-tin boundary is zero
         * and the number of nodes inside muffin-tin is equal to n-l-1. This will be the top
         * of the band. */
        for (int i = 0; i < 1000; i++) {
            int nnd{0};

            switch (rel__) {
                case relativity_t::none: {
                    nnd = integrate_forward_rk4<relativity_t::none, false>(enu, l_, 0, chi_p, chi_q, p, dpdr, q, dqdr);
                    break;
                }
                case relativity_t::koelling_harmon: {
                    nnd = integrate_forward_rk4<relativity_t::koelling_harmon, false>(enu, l_, 0, chi_p, chi_q, p, dpdr,
                                                                                      q, dqdr);
                    break;
                }
                case relativity_t::zora: {
                    nnd = integrate_forward_rk4<relativity_t::zora, false>(enu, l_, 0, chi_p, chi_q, p, dpdr, q, dqdr);
                    break;
                }
                case relativity_t::iora: {
                    nnd = integrate_forward_rk4<relativity_t::iora, false>(enu, l_, 0, chi_p, chi_q, p, dpdr, q, dqdr);
                    break;
                }
                default: {
                    RTE_THROW("unsupported relativity type");
                }
            }
            nnd -= (n_ - l_ - 1);

            enu = (nnd > 0) ? enu - de : enu + de;

            if (i) {
                de = (nnd != nndp) ? de * 0.5 : de * 1.25;
            }
            if (std::abs(de) < 1e-10) {
                found = true;
                break;
            }
            nndp = nnd;
        }
        etop_ = (!found) ? enu_start__ : enu;

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
         * at the muffin-tin boundary. This will be the bottom of the band. */
        de = 1e-4;
        for (int i = 0; i < 100; i++) {
            de *= 1.1;
            enu -= de;
            switch (rel__) {
                case relativity_t::none: {
                    integrate_forward_rk4<relativity_t::none, false>(enu, l_, 0, chi_p, chi_q, p, dpdr, q, dqdr);
                    break;
                }
                case relativity_t::koelling_harmon: {
                    integrate_forward_rk4<relativity_t::koelling_harmon, false>(enu, l_, 0, chi_p, chi_q, p, dpdr, q,
                                                                                dqdr);
                    break;
                }
                case relativity_t::zora: {
                    integrate_forward_rk4<relativity_t::zora, false>(enu, l_, 0, chi_p, chi_q, p, dpdr, q, dqdr);
                    break;
                }
                case relativity_t::iora: {
                    integrate_forward_rk4<relativity_t::iora, false>(enu, l_, 0, chi_p, chi_q, p, dpdr, q, dqdr);
                    break;
                }
                default: {
                    RTE_THROW("unsupported relativity type");
                }
            }
            if (surface_deriv() * sd <= 0) {
                break;
            }
        }

        /* refine bottom energy */
        double e1 = enu;
        double e0 = enu + de;

        for (int i = 0; i < 100; i++) {
            enu = (e1 + e0) / 2.0;
            switch (rel__) {
                case relativity_t::none: {
                    integrate_forward_rk4<relativity_t::none, false>(enu, l_, 0, chi_p, chi_q, p, dpdr, q, dqdr);
                    break;
                }
                case relativity_t::koelling_harmon: {
                    integrate_forward_rk4<relativity_t::koelling_harmon, false>(enu, l_, 0, chi_p, chi_q, p, dpdr, q,
                                                                                dqdr);
                    break;
                }
                case relativity_t::zora: {
                    integrate_forward_rk4<relativity_t::zora, false>(enu, l_, 0, chi_p, chi_q, p, dpdr, q, dqdr);
                    break;
                }
                case relativity_t::iora: {
                    integrate_forward_rk4<relativity_t::iora, false>(enu, l_, 0, chi_p, chi_q, p, dpdr, q, dqdr);
                    break;
                }
                default: {
                    RTE_THROW("unsupported relativity type");
                }
            }
            /* derivative at the boundary */
            if (std::abs(surface_deriv()) < 1e-10) {
                break;
            }

            if (surface_deriv() * sd > 0) {
                e0 = enu;
            } else {
                e1 = enu;
            }
        }

        ebot_ = enu;
        /* last check */
        int nn{0};
        switch (rel__) {
            case relativity_t::none: {
                nn = integrate_forward_rk4<relativity_t::none, false>(enu, l_, 0, chi_p, chi_q, p, dpdr, q, dqdr);
                break;
            }
            case relativity_t::koelling_harmon: {
                nn = integrate_forward_rk4<relativity_t::koelling_harmon, false>(enu, l_, 0, chi_p, chi_q, p, dpdr, q,
                                                                                 dqdr);
                break;
            }
            case relativity_t::zora: {
                nn = integrate_forward_rk4<relativity_t::zora, false>(enu, l_, 0, chi_p, chi_q, p, dpdr, q, dqdr);
                break;
            }
            case relativity_t::iora: {
                nn = integrate_forward_rk4<relativity_t::iora, false>(enu, l_, 0, chi_p, chi_q, p, dpdr, q, dqdr);
                break;
            }
            default: {
                RTE_THROW("unsupported relativity type");
            }
        }

        if (nn != n_ - l_ - 1) {
            FILE* fout = fopen("p.dat", "w");
            for (int ir = 0; ir < np; ir++) {
                double x = radial_grid(ir);
                fprintf(fout, "%16.8f %16.8f %16.8f\n", x, p[ir], q[ir]);
            }
            fclose(fout);

            // printf("n: %i, l: %i, nn: %i", n_, l_, nn);
            std::stringstream s;
            s << "wrong number of nodes: " << nn << " instead of " << n_ - l_ - 1 << std::endl
              << "n: " << n_ << ", l: " << l_ << std::endl
              << "etop: " << etop_ << " ebot: " << ebot_ << std::endl
              << "initial surface derivative: " << sd;

            RTE_THROW(s);
        }

        enu_ = (ebot_ + etop_) / 2.0;
    }

  public:
    /// Constructor
    Enu_finder(relativity_t rel__, int zn__, int n__, int l__, Radial_grid<double> const& radial_grid__,
               std::vector<double> const& v__, double enu_start__)
        : Radial_solver(zn__, v__, radial_grid__)
        , n_(n__)
        , l_(l__)
    {
        assert(l_ < n_);
        find_enu(rel__, enu_start__);
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
