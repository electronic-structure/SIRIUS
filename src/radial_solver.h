// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
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

/** \file radial_solver.h
 *   
 *  \brief Contains declaration and partial implementation of sirius::Radial_solver class.
 */

#ifndef __RADIAL_SOLVER_H__
#define __RADIAL_SOLVER_H__

#include "radial_grid.h"
#include "spline.h"
#include "constants.h"

namespace sirius {

/// Solves a "classical" or scalar relativistic radial Schroedinger equation
/** Second order differential equation is converted into the system of coupled first-order differential equations, 
 *  which are then solved by the Runge–Kutta 4th order method.
 *
 *  \f{eqnarray*}{
 *     P' &=& 2 M Q + \frac{P}{r} \\
 *     Q' &=& (V - E + \frac{\ell(\ell + 1)}{2 M r^2}) P - \frac{Q}{r}
 *  \f}
 *
 *  \todo Correct relativistic DFT 
 */
class Radial_solver
{
    private:

        /// True if scalar-relativistic equation is solved.
        bool relativistic_;
        
        /// Negative charge of the nucleus.
        double zn_;
        
        /// Radial grid.
        Radial_grid const& radial_grid_;
        
        /// Tolerance of bound state energy. 
        double enu_tolerance_;
       
        /// Integrate system of two first-order differential equations forward starting from the origin. 
        int integrate(int l, 
                      double enu, 
                      Spline<double>& ve,
                      Spline<double>& mp, 
                      std::vector<double>& p, 
                      std::vector<double>& dpdr, 
                      std::vector<double>& q, 
                      std::vector<double>& dqdr) const;

    public:

        /// Constructor
        Radial_solver(bool relativistic__, double zn__, Radial_grid const& radial_grid__) 
            : relativistic_(relativistic__), 
              zn_(zn__), 
              radial_grid_(radial_grid__),
              enu_tolerance_(1e-10)
        {
        }
        
        /// Explicitly specify tolerance of the solver.
        inline void set_tolerance(double tolerance__)
        {
            enu_tolerance_ = tolerance__;
        }
        
        /// Find center of band (linearization energy).
        double find_enu(int n, int l, std::vector<double>& v, double enu0);

        /// Find m-th energy derivative of the radial solution.
        /** Return raw data: P and Q functions and their radial derivatives */
        int solve(int l,
                  double enu,
                  int m,
                  const std::vector<double>& v,
                  std::vector<double>& p0,
                  std::vector<double>& p1,
                  std::vector<double>& q0,
                  std::vector<double>& q1);
        
        /// Find m-th energy derivative of the radial solution.
        /** Solve radial equation and return \f$ p(r) = ru(r) \f$, \f$ ru'(r)\f$ and \f$ p'(R) \f$ at the 
         *  muffin-tin boundary.
         *
         *  \param [in] l Oribtal quantum number.
         *  \param [in] m Order of energy derivative.
         *  \param [in] e Integration energy.
         *  \param [in] v Spherical potential.
         *  \param [out] p \f$ p(r) = ru(r) \f$ radial function.
         *  \param [out] rdudr \f$ ru'(r) \f$.
         *  \param [out] dpdr \f$ p'(R) \f$ on the sphere.
         */
        int solve(int l__,
                  int m__,
                  double e__,
                  const std::vector<double>& v__,
                  std::vector<double>& p__,
                  std::vector<double>& rdudr__,
                  double* dpdr__);
        
        /// Find a bound state.
        /** Radial grid must be large enough to fully hold the bound state. */
        double bound_state(int n__,
                           int l__,
                           double enu__,
                           const std::vector<double>& v__,
                           std::vector<double>& p__) const;

        double bound_state(int n__,
                           int l__,
                           double enu__,
                           const std::vector<double>& v__,
                           std::vector<double>& p__,
                           std::vector<double>& rdudr__) const;
};

/// Find a solution to radial Schrodinger equation.
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
 *  So we can generalize the radial Schrodinger equation like this:
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
 *    p''(r) = 2q'(r) + \frac{p'(r)}{r} - \frac{p(r)}{r^2} = 2q'(r) + \frac{2q(r)}{r} + \frac{p(r)}{r^2} - \frac{p(r)}{r^2} 
 *  \f]
 *  \f[
 *    q'(r) = \frac{1}{2}p''(r) - \frac{q(r)}{r} = \big(V_{eff}(r) - E\big) p(r) - \frac{q(r)}{r} - \chi(r)
 *  \f]
 *  Final expression for a linear system of differential equations:
 *  \f{eqnarray*}{
 *    p'(r) &=& 2q(r) + \frac{p(r)}{r} \\
 *    q'(r) &=& \big(V_{eff}(r) - E\big) p(r) - \frac{q(r)}{r} - \chi(r)
 *  \f}
 */
class Radial_solution
{
    private:

        /// Type of relativistic solution (0 - non-relativistic, 1 - scalar-relativistic).
        int relativistic_;
        
        /// Positive charge of the nucleus.
        int zn_;

        /// Orbital quantum number.
        int l_;

        /// Radial grid.
        Radial_grid const& radial_grid_;

    protected: 

        /// Integrate system of two first-order differential equations forward starting from the origin. 
        /** Use Runge-Kutta 4th order method */
        template <bool check_overflow>
        int integrate_forward_rk4(double enu__,
                                  Spline<double> const& ve__,
                                  Spline<double> const& mp__,
                                  std::vector<double>& p__,
                                  std::vector<double>& dpdr__,
                                  std::vector<double>& q__,
                                  std::vector<double>& dqdr__) const
        {
            /* number of mesh points */
            int nr = num_points();
            
            double alpha2 = 0.5 * std::pow(speed_of_light, -2);
            if (!relativistic_) alpha2 = 0.0;

            double enu0 = 0.0;
            if (relativistic_) enu0 = enu__;

            double ll2 = 0.5 * l_ * (l_ + 1);

            double x2 = radial_grid_[0];
            double x2inv = radial_grid_.x_inv(0);
            double v2 = ve__[0] - zn_ / x2;
            double M2 = 1 - (v2 - enu0) * alpha2;

            if (l_ == 0)
            {
                p__[0] = 2 * zn_ * x2;
                q__[0] = -std::pow(zn_, 2) * x2;
            }
            else
            {
                p__[0] = std::pow(x2, l_ + 1);
                q__[0] = std::pow(x2, l_) * l_ / 2;
            }

            //p__[0] = std::pow(radial_grid_[0], l_ + 1);
            //if (l_ == 0)
            //{
            //    q__[0] = -zn_ * radial_grid_[0] / M2 / 2;
            //}
            //else
            //{
            //    q__[0] = std::pow(radial_grid_[0], l_) * l_ / M2 / 2;
            //}

            double p2 = p__[0];
            double q2 = q__[0];
            double mp2 = mp__[0];
            double vl2 = ll2 / M2 / std::pow(x2, 2);

            double v2enuvl2 = (v2 - enu__ + vl2);

            double pk[4];
            double qk[4];

            int last = 0;
            
            for (int i = 0; i < nr - 1; i++)
            {
                double x0 = x2;
                x2 = radial_grid_[i + 1];
                double x0inv = x2inv;
                x2inv = radial_grid_.x_inv(i + 1);
                double h = radial_grid_.dx(i);
                double h1 = h / 2;

                double x1 = x0 + h1;
                double x1inv = 1.0 / x1;
                double p0 = p2;
                double q0 = q2;
                double M0 = M2;
                v2 = ve__[i + 1] - zn_ * x2inv;

                double mp0 = mp2;
                mp2 = mp__[i + 1];
                double mp1 = mp__(i, h1);
                double v1 = ve__(i, h1) - zn_ * x1inv;
                double M1 = 1 - (v1 - enu0) * alpha2;
                M2 = 1 - (v2 - enu0) * alpha2;
                vl2 = ll2 / M2 / std::pow(x2, 2);
                
                double v0enuvl0 = v2enuvl2;
                v2enuvl2 = (v2 - enu__ + vl2);
                
                double vl1 = ll2 / M1 / std::pow(x1, 2);

                double v1enuvl1 = (v1 - enu__ + vl1);
                
                // k0 = F(Y(x), x)
                pk[0] = 2 * M0 * q0 + p0 * x0inv;
                qk[0] = v0enuvl0 * p0 - q0 * x0inv - mp0;

                // k1 = F(Y(x) + k0 * h/2, x + h/2)
                pk[1] = 2 * M1 * (q0 + qk[0] * h1) + (p0 + pk[0] * h1) * x1inv;
                qk[1] = v1enuvl1 * (p0 + pk[0] * h1) - (q0 + qk[0] * h1) * x1inv - mp1;

                // k2 = F(Y(x) + k1 * h/2, x + h/2)
                pk[2] = 2 * M1 * (q0 + qk[1] * h1) + (p0 + pk[1] * h1) * x1inv; 
                qk[2] = v1enuvl1 * (p0 + pk[1] * h1) - (q0 + qk[1] * h1) * x1inv - mp1;

                // k3 = F(Y(x) + k2 * h, x + h)
                pk[3] = 2 * M2 * (q0 + qk[2] * h) + (p0 + pk[2] * h) * x2inv; 
                qk[3] = v2enuvl2 * (p0 + pk[2] * h) - (q0 + qk[2] * h) * x2inv - mp2;
                
                // Y(x + h) = Y(x) + h * (k0 + 2 * k1 + 2 * k2 + k3) / 6
                p2 = p0 + (pk[0] + 2 * (pk[1] + pk[2]) + pk[3]) * h / 6.0;
                q2 = q0 + (qk[0] + 2 * (qk[1] + qk[2]) + qk[3]) * h / 6.0;

                /* don't allow overflow */
                if (check_overflow && std::abs(p2) > 1e10)
                {
                    last = i;
                    break;
                }

                if (!check_overflow && std::abs(p2) > 1e10)
                {
                    std::stringstream s;
                    s << "overflow for atom type with zn = " << zn_ << ", l = " << l_ << ", enu = " << enu__;
                    TERMINATE(s);
                    p2 = std::max(std::min(1e10, p2), -1e10);
                    q2 = std::max(std::min(1e10, q2), -1e10);
                }
               
                p__[i + 1] = p2;
                q__[i + 1] = q2;
            }

            if (check_overflow && last)
            {
                /* find the minimum value of the "tail" */
                double pmax = std::abs(p__[last]);
                for (int j = last - 1; j >= 0; j++)
                {
                    if (std::abs(p__[j]) < pmax)
                    {
                        pmax = std::abs(p__[j]);
                    }
                    else
                    {
                        /* we may go through zero here and miss one node,
                         * so stay on the safe side with one extra point */
                        last = j + 1;
                        break;
                    }
                }
                for (int j = last; j < nr; j++)
                {
                    p__[j] = 0;
                    q__[j] = 0;
                }
            }
            
            /* get number of nodes */
            int nn = 0;
            for (int i = 0; i < nr - 1; i++) if (p__[i] * p__[i + 1] < 0.0) nn++;

            for (int i = 0; i < nr; i++)
            {
                double V = ve__[i] - zn_ * radial_grid_.x_inv(i); 
                double M = 1.0 - (V - enu0) * alpha2;

                /* P' = 2MQ + \frac{P}{r} */
                dpdr__[i] = 2 * M * q__[i] + p__[i] * radial_grid_.x_inv(i);

                /* Q' = (V - E + \frac{\ell(\ell + 1)}{2 M r^2}) P - \frac{Q}{r} */
                dqdr__[i] = (V - enu__ + double(l_ * (l_ + 1)) / (2 * M * std::pow(radial_grid_[i], 2))) * p__[i] - 
                            q__[i] * radial_grid_.x_inv(i) - mp__[i];
            }

            return nn;
        }

        inline double extrapolate_to_zero(int istep, double y, double* x, double* work) const
        {
            double dy = y;
            double result = y;
             
            if (istep == 0) 
            {
                work[0] = y;
            } 
            else 
            {
                double c = y;
                for (int k = 1; k < istep; k++) 
                {
                    double delta = 1.0 / (x[istep - k] - x[istep]);
                    double f1 = x[istep] * delta;
                    double f2 = x[istep - k] * delta;
             
                    double q = work[k];
                    work[k] = dy;
                    delta = c - q;
                    dy = f1 * delta;
                    c = f2 * delta;
                    result += dy;
                }
            }
            work[istep] = dy;
            return result;
        }

        /// Integrate system of two first-order differential equations forward starting from the origin.
        /** Use Bulirsch-Stoer technique with Gragg's modified midpoint method */
        template <bool check_overflow>
        int integrate_forward_gbs(double enu__,
                                  Spline<double> const& ve__,
                                  Spline<double> const& mp__,
                                  std::vector<double>& p__,
                                  std::vector<double>& dpdr__,
                                  std::vector<double>& q__,
                                  std::vector<double>& dqdr__) const
        {
            /* number of mesh points */
            int nr = num_points();
            
            double alpha2 = 0.5 * std::pow((1 / speed_of_light), 2);
            if (!relativistic_) alpha2 = 0.0;

            double enu0 = 0.0;
            if (relativistic_) enu0 = enu__;

            double ll2 = 0.5 * l_ * (l_ + 1);

            double x2 = radial_grid_[0];
            double v2 = ve__[0] - zn_ / x2;
            double M2 = 1 - (v2 - enu0) * alpha2;

            p__[0] = std::pow(radial_grid_[0], l_ + 1);
            if (l_ == 0)
            {
                q__[0] = -zn_ * radial_grid_[0] / M2 / 2;
            }
            else
            {
                q__[0] = std::pow(radial_grid_[0], l_) * l_ / M2 / 2;
            }
            
            double step_size2[20];
            double work_p[20];
            double work_q[20];

            int last = 0;

            for (int ir = 0; ir < nr - 1; ir++)
            {
                double H = radial_grid_.dx(ir);
                double p_est, q_est, p_old, q_old;
                for (int j = 0; j < 12; j++)
                {
                    int num_steps = 2 * (j + 1);
                    double h = H / num_steps;
                    double h2 = h + h;
                    step_size2[j] = std::pow(h, 2);
                    double x0 = radial_grid_[ir];
                    double x0inv = radial_grid_.x_inv(ir);
                    double x1inv = radial_grid_.x_inv(ir + 1);

                    p_old = p__[ir + 1];
                    q_old = q__[ir + 1];

                    double p0 = p__[ir];
                    double q0 = q__[ir];
                    double p1 = p0 + h * (2 * q0 + p0 * x0inv);
                    double q1 = q0 + h * ((ve__[ir] + x0inv * (ll2 * x0inv - zn_) - enu__) * p0 - q0 * x0inv - mp__[ir]);

                    for (int step = 1; step < num_steps; step++)
                    {
                        double x = x0 + h * step;
                        double xinv = 1.0 / x;
                        double p2 = p0 + h2 * (2 * q1 + p1 * xinv);
                        double q2 = q0 + h2 * ((ve__(ir, h * step) + xinv * (ll2 * xinv - zn_) - enu__) * p1 -
                                               q1 * xinv - mp__(ir, h * step));
                        p0 = p1;
                        p1 = p2;
                        q0 = q1;
                        q1 = q2;
                    }
                    p_est = 0.5 * (p0 + p1 + h * (2 * q1 + p1 * x1inv));
                    q_est = 0.5 * (q0 + q1 + h * ((ve__[ir + 1] + x1inv * (ll2 * x1inv - zn_) - enu__) * p1 -
                                                   q1 * x1inv - mp__[ir + 1]));

                    p__[ir + 1] = extrapolate_to_zero(j, p_est, step_size2, work_p);
                    q__[ir + 1] = extrapolate_to_zero(j, q_est, step_size2, work_q);
                    
                    if (j > 1)
                    {
                        if (std::abs(p__[ir + 1] - p_old) < 1e-12 && std::abs(q__[ir + 1] - q_old) < 1e-12) break;
                    }
                }

                /* don't allow overflow */
                if (check_overflow && std::abs(p__[ir + 1]) > 1e10)
                {
                    last = ir;
                    break;
                }

                if (!check_overflow && std::abs(p__[ir + 1]) > 1e10)
                {
                    TERMINATE("overflow");
                }
            }

            if (check_overflow && last)
            {
                /* find the minimum value of the "tail" */
                double pmax = std::abs(p__[last]);
                for (int j = last - 1; j >= 0; j++)
                {
                    if (std::abs(p__[j]) < pmax)
                    {
                        pmax = std::abs(p__[j]);
                    }
                    else
                    {
                        /* we may go through zero here and miss one node,
                         * so stay on the safe side with one extra point */
                        last = j + 1;
                        break;
                    }
                }
                for (int j = last; j < nr; j++)
                {
                    p__[j] = 0;
                    q__[j] = 0;
                }
            }
            
            /* get number of nodes */
            int nn = 0;
            for (int i = 0; i < nr - 1; i++) if (p__[i] * p__[i + 1] < 0.0) nn++;

            for (int i = 0; i < nr; i++)
            {
                double V = ve__[i] - zn_ * radial_grid_.x_inv(i); 
                double M = 1.0 - (V - enu0) * alpha2;

                /* P' = 2MQ + \frac{P}{r} */
                dpdr__[i] = 2 * M * q__[i] + p__[i] * radial_grid_.x_inv(i);

                /* Q' = (V - E + \frac{\ell(\ell + 1)}{2 M r^2}) P - \frac{Q}{r} */
                dqdr__[i] = (V - enu__ + double(l_ * (l_ + 1)) / (2 * M * std::pow(radial_grid_[i], 2))) * p__[i] - 
                            q__[i] * radial_grid_.x_inv(i) - mp__[i];
            }

            return nn;
        }
    
    public:
        
        Radial_solution(int relativistic__, int zn__, int l__, Radial_grid const& radial_grid__)
            : relativistic_(relativistic__),
              zn_(zn__),
              l_(l__),
              radial_grid_(radial_grid__)
        {
        }

        inline int num_points() const
        {
            return radial_grid_.num_points();
        }

        inline int zn() const
        {
            return zn_;
        }

        inline double radial_grid(int i__) const
        {
            return radial_grid_[i__];
        }

        inline Radial_grid const& radial_grid() const
        {
            return radial_grid_;
        }
};

class Bound_state: public Radial_solution
{
    private:
        
        int n_;

        int l_;
        
        /// Tolerance of bound state energy. 
        double enu_tolerance_;

        double enu_;

        Spline<double> p_;

        Spline<double> u_;

        Spline<double> rdudr_;

        void solve(std::vector<double> const& v__, double enu_start__)
        {
            int np = num_points();

            Spline<double> vs(radial_grid());
            for (int i = 0; i < np; i++) vs[i] = v__[i] + zn() / radial_grid(i);
            vs.interpolate();

            Spline<double> mp(radial_grid());
            
            std::vector<double> p(np);
            std::vector<double> q(np);
            std::vector<double> dpdr(np);
            std::vector<double> dqdr(np);
            std::vector<double> rdudr(np);
            
            int s = 1;
            int sp;
            enu_ = enu_start__;
            double denu = 0.1;
            
            /* search for the bound state */
            for (int iter = 0; iter < 1000; iter++)
            {
                int nn = integrate_forward_rk4<true>(enu_, vs, mp, p, dpdr, q, dqdr);
                //int nn = integrate_forward_gbs<true>(enu_, vs, mp, p, dpdr, q, dqdr);
                
                sp = s;
                s = (nn > (n_ - l_ - 1)) ? -1 : 1;
                denu = s * std::abs(denu);
                denu = (s != sp) ? denu * 0.5 : denu * 1.25;
                enu_ += denu;

                if (std::abs(denu) < enu_tolerance_ && iter > 4) break;
            }

            if (std::abs(denu) >= enu_tolerance_) 
            {
                std::stringstream s;
                s << "enu is not converged for n = " << n_ << " and l = " << l_ << std::endl
                  << "enu = " << enu_ << ", denu = " << denu;
                
                TERMINATE(s);
            }

            /* compute r * u'(r) */
            for (int i = 0; i < num_points(); i++) rdudr[i] = dpdr[i] - p[i] / radial_grid(i);

            /* search for the turning point */
            int idxtp = np - 1;
            for (int i = 0; i < np; i++)
            {
                if (v__[i] > enu_)
                {
                    idxtp = i;
                    break;
                }
            }

            /* zero the tail of the wave-function */
            double t1 = 1e100;
            for (int i = idxtp; i < np; i++)
            {
                if (std::abs(p[i]) < t1 && p[i - 1] * p[i] > 0)
                {
                    t1 = std::abs(p[i]);
                }
                else
                {
                    t1 = 0.0;
                    p[i] = 0.0;
                    rdudr[i] = 0.0;
                }
            }

            for (int i = 0; i < np; i++)
            {
                p_[i] = p[i];
                u_[i] = p[i] / radial_grid(i);
                rdudr_[i] = rdudr[i];
            }
            p_.interpolate();
            u_.interpolate();
            rdudr_.interpolate();

            /* p is not divided by r, so we integrate with r^0 prefactor */
            double norm = 1.0 / std::sqrt(inner(p_, p_, 0));
            p_.scale(norm);
            u_.scale(norm);
            rdudr_.scale(norm);

            /* count number of nodes of the function */
            int nn = 0;
            for (int i = 0; i < np - 1; i++) if (p_[i] * p_[i + 1] < 0.0) nn++;

            if (nn != (n_ - l_ - 1))
            {
                FILE* fout = fopen("p.dat", "w");
                for (int ir = 0; ir < np; ir++) 
                {
                    double x = radial_grid(ir);
                    fprintf(fout, "%12.6f %16.8f\n", x, p_[ir]);
                }
                fclose(fout);

                std::stringstream s;
                s << "n = " << n_ << std::endl 
                  << "l = " << l_ << std::endl
                  << "enu = " << enu_ << std::endl
                  << "wrong number of nodes : " << nn << " instead of " << (n_ - l_ - 1);
                TERMINATE(s);
            }
        }
        
    public:
        
        Bound_state(int relativistic__,
                    int zn__,
                    int n__,
                    int l__,
                    Radial_grid const& radial_grid__,
                    std::vector<double> const& v__,
                    double enu_start__)
            : Radial_solution(relativistic__, zn__, l__, radial_grid__),
              n_(n__),
              l_(l__),
              enu_tolerance_(1e-12),
              p_(radial_grid__),
              u_(radial_grid__),
              rdudr_(radial_grid__)
        {
            solve(v__, enu_start__);
        }

        inline double enu() const
        {
            return enu_;
        }

        Spline<double> const& u() const
        {
            return u_;
        }

        Spline<double> const& rdudr() const
        {
            return rdudr_;
        }
};

class Enu_finder: public Radial_solution
{
    private:

        int n_;

        int l_;
        
        double enu_;

        double etop_;
        double ebot_;

        void find_enu(std::vector<double> const& v__, double enu_start__)
        {
            int np = num_points();

            Spline<double> vs(radial_grid());
            for (int i = 0; i < np; i++) vs[i] = v__[i] + zn() / radial_grid(i);
            vs.interpolate();

            Spline<double> mp(radial_grid());
            
            std::vector<double> p(np);
            std::vector<double> q(np);
            std::vector<double> dpdr(np);
            std::vector<double> dqdr(np);
            
            double enu = enu_start__;
            double de = 0.001;
            bool found = false;
            int nndp = 0;

            /* We want to find enu such that the wave-function at the muffin-tin boundary is zero
             * and the number of nodes inside muffin-tin is equal to n-l-1. This will be the top 
             * of the band. */
            for (int i = 0; i < 1000; i++)
            {
                int nnd = integrate_forward_rk4<false>(enu, vs, mp, p, dpdr, q, dqdr) - (n_ - l_ - 1);
                //int nnd = integrate_forward_gbs<false>(enu, vs, mp, p, dpdr, q, dqdr) - (n_ - l_ - 1);

                enu = (nnd > 0) ? enu - de : enu + de;

                if (i > 0)
                {
                    de = (nnd != nndp) ? de * 0.5 : de * 1.25;
                }
                nndp = nnd;
                if (std::abs(de) < 1e-10)
                {
                    found = true;
                    break;
                }
            }
            etop_ = (!found) ? enu_start__ : enu;

            auto ptop = p;

            // TODO: try u'(R) == 0 instead of p'(R) == 0
            
            /* Now we go down in energy and serach for enu such that the wave-function derivative is zero
             * at the muffin-tin boundary. This will be the bottom of the band. */
            de = 0.001;
            found = false;

            double dpdr_R = dpdr[np - 1];
            do
            {
                enu -= de;
                integrate_forward_rk4<false>(enu, vs, mp, p, dpdr, q, dqdr);
                //integrate_forward_gbs<false>(enu, vs, mp, p, dpdr, q, dqdr);
                de *= 1.5;
            } while (dpdr[np - 1] * dpdr_R > 0);

            /* refine bottom energy */
            double e1 = enu;
            double e0 = enu + de;

            while (true)
            {
                enu = (e1 + e0) / 2.0;
                integrate_forward_rk4<false>(enu, vs, mp, p, dpdr, q, dqdr);
                //integrate_forward_gbs<false>(enu, vs, mp, p, dpdr, q, dqdr);
                if (std::abs(dpdr[np - 1]) < 1e-10) break;

                if (dpdr[np - 1] * dpdr_R > 0)
                {
                    e0 = enu;
                }
                else
                {
                    e1 = enu;
                }
            }
        
            ebot_ = enu;
            int nn = integrate_forward_rk4<false>(ebot_, vs, mp, p, dpdr, q, dqdr);
            //int nn = integrate_forward_gbs<false>(ebot_, vs, mp, p, dpdr, q, dqdr);
            if (nn != (n_ - l_ - 1))
            {
                //FILE* fout = fopen("p.dat", "w");
                //for (int ir = 0; ir < np; ir++) 
                //{
                //    double x = radial_grid(ir);
                //    fprintf(fout, "%16.8f %16.8f %16.8f\n", x, ptop[ir], p[ir]);
                //}
                //fclose(fout);

                //printf("n: %i, l: %i, nn: %i", n_, l_, nn);

                TERMINATE("wrong number of nodes");

            }
                
            enu_ = (ebot_ + etop_) / 2.0;
        }

    public:
        
        Enu_finder(int zn__, int n__, int l__, Radial_grid const& radial_grid__, std::vector<double> const& v__, double enu_start__)
            : Radial_solution(0, zn__, l__, radial_grid__),
              n_(n__),
              l_(l__)
        {
            find_enu(v__, enu_start__);
        }

        inline double enu() const
        {
            return enu_;
        }

        inline double ebot() const
        {
            return ebot_;
        }

        inline double etop() const
        {
            return etop_;
        }
};

};

#endif // __RADIAL_SOLVER_H__
