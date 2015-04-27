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

#include <vector>
#include "radial_grid.h"
#include "spline.h"
#include "constants.h"

namespace sirius {

/// Solves a "classical" or scalar relativistic radial Schroedinger equation
/** Second order differential equation is converted into the system of coupled first-order differential equations, 
 *  which are then solved byt the Runge–Kutta 4th order method.
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
        Radial_grid& radial_grid_;
        
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
        Radial_solver(bool relativistic__, double zn__, Radial_grid& radial_grid__) 
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

class Radial_soultion
{
    private:

        /// True if scalar-relativistic equation is solved.
        bool relativistic_;
        
        /// Positive charge of the nucleus.
        int zn_;

        /// Orbital quantum number.
        int l_;

        /// Radial grid.
        Radial_grid const& radial_grid_;

    protected: 

        ///// Integrate system of two first-order differential equations forward starting from the origin. 
        template <bool check_overflow>
        int integrate_forward(double enu__, 
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
            double x2inv = radial_grid_.x_inv(0);
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

            //p__[0] = std::pow(radial_grid_[0], l_ + 1); // * exp(-zn_ * radial_grid_[0] / (l_ + 1));
            //q__[0] = (0.5 / M2) * p__[0] * (l_ / radial_grid_[0] - zn_ / (l_ + 1));

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
    
    public:
        
        Radial_soultion(int zn__, int l__, Radial_grid const& radial_grid__) 
            : relativistic_(false),
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

class Bound_state: public Radial_soultion
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
                int nn = integrate_forward<true>(enu_, vs, mp, p, dpdr, q, dqdr);
                
                sp = s;
                s = (nn > (n_ - l_ - 1)) ? -1 : 1;
                denu = s * std::abs(denu);
                denu = (s != sp) ? denu * 0.5 : denu * 1.25;
                enu_ += denu;

                //std::cout <<"iter="<<iter<<" nn="<<nn<<" denu="<<denu<<" enu="<<enu_<<std::endl;

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
                error_local(__FILE__, __LINE__, s);
            }
        }
        
    public:
        
        Bound_state(int zn__, int n__, int l__, Radial_grid const& radial_grid__, std::vector<double> const& v__, double enu_start__)
            : Radial_soultion(zn__, l__, radial_grid__),
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

class Enu_finder: public Radial_soultion
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
                int nnd = integrate_forward<false>(enu, vs, mp, p, dpdr, q, dqdr) - (n_ - l_ - 1);

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

            //== if (zn() == 38 && n_ == 3 && l_ == 1)
            //== {
            //==     FILE* fout = fopen("p_top.dat", "w");
            //==     for (int ir = 0; ir < np; ir++) 
            //==     {
            //==         double x = radial_grid(ir);
            //==         fprintf(fout, "%16.8f %16.8f\n", x, p[ir]);
            //==     }
            //==     fclose(fout);
            //== }

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
                integrate_forward<false>(enu, vs, mp, p, dpdr, q, dqdr);
                de *= 1.5;
            } while (dpdr[np - 1] * dpdr_R > 0);

            /* refine bottom energy */
            double e1 = enu;
            double e0 = enu + de;

            while (true)
            {
                enu = (e1 + e0) / 2.0;
                integrate_forward<false>(enu, vs, mp, p, dpdr, q, dqdr);
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
            int nn = integrate_forward<false>(ebot_, vs, mp, p, dpdr, q, dqdr);
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
                
            //double b;
            //for (int i = 0; i < 1000; i++)
            //{
            //    int nn = integrate_forward<false>(enu, vs, mp, p, dpdr, q, dqdr);

            //    //double a = dpdr[np - 1];
            //    double a = dpdr[np - 1] / radial_grid(np - 1) - p[np - 1] / std::pow(radial_grid(np - 1), 2);

            //    if (i > 0)
            //    {
            //        de = (a * b < 0) ? -de * 0.5 : de * 1.25;
            //    }
            //    b = a;
            //    enu += de;
            //    if (std::abs(de) < 1e-10)
            //    {
            //        if (nn != (n_ - l_ - 1))
            //        {
            //            FILE* fout = fopen("p_bot_err.dat", "w");
            //            for (int ir = 0; ir < np; ir++) 
            //            {
            //                double x = radial_grid(ir);
            //                fprintf(fout, "%16.8f %16.8f\n", x, p[ir]);
            //            }
            //            fclose(fout);

            //            printf("n: %i, l: %i, nn: %i", n_, l_, nn);

            //            TERMINATE("wrong number of nodes");

            //        }
            //        found = true;
            //        break;
            //    }
            //}
            //ebot_ = (!found) ? enu_start__ : enu;

            //fout = fopen("p_bot.dat", "w");
            //for (int ir = 0; ir < np; ir++) 
            //{
            //    double x = radial_grid(ir);
            //    fprintf(fout, "%16.8f %16.8f\n", x, p[ir]);
            //}
            //fclose(fout);

            enu_ = (ebot_ + etop_) / 2.0;
        }

    public:
        
        Enu_finder(int zn__, int n__, int l__, Radial_grid const& radial_grid__, std::vector<double> const& v__, double enu_start__)
            : Radial_soultion(zn__, l__, radial_grid__),
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

class Unbound_state: public Radial_soultion
{
};

class Confined_state: public Radial_soultion
{
};

};

#endif // __RADIAL_SOLVER_H__
