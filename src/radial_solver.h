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

};

#endif // __RADIAL_SOLVER_H__
