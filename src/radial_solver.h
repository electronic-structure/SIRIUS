// This file is part of SIRIUS
//
// Copyright (c) 2013 Anton Kozhevnikov, Thomas Schulthess
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

#ifndef __RADIAL_SOLVER_H__
#define __RADIAL_SOLVER_H__

#include <vector>
#include "radial_grid.h"
#include "spline.h"
#include "constants.h"

/** \file radial_solver.h
    
    \brief Implementation of radial solver
*/

namespace sirius {

/// Solves a "classical" or scalar relativistic radial Schroedinger equation
/** Second order differential equation is converted into the system of coupled first-order differential equations, 
    which are then solved byt the Runge–Kutta 4th order method.

    \f{eqnarray*}{
       P' &=& 2 M Q + \frac{P}{r} \\
       Q' &=& (V - E + \frac{\ell(\ell + 1)}{2 M r^2}) P - \frac{Q}{r}
    \f}
    
    \todo Correct relativistic DFT 
*/
class Radial_solver
{
    private:

        /// true if scalar-relativistic equation is solved 
        bool relativistic_;
        
        /// negative charge of the nucleus
        double zn_;
        
        /// radial grid
        sirius::Radial_grid& radial_grid_;
        
        double enu_tolerance_;
        
        int integrate(int nr, int l, double enu, sirius::Spline<double>& ve, sirius::Spline<double>& mp, 
                      std::vector<double>& p, std::vector<double>& dpdr, 
                      std::vector<double>& q, std::vector<double>& dqdr);

    public:

        /// Constructor
        Radial_solver(bool relativistic__, double zn__, sirius::Radial_grid& radial_grid__) 
            : relativistic_(relativistic__), 
              zn_(zn__), 
              radial_grid_(radial_grid__),
              enu_tolerance_(1e-10)
        {
        }
        
        inline void set_tolerance(double tolerance__)
        {
            enu_tolerance_ = tolerance__;
        }

        double find_enu(int n, int l, std::vector<double>& v, double enu0);
                        
        int solve_in_mt(int l, double enu, int m, std::vector<double>& v, std::vector<double>& p, 
                        std::vector<double>& hp, double& dpdr_R);
        
        int solve_in_mt(int l, double enu, int m, std::vector<double>& v, std::vector<double>& p0, 
                        std::vector<double>& p1, std::vector<double>& q0, std::vector<double>& q1);

        void bound_state(int n, int l, std::vector<double>& v, double& enu, std::vector<double>& p);
};

};

#endif // __RADIAL_SOLVER_H__
