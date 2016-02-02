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

/** \file radial_grid.cpp
 *
 *  \brief Contains remaining implementation of sirius::Radial_grid class.
 */

#include "radial_grid.h"
#include "utils.h"

namespace sirius
{

std::vector<double> Radial_grid::create_radial_grid_points(radial_grid_t grid_type, int num_points, double rmin, double rmax)
{
    std::vector<double> grid_points(num_points);

    switch (grid_type)
    {
        case linear_grid:
        {
            for (int i = 0; i < num_points; i++) grid_points[i] = rmin + (rmax - rmin) * double(i) / (num_points - 1);
            break;
        }
        case exponential_grid:
        {
            double alpha = 6.0;
            double beta = 1e-6 * num_points / (rmax - rmin);
            for (int i = 0; i < num_points; i++)
            {
                double t = double(i) / (num_points - 1);
                double f = (beta * t + std::exp(std::pow(t, alpha)) - 1) / (std::exp(1.0) - 1 + beta);
                grid_points[i] = rmin + (rmax - rmin) * f;
            }
            //for (int i = 0; i < num_points; i++) grid_points[i] = rmin * pow(rmax / rmin, double(i) / (num_points - 1));
            break;
        }
        case pow2_grid:
        {
            for (int i = 0; i < num_points; i++) grid_points[i] = rmin + (rmax - rmin) * pow(double(i) / (num_points - 1), 2);
            break; 
        }
        case pow3_grid:
        {
            //for (int i = 0; i < num_points; i++) grid_points[i] = rmin + (rmax - rmin) * pow(double(i) / (num_points - 1), 3);
            for (int i = 0; i < num_points; i++) grid_points[i] = rmin + std::pow(double(i) / double(num_points - 1), 3.0) * (rmax - rmin);
            break; 
        }
        case scaled_pow_grid:
        {   
            ///* ratio of last and first dx */
            double S = rmax * 1000;
            double alpha = pow(S, 1.0 / (num_points - 2));
            double x = rmin;
            for (int i = 0; i < num_points; i++)
            {
                grid_points[i] = x;
                x += (rmax - rmin) * (alpha - 1) * std::pow(S, double(i) / (num_points - 2)) / (S * alpha - 1);
            }
            break;
            
            //double dx0 = 1e-7;
            //double alpha = -std::log(dx0 / (rmax - rmin)) / std::log(double(num_points - 1));
            //for (int i = 0; i < num_points; i++)
            //{
            //    grid_points[i] = rmin + (rmax - rmin) * std::pow(double(i) / (num_points - 1), alpha);
            //}
            break;
        }
        default:
        {
            TERMINATE_NOT_IMPLEMENTED
        }

        //== case hyperbolic_grid:
        //== {
        //==     double x = origin;
        //==     int i = 1;
        //==     
        //==     while (x <= infinity + tol)
        //==     {
        //==         grid_points.push_back(x);
        //==         double t = double(i++) / double(num_mt_points - 1);
        //==         x = origin + 2.0 * (mt_radius - origin) * t / (t + 1);
        //==     }
        //==     break;
        //== }
        //== case incremental_grid:
        //== {
        //==     double D = mt_radius - origin;
        //==     double S = 1000.0;
        //==     double dx0 = 2 * D / (2 * (num_mt_points - 1) + (num_mt_points - 1) * (S - 1));
        //==     double alpha = (S - 1) * dx0 / (num_mt_points - 2);

        //==     int i = 0;
        //==     double x = origin;
        //==     while (x <= infinity + tol)
        //==     {
        //==         grid_points.push_back(x);
        //==         x = origin + (dx0 + dx0 + i * alpha) * (i + 1) / 2.0;
        //==         i++;
        //==     }
        //==     break;
        //== }
    }
   
    /* trivial check */
    if (std::abs(rmax - grid_points[num_points - 1]) > 1e-10)
    {
        std::stringstream s;
        s << "Wrong radial grid" << std::endl
          << "  num_points      : " << num_points << std::endl
          << "  rmax            : " << Utils::double_to_string(rmax, 12) << std::endl
          << "  last grid point : " << Utils::double_to_string(grid_points[num_points - 1], 12); 
        TERMINATE(s);
    }

    return grid_points;
}

void Radial_grid::create(radial_grid_t grid_type, int num_points, double rmin, double rmax)
{
    assert(rmin >= 0);
    assert(rmax > 0);

    auto grid_points = create_radial_grid_points(grid_type, num_points, rmin, rmax);
    set_radial_points((int)grid_points.size(), &grid_points[0]);

    switch (grid_type)
    {
        case linear_grid:
        {
            grid_type_name_ = "linear";
            break;
        }
        case exponential_grid:
        {
            grid_type_name_ = "exponential";
            break;
        }
        case scaled_pow_grid:
        {
            grid_type_name_ = "scaled_power_grid";
            break;
        }
        case pow2_grid:
        {
            grid_type_name_ = "power2";
            break;
        }
        case pow3_grid:
        {
            grid_type_name_ = "power3";
            break;
        }
        case hyperbolic_grid:
        {
            grid_type_name_ = "hyperbolic";
            break;
        }
        case incremental_grid:
        {
            grid_type_name_ = "incremental";
            break;
        }
    }
}

void Radial_grid::set_radial_points(int num_points__, double const* x__)
{
    assert(num_points__ > 0);
    
    /* set points */
    x_ = mdarray<double, 1>(num_points__);
    memcpy(&x_(0), x__, num_points__ * sizeof(double));

    for (int i = 0; i < num_points__; i++) x_(i) = Utils::round(x_(i), 13);
    
    /* set x^{-1} */
    x_inv_ = mdarray<double, 1>(num_points__);
    for (int i = 0; i < num_points__; i++) x_inv_(i) = (x_(i) == 0) ? 0 : 1.0 / x_(i);
    
    /* set dx */
    dx_ = mdarray<double, 1>(num_points__ - 1);
    for (int i = 0; i < num_points__ - 1; i++) dx_(i) = x_(i + 1) - x_(i);
    
    //== if (dx_(0) < 1e-7)
    //== {
    //==     std::stringstream s;
    //==     s << "dx step near origin is small : " << Utils::double_to_string(dx_(0));
    //==     warning_global(__FILE__, __LINE__, s);
    //== }
}

};
