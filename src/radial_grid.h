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

/** \file radial_grid.h
 *
 *  \brief Contains declaraion and partial implementation of sirius::Radial_grid class.
 */

#ifndef __RADIAL_GRID_H__
#define __RADIAL_GRID_H__

#include "utils.h"

namespace sirius {

/// Types of radial grid.
enum radial_grid_t
{
    linear_grid = 0,

    exponential_grid = 1,

    pow2_grid = 2,

    pow3_grid = 3,

    scaled_pow_grid = 4,

    lin_exp_grid = 5

    //hyperbolic_grid,

    //incremental_grid
};

/// Radial grid for a muffin-tin or an isolated atom.
class Radial_grid // TODO: rewrite: base class + virtual methods
{
    private:
        
        /// Radial grid points.
        mdarray<double, 1> x_;
        
        /// Inverse values of radial grid points.
        mdarray<double, 1> x_inv_;
        
        /// Radial grid points difference.
        /** \f$ dx_{i} = x_{i+1} - x_{i} \f$ */
        mdarray<double, 1> dx_;
        
        /// Name of the grid type.
        std::string grid_type_name_;
        
        /// Create array of radial grid points.
        std::vector<double> create_radial_grid_points(radial_grid_t grid_type, int num_points, double rmin, double rmax)
        {
            std::vector<double> grid_points(num_points);

            switch (grid_type) {
                case linear_grid: {
                    for (int i = 0; i < num_points; i++) {
                        grid_points[i] = rmin + (rmax - rmin) * double(i) / (num_points - 1);
                    }
                    break;
                }
                case exponential_grid: {
                    /* x_i = x_min * (x_max/x_min) ^ (i / (N - 1)) */
                    for (int i = 0; i < num_points; i++) {
                        grid_points[i] = rmin * std::pow(rmax / rmin, double(i) / (num_points - 1));
                    }
                    break;
                }
                case lin_exp_grid: {
                    /* x_i = x_min + (x_max - x_min) * A(t)
                     * A(t) ~ b * t + Exp[t^a] - 1 */
                    double alpha{6.0};
                    double beta = 1e-6 * num_points / (rmax - rmin);
                    double A = 1.0 / (std::exp(1) + beta - 1);

                    for (int i = 0; i < num_points; i++) {
                        double t = double(i) / (num_points - 1);
                        grid_points[i] = rmin + (rmax - rmin) * (beta * t + std::exp(std::pow(t, alpha)) - 1) * A;
                    }
                    break;
                }
                case pow2_grid: {
                    /* x_i = x_min + (x_max - x_min) * (i / N - 1)^2 */
                    for (int i = 0; i < num_points; i++) {
                        grid_points[i] = rmin + (rmax - rmin) * std::pow(double(i) / (num_points - 1), 2);
                    }
                    break; 
                }
                case pow3_grid: {
                    for (int i = 0; i < num_points; i++) {
                        grid_points[i] = rmin + (rmax - rmin) * std::pow(double(i) / (num_points - 1), 3);
                    }
                    break; 
                }
                case scaled_pow_grid: {
                    /* ratio of last and first dx */
                    double S = rmax * 1000;
                    double alpha = std::pow(S, 1.0 / (num_points - 2));
                    double x = rmin;
                    for (int i = 0; i < num_points; i++)
                    {
                        grid_points[i] = x;
                        x += (rmax - rmin) * (alpha - 1) * std::pow(S, double(i) / (num_points - 2)) / (S * alpha - 1);
                    }
                    break;
                }
                default: {
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
            if (std::abs(rmax - grid_points[num_points - 1]) > 1e-10) {
                std::stringstream s;
                s << "Wrong radial grid" << std::endl
                  << "  num_points      : " << num_points << std::endl
                  << "  rmax            : " << Utils::double_to_string(rmax, 12) << std::endl
                  << "  last grid point : " << Utils::double_to_string(grid_points[num_points - 1], 12); 
                TERMINATE(s);
            }

            return grid_points;
        }

        /// Create the predefined grid.
        void create(radial_grid_t grid_type,
                    int num_points,
                    double rmin,
                    double rmax)
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
                case lin_exp_grid:
                {
                    grid_type_name_ = "linear_exponential";
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
                //case hyperbolic_grid:
                //{
                //    grid_type_name_ = "hyperbolic";
                //    break;
                //}
                //case incremental_grid:
                //{
                //    grid_type_name_ = "incremental";
                //    break;
                //}
            }
        }

        Radial_grid(Radial_grid const& src__) = delete;

        Radial_grid& operator=(Radial_grid const& src__) = delete;

    public:

        Radial_grid(Radial_grid&& src__)
        {
            x_ = std::move(src__.x_);
            dx_ = std::move(src__.dx_);
            x_inv_ = std::move(src__.x_inv_);
            grid_type_name_ = src__.grid_type_name_;
        }

        Radial_grid& operator=(Radial_grid&& src__)
        {
            if (this != &src__)
            {
                x_ = std::move(src__.x_);
                dx_ = std::move(src__.dx_);
                x_inv_ = std::move(src__.x_inv_);
                grid_type_name_ = src__.grid_type_name_;
            }
            return *this;
        }
        
        /// Constructor for an empty grid.
        Radial_grid()
        {
        }
        
        /// Constructor for user provided radial grid.
        Radial_grid(std::vector<double>& x__) : grid_type_name_("custom")
        {
            set_radial_points((int)x__.size(), &x__[0]);
        }

        /// Constructor for user provided radial grid.
        Radial_grid(int num_points__, double const* x__) : grid_type_name_("custom")
        {
            set_radial_points(num_points__, x__);
        }

        /// Constructor for a specific radial grid.
        Radial_grid(radial_grid_t grid_type, int num_points, double rmin, double rmax) : grid_type_name_("")
        {
            create(grid_type, num_points, rmin, rmax);
        }
        
        /// Return \f$ x_{i} \f$.
        inline double operator[](const int i) const
        {
            assert(i < (int)x_.size());
            return x_(i);
        }
        
        /// Return \f$ dx_{i} \f$.
        inline double dx(const int i) const
        {
            assert(i < (int)dx_.size());
            return dx_(i);
        }
        
        /// Return \f$ x_{i}^{-1} \f$.
        inline double x_inv(const int i) const
        {
            assert(i < (int)x_inv_.size());
            return x_inv_(i);
        }
       
        /// Number of grid points.
        inline int num_points() const
        {
            return static_cast<int>(x_.size());
        }

        /// First point of the grid.
        inline double first() const
        {
            return x_(0);
        }

        /// Last point of the grid.
        inline double last() const
        {
            return x_(num_points() - 1);
        }
               
        /// Return name of the grid type.
        inline std::string grid_type_name() const
        {
            return grid_type_name_;
        }

        /// Set new radial points.
        void set_radial_points(int num_points__, double const* points__)
        {
            assert(num_points__ > 0);
            
            /* set points */
            x_ = mdarray<double, 1>(num_points__);
            std::memcpy(&x_(0), points__, num_points__ * sizeof(double));

            for (int i = 0; i < num_points__; i++) {
                x_(i) = Utils::round(x_(i), 13);
            }
            
            /* set x^{-1} */
            x_inv_ = mdarray<double, 1>(num_points__);
            for (int i = 0; i < num_points__; i++) {
                x_inv_(i) = (x_(i) == 0) ? 0 : 1.0 / x_(i);
            }
            
            /* set dx */
            dx_ = mdarray<double, 1>(num_points__ - 1);
            for (int i = 0; i < num_points__ - 1; i++) {
                dx_(i) = x_(i + 1) - x_(i);
            }
            
            //== if (dx_(0) < 1e-7)
            //== {
            //==     std::stringstream s;
            //==     s << "dx step near origin is small : " << Utils::double_to_string(dx_(0));
            //==     warning_global(__FILE__, __LINE__, s);
            //== }
        }

        uint64_t hash() const
        {
            uint64_t h = Utils::hash(&x_(0), x_.size() * sizeof(double));
            h += Utils::hash(&dx_(0), dx_.size() * sizeof(double), h);
            h += Utils::hash(&x_inv_(0), x_inv_.size() * sizeof(double), h);
            return h;
        }

        Radial_grid segment(int num_points__) const
        {
            assert(num_points__ >= 0 && num_points__ <= (int)x_.size());
            Radial_grid r;
            r.grid_type_name_ = grid_type_name_ + " (segment)";
            r.x_ = mdarray<double, 1>(num_points__);
            r.dx_ = mdarray<double, 1>(num_points__ - 1);
            r.x_inv_ = mdarray<double, 1>(num_points__);

            memcpy(&r.x_(0), &x_(0), num_points__ * sizeof(double));
            memcpy(&r.dx_(0), &dx_(0), (num_points__ - 1) * sizeof(double));
            memcpy(&r.x_inv_(0), &x_inv_(0), num_points__ * sizeof(double));

            return std::move(r);
        }

        #ifdef __GPU
        void copy_to_device()
        {
            x_.allocate(memory_t::device);
            x_.copy_to_device();
            dx_.allocate(memory_t::device);
            dx_.copy_to_device();
        }
        #endif

        mdarray<double, 1> const& x() const
        {
            return x_;
        }

        mdarray<double, 1> const& dx() const
        {
            return dx_;
        }
};

};

#endif // __RADIAL_GRID_H__
