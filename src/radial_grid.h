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

#ifndef __RADIAL_GRID_H__
#define __RADIAL_GRID_H__

/** \file radial_grid.h

    \brief Implementation of the muffin-tin radial grids.
*/

namespace sirius {

enum radial_grid_type {linear_grid, exponential_grid, linear_exponential_grid, pow3_grid, hyperbolic_grid};

/// Radial grid for a muffin-tin or an isolated atom.
/** Radial grid is used by the radial solver, spline and other related objects. The grid is constructed in such a way 
    that the point number \f$N_{MT} - 1\f$ is equal to the muffin-tin radius. */
class Radial_grid
{
    private:
        
        /// muffin-tin radius
        double mt_radius_; 
        
        /// number of muffin-tin radial points
        int num_mt_points_;

        /// list of radial points
        std::vector<double> points_;
        
        /// list of 1/r
        std::vector<double> points_inv_;
        
        /// intervals between points
        std::vector<double> deltas_;
        
        // forbid copy constructor
        Radial_grid(const Radial_grid& src);
        
        /// Create array of radial grid points.
        std::vector<double> create_radial_grid(radial_grid_type grid_type, int num_mt_points, double origin, 
                                               double mt_radius, double infinity)
        {
            std::vector<double> grid_points;

            double tol = 1e-10;

            switch (grid_type)
            {
                case linear_grid:
                {
                    double x = origin;
                    double dx = (mt_radius - origin) / (num_mt_points - 1);
                    
                    while (x <= infinity + tol)
                    {
                       grid_points.push_back(x);
                       x += dx;
                    }
                    break;
                }
                case exponential_grid:
                {
                    double x = origin;
                    int i = 1;
                    
                    while (x <= infinity + tol)
                    {
                        grid_points.push_back(x);
                        x = origin * pow((mt_radius / origin), double(i++) / (num_mt_points - 1));
                    }
                    break;
                }
                case linear_exponential_grid:
                {
                    double x = origin;
                    double b = log(mt_radius + 1 - origin);
                    int i = 1;
                    
                    while (x <= infinity + tol)
                    {
                        grid_points.push_back(x);
                        x = origin + exp(b * (i++) / double (num_mt_points - 1)) - 1.0;
                    }
                    break;
                }
                case pow3_grid:
                {
                    double x = origin;
                    int i = 1;
                    
                    while (x <= infinity + tol)
                    {
                        grid_points.push_back(x);
                        double t = double(i++) / double(num_mt_points - 1);
                        x = origin + (mt_radius - origin) * pow(t, 3);
                    }
                    break;
                }
                case hyperbolic_grid:
                {
                    double x = origin;
                    int i = 1;
                    
                    while (x <= infinity + tol)
                    {
                        grid_points.push_back(x);
                        double t = double(i++) / double(num_mt_points - 1);
                        x = origin + 2.0 * (mt_radius - origin) * t / (t + 1);
                    }
                    break;
                }
            }
           
            // trivial check
            if (mt_radius == infinity && num_mt_points != (int)grid_points.size()) 
                error(__FILE__, __LINE__, "Wrong radial grid");

            return grid_points;
        }

        /// Initialize the grid
        /** Total number of points from origin to effective infinity depends on the grid type */
        void initialize(radial_grid_type grid_type, double origin, double infinity)
        {
            assert(mt_radius_ > 0);
            assert(num_mt_points_ > 0);
            assert(origin > 0);
            assert(infinity > 0 && infinity > origin && infinity >= mt_radius_);

            std::vector<double> grid_points = create_radial_grid(grid_type, num_mt_points_, origin, mt_radius_, infinity);
            set_radial_points((int)grid_points.size(), num_mt_points_, &grid_points[0]);
        }

    public:
        
        /// Default constructor
        Radial_grid() : mt_radius_(-1.0), num_mt_points_(0)
        {
        }

        /// Constructor for muffin-tin radial grids
        Radial_grid(radial_grid_type grid_type, int num_mt_points__, double origin, double mt_radius__, double infinity) :
            mt_radius_(mt_radius__), num_mt_points_(num_mt_points__)
        {
            initialize(grid_type, origin, infinity);
        }
        
        /// Constructor for radial grids of isolated atoms (effective infinity is not neccessary)
        Radial_grid(radial_grid_type grid_type, int num_mt_points__, double origin, double mt_radius__) : 
            mt_radius_(mt_radius__), num_mt_points_(num_mt_points__)
        {
            initialize(grid_type, origin, mt_radius_);
        }
        
        inline double operator [](const int i)
        {
            return points_[i];
        }
        
        inline double dr(const int i)
        {
            return deltas_[i];
        }

        inline double rinv(const int i)
        {
            return points_inv_[i];
        }
       
        /// Number of muffin-tin points.
        inline int num_mt_points()
        {
            return num_mt_points_;
        }
        
        /// Total number of radial points.
        inline int size()
        {
            return (int)points_.size();
        }
               
        /// Set new radial points.
        inline void set_radial_points(int num_points__, int num_mt_points__, double* points__)
        {
            num_mt_points_ = num_mt_points__;

            points_.resize(num_points__);
            memcpy(&points_[0], points__, num_points__ * sizeof(real8));
            
            if (fabs(points_[num_mt_points_ - 1] - mt_radius_) > 1e-10) error(__FILE__, __LINE__, "Wrong radial grid");
            
            deltas_.resize(points_.size() - 1);
            for (int i = 0; i < (int)points_.size() - 1; i++) deltas_[i] = points_[i + 1] - points_[i];
            
            points_inv_.resize(points_.size());
            for (int i = 0; i < (int)points_.size(); i++) points_inv_[i] = 1.0 / points_[i];
        }

        /// Get all radial points.
        inline void get_radial_points(double* radial_points)
        {
            memcpy(radial_points, &points_[0], points_.size() * sizeof(real8));
        }

        /// Get muffin-tin radial points and deltas.
        inline void get_r_dr(double* array, int lda)
        {
            memcpy(&array[0], &points_[0], num_mt_points_ * sizeof(real8));
            memcpy(&array[lda], &deltas_[0], (num_mt_points_ - 1) * sizeof(real8));
        }
       
        /// Print basic info.
        void print_info()
        {
            printf("number of muffin-tin points : %i\n", num_mt_points());
            printf("total number of points      : %i\n", size());
            printf("starting point              : %f\n", points_[0]);
            printf("muffin-tin point            : %f\n", points_[num_mt_points() - 1]);
            printf("effective infinity point    : %f\n", points_[size() - 1]);
        }
};

};

#endif // __RADIAL_GRID_H__
