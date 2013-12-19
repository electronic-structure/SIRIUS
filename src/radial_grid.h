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
        
        /// string representation of grid type name
        std::string grid_type_name_;
        
        // forbid copy constructor
        Radial_grid(const Radial_grid& src);
        
        /// Create array of radial grid points.
        std::vector<double> create_radial_grid(radial_grid_t grid_type, int num_mt_points, double origin, 
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
                    double S = 10000.0; // scale - ratio of last and first delta_x inside MT
                    double alpha = pow(S, 1.0 / (num_mt_points - 2));
                    int i = 0;
                    double x = origin;
                    while (x <= infinity + tol)
                    {
                        grid_points.push_back(x);
                        x += (mt_radius - origin) * (alpha - 1) * pow(S, double(i) / (num_mt_points - 2)) / (S * alpha - 1);
                        if (x <= mt_radius) i++;
                    }
                    break;
                }
                case pow_grid:
                {
                    double x = origin;
                    int i = 1;
                    
                    while (x <= infinity + tol)
                    {
                        grid_points.push_back(x);
                        double t = double(i++) / double(num_mt_points - 1);
                        //x = origin + (mt_radius - origin) * pow(t, 1.0 + 1.0 / (t + 1));
                        x = origin + (mt_radius - origin) * pow(t, 2);
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
                case incremental_grid:
                {
                    double D = mt_radius - origin;
                    double S = 1000.0;
                    double dx0 = 2 * D / (2 * (num_mt_points - 1) + (num_mt_points - 1) * (S - 1));
                    double alpha = (S - 1) * dx0 / (num_mt_points - 2);

                    int i = 0;
                    double x = origin;
                    while (x <= infinity + tol)
                    {
                        grid_points.push_back(x);
                        x = origin + (dx0 + dx0 + i * alpha) * (i + 1) / 2.0;
                        i++;
                    }
                    break;
                }
            }
           
            // trivial check
            if (mt_radius == infinity && num_mt_points != (int)grid_points.size()) 
            {
                std::stringstream s;
                s << "Wrong radial grid" << std::endl
                  << "  num_mt_points      : " << num_mt_points << std::endl
                  << "  grid_points.size() : " <<  grid_points.size() << std::endl
                  << "  infinity        : " << infinity << std::endl
                  << "  last grid point : " << grid_points[grid_points.size() - 1]; 
                error_local(__FILE__, __LINE__, s);
            }

            return grid_points;
        }

        /// Initialize the grid
        /** Total number of points from origin to effective infinity depends on the grid type */
        void initialize(radial_grid_t grid_type, double origin, double infinity)
        {
            assert(mt_radius_ > 0);
            assert(num_mt_points_ > 0);
            assert(origin > 0);
            assert(infinity > 0 && infinity > origin && infinity >= mt_radius_);

            std::vector<double> grid_points = create_radial_grid(grid_type, num_mt_points_, origin, mt_radius_, infinity);
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
                case linear_exponential_grid:
                {
                    grid_type_name_ = "linear-exponential";
                    break;
                }
                case pow_grid:
                {
                    grid_type_name_ = "power";
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

    public:
        
        /// Constructor for user provided radial grid
        /** The actual grid points must are set with the subsequent call to set_radial_points() */
        Radial_grid(int num_points, int num_mt_points__, double mt_radius__, double* points__) 
            : mt_radius_(mt_radius__), 
              num_mt_points_(num_mt_points__), 
              grid_type_name_("custom")
        {
            set_radial_points(num_points, points__);
        }

        /// Constructor for muffin-tin radial grids
        Radial_grid(radial_grid_t grid_type, int num_mt_points__, double origin, double mt_radius__, double infinity) 
            : mt_radius_(mt_radius__), 
              num_mt_points_(num_mt_points__), 
              grid_type_name_("")
        {
            initialize(grid_type, origin, infinity);
        }
        
        /// Constructor for radial grids of isolated atoms (effective infinity is not neccessary)
        Radial_grid(radial_grid_t grid_type, int num_mt_points__, double origin, double mt_radius__) 
            : mt_radius_(mt_radius__), 
              num_mt_points_(num_mt_points__), 
              grid_type_name_("")
        {
            initialize(grid_type, origin, mt_radius_);
        }
        
        inline double operator [](const int i)
        {
            assert(i < (int)points_.size());
            return points_[i];
        }
        
        inline double dr(const int i)
        {
            assert(i < (int)deltas_.size());
            return deltas_[i];
        }

        inline double rinv(const int i)
        {
            assert(i < (int)points_inv_.size());
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
        inline void set_radial_points(int num_points__, double* points__)
        {
            assert(num_points__ > 0);

            points_.resize(num_points__);
            memcpy(&points_[0], points__, num_points__ * sizeof(real8));
        
            // check if the last MT point is equal to the MT radius
            if (fabs(points_[num_mt_points_ - 1] - mt_radius_) > 1e-10) 
            {   
                std::stringstream s;
                s << "Wrong radial grid" << std::endl 
                  << "  num_points     : " << num_points__ << std::endl
                  << "  num_mt_points  : " << num_mt_points_  << std::endl
                  << "  MT radius      : " << mt_radius_ << std::endl
                  << "  MT point value : " << points_[num_mt_points_ - 1];
                 
                error_local(__FILE__, __LINE__, s);
            }
            
            deltas_.resize(points_.size() - 1);
            for (int i = 0; i < (int)points_.size() - 1; i++) deltas_[i] = points_[i + 1] - points_[i];
            
            points_inv_.resize(points_.size());
            for (int i = 0; i < (int)points_.size(); i++) points_inv_[i] = (points_[i] == 0) ? 0 : 1.0 / points_[i];
        }

        /// Get all radial points.
        /// \todo make it safe and universal
        //== inline void get_radial_points(double* radial_points)
        //== {
        //==     memcpy(radial_points, &points_[0], points_.size() * sizeof(real8));
        //== }

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

        std::string grid_type_name()
        {
            return grid_type_name_;
        }
};

};

#endif // __RADIAL_GRID_H__
