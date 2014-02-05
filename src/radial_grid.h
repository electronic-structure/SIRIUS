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

#include <string.h>
#include <vector>
#include <string>
#include "typedefs.h"
#include "error_handling.h"

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
                                               double mt_radius, double infinity);

        /// Initialize the grid
        /** Total number of points from origin to effective infinity depends on the grid type */
        void initialize(radial_grid_t grid_type, double origin, double infinity);

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
               
        /// Get muffin-tin radial points and deltas.
        inline void get_r_dr(double* array, int lda)
        {
            memcpy(&array[0], &points_[0], num_mt_points_ * sizeof(real8));
            memcpy(&array[lda], &deltas_[0], (num_mt_points_ - 1) * sizeof(real8));
        }
       
        inline std::string grid_type_name()
        {
            return grid_type_name_;
        }

        /// Set new radial points.
        void set_radial_points(int num_points__, double* points__);

        /// Print basic info.
        void print_info();
};

};

#endif // __RADIAL_GRID_H__
