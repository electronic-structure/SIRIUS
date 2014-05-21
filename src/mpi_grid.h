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

/** \file mpi_grid.h
 *
 *  \brief Contains declaration and implementation of MPI_grid class.
 */

#ifndef __MPI_GRID_H__
#define __MPI_GRID_H__

#include <mpi.h>
#include "error_handling.h"

/// MPI grid interface
/** The following terminology is used. Suppose we have a 4x5 grid of MPI ranks. We say it's a two-\em dimensional
 *  grid with the first dimension of the size 4 and the second dimensoion of the size 5. The \em actual number of
 *  grid dimensions is two, however we may also consider the grid as being a D-dimensional (D >= 2) with implicit 
 *  dimension sizes equal to one, e.g. 4x5 := 4x5x1x1x1... The communication happens along single or multiple 
 *  \em directions along the grid dimensions. We specify directions wth bits, eg. directions=00000101 reads as 
 *  "communication along 1-st and 3-rd dimensions".
 *  \image html mpi_grid_comm.png "Communication along dimension d0 (between ranks of d0)." 
 *  In the provided example the corresponding communicator is MPI_grid::communicator(1 << d0), where d0 is the integer 
 *   index of dimension. 
 */
class MPI_grid
{
    private:
        
        /// Dimensions of the grid.
        std::vector<int> dimensions_;

        /// Coordinates of the MPI rank in the grid.
        std::vector<int> coordinates_;

        /// parent communicator
        MPI_Comm parent_communicator_;

        /// grid communicator of the enrire grid returned by MPI_Cart_create
        MPI_Comm base_grid_communicator_;

        /// grid communicators
        /** Grid comminicators are built for all possible combinations of 
            directions, i.e. 001, 010, 011, etc. First communicator is the 
            trivial "self" communicator; the last communicator handles the 
            entire grid. */
        std::vector<MPI_Comm> communicators_;

        /// number of MPI ranks in each communicator
        std::vector<int> communicator_size_;

        /// true if this is the root of the communicator group
        std::vector<bool> communicator_root_;

        /// return valid directions for the current grid dimensionality
        inline int valid_directions(int directions)
        {
            return (directions & ((1 << dimensions_.size()) - 1));
        }

        // forbid copy constructor
        MPI_grid(const MPI_grid& src);

        // forbid assignment operator
        MPI_grid& operator=(const MPI_grid& src);

    public:

        // default constructor
        MPI_grid() : parent_communicator_(MPI_COMM_WORLD), base_grid_communicator_(MPI_COMM_NULL) 
        {
        }

        MPI_grid(const std::vector<int> dimensions__, MPI_Comm parent_communicator__) : 
            dimensions_(dimensions__), parent_communicator_(parent_communicator__), 
            base_grid_communicator_(MPI_COMM_NULL)
        {
            initialize();
        }

        ~MPI_grid()
        {
            finalize();
        }

        void initialize(const std::vector<int> dimensions__)
        {
            dimensions_ = dimensions__;
            initialize();
        }
        
        /// Initialize the grid.
        void initialize()
        {
            if (dimensions_.size() == 0) error_local(__FILE__, __LINE__, "no dimensions for the grid");

            int sz = 1;
            for (int i = 0; i < (int)dimensions_.size(); i++) sz *= dimensions_[i];
            
            if (Platform::num_mpi_ranks(parent_communicator_) != sz)
            {
                std::stringstream s;
                s << "Number of MPI ranks doesn't match the size of the grid." << std::endl
                  << "  grid dimensions :";
                for (int i = 0; i < (int)dimensions_.size(); i++) s << " " << dimensions_[i];
                s << std::endl
                  << "  available number of MPI ranks : " << Platform::num_mpi_ranks(parent_communicator_);

                error_local(__FILE__, __LINE__, s);
            }
            
            // communicator of the entire grid
            std::vector<int> periods(dimensions_.size(), 0);
            MPI_Cart_create(parent_communicator_, (int)dimensions_.size(), &dimensions_[0], &periods[0], 0, 
                            &base_grid_communicator_);

            if (in_grid()) 
            {
                // total number of communicators
                int num_comm = 1 << dimensions_.size();

                communicators_ = std::vector<MPI_Comm>(num_comm, MPI_COMM_NULL);

                coordinates_ = std::vector<int>(dimensions_.size(), -1);

                communicator_size_ = std::vector<int>(num_comm, 0);

                communicator_root_ = std::vector<bool>(num_comm, false);

                // get coordinates
                MPI_Cart_get(base_grid_communicator_, (int)dimensions_.size(), &dimensions_[0], &periods[0], 
                             &coordinates_[0]);

                // get all possible communicators
                for (int i = 1; i < num_comm; i++) 
                {
                    bool is_root = true;
                    int comm_size = 1;
                    std::vector<int> flg(dimensions_.size(), 0);

                    // each bit represents a directions
                    for (int j = 0; j < (int)dimensions_.size(); j++) 
                    {
                        if (i & (1<<j)) 
                        {
                            flg[j] = 1;
                            is_root = is_root && (coordinates_[j] == 0);
                            comm_size *= dimensions_[j];
                        }
                    }

                    communicator_root_[i] = is_root;

                    communicator_size_[i] = comm_size;

                    // subcommunicators
                    MPI_Cart_sub(base_grid_communicator_, &flg[0], &communicators_[i]);
                }
                
                // explicitly set the size of "self" communicator
                communicator_size_[0] = 1;
                
                // explicitly set the root of "self" communicator
                communicator_root_[0] = true;

                // expicitly set the "self" communicator
                communicators_[0] = MPI_COMM_SELF;

                // double check the size of communicators
                for (int i = 1; i < num_comm; i++)
                {
                    if (Platform :: num_mpi_ranks(communicators_[i]) != communicator_size_[i]) 
                        error_local(__FILE__, __LINE__, "communicator sizes don't match");
                }

                for (int i = 0; i < (int)dimensions_.size(); i++)
                {
                    if (Platform::mpi_rank(communicator(1 << i)) != coordinate(i))
                        error_local(__FILE__, __LINE__, "ranks don't match");
                }

                if (cart_rank(communicator(), coordinates()) != Platform::mpi_rank(communicator()))
                    error_local(__FILE__, __LINE__, "cartesian and communicator ranks don't match");
            }
        }

        void finalize()
        {
            for (int i = 1; i < (int)communicators_.size(); i++) MPI_Comm_free(&communicators_[i]);

            if (in_grid()) MPI_Comm_free(&base_grid_communicator_);

            communicators_.clear();
            communicator_root_.clear();
            communicator_size_.clear();
            coordinates_.clear();
            dimensions_.clear();
        }

        /// Total number of ranks along specified directions
        inline int size(int directions = 0xFF)
        {
            return communicator_size_[valid_directions(directions)];
        }

        /// true if MPI rank belongs to the grid
        inline bool in_grid()
        {
            return (base_grid_communicator_ != MPI_COMM_NULL);
        }

        /// true if MPI rank is the root of the grid
        inline bool root(int directions = 0xFF)
        {
            return communicator_root_[valid_directions(directions)];
        }
        
        /// Actual coordinates of the calling MPI rank
        std::vector<int> coordinates()
        {
            return coordinates_;
        }

        /// Coordinates along chosen directions
        std::vector<int> coordinates(int directions)
        {
            return sub_coordinates(directions);
        }
        
        /// Coordinate along a given dimension
        inline int coordinate(int idim)
        {
            return (idim < (int)coordinates_.size()) ? coordinates_[idim] : 0;
        }
       
        /// Actual grid dimensions 
        inline std::vector<int> dimensions()
        {
            return dimensions_;
        }
        
        /// Grid dimensions along chosen directions 
        inline std::vector<int> dimensions(int directions)
        {
            return sub_dimensions(directions);
        }

        /// Size of a given dimensions 
        inline int dimension_size(int idim)
        {
            return (idim < (int)dimensions_.size()) ? dimensions_[idim] : 1;
        }

        /// Actual number of grid dimensions
        inline int num_dimensions()
        {
            return (int)dimensions_.size();
        }

        /// Check if MPI ranks are at the side of the grid
        /** Side ranks are those for which coordinates along remaining directions are zero. */
        inline bool side(int directions)
        {
            if (!in_grid()) return false;

            bool flg = true; 

            for (int i = 0; i < (int)dimensions_.size(); i++) 
            {
                if (!(directions & (1 << i)) && coordinates_[i]) flg = false;
            }

            return flg;
        }

        /// Get vector of sub-dimensions
        inline std::vector<int> sub_dimensions(int directions)
        {
            std::vector<int> sd;

            for (int i = 0; i < 8; i++)
            {
                if (directions & (1 << i)) sd.push_back(dimension_size(i));
            }

            return sd;
        }

        /// Get vector of sub-coordinates
        inline std::vector<int> sub_coordinates(int directions)
        {
            std::vector<int> sc;

            for (int i = 0; i < 8; i++)
            {
                if (directions & (1 << i)) sc.push_back(coordinate(i));
            }

            return sc;
        }

        int cart_rank(const MPI_Comm& comm, std::vector<int> coords)
        {
            int r;
            if (comm == MPI_COMM_SELF) return 0;
            
            MPI_Cart_rank(comm, &coords[0], &r);

            return r;
        }

        void barrier(int directions = 0xFF)
        {
           Platform::barrier(communicators_[valid_directions(directions)]);
        }

        inline MPI_Comm& communicator(int directions = 0xFF)
        {
            assert(communicators_.size() != 0);

            return communicators_[valid_directions(directions)];
        }
};

/// A bundle of MPI communicators.
/** Each MPI rank is assigned to one of the MPI subgroups. */
class MPI_comm_bundle
{
    private:
        
        /// MPI subgroup communicator
        MPI_Comm communicator_;
        
        /// Total number of communicators.
        int size_;
        
        /// ID of a communication subgroup.
        int id_;

    public:
        
        MPI_comm_bundle() : communicator_(MPI_COMM_NULL), size_(-1), id_(-1)
        {
        }

        ~MPI_comm_bundle()
        {
            finalize();
        }

        void finalize()
        {
            if (communicator_ != MPI_COMM_NULL) MPI_Comm_free(&communicator_);
            communicator_ = MPI_COMM_NULL;
        }
        
        /// Split parent communicator into bundle of smaller communicators with possibly different sizes. 
        void split(int max_size_, MPI_Comm base_comm = MPI_COMM_WORLD)
        {
            /* number of communicators can't be larger than the number of MPI ranks */
            size_ = std::min(Platform::num_mpi_ranks(base_comm), max_size_);
            int rank = Platform::mpi_rank(base_comm);
            /* assign ID in cyclic fasion */
            id_ = rank % size_;
            /* split parent communicator */
            MPI_Comm_split(base_comm, id_, rank, &communicator_);
        }
        
        /// Return sub-communicator.
        inline MPI_Comm& communicator()
        {
            return communicator_;
        }

        /// Return size of the communicator bundle.
        inline int size()
        {
            return size_;
        }

        /// Return id of a subgroup for a given MPI rank.
        inline int id()
        {
            return id_;
        }
};


#endif // __MPI_GRID_H__
