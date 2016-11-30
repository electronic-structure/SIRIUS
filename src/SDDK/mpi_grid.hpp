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

/** \file mpi_grid.hpp
 *
 *  \brief Contains declaration and implementation of MPI_grid class.
 */

#ifndef __MPI_GRID_HPP__
#define __MPI_GRID_HPP__

#include "communicator.hpp"
//#include "runtime.h"

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

    /// Parent communicator
    Communicator const& parent_communicator_;

    /// Grid communicator of the enrire grid returned by MPI_Cart_create
    Communicator base_grid_communicator_;

    /// grid communicators
    /** Grid comminicators are built for all possible combinations of directions, i.e. 001, 010, 011, etc.
     *  First communicator is the trivial "self" communicator; the last communicator handles the entire grid. */
    std::vector<Communicator> communicators_;

    /// Number of MPI ranks in each communicator.
    std::vector<int> communicator_size_;

    /// True if this is the root of the communicator group.
    std::vector<bool> communicator_root_;

    /// Return valid directions for the current grid dimensionality.
    inline int valid_directions(int directions__) const
    {
        return (directions__ & ((1 << dimensions_.size()) - 1));
    }

    /// Initialize the grid.
    void initialize()
    {
        PROFILE();

        if (dimensions_.size() == 0) {
            TERMINATE("no dimensions provided for the MPI grid");
        }

        int sz{1};
        for (size_t i = 0; i < dimensions_.size(); i++) {
            sz *= dimensions_[i];
        }

        if (parent_communicator_.size() != sz) {
            std::stringstream s;
            s << "Number of MPI ranks doesn't match the size of the grid." << std::endl << "  grid dimensions :";
            for (auto& e : dimensions_) {
                s << " " << e;
            }
            s << std::endl << "  available number of MPI ranks : " << parent_communicator_.size();

            TERMINATE(s);
        }

        /* communicator of the entire grid */
        std::vector<int> periods(dimensions_.size(), 0);
        CALL_MPI(MPI_Cart_create, (parent_communicator_.mpi_comm(), (int)dimensions_.size(), &dimensions_[0],
                                   &periods[0], 0, &base_grid_communicator_.mpi_comm()));

        /* total number of communicators inside the grid */
        int num_comm = 1 << dimensions_.size();

        communicators_ = std::vector<Communicator>(num_comm); //.resize(num_comm);

        coordinates_ = std::vector<int>(dimensions_.size(), -1);

        communicator_size_ = std::vector<int>(num_comm, 0);

        communicator_root_ = std::vector<bool>(num_comm, false);

        /* get coordinates */
        CALL_MPI(MPI_Cart_get, (base_grid_communicator_.mpi_comm(), (int)dimensions_.size(), &dimensions_[0],
                                &periods[0], &coordinates_[0]));

        /* get all possible communicators */
        for (int i = 1; i < num_comm; i++) {
            bool is_root  = true;
            int comm_size = 1;
            std::vector<int> flg(dimensions_.size(), 0);

            /* each bit represents a directions */
            for (int j = 0; j < (int)dimensions_.size(); j++) {
                if (i & (1 << j)) {
                    flg[j]  = 1;
                    is_root = is_root && (coordinates_[j] == 0);
                    comm_size *= dimensions_[j];
                }
            }

            communicator_root_[i] = is_root;

            communicator_size_[i] = comm_size;

            /* subcommunicators */
            CALL_MPI(MPI_Cart_sub, (base_grid_communicator_.mpi_comm(), &flg[0], &communicators_[i].mpi_comm()));
        }

        /* explicitly set the size of "self" communicator */
        communicator_size_[0] = 1;

        /* explicitly set the root of "self" communicator */
        communicator_root_[0] = true;

        /* expicitly set the "self" communicator */
        communicators_[0].mpi_comm() = MPI_COMM_SELF;

        /* double check the size of communicators */
        for (int i = 1; i < num_comm; i++) {
            if (communicators_[i].size() != communicator_size_[i]) {
                TERMINATE("communicator sizes don't match");
            }
        }

        for (int i = 0; i < (int)dimensions_.size(); i++) {
            if (communicator(1 << i).rank() != coordinate(i)) {
                TERMINATE("ranks don't match");
            }
        }

        if (communicator().cart_rank(coordinates_) != parent_communicator_.rank()) {
            TERMINATE("cartesian and communicator ranks don't match");
        }
    }

    void finalize()
    {
        PROFILE();

        communicators_.clear();
        communicator_root_.clear();
        communicator_size_.clear();
        coordinates_.clear();
        dimensions_.clear();
    }

    /* forbid copy constructor */
    MPI_grid(MPI_grid const& src) = delete;
    /* forbid assigment operator */
    MPI_grid& operator=(MPI_grid const& src) = delete;

  public:
    MPI_grid(std::vector<int> dimensions__, Communicator const& parent_communicator__)
        : dimensions_(dimensions__), parent_communicator_(parent_communicator__)
    {
        PROFILE();
        initialize();
    }

    MPI_grid(Communicator const& parent_communicator__)
        : dimensions_({parent_communicator__.size()}), parent_communicator_(parent_communicator__)
    {
        PROFILE();
        initialize();
    }

    ~MPI_grid()
    {
        PROFILE();
        finalize();
    }

    /// Total number of ranks along specified directions
    inline int size(int directions = 0xFF) const
    {
        return communicator_size_[valid_directions(directions)];
    }

    //== /// true if MPI rank is the root of the grid
    //== inline bool root(int directions = 0xFF)
    //== {
    //==     return communicator_root_[valid_directions(directions)];
    //== }

    /// Coordinate along a given dimension
    inline int coordinate(int idim) const
    {
        return (idim < (int)coordinates_.size()) ? coordinates_[idim] : 0;
    }

    /// Size of a given dimensions
    inline int dimension_size(int idim) const
    {
        return (idim < (int)dimensions_.size()) ? dimensions_[idim] : 1;
    }

    /// Actual number of grid dimensions
    inline int num_dimensions() const
    {
        return (int)dimensions_.size();
    }

    inline Communicator const& communicator(int directions__ = 0xFF) const
    {
        assert(communicators_.size() != 0);

        return communicators_[valid_directions(directions__)];
    }
};

/// A bundle of MPI communicators.
/** Each MPI rank is assigned to one of the MPI subgroups. */
class Communicator_bundle
{
  private:
    /// MPI subgroup communicator
    Communicator comm_;

    /// Total number of communicators.
    int size_;

    /// ID of a communication subgroup.
    int id_;

  public:
    Communicator_bundle() : size_(-1), id_(-1)
    {
    }

    Communicator_bundle(Communicator const& base_comm__, int num_elements__)
    {
        /* number of communicators can't be larger than the number of MPI ranks */
        size_    = std::min(base_comm__.size(), num_elements__);
        int rank = base_comm__.rank();
        /* assign ID in cyclic fasion */
        id_ = rank % size_;
        /* split parent communicator */
        CALL_MPI(MPI_Comm_split, (base_comm__.mpi_comm(), id_, rank, &comm_.mpi_comm()));
    }

    /// Return sub-communicator.
    inline Communicator const& comm() const
    {
        return comm_;
    }

    /// Return size of the communicator bundle.
    inline int size() const
    {
        return size_;
    }

    /// Return id of a subgroup for a given MPI rank.
    inline int id() const
    {
        return id_;
    }
};

#endif // __MPI_GRID_HPP__
