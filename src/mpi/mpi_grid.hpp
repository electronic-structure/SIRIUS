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
 *  \brief Contains declaration and implementation of sddk::MPI_grid class.
 */

#ifndef __MPI_GRID_HPP__
#define __MPI_GRID_HPP__

#include "communicator.hpp"
#include "utils/utils.hpp"

namespace sddk {

// TODO: template over number of dimensions

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

    /// Parent communicator
    Communicator const& parent_communicator_;

    /// Grid communicator of the enrire grid returned by MPI_Cart_create
    Communicator base_grid_communicator_;

    /// Grid communicators
    /** Grid comminicators are built for all possible combinations of directions, i.e. 001, 010, 011, etc.
     *  First communicator is the trivial "self" communicator; the last communicator handles the entire grid. */
    std::vector<Communicator> communicators_;

    /// Return valid directions for the current grid dimensionality.
    inline int valid_directions(int directions__) const
    {
        return (directions__ & ((1 << dimensions_.size()) - 1));
    }

    /// Initialize the grid.
    void initialize()
    {
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
        base_grid_communicator_ = parent_communicator_.cart_create((int)dimensions_.size(), dimensions_.data(),
                                                                   periods.data());

        /* total number of communicators inside the grid */
        int num_comm = 1 << dimensions_.size();

        communicators_ = std::vector<Communicator>(num_comm);

        /* get all possible communicators */
        for (int i = 1; i < num_comm; i++) {
            //bool is_root  = true;
            //int comm_size = 1;
            std::vector<int> flg(dimensions_.size(), 0);

            /* each bit represents a directions */
            for (int j = 0; j < (int)dimensions_.size(); j++) {
                if (i & (1 << j)) {
                    flg[j]  = 1;
                }
            }
            /* subcommunicators */
            communicators_[i] = base_grid_communicator_.cart_sub(flg.data());
        }

        /* expicitly set the "self" communicator */
        communicators_[0] = Communicator(MPI_COMM_SELF);
    }

    /* forbid copy constructor */
    MPI_grid(MPI_grid const& src) = delete;
    /* forbid assigment operator */
    MPI_grid& operator=(MPI_grid const& src) = delete;

  public:
    MPI_grid(std::vector<int> dimensions__, Communicator const& parent_communicator__)
        : dimensions_(dimensions__)
        , parent_communicator_(parent_communicator__)
    {
        initialize();
    }

    /// Actual number of grid dimensions
    inline int num_dimensions() const
    {
        return static_cast<int>(dimensions_.size());
    }

    inline Communicator const& communicator(int directions__ = 0xFF) const
    {
        assert(communicators_.size() != 0);
        return communicators_[valid_directions(directions__)];
    }
};

} // namespace sddk

#endif // __MPI_GRID_HPP__
