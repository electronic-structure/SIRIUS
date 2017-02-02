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

/** \file blacs_grid.hpp
 *
 *  \brief Contains declaration and implementation of BLACS_grid class.
 */

#ifndef __BLACS_GRID_HPP__
#define __BLACS_GRID_HPP__

#include <memory>
#include "mpi_grid.hpp"
#include "linalg_base.hpp"

namespace sddk {

/// BLACS grid wrapper.
class BLACS_grid
{
  private:
    Communicator const& comm_;

    std::unique_ptr<MPI_grid> mpi_grid_;

    int num_ranks_row_{-1};

    int num_ranks_col_{-1};

    int rank_row_{-1};

    int rank_col_{-1};

    int blacs_handler_{-1};

    int blacs_context_{-1};

    /* forbid copy constructor */
    BLACS_grid(BLACS_grid const& src) = delete;
    /* forbid assigment operator */
    BLACS_grid& operator=(BLACS_grid const& src) = delete;

  public:
    BLACS_grid(Communicator const& comm__, int num_ranks_row__, int num_ranks_col__)
        : comm_(comm__)
        , num_ranks_row_(num_ranks_row__)
        , num_ranks_col_(num_ranks_col__)
    {
        PROFILE("sddk::BLACS_grid::BLACS_grid");

        mpi_grid_ = std::unique_ptr<MPI_grid>(new MPI_grid({num_ranks_row__, num_ranks_col__}, comm_));

        rank_row_ = mpi_grid_->coordinate(0);
        rank_col_ = mpi_grid_->coordinate(1);

        #ifdef __SCALAPACK
        /* create handler first */
        blacs_handler_ = linalg_base::create_blacs_handler(mpi_grid_->communicator().mpi_comm());

        std::vector<int> map_ranks(num_ranks_row__ * num_ranks_col__);
        for (int j = 0; j < num_ranks_col__; j++) {
            for (int i = 0; i < num_ranks_row__; i++) {
                map_ranks[i + j * num_ranks_row__] = mpi_grid_->communicator().cart_rank({i, j});
            }
        }

        /* create context */
        blacs_context_ = blacs_handler_;
        linalg_base::gridmap(&blacs_context_, &map_ranks[0], num_ranks_row__, num_ranks_row__, num_ranks_col__);

        /* check the grid */
        int nrow1, ncol1, irow1, icol1;
        linalg_base::gridinfo(blacs_context_, &nrow1, &ncol1, &irow1, &icol1);

        if (rank_row_ != irow1 || rank_col_ != icol1 || num_ranks_row__ != nrow1 || num_ranks_col__ != ncol1) {
            std::stringstream s;
            s << "wrong grid" << std::endl
              << "            row | col | nrow | ncol " << std::endl
              << " mpi_grid " << rank_row_ << " " << rank_col_ << " " << num_ranks_row__ << " " << num_ranks_col__
              << std::endl
              << " blacs    " << irow1 << " " << icol1 << " " << nrow1 << " " << ncol1;
            TERMINATE(s);
        }
        #endif
    }

    ~BLACS_grid()
    {
        PROFILE("sddk::BLACS_grid::~BLACS_grid");

        #ifdef __SCALAPACK
        linalg_base::gridexit(blacs_context_);
        linalg_base::free_blacs_handler(blacs_handler_);
        #endif
    }

    inline int context() const
    {
        return blacs_context_;
    }

    inline Communicator const& comm() const
    {
        return comm_;
    }

    inline Communicator const& comm_row() const
    {
        return mpi_grid_->communicator(1 << 0);
    }

    inline Communicator const& comm_col() const
    {
        return mpi_grid_->communicator(1 << 1);
    }

    inline int num_ranks_row() const
    {
        return num_ranks_row_;
    }

    inline int rank_row() const
    {
        return rank_row_;
    }

    inline int num_ranks_col() const
    {
        return num_ranks_col_;
    }

    inline int rank_col() const
    {
        return rank_col_;
    }

    inline int cart_rank(int irow__, int icol__) const
    {
        return mpi_grid_->communicator().cart_rank({irow__, icol__});
    }

    MPI_grid const& mpi_grid() const
    {
        return *mpi_grid_;
    }
};

} // namespace sddk

#endif // __BLACS_GRID_HPP__
