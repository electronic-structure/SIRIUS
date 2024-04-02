/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file blacs_grid.hpp
 *
 *  \brief Contains declaration and implementation of sddk::BLACS_grid class.
 */

#ifndef __BLACS_GRID_HPP__
#define __BLACS_GRID_HPP__

#include <memory>
#include "core/mpi/mpi_grid.hpp"
#include "core/rte/rte.hpp"
#include "linalg_base.hpp"

#ifdef SIRIUS_DLAF
#include "dlaf_c/grid.h"
#endif

namespace sirius {

namespace la {

/// BLACS grid wrapper.
class BLACS_grid
{
  private:
    mpi::Communicator const& comm_;

    std::unique_ptr<mpi::Grid> mpi_grid_;

#ifdef SIRIUS_SCALAPACK
    int blacs_handler_{-1};
#endif

    int blacs_context_{-1};

    std::vector<int> rank_map_;

    /* forbid copy constructor */
    BLACS_grid(BLACS_grid const& src) = delete;
    /* forbid assignment operator */
    BLACS_grid&
    operator=(BLACS_grid const& src) = delete;

  public:
    BLACS_grid(mpi::Communicator const& comm__, int num_ranks_row__, int num_ranks_col__)
        : comm_(comm__)
    {
        mpi_grid_ = std::make_unique<mpi::Grid>(std::vector<int>({num_ranks_row__, num_ranks_col__}), comm_);
        rank_map_.resize(num_ranks_row__ * num_ranks_col__);

#ifdef SIRIUS_SCALAPACK
        /* create handler first */
        blacs_handler_ = linalg_base::create_blacs_handler(mpi_grid_->communicator().native());

        for (int j = 0; j < num_ranks_col__; j++) {
            for (int i = 0; i < num_ranks_row__; i++) {
                rank_map_[i + j * num_ranks_row__] = mpi_grid_->communicator().cart_rank({i, j});
            }
        }

        /* create context */
        blacs_context_ = blacs_handler_;
        linalg_base::gridmap(&blacs_context_, &rank_map_[0], num_ranks_row__, num_ranks_row__, num_ranks_col__);

        /* check the grid */
        int nrow1, ncol1, irow1, icol1;
        linalg_base::gridinfo(blacs_context_, &nrow1, &ncol1, &irow1, &icol1);

        if (rank_row() != irow1 || rank_col() != icol1 || num_ranks_row() != nrow1 || num_ranks_col() != ncol1) {
            std::stringstream s;
            s << "wrong grid" << std::endl
              << "            row | col | nrow | ncol " << std::endl
              << " mpi_grid " << rank_row() << " " << rank_col() << " " << num_ranks_row() << " " << num_ranks_col()
              << std::endl
              << " blacs    " << irow1 << " " << icol1 << " " << nrow1 << " " << ncol1;
            RTE_THROW(s);
        }

#ifdef SIRIUS_DLAF
        dlaf_create_grid_from_blacs(blacs_context_);
#endif

#else
        for (int i = 0; i < static_cast<int>(rank_map_.size()); i++) {
            rank_map_[i] = i;
        }
#ifdef SIRIUS_DLAF
        blacs_context_ = dlaf_create_grid(comm_.native(), num_ranks_row__, num_ranks_col__, 'R');
#endif
#endif
    }

    ~BLACS_grid()
    {
        int mpi_finalized;
        MPI_Finalized(&mpi_finalized);
        if (mpi_finalized == 0) {
#ifdef SIRIUS_SCALAPACK
            linalg_base::gridexit(blacs_context_);
            linalg_base::free_blacs_handler(blacs_handler_);
#endif
        }
#ifdef SIRIUS_DLAF
        dlaf_free_grid(blacs_context_);
#endif
    }

    inline int
    context() const
    {
        return blacs_context_;
    }

    inline auto const&
    comm() const
    {
        return comm_;
    }

    inline auto const&
    comm_row() const
    {
        return mpi_grid_->communicator(1 << 0);
    }

    inline auto const&
    comm_col() const
    {
        return mpi_grid_->communicator(1 << 1);
    }

    inline int
    num_ranks_row() const
    {
        return comm_row().size();
    }

    inline int
    rank_row() const
    {
        return comm_row().rank();
    }

    inline int
    num_ranks_col() const
    {
        return comm_col().size();
    }

    inline int
    rank_col() const
    {
        return comm_col().rank();
    }

    inline int
    cart_rank(int irow__, int icol__) const
    {
        return mpi_grid_->communicator().cart_rank({irow__, icol__});
    }

    auto const&
    mpi_grid() const
    {
        return *mpi_grid_;
    }

    auto const&
    rank_map() const
    {
        return rank_map_;
    }
};

} // namespace la

} // namespace sirius

#endif // __BLACS_GRID_HPP__
