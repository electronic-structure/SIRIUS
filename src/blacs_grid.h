#ifndef __BLACS_GRID_H__
#define __BLACS_GRID_H__

#include "mpi_grid.h"
#include "linalg_base.h"

class BLACS_grid
{
    private:

        Communicator const& comm_;

        MPI_grid* mpi_grid_;

        int num_ranks_row_;

        int num_ranks_col_;

        int rank_row_;

        int rank_col_;
        
        int blacs_handler_;

        int blacs_context_;

        /* forbid copy constructor */
        BLACS_grid(BLACS_grid const& src) = delete;
        /* forbid assigment operator */
        BLACS_grid& operator=(BLACS_grid const& src) = delete; 

    public:
        
        BLACS_grid(Communicator const& comm__, int num_ranks_row__, int num_ranks_col__)
            : comm_(comm__),
              num_ranks_row_(num_ranks_row__),
              num_ranks_col_(num_ranks_col__),
              blacs_handler_(-1),
              blacs_context_(-1)
        {
            PROFILE();

            mpi_grid_ = new MPI_grid({num_ranks_row__, num_ranks_col__}, comm_);

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
                  << " mpi_grid " << rank_row_ << " " << rank_col_ << " " << num_ranks_row__ << " " << num_ranks_col__ << std::endl  
                  << " blacs    " << irow1 << " " << icol1 << " " << nrow1 << " " << ncol1;
                TERMINATE(s);
            }
            #endif
        }

        ~BLACS_grid()
        {
            PROFILE();

            #ifdef __SCALAPACK
            linalg_base::gridexit(blacs_context_);
            linalg_base::free_blacs_handler(blacs_handler_);
            #endif
            delete mpi_grid_;
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

        MPI_grid const* mpi_grid() const
        {
            return mpi_grid_;
        }
};

#endif // __BLACS_GRID_H__
