#ifndef __WAVE_FUNCTIONS_H__
#define __WAVE_FUNCTIONS_H__

namespace sirius {

class Wave_functions // TODO: don't allocate buffers in the case of 1 rank
{
    private:
        
        /// Number of wave-functions.
        int num_wfs_;

        Gvec const& gvec_;
        
        /// MPI grid for wave-function storage.
        /** Assume that the 1st dimension is used to distribute wave-functions and 2nd to distribute G-vectors */
        MPI_grid const& mpi_grid_;

        /// Entire communicator.
        Communicator const& comm_;

        mdarray<double_complex, 1> primary_data_storage_;
        int primary_ld_;

        mdarray<double_complex, 1> swapped_data_storage_;
        int swapped_ld_;

        matrix<double_complex> primary_data_storage_as_matrix_;

        matrix<double_complex> swapped_data_storage_as_matrix_;

        mdarray<double_complex, 1> send_recv_buf_;
        
        splindex<block> spl_num_wfs_;

        splindex<block> spl_n_;

        int num_gvec_loc_;

        int rank_;
        int rank_row_;
        int num_ranks_col_;

        block_data_descriptor gvec_slab_distr_;

    public:

        Wave_functions(int num_wfs__, Gvec const& gvec__, MPI_grid const& mpi_grid__, bool swappable__)
            : num_wfs_(num_wfs__),
              gvec_(gvec__),
              mpi_grid_(mpi_grid__),
              comm_(mpi_grid_.communicator())
        {
            PROFILE();

            /* number of column ranks */
            num_ranks_col_ = mpi_grid_.communicator(1 << 0).size();

            spl_num_wfs_ = splindex<block>(num_wfs_, num_ranks_col_, mpi_grid_.communicator(1 << 0).rank());

            num_gvec_loc_ = gvec_.num_gvec(mpi_grid_.communicator().rank());

            /* primary storage of PW wave functions: slabs */ 
            primary_data_storage_ = mdarray<double_complex, 1>(num_gvec_loc_ * num_wfs_);
            primary_ld_ = num_gvec_loc_;
            swapped_ld_ = gvec_.num_gvec_fft();

            primary_data_storage_as_matrix_ = mdarray<double_complex, 2>(&primary_data_storage_[0], primary_ld_, num_wfs_);

            if (swappable__) // TODO: optimize memory allocation size
            {
                swapped_data_storage_ = mdarray<double_complex, 1>(gvec_.num_gvec_fft() * spl_num_wfs_.local_size());
                swapped_data_storage_as_matrix_ = mdarray<double_complex, 2>(&swapped_data_storage_[0], swapped_ld_, spl_num_wfs_.local_size());
                //int buf_size = std::max(gvec_.num_gvec_fft() * spl_num_wfs_.local_size(),
                //                        num_gvec_loc_ * num_wfs_);
                int buf_size = gvec_.num_gvec_fft() * spl_num_wfs_.local_size(); 
                send_recv_buf_ = mdarray<double_complex, 1>(buf_size);
            }
            
            /* flat rank id */
            rank_ = comm_.rank();
            /* row rank */
            rank_row_ = mpi_grid_.communicator(1 << 1).rank();

            /* store the number of G-vectors to be received by this rank */
            gvec_slab_distr_ = block_data_descriptor(num_ranks_col_);
            for (int i = 0; i < num_ranks_col_; i++)
            {
                gvec_slab_distr_.counts[i] = gvec_.num_gvec(rank_row_ * num_ranks_col_ + i);
            }
            gvec_slab_distr_.calc_offsets();

            assert(gvec_slab_distr_.offsets[num_ranks_col_ - 1] + gvec_slab_distr_.counts[num_ranks_col_ - 1] == gvec__.num_gvec_fft());
        }

        ~Wave_functions()
        {
        }

        void swap_forward(int idx0__, int n__)
        {
            PROFILE();

            Timer t("sirius::Wave_functions::swap_forward");

            /* this is how n wave-functions will be distributed between panels */
            spl_n_ = splindex<block>(n__, num_ranks_col_, mpi_grid_.communicator(1 << 0).rank());
            /* local number of columns */
            int n_loc = spl_n_.local_size();

            /* send parts of slab
             * +---+---+--+
             * |   |   |  |  <- irow = 0
             * +---+---+--+
             * |   |   |  |
             * ............
             * ranks in flat and 2D grid are related as: rank = irow * ncol + icol */
            for (int icol = 0; icol < num_ranks_col_; icol++)
            {
                int dest_rank = comm_.cart_rank({icol, rank_ / num_ranks_col_});
                comm_.isend(&primary_data_storage_[primary_ld_ * (idx0__ + spl_n_.global_offset(icol))],
                            num_gvec_loc_ * spl_n_.local_size(icol),
                            dest_rank, rank_ % num_ranks_col_);
            }
            
            /* receive parts of panel
             *                 n_loc
             *                 +---+  
             *                 |   |
             * gvec_slab_distr +---+
             *                 |   | 
             *                 +---+ */
            if (num_ranks_col_ > 1)
            {
                for (int i = 0; i < num_ranks_col_; i++)
                {
                    int src_rank = rank_row_ * num_ranks_col_ + i;
                    comm_.recv(&send_recv_buf_[gvec_slab_distr_.offsets[i] * n_loc], gvec_slab_distr_.counts[i] * n_loc, src_rank, i);
                }
                
                /* reorder received blocks to make G-vector index continuous */
                #pragma omp parallel for
                for (int i = 0; i < n_loc; i++)
                {
                    for (int j = 0; j < num_ranks_col_; j++)
                    {
                        std::memcpy(&swapped_data_storage_[gvec_slab_distr_.offsets[j] + swapped_ld_ * i],
                                    &send_recv_buf_[gvec_slab_distr_.offsets[j] * n_loc + gvec_slab_distr_.counts[j] * i],
                                    gvec_slab_distr_.counts[j] * sizeof(double_complex));
                    }
                }
            }
            else
            {
                int src_rank = rank_row_ * num_ranks_col_;
                comm_.recv(&swapped_data_storage_[0], gvec_slab_distr_.counts[0] * n_loc, src_rank, 0);
            }
        }

        void swap_backward(int idx0__, int n__)
        {
            PROFILE();

            Timer t("sirius::Wave_functions::swap_backward");
        
            /* this is how n wave-functions are distributed between panels */
            splindex<block> spl_n(n__, num_ranks_col_, mpi_grid_.communicator(1 << 0).rank());
            /* local number of columns */
            int n_loc = spl_n.local_size();

            //==std::vector<MPI_Request> req(num_ranks_col_);
            //==/* post a non-blocking recieve request */
            //==for (int icol = 0; icol < num_ranks_col_; icol++)
            //=={
            //==    int src_rank = comm_.cart_rank({icol, rank_ / num_ranks_col_});
            //==    comm_.irecv(&primary_data_storage_[primary_ld_ * (idx0__ + spl_n.global_offset(icol))],
            //==                num_gvec_loc_ * spl_n.local_size(icol),
            //==                src_rank, rank_ % num_ranks_col_, &req[icol]);
            //==}
            
            if (num_ranks_col_ > 1)
            {
                /* reorder sending blocks */
                #pragma omp parallel for
                for (int i = 0; i < n_loc; i++)
                {
                    for (int j = 0; j < num_ranks_col_; j++)
                    {
                        std::memcpy(&send_recv_buf_[gvec_slab_distr_.offsets[j] * n_loc + gvec_slab_distr_.counts[j] * i],
                                    &swapped_data_storage_[gvec_slab_distr_.offsets[j] + swapped_ld_ * i],
                                    gvec_slab_distr_.counts[j] * sizeof(double_complex));
                    }
                }
        
                for (int i = 0; i < num_ranks_col_; i++)
                {
                    int dest_rank = rank_row_ * num_ranks_col_ + i;
                    comm_.isend(&send_recv_buf_[gvec_slab_distr_.offsets[i] * n_loc], gvec_slab_distr_.counts[i] * n_loc, dest_rank, i);
                }
            }
            else
            {
                int dest_rank = rank_row_ * num_ranks_col_;
                comm_.isend(&swapped_data_storage_[0], gvec_slab_distr_.counts[0] * n_loc, dest_rank, 0);
            }
            
            for (int icol = 0; icol < num_ranks_col_; icol++)
            {
                int src_rank = comm_.cart_rank({icol, rank_ / num_ranks_col_});
                comm_.recv(&primary_data_storage_[primary_ld_ * (idx0__ + spl_n.global_offset(icol))],
                           num_gvec_loc_ * spl_n.local_size(icol),
                           src_rank, rank_ % num_ranks_col_);
            }
            //==std::vector<MPI_Status> stat(num_ranks_col_);
            //==MPI_Waitall(num_ranks_col_, &req[0], &stat[0]);
        }

        inline double_complex& operator()(int igloc__, int i__)
        {
            assert(igloc__ + primary_ld_ * i__ < (int)primary_data_storage_.size()); 
            return primary_data_storage_[igloc__ + primary_ld_ * i__];
        }

        inline double_complex* operator[](int i__)
        {
            assert(swapped_ld_ * i__ < (int)swapped_data_storage_.size());
            return &swapped_data_storage_[swapped_ld_ * i__];
        }

        inline int num_gvec_loc() const
        {
            return num_gvec_loc_;
        }

        inline Gvec const& gvec() const
        {
            return gvec_;
        }

        inline splindex<block> const& spl_num_swapped() const
        {
            return spl_n_;
        }

        matrix<double_complex>& primary_data_storage_as_matrix()
        {
            return primary_data_storage_as_matrix_;
        }

        inline void copy_from(Wave_functions const& src__, int i0__, int n__, int j0__)
        {
            std::memcpy(&primary_data_storage_[primary_ld_ * j0__],
                        &src__.primary_data_storage_[primary_ld_ * i0__],
                        primary_ld_ * n__ * sizeof(double_complex));
        }

        inline void copy_from(Wave_functions const& src__, int i0__, int n__)
        {
            copy_from(src__, i0__, n__, i0__);
        }

        inline void transform_from(Wave_functions& wf__, int nwf__, matrix<double_complex>& mtrx__, int n__)
        {
            assert(num_gvec_loc() == wf__.num_gvec_loc());

            linalg<CPU>::gemm(0, 0, num_gvec_loc(), n__, nwf__, &wf__.primary_data_storage_[0], wf__.primary_ld_,
                              &mtrx__(0, 0), mtrx__.ld(), &primary_data_storage_[0], primary_ld_);
        }

        inline void inner(int i0__, int m__, Wave_functions& ket__, int j0__, int n__,
                          double_complex* result__, int ld__)
        {
            assert(num_gvec_loc() == ket__.num_gvec_loc());
            static std::vector<double_complex> buf;

            if (comm_.size() == 1)
            {
                linalg<CPU>::gemm(2, 0, m__, n__, num_gvec_loc(), &(*this)(0, i0__), num_gvec_loc(),
                                  &ket__(0, j0__), num_gvec_loc(), result__, ld__);
            }
            else
            {
                buf.resize(m__ * n__);
                linalg<CPU>::gemm(2, 0, m__, n__, num_gvec_loc(), &(*this)(0, i0__), num_gvec_loc(),
                                  &ket__(0, j0__), num_gvec_loc(), &buf[0], m__);
                comm_.allreduce(&buf[0], m__ * n__);
                for (int i = 0; i < n__; i++)
                {
                    std::memcpy(&result__[i * ld__], &buf[i * m__], m__ * sizeof(double_complex));
                }
            }
        }

        inline Communicator const& comm() const
        {
            return comm_;
        }
};

};

#endif
