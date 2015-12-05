// Copyright (c) 2013-2015 Anton Kozhevnikov, Thomas Schulthess
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

/** \file wave_functions.h
 *   
 *  \brief Contains declaration and implementation of sirius::Wave_functions class.
 */

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

        mdarray<double_complex, 2> wf_coeffs_;

        mdarray<double_complex, 2> wf_coeffs_swapped_;

        mdarray<double_complex, 1> send_recv_buf_;
        
        splindex<block> spl_n_;

        int num_gvec_loc_;

        int rank_;
        int rank_row_;
        int num_ranks_col_;

        block_data_descriptor gvec_slab_distr_;

        mdarray<double_complex, 1> inner_prod_buf_;

    public:

        Wave_functions(int num_wfs__, Gvec const& gvec__, MPI_grid const& mpi_grid__)
            : num_wfs_(num_wfs__),
              gvec_(gvec__),
              mpi_grid_(mpi_grid__),
              comm_(mpi_grid_.communicator()),
              rank_(-1),
              rank_row_(-1),
              num_ranks_col_(-1)
        {
            PROFILE();

            /* local number of G-vectors */
            num_gvec_loc_ = gvec_.num_gvec(mpi_grid_.communicator().rank());

            /* primary storage of PW wave functions: slabs */ 
            wf_coeffs_ = mdarray<double_complex, 2>(num_gvec_loc_, num_wfs_, "wf_coeffs_");
        }

        Wave_functions(int num_wfs__, int max_num_wfs_swapped__, Gvec const& gvec__, MPI_grid const& mpi_grid__)
            : num_wfs_(num_wfs__),
              gvec_(gvec__),
              mpi_grid_(mpi_grid__),
              comm_(mpi_grid_.communicator())
        {
            PROFILE();

            /* flat rank id */
            rank_ = comm_.rank();

            /* number of column ranks */
            num_ranks_col_ = mpi_grid_.communicator(1 << 0).size();

            /* row rank */
            rank_row_ = mpi_grid_.communicator(1 << 1).rank();
            
            /* local number of G-vectors */
            num_gvec_loc_ = gvec_.num_gvec(rank_);

            /* primary storage of PW wave functions: slabs */ 
            wf_coeffs_ = mdarray<double_complex, 2>(num_gvec_loc_, num_wfs_, "wf_coeffs_");

            splindex<block> spl_wf(max_num_wfs_swapped__, num_ranks_col_, mpi_grid_.communicator(1 << 0).rank());

            if (comm_.size() > 1)
            {
                wf_coeffs_swapped_ = mdarray<double_complex, 2>(gvec_.num_gvec_fft(), spl_wf.local_size(), "wf_coeffs_swapped_");
                send_recv_buf_ = mdarray<double_complex, 1>(wf_coeffs_swapped_.size(), "send_recv_buf_");
            }

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

            /* this is how n wave-functions will be distributed between panels */
            spl_n_ = splindex<block>(n__, num_ranks_col_, mpi_grid_.communicator(1 << 0).rank());

            /* trivial case */
            if (comm_.size() == 1)
            {
                wf_coeffs_swapped_ = mdarray<double_complex, 2>(&wf_coeffs_(0, idx0__), num_gvec_loc_, n__);
                return;
            }

            Timer t("sirius::Wave_functions::swap_forward", comm_);

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
                comm_.isend(&wf_coeffs_(0, idx0__ + spl_n_.global_offset(icol)),
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
                        std::memcpy(&wf_coeffs_swapped_(gvec_slab_distr_.offsets[j], i),
                                    &send_recv_buf_[gvec_slab_distr_.offsets[j] * n_loc + gvec_slab_distr_.counts[j] * i],
                                    gvec_slab_distr_.counts[j] * sizeof(double_complex));
                    }
                }
            }
            else
            {
                int src_rank = rank_row_ * num_ranks_col_;
                comm_.recv(&wf_coeffs_swapped_(0, 0), gvec_slab_distr_.counts[0] * n_loc, src_rank, 0);
            }
        }

        void swap_backward(int idx0__, int n__)
        {
            PROFILE();

            if (comm_.size() == 1) return;
            
            Timer t("sirius::Wave_functions::swap_backward", comm_);
        
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
                                    &wf_coeffs_swapped_(gvec_slab_distr_.offsets[j], i),
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
                comm_.isend(&wf_coeffs_swapped_(0, 0), gvec_slab_distr_.counts[0] * n_loc, dest_rank, 0);
            }
            
            for (int icol = 0; icol < num_ranks_col_; icol++)
            {
                int src_rank = comm_.cart_rank({icol, rank_ / num_ranks_col_});
                //double t = -omp_get_wtime();
                comm_.recv(&wf_coeffs_(0, idx0__ + spl_n.global_offset(icol)),
                           num_gvec_loc_ * spl_n.local_size(icol),
                           src_rank, rank_ % num_ranks_col_);
                //t += omp_get_wtime();
                //DUMP("recieve from %i, %li bytes, %f GB/s",
                //     src_rank, 
                //     num_gvec_loc_ * spl_n.local_size(icol) * sizeof(double_complex),
                //     num_gvec_loc_ * spl_n.local_size(icol) * sizeof(double_complex) / double(1 << 30) / t);
            }
            //==std::vector<MPI_Status> stat(num_ranks_col_);
            //==MPI_Waitall(num_ranks_col_, &req[0], &stat[0]);
        }

        inline double_complex& operator()(int igloc__, int i__)
        {
            return wf_coeffs_(igloc__, i__);
        }

        inline double_complex* operator[](int i__)
        {
            return &wf_coeffs_swapped_(0, i__);
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

        inline void copy_from(Wave_functions const& src__, int i0__, int n__, int j0__)
        {
            std::memcpy(&wf_coeffs_(0, j0__), &src__.wf_coeffs_(0, i0__), num_gvec_loc_ * n__ * sizeof(double_complex));
        }

        inline void copy_from(Wave_functions const& src__, int i0__, int n__)
        {
            copy_from(src__, i0__, n__, i0__);
        }

        inline void transform_from(Wave_functions& wf__, int nwf__, matrix<double_complex>& mtrx__, int n__)
        {
            assert(num_gvec_loc() == wf__.num_gvec_loc());

            linalg<CPU>::gemm(0, 0, num_gvec_loc(), n__, nwf__, &wf__(0, 0), num_gvec_loc(),
                              &mtrx__(0, 0), mtrx__.ld(), &wf_coeffs_(0, 0), num_gvec_loc());
        }

        inline void inner(int i0__, int m__, Wave_functions& ket__, int j0__, int n__,
                          mdarray<double_complex, 2>& result__, int irow__, int icol__, processing_unit_t pu__ = CPU)
        {
            PROFILE();

            assert(num_gvec_loc() == ket__.num_gvec_loc());

            /* single rank, CPU: store result directly in the output matrix */
            if (comm_.size() == 1 && pu__ == CPU)
            {
                linalg<CPU>::gemm(2, 0, m__, n__, num_gvec_loc(), &wf_coeffs_(0, i0__), num_gvec_loc(),
                                  &ket__(0, j0__), num_gvec_loc(), &result__(irow__, icol__), result__.ld());
            }
            else
            {
                /* reallocate buffer if necessary */
                if (static_cast<size_t>(m__ * n__) > inner_prod_buf_.size())
                {
                    inner_prod_buf_ = mdarray<double_complex, 1>(m__ * n__);
                    #ifdef __GPU
                    if (pu__ == GPU) inner_prod_buf_.allocate_on_device();
                    #endif
                }
                switch (pu__)
                {
                    case CPU:
                    {
                        linalg<CPU>::gemm(2, 0, m__, n__, num_gvec_loc(), &wf_coeffs_(0, i0__), num_gvec_loc(),
                                          &ket__(0, j0__), num_gvec_loc(), &inner_prod_buf_[0], m__);
                        break;
                    }
                    case GPU:
                    {
                        #ifdef __GPU
                        linalg<GPU>::gemm(2, 0, m__, n__, num_gvec_loc(), wf_coeffs_.at<GPU>(0, i0__), num_gvec_loc(),
                                          ket__.wf_coeffs_.at<GPU>(0, j0__), num_gvec_loc(), inner_prod_buf_.at<GPU>(), m__);
                        inner_prod_buf_.copy_to_host(m__ * n__);
                        #else
                        TERMINATE_NO_GPU
                        #endif
                        break;
                    }
                    if (comm_.size() > 1) comm_.allreduce(&inner_prod_buf_[0], m__ * n__);
                }

                for (int i = 0; i < n__; i++)
                    std::memcpy(&result__(irow__, icol__ + i), &inner_prod_buf_[i * m__], m__ * sizeof(double_complex));
            }
        }

        inline Communicator const& comm() const
        {
            return comm_;
        }

        mdarray<double_complex, 2>& coeffs()
        {
            return wf_coeffs_;
        }

        #ifdef __GPU
        void allocate_on_device()
        {
            wf_coeffs_.allocate_on_device();
        }

        void deallocate_on_device()
        {
            wf_coeffs_.deallocate_on_device();
        }

        void copy_to_device(int i0__, int n__)
        {
            acc::copyin(wf_coeffs_.at<GPU>(0, i0__), wf_coeffs_.at<CPU>(0, i0__), n__ * num_gvec_loc());
        }
        #endif
};

};

#endif
