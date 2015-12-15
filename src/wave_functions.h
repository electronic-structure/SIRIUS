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

#include "gvec.h"
#include "mpi_grid.h"
#include "linalg.h"

namespace sirius {

template <bool mt_spheres>
class Wave_functions
{
};

template<>
class Wave_functions<false> // TODO: don't allocate buffers in the case of 1 rank
{
    private:
        
        /// Number of wave-functions.
        int num_wfs_;

        Gvec const& gvec_;
        
        /// MPI grid for wave-function storage.
        /** Assume that the 1st dimension is used to distribute wave-functions and 2nd to distribute G-vectors */
        MPI_grid const& mpi_grid_;

        processing_unit_t pu_;

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

        Wave_functions(int num_wfs__, Gvec const& gvec__, MPI_grid const& mpi_grid__, processing_unit_t pu__)
            : num_wfs_(num_wfs__),
              gvec_(gvec__),
              mpi_grid_(mpi_grid__),
              pu_(pu__),
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

        Wave_functions(int num_wfs__, int max_num_wfs_swapped__, Gvec const& gvec__, MPI_grid const& mpi_grid__, processing_unit_t pu__)
            : num_wfs_(num_wfs__),
              gvec_(gvec__),
              mpi_grid_(mpi_grid__),
              pu_(pu__),
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
                wf_coeffs_swapped_ = mdarray<double_complex, 2>(gvec_.num_gvec_fft(), spl_wf.local_size(0), "wf_coeffs_swapped_");
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

        void swap_forward(int idx0__, int n__);

        void swap_backward(int idx0__, int n__);

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
            if (pu_ == CPU)
            {
                std::memcpy(&wf_coeffs_(0, j0__), &src__.wf_coeffs_(0, i0__), num_gvec_loc_ * n__ * sizeof(double_complex));
            }
            #ifdef __GPU
            if (pu_ == GPU)
            {
                acc::copy(wf_coeffs_.at<GPU>(0, j0__), src__.wf_coeffs_.at<GPU>(0, i0__), num_gvec_loc_ * n__);
            }
            #endif
        }

        inline void copy_from(Wave_functions const& src__, int i0__, int n__)
        {
            copy_from(src__, i0__, n__, i0__);
        }

        inline void transform_from(Wave_functions& wf__, int nwf__, matrix<double_complex>& mtrx__, int n__)
        {
            assert(num_gvec_loc() == wf__.num_gvec_loc());

            if (pu_ == CPU)
            {
                linalg<CPU>::gemm(0, 0, num_gvec_loc(), n__, nwf__, &wf__(0, 0), num_gvec_loc(),
                                  &mtrx__(0, 0), mtrx__.ld(), &wf_coeffs_(0, 0), num_gvec_loc());
            }
            #ifdef __GPU
            if (pu_ == GPU)
            {
                linalg<GPU>::gemm(0, 0, num_gvec_loc(), n__, nwf__, wf__.coeffs().at<GPU>(), num_gvec_loc(),
                                  mtrx__.at<GPU>(), mtrx__.ld(), wf_coeffs_.at<GPU>(), num_gvec_loc());
            }
            #endif
        }

        void inner(int i0__, int m__, Wave_functions& ket__, int j0__, int n__,
                   mdarray<double_complex, 2>& result__, int irow__, int icol__);

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

        void copy_to_host(int i0__, int n__)
        {
            acc::copyout(wf_coeffs_.at<CPU>(0, i0__), wf_coeffs_.at<GPU>(0, i0__), n__ * num_gvec_loc());
        }
        #endif
};

template<>
class Wave_functions<true>
{
    private:
        
        int wf_size_;
        int num_wfs_;
        int bs_;
        BLACS_grid const& blacs_grid_;
        BLACS_grid const& blacs_grid_slice_;

        dmatrix<double_complex> wf_coeffs_;

        dmatrix<double_complex> wf_coeffs_swapped_;

        mdarray<double_complex, 1> swp_buf_;

        splindex<block> spl_n_;

    public:

        Wave_functions(int wf_size__, int num_wfs__, int bs__, BLACS_grid const& blacs_grid__, BLACS_grid const& blacs_grid_slice__)
            : wf_size_(wf_size__),
              num_wfs_(num_wfs__),
              bs_(bs__),
              blacs_grid_(blacs_grid__),
              blacs_grid_slice_(blacs_grid_slice__)
        {
            assert(blacs_grid_slice__.num_ranks_row() == 1);

            wf_coeffs_ = dmatrix<double_complex>(wf_size_, num_wfs_, blacs_grid_, bs_, bs_);

            int bs1 = splindex_base<int>::block_size(num_wfs_, blacs_grid_slice_.num_ranks_col());
            swp_buf_ = mdarray<double_complex, 1>(wf_size__ * bs1);
        }

        void set_num_swapped(int n__)
        {
            /* this is how n wave-functions will be distributed between panels */
            spl_n_ = splindex<block>(n__, blacs_grid_slice_.num_ranks_col(), blacs_grid_slice_.rank_col());

            int bs = splindex_base<int>::block_size(n__, blacs_grid_slice_.num_ranks_col());
            wf_coeffs_swapped_ = dmatrix<double_complex>(&swp_buf_[0], wf_size_, n__, blacs_grid_slice_, 1, bs);
        }

        void swap_forward(int idx0__, int n__)
        {
            PROFILE_WITH_TIMER("sirius::Wave_functions::swap_forward");
            set_num_swapped(n__);
            linalg<CPU>::gemr2d(wf_size_, n__, wf_coeffs_, 0, idx0__, wf_coeffs_swapped_, 0, 0, blacs_grid_.context());
        }

        void swap_backward(int idx0__, int n__)
        {
            PROFILE_WITH_TIMER("sirius::Wave_functions::swap_backward");
            linalg<CPU>::gemr2d(wf_size_, n__, wf_coeffs_swapped_, 0, 0, wf_coeffs_, 0, idx0__, blacs_grid_.context());
        }

        inline splindex<block> const& spl_num_swapped() const
        {
            return spl_n_;
        }

        dmatrix<double_complex>& coeffs()
        {
            return wf_coeffs_;
        }

        inline int wf_size() const
        {
            return wf_size_;
        }

        inline double_complex& operator()(int i__, int j__)
        {
            return wf_coeffs_(i__, j__);
        }

        inline double_complex* operator[](int i__)
        {
            return &wf_coeffs_swapped_(0, i__);
        }
};

};

#endif
