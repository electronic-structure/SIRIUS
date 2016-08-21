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
class Wave_functions<false>
{
    private:
        
        /// Type of processing unit used.
        processing_unit_t pu_;
        
        /// Local number of G-vectors.
        int num_gvec_loc_;

        /// Number of wave-functions.
        int num_wfs_;

        /// Primary storage of PW wave functions.
        mdarray<double_complex, 2> wf_coeffs_;

        /// Swapped wave-functions.
        /** This wave-function distribution is used by the FFT driver */
        mdarray<double_complex, 2> wf_coeffs_swapped_;
        
        /// Raw buffer for the swapperd wave-functions.
        mdarray<double_complex, 1> wf_coeffs_swapped_buf_;

        /// Raw send-recieve buffer.
        mdarray<double_complex, 1> send_recv_buf_;
        
        /// Distribution of swapped wave-functions.
        splindex<block> spl_n_;
        
        /// Raw buffer for the inner product.
        mdarray<double, 1> inner_prod_buf_;

    public:

        Wave_functions(int num_gvec_loc__, int num_wfs__, processing_unit_t pu__)
            : pu_(pu__),
              num_gvec_loc_(num_gvec_loc__),
              num_wfs_(num_wfs__)
        {
            PROFILE();
            /* primary storage of PW wave functions: slabs */ 
            wf_coeffs_ = mdarray<double_complex, 2>(num_gvec_loc_, num_wfs_, "wf_coeffs_");
        }

        ~Wave_functions()
        {
        }

        void swap_forward(int idx0__, int n__, Gvec_partition const& gvec__, Communicator const& comm_col__);

        void swap_backward(int idx0__, int n__, Gvec_partition const& gvec__, Communicator const& comm_col__);

        inline double_complex& operator()(int igloc__, int i__)
        {
            return wf_coeffs_(igloc__, i__);
        }

        inline double_complex* operator[](int i__)
        {
            return &wf_coeffs_swapped_(0, i__);
        }

        mdarray<double_complex, 2>& coeffs_swapped()
        {
            return wf_coeffs_swapped_;
        }

        inline int num_gvec_loc() const
        {
            return num_gvec_loc_;
        }

        inline splindex<block> const& spl_num_swapped() const
        {
            return spl_n_;
        }

        inline void copy_from(Wave_functions const& src__, int i0__, int n__, int j0__)
        {
            switch (pu_) {
                case CPU: {
                    std::memcpy(&wf_coeffs_(0, j0__), &src__.wf_coeffs_(0, i0__), num_gvec_loc_ * n__ * sizeof(double_complex));
                    break;
                }
                case GPU: {
                    #ifdef __GPU
                    acc::copy(wf_coeffs_.at<GPU>(0, j0__), src__.wf_coeffs_.at<GPU>(0, i0__), num_gvec_loc_ * n__);
                    #endif
                    break;
                }
            }
        }

        inline void copy_from(Wave_functions const& src__, int i0__, int n__)
        {
            copy_from(src__, i0__, n__, i0__);
        }
        
        template <typename T>
        void transform_from(Wave_functions& wf__, int nwf__, matrix<T>& mtrx__, int n__);

        template <typename T>
        void inner(int i0__, int m__, Wave_functions& ket__, int j0__, int n__,
                   mdarray<T, 2>& result__, int irow__, int icol__, Communicator const& comm);

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
            if (blacs_grid_.comm().size() > 1)
                swp_buf_ = mdarray<double_complex, 1>(wf_size__ * bs1);
        }

        void set_num_swapped(int n__)
        {
            /* this is how n wave-functions will be distributed between panels */
            spl_n_ = splindex<block>(n__, blacs_grid_slice_.num_ranks_col(), blacs_grid_slice_.rank_col());

            int bs = splindex_base<int>::block_size(n__, blacs_grid_slice_.num_ranks_col());
            if (blacs_grid_.comm().size() > 1)
            {
                wf_coeffs_swapped_ = dmatrix<double_complex>(&swp_buf_[0], wf_size_, n__, blacs_grid_slice_, 1, bs);
            }
            else
            {
                wf_coeffs_swapped_ = dmatrix<double_complex>(&wf_coeffs_(0, 0), wf_size_, n__, blacs_grid_slice_, 1, bs);
            }
        }

        void swap_forward(int idx0__, int n__)
        {
            PROFILE_WITH_TIMER("sirius::Wave_functions::swap_forward");
            set_num_swapped(n__);
            if (blacs_grid_.comm().size() > 1)
            {
                #ifdef __SCALAPACK
                linalg<CPU>::gemr2d(wf_size_, n__, wf_coeffs_, 0, idx0__, wf_coeffs_swapped_, 0, 0, blacs_grid_.context());
                #else
                TERMINATE_NO_SCALAPACK
                #endif
            }
        }

        void swap_backward(int idx0__, int n__)
        {
            PROFILE_WITH_TIMER("sirius::Wave_functions::swap_backward");
            if (blacs_grid_.comm().size() > 1)
            {
                #ifdef __SCALAPACK
                linalg<CPU>::gemr2d(wf_size_, n__, wf_coeffs_swapped_, 0, 0, wf_coeffs_, 0, idx0__, blacs_grid_.context());
                #else
                TERMINATE_NO_SCALAPACK
                #endif
            }
        }

        inline splindex<block> const& spl_num_swapped() const
        {
            return spl_n_;
        }

        dmatrix<double_complex>& coeffs()
        {
            return wf_coeffs_;
        }

        dmatrix<double_complex>& coeffs_swapped()
        {
            return wf_coeffs_swapped_;
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
