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

/** \file matrix_storage.h
 *
 *  \brief Contains definition and implementaiton of matrix_storage class.
 */

#ifndef __MATRIX_STORAGE_H__
#define __MATRIX_STORAGE_H__

#include "gvec.h"

namespace sirius {

enum class matrix_storage_t
{
    fft_slab,
    block_cyclic
};

template <typename T, matrix_storage_t kind>
class matrix_storage;

template <typename T>
class matrix_storage<T, matrix_storage_t::fft_slab> 
{
    protected:

        /// Type of processing unit used.
        device_t pu_;
        
        /// Local number of rows.
        int num_rows_loc_;

        /// Total number of columns.
        int num_cols_;

        /// Primary storage of matrix.
        mdarray<T, 2> prime_;

        /// Auxiliary matrix storage.
        /** This distribution is used by the FFT driver */
        mdarray<T, 2> spare_;

        /// Raw buffer for the spare storage.
        mdarray<T, 1> spare_buf_;

        /// Raw send-recieve buffer.
        mdarray<T, 1> send_recv_buf_;

        /// Column distribution in auxiliary matrix.
        splindex<block> spl_num_col_;

    public:

        matrix_storage(int num_rows_loc__,
                       int num_cols__,
                       device_t pu__)
            : pu_(pu__),
              num_rows_loc_(num_rows_loc__),
              num_cols_(num_cols__)
        {
            PROFILE();
            /* primary storage of PW wave functions: slabs */ 
            prime_ = mdarray<T, 2>(num_rows_loc_, num_cols_, memory_t::host, "matrix_storage.prime_");
        }

        void remap_forward(int idx0__,
                           int n__,
                           Gvec_partition const& gvec__,
                           Communicator const& comm_col__)
        {
            PROFILE_WITH_TIMER("sirius::matrix_storage::remap_forward");

            /* this is how n columns of the matrix will be distributed between columns of the MPI grid */
            spl_num_col_ = splindex<block>(n__, comm_col__.size(), comm_col__.rank());

            /* trivial case */
            if (comm_col__.size() == 1) {
                if (pu_ == GPU && prime_.on_device()) {
                    spare_ = mdarray<T, 2>(prime_.template at<CPU>(0, idx0__), prime_.template at<GPU>(0, idx0__), num_rows_loc_, n__);
                } else {
                    spare_ = mdarray<T, 2>(prime_.template at<CPU>(0, idx0__), num_rows_loc_, n__);
                }
                return;
            } else {
                /* maximum local number of matrix columns */
                int max_n_loc = splindex_base<int>::block_size(n__, comm_col__.size());
                /* upper limit for the size of swapped spare matrix */
                size_t sz = gvec__.gvec_count_fft() * max_n_loc;
                /* reallocate buffers if necessary */
                if (spare_buf_.size() < sz) {
                    spare_buf_ = mdarray<T, 1>(sz);
                    send_recv_buf_ = mdarray<T, 1>(sz, memory_t::host, "send_recv_buf_");
                }
                spare_ = mdarray<T, 2>(spare_buf_.template at<CPU>(), gvec__.gvec_count_fft(), max_n_loc);
            }
            
            /* local number of columns */
            int n_loc = spl_num_col_.local_size();
            
            /* send and recieve dimensions */
            block_data_descriptor sd(comm_col__.size()), rd(comm_col__.size());
            for (int j = 0; j < comm_col__.size(); j++) {
                sd.counts[j] = spl_num_col_.local_size(j)                 * gvec__.gvec_fft_slab().counts[comm_col__.rank()];
                rd.counts[j] = spl_num_col_.local_size(comm_col__.rank()) * gvec__.gvec_fft_slab().counts[j];
            }
            sd.calc_offsets();
            rd.calc_offsets();

            comm_col__.alltoall(prime_.template at<CPU>(0, idx0__), sd.counts.data(), sd.offsets.data(),
                                send_recv_buf_.template at<CPU>(), rd.counts.data(), rd.offsets.data());
                              
            /* reorder recieved blocks */
            #pragma omp parallel for
            for (int i = 0; i < n_loc; i++) {
                for (int j = 0; j < comm_col__.size(); j++) {
                    int offset = gvec__.gvec_fft_slab().offsets[j];
                    int count  = gvec__.gvec_fft_slab().counts[j];
                    std::memcpy(&spare_(offset, i), &send_recv_buf_[offset * n_loc + count * i], count * sizeof(T));
                }
            }
        }

        void remap_backward(int idx0__, int n__, Gvec_partition const& gvec__, Communicator const& comm_col__)
        {
            PROFILE_WITH_TIMER("sirius::matrix_storage::remap_backward");

            if (comm_col__.size() == 1) {
                return;
            }

            assert(n__ == spl_num_col_.global_index_size());

            /* this is how n wave-functions are distributed between column ranks */
            splindex<block> spl_n(n__, comm_col__.size(), comm_col__.rank());
            /* local number of columns */
            int n_loc = spl_n.local_size();

            /* reorder sending blocks */
            #pragma omp parallel for
            for (int i = 0; i < n_loc; i++) {
                for (int j = 0; j < comm_col__.size(); j++) {
                    int offset = gvec__.gvec_fft_slab().offsets[j];
                    int count  = gvec__.gvec_fft_slab().counts[j];
                    std::memcpy(&send_recv_buf_[offset * n_loc + count * i], &spare_(offset, i), count * sizeof(T));
                }
            }

            /* send and recieve dimensions */
            block_data_descriptor sd(comm_col__.size()), rd(comm_col__.size());
            for (int j = 0; j < comm_col__.size(); j++) {
                sd.counts[j] = spl_n.local_size(comm_col__.rank()) * gvec__.gvec_fft_slab().counts[j];
                rd.counts[j] = spl_n.local_size(j)                 * gvec__.gvec_fft_slab().counts[comm_col__.rank()];
            }
            sd.calc_offsets();
            rd.calc_offsets();

            comm_col__.alltoall(send_recv_buf_.template at<CPU>(), sd.counts.data(), sd.offsets.data(),
                                prime_.template at<CPU>(0, idx0__), rd.counts.data(), rd.offsets.data());
        }

        inline T& operator()(int irow__, int icol__)
        {
            return prime_(irow__, icol__);
        }

        mdarray<T, 2>& prime()
        {
            return prime_;
        }

        mdarray<T, 2>& spare()
        {
            return spare_;
        }
        
        /// Local number of rows in prime matrix.
        inline int num_rows_loc() const
        {
            return num_rows_loc_;
        }

        inline splindex<block> const& spl_num_col() const
        {
            return spl_num_col_;
        }

        //inline void copy_from(Wave_functions const& src__, int i0__, int n__, int j0__)
        //{
        //    switch (pu_) {
        //        case CPU: {
        //            std::memcpy(&wf_coeffs_(0, j0__), &src__.wf_coeffs_(0, i0__), num_gvec_loc_ * n__ * sizeof(double_complex));
        //            break;
        //        }
        //        case GPU: {
        //            #ifdef __GPU
        //            acc::copy(wf_coeffs_.at<GPU>(0, j0__), src__.wf_coeffs_.at<GPU>(0, i0__), num_gvec_loc_ * n__);
        //            #endif
        //            break;
        //        }
        //    }
        //}

        //inline void copy_from(Wave_functions const& src__, int i0__, int n__)
        //{
        //    copy_from(src__, i0__, n__, i0__);
        //}
        //
        //template <typename T>
        //void transform_from(Wave_functions& wf__, int nwf__, matrix<T>& mtrx__, int n__);

        //template <typename T>
        //void inner(int i0__, int m__, Wave_functions& ket__, int j0__, int n__,
        //           mdarray<T, 2>& result__, int irow__, int icol__, Communicator const& comm);

        //#ifdef __GPU
        //void allocate_on_device()
        //{
        //    wf_coeffs_.allocate(memory_t::device);
        //}

        //void deallocate_on_device()
        //{
        //    wf_coeffs_.deallocate_on_device();
        //}

        //void copy_to_device(int i0__, int n__)
        //{
        //    acc::copyin(wf_coeffs_.at<GPU>(0, i0__), wf_coeffs_.at<CPU>(0, i0__), n__ * num_gvec_loc());
        //}

        //void copy_to_host(int i0__, int n__)
        //{
        //    acc::copyout(wf_coeffs_.at<CPU>(0, i0__), wf_coeffs_.at<GPU>(0, i0__), n__ * num_gvec_loc());
        //}
        //#endif

};

template <typename T>
class matrix_storage<T, matrix_storage_t::block_cyclic> 
{

};

}

#endif // __MATRIX_STORAGE_H__

