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
    private:

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

        /// Column distribution in auxiliary matrix.
        splindex<block> spl_num_col_;

    public:

        matrix_storage(int num_rows_loc__, int num_cols__, device_t pu__)
            : pu_(pu__),
              num_rows_loc_(num_rows_loc__),
              num_cols_(num_cols__)
        {
            PROFILE();
            /* primary storage of PW wave functions: slabs */ 
            prime_ = mdarray<T, 2>(num_rows_loc_, num_cols_, memory_t::host, "matrix_storage.prime_");
        }

        //void remap_forward(int idx0__, int n__, Gvec_partition const& gvec__, Communicator const& comm_col__);

        //void remap_backward(int idx0__, int n__, Gvec_partition const& gvec__, Communicator const& comm_col__);

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

#endif // __MATRIX_STORAGE_H__

