// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

/** \file matrix_storage.hpp
 *
 *  \brief Contains definition and implementaiton of sddk::matrix_storage class.
 */

#ifndef __MATRIX_STORAGE_HPP__
#define __MATRIX_STORAGE_HPP__

#include "gvec.hpp"
#include "dmatrix.hpp"

#ifdef __GPU
extern "C" void add_checksum_gpu(std::complex<double> const* wf__,
                                 int                   num_rows_loc__,
                                 int                   nwf__,
                                 std::complex<double>* result__);
#endif

namespace sddk {

enum class matrix_storage_t
{
    slab,
    block_cyclic
};

/// Class declaration.
template <typename T, matrix_storage_t kind>
class matrix_storage;

/// Specialization of matrix storage class for slab data distribution.
/** \tparam T data type */
template <typename T>
class matrix_storage<T, matrix_storage_t::slab>
{
  private:

    /// G-vector partitioning.
    Gvec_partition const* gvp_{nullptr};

    /// Local number of rows.
    int num_rows_loc_{0};

    /// Total number of columns.
    int num_cols_;

    /// Primary storage of matrix.
    mdarray<T, 2> prime_;

    /// Auxiliary matrix storage.
    /** This distribution is used by the FFT driver */
    mdarray<T, 2> extra_;

    /// Raw buffer for the extra storage.
    mdarray<T, 1> extra_buf_;

    /// Raw send-recieve buffer.
    mdarray<T, 1> send_recv_buf_;

    /// Column distribution in auxiliary matrix.
    splindex<splindex_t::block> spl_num_col_;

  public:
    /// Constructor.
    matrix_storage(Gvec_partition const& gvp__, int num_cols__)
        : gvp_(&gvp__)
        , num_rows_loc_(gvp__.gvec().count())
        , num_cols_(num_cols__)
    {
        PROFILE("sddk::matrix_storage::matrix_storage");
        /* primary storage of PW wave functions: slabs */
        prime_ = mdarray<T, 2>(num_rows_loc_, num_cols_, memory_t::host, "matrix_storage.prime_");
    }

    matrix_storage(int num_rows_loc__, int num_cols__)
        : num_rows_loc_(num_rows_loc__)
        , num_cols_(num_cols__)
    {
        PROFILE("sddk::matrix_storage::matrix_storage");
        /* primary storage of PW wave functions: slabs */
        prime_ = mdarray<T, 2>(num_rows_loc_, num_cols_, memory_t::host, "matrix_storage.prime_");
    }

    /// Constructor.
    /** Memory for prime storage is allocated from the memory pool */
    matrix_storage(memory_pool& mp__, Gvec_partition const& gvp__, int num_cols__)
        : gvp_(&gvp__)
        , num_rows_loc_(gvp__.gvec().count())
        , num_cols_(num_cols__)
    {
        PROFILE("sddk::matrix_storage::matrix_storage");
        /* primary storage of PW wave functions: slabs */
        prime_ = mdarray<T, 2>(num_rows_loc_, num_cols_, mp__, "matrix_storage.prime_");
    }

    /// Check if data needs to be remapped.
    /** Data is not remapped when communicator that is orthogonal to FFT communicator is trivial.
     *  In this case the FFT communicator coincides with the entire communicator used to distribute wave functions
     *  and the data is ready for FFT transformations. */
    inline bool is_remapped() const
    {
        return (gvp_->comm_ortho_fft().size() > 1);
    }

    /// Set the dimensions of the extra matrix storage.
    /** \param [in] n        Number of matrix columns to distribute.
     *  \param [in] idx0     Starting column of the matrix.
     *  \param [in] mp       Memory pool that can be used for fast reallocation.
     *
     *  \image html matrix_storage.png "Redistribution of wave-functions between MPI ranks"
     *
     *  The extra storage is always created in the CPU memory. If the data distribution doesn't
     *  change (no swapping between comm_col ranks is performed) – then the extra storage will mirror
     *  the prime storage (both on CPU and GPU) irrespective of the target processing unit. If
     *  data remapping is necessary extra storage is allocated only in the host memory because MPI is done using the
     *  host pointers.
     */
    void set_num_extra(int n__, int idx0__, memory_pool* mp__);

    /// Remap data from prime to extra storage.
    /** \param [in] n         Number of matrix columns to distribute.
     *  \param [in] idx0      Starting column of the matrix.
     *
     *  Prime storage is expected on the CPU (for the MPI a2a communication). */
    void remap_forward(int n__, int idx0__, memory_pool* mp__);

    /// Remap data from extra to prime storage.
    /** \param [in] n         Number of matrix columns to collect.
     *  \param [in] idx0      Starting column of the matrix.
     *
     *  Extra storage is expected on the CPU (for the MPI a2a communication). If the prime storage is allocated on GPU
     *  remapped data will be copied to GPU. */
    void remap_backward(int n__, int idx0__);

    void remap_from(dmatrix<T> const& mtrx__, int irow0__);

    inline T& prime(int irow__, int jcol__)
    {
        return prime_(irow__, jcol__);
    }

    inline T const& prime(int irow__, int jcol__) const
    {
        return prime_(irow__, jcol__);
    }

    mdarray<T, 2>& prime()
    {
        return prime_;
    }

    mdarray<T, 2> const& prime() const
    {
        return prime_;
    }

    mdarray<T, 2>& extra()
    {
        return extra_;
    }

    mdarray<T, 2> const& extra() const
    {
        return extra_;
    }

    /// Local number of rows in prime matrix.
    inline int num_rows_loc() const
    {
        return num_rows_loc_;
    }

    inline splindex<splindex_t::block> const& spl_num_col() const
    {
        return spl_num_col_;
    }

    /// Allocate prime storage.
    void allocate(memory_pool& mp__)
    {
        prime_.allocate(mp__);
    }

    /// Allocate prime storage.
    void allocate(memory_t mem__)
    {
        prime_.allocate(mem__);
    }

    /// Deallocate storage.
    void deallocate(memory_t mem__)
    {
        prime_.deallocate(mem__);
    }

    void scale(memory_t mem__, int i0__, int n__, double beta__);

    inline void zero(memory_t mem__, int i0__, int n__)
    {
        prime_.zero(mem__, i0__ * num_rows_loc(), n__ * num_rows_loc());
    }

    /// Copy prime storage to device memory.
    void copy_to(memory_t mem__, int i0__, int n__)
    {
        if (num_rows_loc()) {
            prime_.copy_to(mem__, i0__ * num_rows_loc(),  n__ * num_rows_loc());
        }
    }

    double_complex checksum(device_t pu__, int i0__, int n__) const;
};

} // namespace sddk

#endif // __MATRIX_STORAGE_HPP__
