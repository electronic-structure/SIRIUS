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

/** \file wave_functions.hpp
 *
 *  \brief Contains declaration and implementation of wave_functions class.
 */

#ifndef __WAVE_FUNCTIONS_HPP__
#define __WAVE_FUNCTIONS_HPP__

#include <cstdlib>
#include <iostream>
#include "linalg.hpp"

namespace sddk {

#ifdef __GPU
extern "C" void add_square_sum_gpu(double_complex const* wf__,
                                   int num_rows_loc__,
                                   int nwf__,
                                   int reduced__,
                                   int mpi_rank__,
                                   double* result__);

extern "C" void add_checksum_gpu(double_complex* wf__,
                                 int num_rows_loc__,
                                 int nwf__,
                                 double_complex* result__);
#endif

const int sddk_default_block_size = 256;

/// Wave-functions representation.
/** Wave-functions consist of two parts: plane-wave part and mufin-tin part. Both are the matrix_storage objects
 *  with the slab distribution. */
class wave_functions
{
    private:

        device_t pu_;

        /// Communicator which is used to distribute G+k vectors and MT spheres.
        Communicator const& comm_;

        /// G+k vectors of the wave-function.
        Gvec const& gkvec_;

        splindex<block> spl_num_atoms_;

        block_data_descriptor mt_coeffs_distr_;

        std::vector<int> offset_mt_coeffs_;

        /// Total number of muffin-tin coefficients.
        int num_mt_coeffs_{0};

        /// Total number of wave-functions.
        int num_wf_{0};

        /// Plane-wave part of wave-functions.
        std::unique_ptr<matrix_storage<double_complex, matrix_storage_t::slab>> pw_coeffs_{nullptr};

        /// Muffin-tin part of wave-functions.
        std::unique_ptr<matrix_storage<double_complex, matrix_storage_t::slab>> mt_coeffs_{nullptr};

        bool has_mt_{false};

    public:

        /// Constructor for PW wave-functions.
        wave_functions(device_t    pu__,
                       Gvec const& gkvec__,
                       int         num_wf__)
            : pu_(pu__)
            , comm_(gkvec__.comm())
            , gkvec_(gkvec__)
            , num_wf_(num_wf__)
        {
            pw_coeffs_ = std::unique_ptr<matrix_storage<double_complex, matrix_storage_t::slab>>(
                new matrix_storage<double_complex, matrix_storage_t::slab>(gkvec_.count(), num_wf_, gkvec_.comm_ortho_fft()));
        }

        /// Constructor for PW wave-functions.
        wave_functions(double_complex* ptr__,
                       device_t        pu__,
                       Gvec const&     gkvec__,
                       int             num_wf__)
            : pu_(pu__)
            , comm_(gkvec__.comm())
            , gkvec_(gkvec__)
            , num_wf_(num_wf__)
        {
            pw_coeffs_ = std::unique_ptr<matrix_storage<double_complex, matrix_storage_t::slab>>(
                new matrix_storage<double_complex, matrix_storage_t::slab>(ptr__, gkvec_.count(), num_wf_, gkvec_.comm_ortho_fft()));
        }

        /// Constructor for LAPW wave-functions.
        wave_functions(device_t                pu__,
                       Gvec const&             gkvec__,
                       int                     num_atoms__,
                       std::function<int(int)> mt_size__,
                       int                     num_wf__)
            : pu_(pu__)
            , comm_(gkvec__.comm())
            , gkvec_(gkvec__)
            , num_wf_(num_wf__)
            , has_mt_(true)
        {
            pw_coeffs_ = std::unique_ptr<matrix_storage<double_complex, matrix_storage_t::slab>>(
                new matrix_storage<double_complex, matrix_storage_t::slab>(gkvec_.count(), num_wf_, gkvec_.comm_ortho_fft()));

            spl_num_atoms_ = splindex<block>(num_atoms__, comm_.size(), comm_.rank());
            mt_coeffs_distr_ = block_data_descriptor(comm_.size());

            for (int ia = 0; ia < num_atoms__; ia++) {
                int rank = spl_num_atoms_.local_rank(ia);
                if (rank == comm_.rank()) {
                    offset_mt_coeffs_.push_back(mt_coeffs_distr_.counts[rank]);
                }
                mt_coeffs_distr_.counts[rank] += mt_size__(ia);

            }
            mt_coeffs_distr_.calc_offsets();

            num_mt_coeffs_ = mt_coeffs_distr_.offsets.back() + mt_coeffs_distr_.counts.back();

            mt_coeffs_ = std::unique_ptr<matrix_storage<double_complex, matrix_storage_t::slab>>(
                new matrix_storage<double_complex, matrix_storage_t::slab>(mt_coeffs_distr_.counts[comm_.rank()],
                                                                           num_wf_, mpi_comm_null()));
        }

        inline matrix_storage<double_complex, matrix_storage_t::slab>& pw_coeffs()
        {
            assert(pw_coeffs_ != nullptr);
            return *pw_coeffs_;
        }

        inline matrix_storage<double_complex, matrix_storage_t::slab> const& pw_coeffs() const
        {
            assert(pw_coeffs_ != nullptr);
            return *pw_coeffs_;
        }

        inline matrix_storage<double_complex, matrix_storage_t::slab>& mt_coeffs()
        {
            assert(mt_coeffs_ != nullptr);
            return *mt_coeffs_;
        }

        inline matrix_storage<double_complex, matrix_storage_t::slab> const& mt_coeffs() const
        {
            assert(mt_coeffs_ != nullptr);
            return *mt_coeffs_;
        }

        inline bool has_mt() const
        {
            return has_mt_;
        }

        inline int num_wf() const
        {
            return num_wf_;
        }

        inline splindex<block> const& spl_num_atoms() const
        {
            return spl_num_atoms_;
        }

        inline int offset_mt_coeffs(int ialoc__) const
        {
            return offset_mt_coeffs_[ialoc__];
        }

        /// Copy values from another wave-function.
        /** \param [in] src Input wave-function.
         *  \param [in] i0  Starting index of wave-functions in src.
         *  \param [in] n   Number of wave-functions to copy.
         *  \param [in] j0  Starting index of wave-functions in destination.
         *  \param [in] pu  Type of processging unit which copies data. */
        inline void copy_from(wave_functions const& src__,
                              int i0__,
                              int n__,
                              int j0__,
                              device_t pu__)
        {
            switch (pu__) {
                case CPU: {
                    /* copy PW part */
                    std::memcpy(pw_coeffs().prime().at<CPU>(0, j0__),
                                src__.pw_coeffs().prime().at<CPU>(0, i0__),
                                pw_coeffs().num_rows_loc() * n__ * sizeof(double_complex));
                    /* copy MT part */
                    if (has_mt_ && mt_coeffs().num_rows_loc()) {
                        std::memcpy(mt_coeffs().prime().at<CPU>(0, j0__),
                                    src__.mt_coeffs().prime().at<CPU>(0, i0__),
                                    mt_coeffs().num_rows_loc() * n__ * sizeof(double_complex));
                    }
                    break;
                }
                case GPU: {
                    #ifdef __GPU
                    /* copy PW part */
                    acc::copy(pw_coeffs().prime().at<GPU>(0, j0__),
                              src__.pw_coeffs().prime().at<GPU>(0, i0__),
                              pw_coeffs().num_rows_loc() * n__);
                    /* copy MT part */
                    if (has_mt_ && mt_coeffs().num_rows_loc()) {
                        acc::copy(mt_coeffs().prime().at<GPU>(0, j0__),
                                  src__.mt_coeffs().prime().at<GPU>(0, i0__),
                                  mt_coeffs().num_rows_loc() * n__);
                    }
                    #endif
                    break;
                }
            }
        }

        inline void copy_from(wave_functions const& src__, int i0__, int n__, device_t pu__)
        {
            copy_from(src__, i0__, n__, i0__, pu__);
        }

        /// Compute L2 norm of first n wave-functions.
        inline mdarray<double,1> l2norm(device_t pu__, int n__) const
        {
            assert(n__ != 0);

            mdarray<double, 1> norm(n__, memory_t::host, "l2norm");
            norm.zero();

            switch (pu__) {
                case CPU: {
                    #pragma omp parallel for
                    for (int i = 0; i < n__; i++) {
                        for (int ig = 0; ig < pw_coeffs().num_rows_loc(); ig++) {
                            norm[i] += (std::pow(pw_coeffs().prime(ig, i).real(), 2) + std::pow(pw_coeffs().prime(ig, i).imag(), 2));
                        }
                        if (gkvec_.reduced()) {
                            if (comm_.rank() == 0) {
                                norm[i] = 2 * norm[i] - std::pow(pw_coeffs().prime(0, i).real(), 2);
                            } else {
                                norm[i] *= 2;
                            }
                        }
                        if (has_mt_ && mt_coeffs().num_rows_loc()) {
                            for (int j = 0; j < mt_coeffs().num_rows_loc(); j++) {
                                norm[i] += (std::pow(mt_coeffs().prime(j, i).real(), 2) + std::pow(mt_coeffs().prime(j, i).imag(), 2));
                            }
                        }
                    }
                    break;
                }
                case GPU: {
                    #ifdef __GPU
                    norm.allocate(memory_t::device);
                    norm.zero<memory_t::device>();
                    add_square_sum_gpu(pw_coeffs().prime().at<GPU>(), pw_coeffs().num_rows_loc(), n__,
                                       gkvec_.reduced(), comm_.rank(), norm.at<GPU>());
                    if (has_mt_ && mt_coeffs().num_rows_loc()) {
                        add_square_sum_gpu(mt_coeffs().prime().at<GPU>(), mt_coeffs().num_rows_loc(), n__,
                                           0, comm_.rank(), norm.at<GPU>());
                    }
                    norm.copy<memory_t::device, memory_t::host>();
                    #endif
                    break;
                }
            }

            comm_.allreduce(norm.at<CPU>(), n__);
            for (int i = 0; i < n__; i++) {
                norm[i] = std::sqrt(norm[i]);
            }

            return std::move(norm);
        }

        /// Compute L2 norm of first n wave-functions.
        inline mdarray<double,1> l2norm(int n__) const
        {
            return l2norm(pu_, n__);
        }

        Communicator const& comm() const
        {
            return comm_;
        }

        device_t pu() const
        {
            return pu_;
        }

        Gvec const& gkvec() const
        {
            return gkvec_;
        }

        inline double_complex checksum_pw(int i0__, int n__, device_t pu__)
        {
            assert(n__ != 0);
            double_complex cs(0, 0);
            switch (pu__) {
                case CPU: {
                    for (int i = 0; i < n__; i++) {
                        for (int j = 0; j < pw_coeffs().num_rows_loc(); j++) {
                            cs += pw_coeffs().prime(j, i0__ + i);
                        }
                    }
                    break;
                }
                case GPU: {
                    mdarray<double_complex, 1> cs1(n__, memory_t::host | memory_t::device, "checksum");
                    cs1.zero<memory_t::device>();
                    #ifdef __GPU
                    add_checksum_gpu(pw_coeffs().prime().at<GPU>(0, i0__), pw_coeffs().num_rows_loc(), n__, cs1.at<GPU>());
                    cs1.copy_to_host();
                    cs = cs1.checksum();
                    #endif
                    break;
                }
            }
            comm_.allreduce(&cs, 1);
            return cs;
        }

        /// Compute the checksum for n wave-functions starting from i0.
        inline double_complex checksum(int i0__, int n__)
        {
            assert(n__ != 0);
            double_complex cs(0, 0);
            #ifdef __GPU
            if (pu_ == GPU) {
                mdarray<double_complex, 1> cs1(n__, memory_t::host | memory_t::device, "checksum");
                cs1.zero<memory_t::device>();
                add_checksum_gpu(pw_coeffs().prime().at<GPU>(0, i0__), pw_coeffs().num_rows_loc(), n__, cs1.at<GPU>());
                if (has_mt_ && mt_coeffs().num_rows_loc()) {
                    add_checksum_gpu(mt_coeffs().prime().at<GPU>(0, i0__), mt_coeffs().num_rows_loc(), n__, cs1.at<GPU>());
                }
                cs1.copy_to_host();
                cs = cs1.checksum();
            }
            #endif
            if (pu_ == CPU) {
                for (int i = 0; i < n__; i++) {
                    for (int j = 0; j < pw_coeffs().num_rows_loc(); j++) {
                        cs += pw_coeffs().prime(j, i0__ + i);
                    }
                }
                if (has_mt_ && mt_coeffs().num_rows_loc()) {
                    for (int i = 0; i < n__; i++) {
                        for (int j = 0; j < mt_coeffs().num_rows_loc(); j++) {
                            cs += mt_coeffs().prime(j, i0__ + i);
                        }
                    }
                }
            }
            comm_.allreduce(&cs, 1);
            return cs;
        }

        #ifdef __GPU
        void allocate_on_device() {
            pw_coeffs().allocate_on_device();
            if (has_mt_ && mt_coeffs().num_rows_loc()) {
                mt_coeffs().allocate_on_device();
            }
        }

        void deallocate_on_device() {
            pw_coeffs().deallocate_on_device();
            if (has_mt_ && mt_coeffs().num_rows_loc()) {
                mt_coeffs().deallocate_on_device();
            }
        }

        void copy_to_device(int i0__, int n__) {
            pw_coeffs().copy_to_device(i0__, n__);
            if (has_mt_ && mt_coeffs().num_rows_loc()) {
                mt_coeffs().copy_to_device(i0__, n__);
            }
        }

        void copy_to_host(int i0__, int n__) {
            pw_coeffs().copy_to_host(i0__, n__);
            if (has_mt_ && mt_coeffs().num_rows_loc()) {
                mt_coeffs().copy_to_host(i0__, n__);
            }
        }
        #endif

        //template <memory_t mem_type>
        //inline void zero(int i0__, int n__)
        //{
        //    pw_coeffs().prime().zero<mem_type>(i0__, n__);
        //    if (has_mt_) {
        //        mt_coeffs().prime().zero<mem_type>(i0__, n__);
        //    }
        //}
};

/// A set of wave-functions.
/** This class is used to store several idential sets of wave-functions (for example, the spinor components). */
class Wave_functions
{
  private:
    /// Set of wave-functions.
    std::vector<std::unique_ptr<wave_functions>> components_;

    /// G+k vectors of the wave-function.
    Gvec const& gkvec_;

  public:

    /// Constructor for PW wave-functions.
    Wave_functions(device_t    pu__,
                   Gvec const& gkvec__,
                   int         num_wf__,
                   int         num_components__)
        : gkvec_(gkvec__)
    {
        for (int i = 0; i < num_components__; i++) {
            components_.push_back(std::unique_ptr<wave_functions>(new wave_functions(pu__, gkvec__, num_wf__)));
        }
    }

    /// Constructor for PW wave-functions.
    Wave_functions(double_complex* ptr__,
                   device_t        pu__,
                   Gvec const&     gkvec__,
                   int             num_wf__,
                   int             num_components__)
        : gkvec_(gkvec__)
    {
        int offs{0};
        for (int i = 0; i < num_components__; i++) {
            components_.push_back(std::unique_ptr<wave_functions>(new wave_functions(ptr__ + offs, pu__, gkvec__, num_wf__)));
            offs += gkvec__.count() * num_wf__;
        }
    }

    /// Constructor for LAPW wave-functions.
    Wave_functions(device_t                pu__,
                   Gvec const&             gkvec__,
                   int                     num_atoms__,
                   std::function<int(int)> mt_size__,
                   int                     num_wf__,
                   int                     num_components__)
        : gkvec_(gkvec__)
    {
        for (int i = 0; i < num_components__; i++) {
            components_.push_back(std::unique_ptr<wave_functions>(new wave_functions(pu__, gkvec__, num_atoms__, mt_size__, num_wf__)));
        }
    }

    inline int num_components() const
    {
        return static_cast<int>(components_.size());
    }

    /// Return a single component.
    wave_functions& component(int idx__)
    {
        assert(idx__ >= 0 && idx__ < num_components());
        return *components_[idx__];
    }

    wave_functions const& component(int idx__) const
    {
        assert(idx__ >= 0 && idx__ < num_components());
        return *components_[idx__];
    }
    
    /// Return a reference to wave-function component.
    inline wave_functions& operator[](int idx__)
    {
        assert(idx__ >= 0 && idx__ < num_components());
        return *components_[idx__];
    }

    /// Return a const reference to wave-function component.
    inline wave_functions const& operator[](int idx__) const
    {
        assert(idx__ >= 0 && idx__ < num_components());
        return *components_[idx__];
    }

    Gvec const& gkvec() const
    {
        return gkvec_;
    }

    inline mdarray<double,1> l2norm(device_t pu__, int n__)
    {
        auto norm1 = component(0).l2norm(pu__, n__);
        if (num_components() == 2) {
            auto norm2 = component(1).l2norm(pu__, n__);
            for (int i = 0; i < n__; i++) {
                norm1[i] = std::sqrt(norm1[i] * norm1[i] + norm2[i] * norm2[i]);
            }
        }
        return std::move(norm1);
    }

    inline mdarray<double,1> l2norm(int n__)
    {
        return l2norm(component(0).pu(), n__);
    }

    inline void copy_from(Wave_functions const& src__,
                          int i0__,
                          int n__,
                          int j0__,
                          device_t pu__)
    {
        for (int ispn = 0; ispn < num_components(); ispn++) {
            component(ispn).copy_from(src__.component(ispn), i0__, n__, j0__, pu__);
        }
    }
};

#include "wf_inner.hpp"
#include "wf_trans.hpp"
#include "wf_ortho.hpp"

} // namespace sddk

#endif
