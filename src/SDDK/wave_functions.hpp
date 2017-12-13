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
#include "eigenproblem.h"
#include "hdf5_tree.hpp"

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

namespace sddk {

/// Wave-functions representation.
/** Wave-functions consist of two parts: plane-wave part and mufin-tin part. Both are the matrix_storage objects
 *  with the slab distribution. Wave-functions have one or two spin components. In case of collinear magnetism
 *  each component represents a pure (up- or dn-) spinor state and they are independent. In non-collinear case 
 *  the two components represent a full spinor state. 
 *
 *  In case of collinear magnetism we can work with auxiliary scalar wave-functions and update up- or dn- components
 *  of pure spinor wave-functions independently. We can also apply uu or dd block of Hamiltonian. In this case it is 
 *  reasonable to implement the following convention: for scalar wave-function (num_sc = 1) it's value is returned 
 *  for any spin index (ispn = 0 or ispn = 1).
 */
class Wave_functions
{
  private:

    /// Communicator used to distribute G+k vectors and atoms.
    Communicator const& comm_;

    /// G+k vectors of the wave-function.
    Gvec const& gkvec_;

    splindex<block> spl_num_atoms_;

    /// Distribution of muffin-tin coefficients between ranks.
    block_data_descriptor mt_coeffs_distr_;

    std::vector<int> offset_mt_coeffs_;

    /// Total number of muffin-tin coefficients.
    int num_mt_coeffs_{0};

    /// Total number of wave-functions.
    int num_wf_{0};

    /// Number of spin components (1 or 2).
    int num_sc_{1};

    /// Plane-wave part of wave-functions.
    std::array<std::unique_ptr<matrix_storage<double_complex, matrix_storage_t::slab>>, 2> pw_coeffs_{{nullptr, nullptr}};

    /// Muffin-tin part of wave-functions.
    std::array<std::unique_ptr<matrix_storage<double_complex, matrix_storage_t::slab>>, 2> mt_coeffs_{{nullptr, nullptr}};

    bool has_mt_{false};
    
    /// Lower boundary for the spin component index by spin index.
    inline int s0(int ispn__) const
    {
        assert(ispn__ == 0 || ispn__ == 1 || ispn__ == 2);

        if (ispn__ == 2) {
            return 0;
        } else {
            return ispn__;
        }
    }

    /// Upper boundary for the spin component index by spin index.
    inline int s1(int ispn__) const
    {
        assert(ispn__ == 0 || ispn__ == 1 || ispn__ == 2);

        if (ispn__ == 2) {
            return (num_sc_ == 1) ? 0 : 1;
        } else {
            return ispn__;
        }
    }

    /// Spin-component index by spin index.
    inline int isc(int ispn__) const
    {
        assert(ispn__ == 0 || ispn__ == 1);
        return (num_sc_ == 1) ? 0 : ispn__;
    }

    inline mdarray<double, 1> sumsqr(device_t pu__, int ispn__, int n__) const
    {
        mdarray<double, 1> s(n__, memory_t::host, "sumsqr");
        s.zero();
        if (pu__ == GPU) {
            s.allocate(memory_t::device);
            s.zero<memory_t::device>();
        }

        for (int is = s0(ispn__); is <= s1(ispn__); is++) {
            switch (pu__) {
                case CPU: {
                    #pragma omp parallel for
                    for (int i = 0; i < n__; i++) {
                        for (int ig = 0; ig < pw_coeffs(is).num_rows_loc(); ig++) {
                            s[i] += (std::pow(pw_coeffs(is).prime(ig, i).real(), 2) + std::pow(pw_coeffs(is).prime(ig, i).imag(), 2));
                        }
                        if (gkvec_.reduced()) {
                            if (comm_.rank() == 0) {
                                s[i] = 2 * s[i] - std::pow(pw_coeffs(is).prime(0, i).real(), 2);
                            } else {
                                s[i] *= 2;
                            }
                        }
                        if (has_mt()) {
                            for (int j = 0; j < mt_coeffs(is).num_rows_loc(); j++) {
                               s[i] += (std::pow(mt_coeffs(is).prime(j, i).real(), 2) + std::pow(mt_coeffs(is).prime(j, i).imag(), 2));
                            }
                        }
                    }
                    break;
                }
                case GPU: {
                    #ifdef __GPU
                    add_square_sum_gpu(pw_coeffs(is).prime().at<GPU>(), pw_coeffs(is).num_rows_loc(), n__,
                                       gkvec_.reduced(), comm_.rank(), s.at<GPU>());
                    if (has_mt()) {
                        add_square_sum_gpu(mt_coeffs(is).prime().at<GPU>(), mt_coeffs(is).num_rows_loc(), n__,
                                           0, comm_.rank(), s.at<GPU>());
                    }
                    #endif
                    break;
                }
            }
        }
        if (pu__ == GPU) {
            s.copy<memory_t::device, memory_t::host>();
        }
        comm_.allreduce(s.at<CPU>(), n__);
        return std::move(s);
    }

  public:
    /// Constructor for PW wave-functions.
    Wave_functions(Gvec const& gkvec__,
                   int         num_wf__,
                   int         num_sc__ = 1)
        : comm_(gkvec__.comm())
        , gkvec_(gkvec__)
        , num_wf_(num_wf__)
        , num_sc_(num_sc__)
    {
        if (!(num_sc__ == 1 || num_sc__ == 2)) {
            TERMINATE("wrong number of spin components");
        }

        for (int ispn = 0; ispn < num_sc_; ispn++) {
            pw_coeffs_[ispn] = std::unique_ptr<matrix_storage<double_complex, matrix_storage_t::slab>>(
                new matrix_storage<double_complex, matrix_storage_t::slab>(gkvec_.count(), num_wf_, gkvec_.comm_ortho_fft()));
        }
    }

    /// Constructor for PW wave-functions.
    Wave_functions(double_complex* ptr__,
                   Gvec const&     gkvec__,
                   int             num_wf__,
                   int             num_sc__ = 1)
        : comm_(gkvec__.comm())
        , gkvec_(gkvec__)
        , num_wf_(num_wf__)
        , num_sc_(num_sc__)
    {
        if (!(num_sc__ == 1 || num_sc__ == 2)) {
            TERMINATE("wrong number of spin components");
        }

        for (int ispn = 0; ispn < num_sc_; ispn++) {
            pw_coeffs_[ispn] = std::unique_ptr<matrix_storage<double_complex, matrix_storage_t::slab>>(
                new matrix_storage<double_complex, matrix_storage_t::slab>(ptr__, gkvec_.count(), num_wf_, gkvec_.comm_ortho_fft()));
                ptr__ += gkvec_.count() * num_wf_;
        }
    }

    /// Constructor for LAPW wave-functions.
    Wave_functions(Gvec const&             gkvec__,
                   int                     num_atoms__,
                   std::function<int(int)> mt_size__,
                   int                     num_wf__,
                   int                     num_sc__ = 1)
        : comm_(gkvec__.comm())
        , gkvec_(gkvec__)
        , num_wf_(num_wf__)
        , num_sc_(num_sc__)
        , has_mt_(true)
    {
        if (!(num_sc__ == 1 || num_sc__ == 2)) {
            TERMINATE("wrong number of spin components");
        }

        for (int ispn = 0; ispn < num_sc_; ispn++) {
            pw_coeffs_[ispn] = std::unique_ptr<matrix_storage<double_complex, matrix_storage_t::slab>>(
                new matrix_storage<double_complex, matrix_storage_t::slab>(gkvec_.count(), num_wf_, gkvec_.comm_ortho_fft()));
        }

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
        
        for (int ispn = 0; ispn < num_sc_; ispn++) {
            mt_coeffs_[ispn] = std::unique_ptr<matrix_storage<double_complex, matrix_storage_t::slab>>(
                new matrix_storage<double_complex, matrix_storage_t::slab>(mt_coeffs_distr_.counts[comm_.rank()],
                                                                           num_wf_, mpi_comm_null()));
        }
    }

    Communicator const& comm() const
    {
        return comm_;
    }

    Gvec const& gkvec() const
    {
        return gkvec_;
    }

    inline int num_mt_coeffs() const
    {
        return num_mt_coeffs_;
    }

    inline matrix_storage<double_complex, matrix_storage_t::slab>& pw_coeffs(int ispn__)
    {
        return *pw_coeffs_[isc(ispn__)];
    }

    inline matrix_storage<double_complex, matrix_storage_t::slab> const& pw_coeffs(int ispn__) const
    {
        return *pw_coeffs_[isc(ispn__)];
    }

    inline matrix_storage<double_complex, matrix_storage_t::slab>& mt_coeffs(int ispn__)
    {
        return *mt_coeffs_[isc(ispn__)];
    }

    inline matrix_storage<double_complex, matrix_storage_t::slab> const& mt_coeffs(int ispn__) const
    {
        return *mt_coeffs_[isc(ispn__)];
    }

    inline bool has_mt() const
    {
        return has_mt_ && (mt_coeffs_distr_.counts[comm_.rank()] > 0);
    }

    inline int num_wf() const
    {
        return num_wf_;
    }

    inline int num_sc() const
    {
        return num_sc_;
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
    /** \param [in] pu   Type of processging unit which copies data.
     *  \param [in] n    Number of wave-functions to copy.
     *  \param [in] src  Input wave-function.
     *  \param [in] ispn Spin component on source wave-functions.
     *  \param [in] i0   Starting index of wave-functions in src.
     *  \param [in] jspn Spin component on destination wave-functions.
     *  \param [in] j0   Starting index of wave-functions in destination. */
    inline void copy_from(device_t              pu__,
                          int                   n__,
                          Wave_functions const& src__,
                          int                   ispn__,
                          int                   i0__,
                          int                   jspn__,
                          int                   j0__)
    {
        assert(ispn__ == 0 || ispn__ == 1);
        assert(jspn__ == 0 || jspn__ == 1);

        int ngv =  pw_coeffs(jspn__).num_rows_loc();
        int nmt = has_mt() ? mt_coeffs(jspn__).num_rows_loc() : 0;

        switch (pu__) {
            case CPU: {
                /* copy PW part */
                std::copy(src__.pw_coeffs(ispn__).prime().at<CPU>(0, i0__),
                          src__.pw_coeffs(ispn__).prime().at<CPU>(0, i0__) + ngv * n__,
                          pw_coeffs(jspn__).prime().at<CPU>(0, j0__));
                /* copy MT part */
                if (has_mt()) {
                    std::copy(src__.mt_coeffs(ispn__).prime().at<CPU>(0, i0__),
                              src__.mt_coeffs(ispn__).prime().at<CPU>(0, i0__) + nmt * n__,
                              mt_coeffs(jspn__).prime().at<CPU>(0, j0__));
                }
                break;
            }
            case GPU: {
                #ifdef __GPU
                /* copy PW part */
                acc::copy(pw_coeffs(jspn__).prime().at<GPU>(0, j0__),
                          src__.pw_coeffs(ispn__).prime().at<GPU>(0, i0__),
                          ngv * n__);
                /* copy MT part */
                if (has_mt()) {
                    acc::copy(mt_coeffs(jspn__).prime().at<GPU>(0, j0__),
                              src__.mt_coeffs(ispn__).prime().at<GPU>(0, i0__),
                              nmt * n__);
                }
                #endif
                break;
            }
        }
    }

    /// Compute the checksum of the spin-components.
    /** Checksum of the n wave-function spin components is computed starting from i0. 
     *  Only plane-wave coefficients are considered. */
    inline double_complex checksum_pw(device_t pu__,
                                      int      ispn__,
                                      int      i0__,
                                      int      n__)
    {
        assert(n__ != 0);
        double_complex cs(0, 0);
        for (int s = s0(ispn__); s <= s1(ispn__); s++) {
            cs += pw_coeffs(s).checksum(pu__, i0__, n__);
        }
        comm_.allreduce(&cs, 1);
        return cs;
    }

    /// Checksum of muffin-tin coefficients.
    inline double_complex checksum_mt(device_t pu__, int ispn__, int i0__, int n__)
    {
        assert(n__ != 0);
        double_complex cs(0, 0);
        if (!has_mt()) {
            return cs;
        }
        for (int s = s0(ispn__); s <= s1(ispn__); s++) {
            cs += mt_coeffs(s).checksum(pu__, i0__, n__);
        }
        comm_.allreduce(&cs, 1);
        return cs;
    }

    inline double_complex checksum(device_t pu__, int ispn__, int i0__, int n__)
    {
        return checksum_pw(pu__, ispn__, i0__, n__) + checksum_mt(pu__, ispn__, i0__, n__);
    }

    inline void zero_pw(device_t pu__, int ispn__, int i0__, int n__)
    {
        for (int s = s0(ispn__); s <= s1(ispn__); s++) {
            switch (pu__) {
                case CPU: {
                    pw_coeffs(s).zero<memory_t::host>(i0__, n__);
                    break;
                }
                case GPU: {
                    pw_coeffs(s).zero<memory_t::device>(i0__, n__);
                    break;
                }
            }
        }
    }

    inline void zero_mt(device_t pu__, int ispn__, int i0__, int n__)
    {
        if (!has_mt()) {
            return;
        }
        for (int s = s0(ispn__); s <= s1(ispn__); s++) {
            switch (pu__) {
                case CPU: {
                    mt_coeffs(s).zero<memory_t::host>(i0__, n__);
                    break;
                }
                case GPU: {
                    mt_coeffs(s).zero<memory_t::device>(i0__, n__);
                    break;
                }
            }
        }
    }

    inline void zero(device_t pu__, int ispn__, int i0__, int n__)
    {
        zero_pw(pu__, ispn__, i0__, n__);
        zero_mt(pu__, ispn__, i0__, n__);
    }

    inline void scale(device_t pu__, int ispn__, int i0__, int n__, double beta__)
    {
        for (int s = s0(ispn__); s <= s1(ispn__); s++) {
            switch (pu__) {
                case CPU: {
                    pw_coeffs(s).scale<memory_t::host>(i0__, n__, beta__);
                    if (has_mt()) {
                        mt_coeffs(s).scale<memory_t::host>(i0__, n__, beta__);
                    }
                    break;
                }
                case GPU: {
                    pw_coeffs(s).scale<memory_t::device>(i0__, n__, beta__);
                    if (has_mt()) {
                        mt_coeffs(s).scale<memory_t::device>(i0__, n__, beta__);
                    }
                    break;
                }
            }
        }
    }

    inline mdarray<double,1> l2norm(device_t pu__, int ispn__, int n__) const
    {
        assert(n__ != 0);

        auto norm = sumsqr(pu__, ispn__, n__);
        for (int i = 0; i < n__; i++) {
            norm[i] = std::sqrt(norm[i]);
        }

        return std::move(norm);
    }

    #ifdef __GPU
    void allocate_on_device(int ispn__) {
        for (int s = s0(ispn__); s <= s1(ispn__); s++) {
            pw_coeffs(s).allocate_on_device();
            if (has_mt()) {
                mt_coeffs(s).allocate_on_device();
            }
        }
    }

    void deallocate_on_device(int ispn__) {
        for (int s = s0(ispn__); s <= s1(ispn__); s++) {
            pw_coeffs(s).deallocate_on_device();
            if (has_mt()) {
                mt_coeffs(s).deallocate_on_device();
            }
        }
    }

    void copy_to_device(int ispn__, int i0__, int n__) {
        for (int s = s0(ispn__); s <= s1(ispn__); s++) {
            pw_coeffs(s).copy_to_device(i0__, n__);
            if (has_mt()) {
                mt_coeffs(s).copy_to_device(i0__, n__);
            }
        }
    }

    void copy_to_host(int ispn__, int i0__, int n__) {
        for (int s = s0(ispn__); s <= s1(ispn__); s++) {
            pw_coeffs(s).copy_to_host(i0__, n__);
            if (has_mt()) {
                mt_coeffs(s).copy_to_host(i0__, n__);
            }
        }
    }
    #endif

};

#include "wf_inner.hpp"
#include "wf_trans.hpp"
#include "wf_ortho.hpp"

}

#endif
