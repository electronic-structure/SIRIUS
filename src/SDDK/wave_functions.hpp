// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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
 *  \brief Contains declaration and implementation of Wave_functions class.
 */

#ifndef __WAVE_FUNCTIONS_HPP__
#define __WAVE_FUNCTIONS_HPP__

#include <cstdlib>
#include <iostream>
#include <omp.h>
#include "linalg.hpp"
#include "eigenproblem.hpp"
#include "hdf5_tree.hpp"
#include "utils/env.hpp"
#include "gvec.hpp"
#include "matrix_storage.hpp"
#ifdef __GPU
extern "C" void add_square_sum_gpu(double_complex const* wf__, int num_rows_loc__, int nwf__, int reduced__,
                                   int mpi_rank__, double* result__);

extern "C" void add_checksum_gpu(double_complex const* wf__, int num_rows_loc__, int nwf__, double_complex* result__);
#endif

const int sddk_inner_default_block_size = 1024;
const int sddk_trans_default_block_size = 2048;

namespace sddk {

/// Helper class to wrap spin index range.
/** Depending on the collinear or non-collinear case, the spin index range of the wave-functions is either
 *  [0, 0] or [1, 1] (trivial cases of single spin channel) or [0, 1] (spinor wave-functions). */
class spin_range : public std::vector<int>
{
  private:
    int idx_;
  public:
    explicit spin_range(int idx__)
        : idx_(idx__)
    {
        if (!(idx_ == 0 || idx_ == 1 || idx_ == 2)) {
            throw std::runtime_error("wrong spin index");
        }
        if (idx_ == 2) {
            this->reserve(2);
            this->push_back(0);
            this->push_back(1);
        } else {
            this->reserve(1);
            this->push_back(idx_);
        }
    }
    inline int operator()() const
    {
        return idx_;
    }
};

//TODO introduce band range to desctibe a set of bands in the interval [N1, N2]

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
 *
 *  Example below shows how the wave-functions are used:

    \code{.cpp}
    // alias for wave-functions
    auto& psi = kp__->spinor_wave_functions();
    // create hpsi
    Wave_functions hpsi(kp__->gkvec_partition(), ctx_.num_bands(), num_sc);
    // create hpsi
    Wave_functions spsi(kp__->gkvec_partition(), ctx_.num_bands(), num_sc);

    // if preferred memory is on GPU
    if (is_device_memory(ctx_.preferred_memory_t())) {
        // alias for memory pool
        auto& mpd = ctx_.mem_pool(memory_t::device);
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            // allocate GPU memory
            psi.pw_coeffs(ispn).allocate(mpd);
            // copy to GPU
            psi.pw_coeffs(ispn).copy_to(memory_t::device, 0, ctx_.num_bands());
        }
        // set the preferred memory type
        psi.preferred_memory_t(ctx_.preferred_memory_t());
        // allocate hpsi and spsi on GPU
        for (int i = 0; i < num_sc; i++) {
            hpsi.pw_coeffs(i).allocate(mpd);
            spsi.pw_coeffs(i).allocate(mpd);
        }
        // set preferred memory for hpsi
        hpsi.preferred_memory_t(ctx_.preferred_memory_t());
        // set preferred memory for spsi
        spsi.preferred_memory_t(ctx_.preferred_memory_t());
    }
    // prepare beta projectors
    kp__->beta_projectors().prepare();
    for (int ispin_step = 0; ispin_step < ctx_.num_spin_dims(); ispin_step++) {
        if (nc_mag) {
            // apply Hamiltonian and S operators to both components of wave-functions
            H__.apply_h_s<T>(kp__, 2, 0, ctx_.num_bands(), psi, &hpsi, &spsi);
        } else {
            // apply Hamiltonian and S operators to a single components of wave-functions
            H__.apply_h_s<T>(kp__, ispin_step, 0, ctx_.num_bands(), psi, &hpsi, &spsi);
        }

        for (int ispn = 0; ispn < num_sc; ispn++) {
            // copy to host if needed
            if (is_device_memory(ctx_.preferred_memory_t())) {
                hpsi.copy_to(ispn, memory_t::host, 0, ctx_.num_bands());
                spsi.copy_to(ispn, memory_t::host, 0, ctx_.num_bands());
            }
        }
        // do something with hpsi and spsi
    }
    // free beta-projectors
    kp__->beta_projectors().dismiss();
    if (is_device_memory(ctx_.preferred_memory_t())) {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            // deallocate wave-functions on GPU
            psi.pw_coeffs(ispn).deallocate(memory_t::device);
        }
        // set preferred memory to CPU
        psi.preferred_memory_t(memory_t::host);
    }
    \endcode
 */
class Wave_functions
{
  private:
    /// Communicator used to distribute G+k vectors and atoms.
    Communicator const& comm_;

    /// G+k vectors of the wave-function.
    Gvec_partition const& gkvecp_;

    splindex<splindex_t::block> spl_num_atoms_;

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
    std::array<std::unique_ptr<matrix_storage<double_complex, matrix_storage_t::slab>>, 2> pw_coeffs_{
        {nullptr, nullptr}};

    /// Muffin-tin part of wave-functions.
    std::array<std::unique_ptr<matrix_storage<double_complex, matrix_storage_t::slab>>, 2> mt_coeffs_{
        {nullptr, nullptr}};

    bool has_mt_{false};

    /// Preferred memory type for this wave functions.
    memory_t preferred_memory_t_{memory_t::host};

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

    /// Compute the sum of squares of expansion coefficients.
    /** The result is always returned in the host memory */
    mdarray<double, 1> sumsqr(device_t pu__, spin_range spins__, int n__) const;

  public:
    /// Constructor for PW wave-functions.
    /** Memory to store plane-wave coefficients is allocated from the heap. */
    Wave_functions(Gvec_partition const& gkvecp__, int num_wf__, memory_t preferred_memory_t__, int num_sc__ = 1);

    /// Constructor for PW wave-functions.
    /** Memory to store plane-wave coefficients is allocated from the memory pool. */
    Wave_functions(memory_pool& mp__, Gvec_partition const& gkvecp__, int num_wf__, memory_t preferred_memory_t__,
                   int num_sc__ = 1);

    /// Constructor for LAPW wave-functions.
    Wave_functions(Gvec_partition const& gkvecp__, int num_atoms__, std::function<int(int)> mt_size__, int num_wf__,
                   memory_t preferred_memory_t__, int num_sc__ = 1);

    /// Communicator of the G+k vector distribution.
    Communicator const& comm() const
    {
        return comm_;
    }

    /// G+k vectors of the wave-functions.
    Gvec const& gkvec() const
    {
        return gkvecp_.gvec();
    }

    Gvec_partition const& gkvec_partition() const
    {
        return gkvecp_;
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

    inline splindex<splindex_t::block> const& spl_num_atoms() const
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
    void copy_from(device_t pu__, int n__, Wave_functions const& src__, int ispn__, int i0__, int jspn__, int j0__);

    /// Copy from and to preferred memory.
    void copy_from(Wave_functions const& src__, int n__, int ispn__, int i0__, int jspn__, int j0__);

    /// Compute the checksum of the spin-components.
    /** Checksum of the n wave-function spin components is computed starting from i0.
     *  Only plane-wave coefficients are considered. */
    double_complex checksum_pw(device_t pu__, int ispn__, int i0__, int n__) const;

    /// Checksum of muffin-tin coefficients.
    double_complex checksum_mt(device_t pu__, int ispn__, int i0__, int n__) const;

    inline double_complex checksum(device_t pu__, int ispn__, int i0__, int n__) const
    {
        return checksum_pw(pu__, ispn__, i0__, n__) + checksum_mt(pu__, ispn__, i0__, n__);
    }

    void zero_pw(device_t pu__, int ispn__, int i0__, int n__);

    void zero_mt(device_t pu__, int ispn__, int i0__, int n__);

    inline void zero(device_t pu__, int ispn__, int i0__, int n__) // TODO: pass memory_t
    {
        zero_pw(pu__, ispn__, i0__, n__);
        zero_mt(pu__, ispn__, i0__, n__);
    }

    void scale(memory_t mem__, int ispn__, int i0__, int n__, double beta__);

    mdarray<double, 1> l2norm(device_t pu__, spin_range spins__, int n__) const;

    /// Normalize the functions.
    void normalize(device_t pu__, spin_range spins__, int n__);

    void allocate(spin_range spins__, memory_t mem__);

    void deallocate(spin_range spins__, memory_t mem__);

    void copy_to(spin_range spins__, memory_t mem__, int i0__, int n__);

    inline memory_t preferred_memory_t() const
    {
        return preferred_memory_t_;
    }

    void print_checksum(device_t pu__, std::string label__, int N__, int n__) const;
};


} // namespace sddk

#endif
