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
#include "linalg.hpp"
#include "eigenproblem.hpp"
#include "hdf5_tree.hpp"
#include "utils/env.hpp"
#ifdef __GPU
extern "C" void add_square_sum_gpu(double_complex const* wf__, int num_rows_loc__, int nwf__, int reduced__,
                                   int mpi_rank__, double* result__);

extern "C" void add_checksum_gpu(double_complex* wf__, int num_rows_loc__, int nwf__, double_complex* result__);
#endif

const int sddk_default_block_size = 256;

namespace sddk {

/// Helper function: get a list of spin-indices.
inline std::vector<int> get_spins(int ispn__)
{
    return (ispn__ == 2) ? std::vector<int>({0, 1}) : std::vector<int>({ispn__});
}

/// Helper class to wrap spin index (integer number).
class spin_idx
{
  private:
    int idx_;
  public:
    explicit spin_idx(int idx__)
        : idx_(idx__)
    {
        if (!(idx_ == 0 || idx_ == 1 || idx_ == 2)) {
            TERMINATE("wrong spin index");
        }
    }
    inline int operator()() const
    {
        return idx_;
    }
};

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

    inline mdarray<double, 1> sumsqr(device_t pu__, int ispn__, int n__) const
    {
        mdarray<double, 1> s(n__, memory_t::host, "sumsqr");
        s.zero();
        if (pu__ == device_t::GPU) {
            s.allocate(memory_t::device);
            s.zero(memory_t::device);
        }

        for (int is = s0(ispn__); is <= s1(ispn__); is++) {
            switch (pu__) {
                case device_t::CPU: {
                    #pragma omp parallel for
                    for (int i = 0; i < n__; i++) {
                        for (int ig = 0; ig < pw_coeffs(is).num_rows_loc(); ig++) {
                            s[i] += (std::pow(pw_coeffs(is).prime(ig, i).real(), 2) +
                                     std::pow(pw_coeffs(is).prime(ig, i).imag(), 2));
                        }
                        if (gkvecp_.gvec().reduced()) {
                            if (comm_.rank() == 0) {
                                s[i] = 2 * s[i] - std::pow(pw_coeffs(is).prime(0, i).real(), 2);
                            } else {
                                s[i] *= 2;
                            }
                        }
                        if (has_mt()) {
                            for (int j = 0; j < mt_coeffs(is).num_rows_loc(); j++) {
                                s[i] += (std::pow(mt_coeffs(is).prime(j, i).real(), 2) +
                                         std::pow(mt_coeffs(is).prime(j, i).imag(), 2));
                            }
                        }
                    }
                    break;
                }
                case device_t::GPU: {
#ifdef __GPU
                    add_square_sum_gpu(pw_coeffs(is).prime().at(memory_t::device), pw_coeffs(is).num_rows_loc(), n__,
                                       gkvecp_.gvec().reduced(), comm_.rank(), s.at(memory_t::device));
                    if (has_mt()) {
                        add_square_sum_gpu(mt_coeffs(is).prime().at(memory_t::device), mt_coeffs(is).num_rows_loc(), n__, 0,
                                           comm_.rank(), s.at(memory_t::device));
                    }
#endif
                    break;
                }
            }
        }
        if (pu__ == device_t::GPU) {
            s.copy_to(memory_t::host);
        }
        comm_.allreduce(s.at(memory_t::host), n__);
        return std::move(s);
    }

  public:
    /// Constructor for PW wave-functions.
    Wave_functions(Gvec_partition const& gkvecp__, int num_wf__, memory_t preferred_memory_t__, int num_sc__ = 1)
        : comm_(gkvecp__.gvec().comm())
        , gkvecp_(gkvecp__)
        , num_wf_(num_wf__)
        , num_sc_(num_sc__)
        , preferred_memory_t_(preferred_memory_t__)
    {
        if (!(num_sc__ == 1 || num_sc__ == 2)) {
            TERMINATE("wrong number of spin components");
        }

        for (int ispn = 0; ispn < num_sc_; ispn++) {
            pw_coeffs_[ispn] = std::unique_ptr<matrix_storage<double_complex, matrix_storage_t::slab>>(
                new matrix_storage<double_complex, matrix_storage_t::slab>(gkvecp_, num_wf_));
        }
    }

    /// Constructor for PW wave-functions.
    Wave_functions(memory_pool& mp__, Gvec_partition const& gkvecp__, int num_wf__, memory_t preferred_memory_t__,
                   int num_sc__ = 1)
        : comm_(gkvecp__.gvec().comm())
        , gkvecp_(gkvecp__)
        , num_wf_(num_wf__)
        , num_sc_(num_sc__)
        , preferred_memory_t_(preferred_memory_t__)
    {
        if (!(num_sc__ == 1 || num_sc__ == 2)) {
            TERMINATE("wrong number of spin components");
        }

        for (int ispn = 0; ispn < num_sc_; ispn++) {
            pw_coeffs_[ispn] = std::unique_ptr<matrix_storage<double_complex, matrix_storage_t::slab>>(
                new matrix_storage<double_complex, matrix_storage_t::slab>(mp__, gkvecp_, num_wf_));
        }
    }

    /// Constructor for LAPW wave-functions.
    Wave_functions(Gvec_partition const& gkvecp__, int num_atoms__, std::function<int(int)> mt_size__, int num_wf__,
                   memory_t preferred_memory_t__, int num_sc__ = 1)
        : comm_(gkvecp__.gvec().comm())
        , gkvecp_(gkvecp__)
        , num_wf_(num_wf__)
        , num_sc_(num_sc__)
        , has_mt_(true)
        , preferred_memory_t_(preferred_memory_t__)
    {
        if (!(num_sc__ == 1 || num_sc__ == 2)) {
            TERMINATE("wrong number of spin components");
        }

        for (int ispn = 0; ispn < num_sc_; ispn++) {
            pw_coeffs_[ispn] = std::unique_ptr<matrix_storage<double_complex, matrix_storage_t::slab>>(
                new matrix_storage<double_complex, matrix_storage_t::slab>(gkvecp_, num_wf_));
        }

        spl_num_atoms_   = splindex<block>(num_atoms__, comm_.size(), comm_.rank());
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
                                                                           num_wf_));
        }
    }

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
    inline void copy_from(device_t pu__, int n__, Wave_functions const& src__, int ispn__, int i0__, int jspn__,
                          int j0__)
    {
        assert(ispn__ == 0 || ispn__ == 1);
        assert(jspn__ == 0 || jspn__ == 1);

        int ngv = pw_coeffs(jspn__).num_rows_loc();
        int nmt = has_mt() ? mt_coeffs(jspn__).num_rows_loc() : 0;

        switch (pu__) {
            case CPU: {
                /* copy PW part */
                std::copy(src__.pw_coeffs(ispn__).prime().at(memory_t::host, 0, i0__),
                          src__.pw_coeffs(ispn__).prime().at(memory_t::host, 0, i0__) + ngv * n__,
                          pw_coeffs(jspn__).prime().at(memory_t::host, 0, j0__));
                /* copy MT part */
                if (has_mt()) {
                    std::copy(src__.mt_coeffs(ispn__).prime().at(memory_t::host, 0, i0__),
                              src__.mt_coeffs(ispn__).prime().at(memory_t::host, 0, i0__) + nmt * n__,
                              mt_coeffs(jspn__).prime().at(memory_t::host, 0, j0__));
                }
                break;
            }
            case GPU: {
#ifdef __GPU
                /* copy PW part */
                acc::copy(pw_coeffs(jspn__).prime().at(memory_t::device, 0, j0__),
                          src__.pw_coeffs(ispn__).prime().at(memory_t::device, 0, i0__),
                          ngv * n__);
                /* copy MT part */
                if (has_mt()) {
                    acc::copy(mt_coeffs(jspn__).prime().at(memory_t::device, 0, j0__),
                              src__.mt_coeffs(ispn__).prime().at(memory_t::device, 0, i0__), nmt * n__);
                }
#endif
                break;
            }
        }
    }

    inline void copy_from(Wave_functions const& src__, int n__, int ispn__, int i0__, int jspn__, int j0__)
    {
        assert(ispn__ == 0 || ispn__ == 1);
        assert(jspn__ == 0 || jspn__ == 1);

        int ngv = pw_coeffs(jspn__).num_rows_loc();
        int nmt = has_mt() ? mt_coeffs(jspn__).num_rows_loc() : 0;

        copy(src__.preferred_memory_t(), src__.pw_coeffs(ispn__).prime().at(src__.preferred_memory_t(), 0, i0__),
             preferred_memory_t(), pw_coeffs(jspn__).prime().at(preferred_memory_t(), 0, j0__), ngv * n__);
        if (has_mt()) {
            copy(src__.preferred_memory_t(), src__.mt_coeffs(ispn__).prime().at(src__.preferred_memory_t(), 0, i0__),
                 preferred_memory_t(), mt_coeffs(jspn__).prime().at(preferred_memory_t(), 0, j0__), nmt * n__);
        }
    }

    /// Compute the checksum of the spin-components.
    /** Checksum of the n wave-function spin components is computed starting from i0.
     *  Only plane-wave coefficients are considered. */
    inline double_complex checksum_pw(device_t pu__, int ispn__, int i0__, int n__)
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

    inline void zero_pw(device_t pu__, int ispn__, int i0__, int n__) // TODO: pass memory_t
    {
        for (int s = s0(ispn__); s <= s1(ispn__); s++) {
            switch (pu__) {
                case CPU: {
                    pw_coeffs(s).zero(memory_t::host, i0__, n__);
                    break;
                }
                case GPU: {
                    pw_coeffs(s).zero(memory_t::device, i0__, n__);
                    break;
                }
            }
        }
    }

    inline void zero_mt(device_t pu__, int ispn__, int i0__, int n__) // TODO: pass memory_t
    {
        if (!has_mt()) {
            return;
        }
        for (int s = s0(ispn__); s <= s1(ispn__); s++) {
            switch (pu__) {
                case CPU: {
                    mt_coeffs(s).zero(memory_t::host, i0__, n__);
                    break;
                }
                case GPU: {
                    mt_coeffs(s).zero(memory_t::device, i0__, n__);
                    break;
                }
            }
        }
    }

    inline void zero(device_t pu__, int ispn__, int i0__, int n__) // TODO: pass memory_t
    {
        zero_pw(pu__, ispn__, i0__, n__);
        zero_mt(pu__, ispn__, i0__, n__);
    }

    inline void scale(memory_t mem__, int ispn__, int i0__, int n__, double beta__)
    {
        for (int s = s0(ispn__); s <= s1(ispn__); s++) {
            pw_coeffs(s).scale(mem__, i0__, n__, beta__);
            if (has_mt()) {
                mt_coeffs(s).scale(mem__, i0__, n__, beta__);
            }
        }
    }

    inline mdarray<double, 1> l2norm(device_t pu__, int ispn__, int n__) const
    {
        assert(n__ != 0);

        auto norm = sumsqr(pu__, ispn__, n__);
        for (int i = 0; i < n__; i++) {
            norm[i] = std::sqrt(norm[i]);
        }

        return std::move(norm);
    }

    void allocate(spin_idx sid__, memory_t mem__)
    {
        for (int s = s0(sid__()); s <= s1(sid__()); s++) {
            pw_coeffs(s).allocate(mem__);
            if (has_mt()) {
                mt_coeffs(s).allocate(mem__);
            }
        }
    }

    void deallocate(spin_idx sid__, memory_t mem__)
    {
        for (int s = s0(sid__()); s <= s1(sid__()); s++) {
            pw_coeffs(s).deallocate(mem__);
            if (has_mt()) {
                mt_coeffs(s).deallocate(mem__);
            }
        }
    }

    void copy_to(spin_idx ispn__, memory_t mem__, int i0__, int n__)
    {
        for (int s = s0(ispn__()); s <= s1(ispn__()); s++) {
            pw_coeffs(s).copy_to(mem__, i0__, n__);
            if (has_mt()) {
                mt_coeffs(s).copy_to(mem__, i0__, n__);
            }
        }
    }

    inline memory_t preferred_memory_t() const
    {
        return preferred_memory_t_;
    }
};

#include "wf_inner.hpp"
#include "wf_trans.hpp"
#include "wf_ortho.hpp"

} // namespace sddk

#endif
