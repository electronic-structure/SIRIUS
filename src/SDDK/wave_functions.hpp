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
#include <costa/layout.hpp>
#include <costa/grid2grid/transformer.hpp>
#include "linalg/linalg.hpp"
#include "SDDK/hdf5_tree.hpp"
#include "SDDK/gvec.hpp"
#include "utils/env.hpp"
#include "utils/rte.hpp"
#include "matrix_storage.hpp"
#include "type_definition.hpp"

#ifdef SIRIUS_GPU
extern "C" void add_square_sum_gpu_double(double_complex const* wf__, int num_rows_loc__, int nwf__, int reduced__,
                                   int mpi_rank__, double* result__);

extern "C" void add_square_sum_gpu_float(std::complex<float> const* wf__, int num_rows_loc__, int nwf__, int reduced__,
                                   int mpi_rank__, float* result__);

extern "C" void scale_matrix_columns_gpu_double(int nrow__, int ncol__, std::complex<double>* mtrx__, double* a__);

extern "C" void scale_matrix_columns_gpu_float(int nrow__, int ncol__, std::complex<float>* mtrx__, float* a__);
#endif

const int sddk_inner_default_block_size = 1024;
const int sddk_trans_default_block_size = 2048;

namespace sddk {

// C++ wrappers for gpu kernels
void add_square_sum_gpu(std::complex<double> const* wf__, int num_rows_loc__, int nwf__, int reduced__, int mpi_rank__, double* result__);
void add_square_sum_gpu(std::complex<float> const* wf__, int num_rows_loc__, int nwf__, int reduced__, int mpi_rank__, float* result__);
void scale_matrix_columns_gpu(int nrow__, int ncol__, std::complex<double>* mtrx__, double* a__);
void scale_matrix_columns_gpu(int nrow__, int ncol__, std::complex<float>* mtrx__, float* a__);

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
template <typename T>   // template type is using real type that determine the precision, wavefunction is always complex
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
    std::array<std::unique_ptr<matrix_storage<std::complex<T>, matrix_storage_t::slab>>, 2> pw_coeffs_{
        {nullptr, nullptr}};

    /// Muffin-tin part of wave-functions.
    std::array<std::unique_ptr<matrix_storage<std::complex<T>, matrix_storage_t::slab>>, 2> mt_coeffs_{
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

    /// Constructor for LAPW wave-functions.
    /** Memory to store wave-function coefficients is allocated from the memory pool. */
    Wave_functions(memory_pool& mp__, Gvec_partition const& gkvecp__, int num_atoms__,
                   std::function<int(int)> mt_size__, int num_wf__,
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

    inline matrix_storage<std::complex<T>, matrix_storage_t::slab>& pw_coeffs(int ispn__)
    {
        return *pw_coeffs_[isc(ispn__)];
    }

    inline matrix_storage<std::complex<T>, matrix_storage_t::slab> const& pw_coeffs(int ispn__) const
    {
        return *pw_coeffs_[isc(ispn__)];
    }

    inline matrix_storage<std::complex<T>, matrix_storage_t::slab>& mt_coeffs(int ispn__)
    {
        return *mt_coeffs_[isc(ispn__)];
    }

    inline matrix_storage<std::complex<T>, matrix_storage_t::slab> const& mt_coeffs(int ispn__) const
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

    inline memory_t preferred_memory_t() const
    {
        return preferred_memory_t_;
    }

    inline auto checksum(device_t pu__, int ispn__, int i0__, int n__) const
    {
        return checksum_pw(pu__, ispn__, i0__, n__) + checksum_mt(pu__, ispn__, i0__, n__);
    }

    inline void zero(device_t pu__, int ispn__, int i0__, int n__) // TODO: pass memory_t
    {
        this->zero_pw(pu__, ispn__, i0__, n__);
        this->zero_mt(pu__, ispn__, i0__, n__);
    }

    inline void zero(device_t pu__)
    {
        for (int is = 0; is < this->num_sc(); is++) {
            this->zero(pu__, is, 0, this->num_wf());
        }
    }

    // compute a dot, i.e. diag(this' * phi).
    mdarray<std::complex<T>, 1> dot(device_t pu__, spin_range spins__, Wave_functions<T> const &phi, int n__) const;

    // compute this[:, i] = alpha * phi[:, i] + beta * this[:, i]
    template<class Ta>
    void axpby(device_t pu__, spin_range spins__, Ta alpha, Wave_functions<T> const &phi, Ta beta, int n__);

    // compute this[:, i] = phi[:, i] + beta[i] * this[:, i], kinda like an axpy
    template<class Ta>
    void xpby(device_t pu__, spin_range spins__, Wave_functions<T> const &phi, std::vector<Ta> const &betas, int n__);

    // compute this[:, i] = alpha[i] * phi[:, i] + this[:, i]
    template<class Ta>
    void axpy(device_t pu__, spin_range spins__, std::vector<Ta> const &alphas, Wave_functions<T> const &phi, int n__);

    // compute this[:, ids[i]] = alpha[i] * phi[:, i] + this[:, i]
    template<class Ta>
    void axpy_scatter(device_t pu__, spin_range spins__, std::vector<Ta> const &alphas, Wave_functions<T> const &phi, std::vector<size_t> const &ids, int n__);

    /// Compute the sum of squares of expansion coefficients.
    /** The result is always returned in the host memory */
    mdarray<T, 1> sumsqr(device_t pu__, spin_range spins__, int n__) const;


    /// Copy values from another wave-function.
    /** \param [in] pu   Type of processging unit which copies data.
     *  \param [in] n    Number of wave-functions to copy.
     *  \param [in] src  Input wave-function.
     *  \param [in] ispn Spin component on source wave-functions.
     *  \param [in] i0   Starting index of wave-functions in src.
     *  \param [in] jspn Spin component on destination wave-functions.
     *  \param [in] j0   Starting index of wave-functions in destination. */
    void copy_from(device_t pu__, int n__, Wave_functions<T> const& src__, int ispn__, int i0__, int jspn__, int j0__);

    template <typename F>
    void copy_from(device_t pu__, int n__, Wave_functions<F> const& src__, int ispn__, int i0__, int jspn__, int j0__) {
        assert(ispn__ == 0 || ispn__ == 1);
        assert(jspn__ == 0 || jspn__ == 1);
        std::cout << "=== WARNING at line " << __LINE__ << " of file " << __FILE__ << " ===" << std::endl;
        std::cout << "    Copying Wavefunction with different type, possible lost of data precision" << std::endl;

        int ngv = pw_coeffs(jspn__).num_rows_loc();
        int nmt = has_mt() ? mt_coeffs(jspn__).num_rows_loc() : 0;

        switch (pu__) {
            case device_t::CPU: {
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
            case device_t::GPU: {
                throw std::runtime_error("Copy mixed precision type not supported in device memory");
                break;
            }
        }
    }

    /// Copy from and to preferred memory.
    void copy_from(Wave_functions<T> const& src__, int n__, int ispn__, int i0__, int jspn__, int j0__);

    template <typename F>
    void copy_from(Wave_functions<F> const& src__, int n__, int ispn__, int i0__, int jspn__, int j0__) {
        assert(ispn__ == 0 || ispn__ == 1);
        assert(jspn__ == 0 || jspn__ == 1);
        std::cout << "=== WARNING at line " << __LINE__ << " of file " << __FILE__ << " ===" << std::endl;
        std::cout << "    Copying Wavefunction with different type, possible lost of data precision" << std::endl;

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
    std::complex<T> checksum_pw(device_t pu__, int ispn__, int i0__, int n__) const;

    /// Checksum of muffin-tin coefficients.
    std::complex<T> checksum_mt(device_t pu__, int ispn__, int i0__, int n__) const;

    void zero_pw(device_t pu__, int ispn__, int i0__, int n__);

    void zero_mt(device_t pu__, int ispn__, int i0__, int n__);

    void scale(memory_t mem__, int ispn__, int i0__, int n__, T beta__);

    sddk::mdarray<T, 1> l2norm(device_t pu__, spin_range spins__, int n__) const;

    /// Normalize the functions.
    void normalize(device_t pu__, spin_range spins__, int n__);

    void allocate(spin_range spins__, memory_t mem__);

    void allocate(spin_range spins__, memory_pool& mp__);

    void deallocate(spin_range spins__, memory_t mem__);

    void copy_to(spin_range spins__, memory_t mem__, int i0__, int n__);

    template <typename OUT>
    void print_checksum(device_t pu__, std::string label__, int N__, int n__, OUT&& out__) const
    {
        for (int ispn = 0; ispn < num_sc(); ispn++) {
            auto cs1 = this->checksum_pw(pu__, ispn, N__, n__);
            auto cs2 = this->checksum_mt(pu__, ispn, N__, n__);
            if (this->comm().rank() == 0) {
                out__ << "checksum (" << label__ << "_pw_" << ispn << ") : " << cs1 << std::endl;
                if (this->has_mt_) {
                    out__ << "checksum (" << label__ << "_mt_" << ispn << ") : " << cs2 << std::endl;
                }
                out__ << "checksum (" << label__ << "_" << ispn << ") : " << cs1 + cs2 << std::endl;
            }
        }
    }

    /// Prepare wave-functions on the device.
    void prepare(spin_range spins__, bool with_copy__, memory_pool* mp__ = nullptr)
    {
        /* if operations on wave-functions are done on GPUs */
        if (is_device_memory(preferred_memory_t_)) {
            if (mp__) {
                if (!is_device_memory(mp__->memory_type())) {
                    RTE_THROW("not a device memory pool");
                }
                this->allocate(spins__, *mp__);
            } else {
                this->allocate(spins__, preferred_memory_t_);
            }
            if (with_copy__) {
                this->copy_to(spins__, preferred_memory_t_, 0, this->num_wf());
            }
        }
    }

    void dismiss(spin_range spins__, bool with_copy__)
    {
        if (is_device_memory(preferred_memory_t_)) {
            if (with_copy__) {
                this->copy_to(spins__, memory_t::host, 0, this->num_wf());
            }
            this->deallocate(spins__, preferred_memory_t_);
        }
    }
};

} // namespace sddk

namespace wf {

template <typename T, typename Tag>
class strong_type
{
  private:
    T val_;
  public:
    explicit strong_type(T const& val__)
        : val_{val__}
    {
    }

    explicit strong_type(T&& val__) 
        : val_{std::move(val__)}
    {
    }

    T const& get() const
    {
        return val_;
    }

    bool operator!=(strong_type<T, Tag> const& rhs__)
    {
        return this->val_ != rhs__.val_;
    }

    bool operator==(strong_type<T, Tag> const& rhs__)
    {
        return this->val_ == rhs__.val_;
    }

    strong_type<T, Tag>& operator++(int)
    {
        this->val_++;
        return *this;
    }
};

using spin_index = strong_type<int, struct __spin_index_tag>;
using atom_index = strong_type<int, struct __atom_index_tag>;
using band_index = strong_type<int, struct __band_index_tag>;

using num_bands = strong_type<int, struct __num_bands_tag>;
using num_spins = strong_type<int, struct __num_spins_tag>;
using num_mag_dims = strong_type<int, struct __num_mag_dims_tag>;

class band_range
{
  private:
    int begin_;
    int end_;
  public:
    band_range(int begin__, int end__)
        : begin_{begin__}
        , end_{end__}
    {
        RTE_ASSERT(begin_ >= 0);
        RTE_ASSERT(end_ >= 0);
        RTE_ASSERT(begin_ <= end_);
    }
    band_range(int size__)
        : begin_{0}
        , end_{size__}
    {
        RTE_ASSERT(size__ > 0);
    }
    inline auto begin() const
    {
        return begin_;
    }
    inline auto end() const
    {
        return end_;
    }
    inline auto size() const
    {
        return end_ - begin_;
    }
};

// Only 3 combinations of spin range are allowed:
// [0, 1)
// [1, 2)
// [0, 2)
class spin_range
{
  private:
    int begin_;
    int end_;
    int spinor_index_;
  public:
    spin_range(int begin__, int end__)
        : begin_{begin__}
        , end_{end__}
    {
        RTE_ASSERT(begin_ >= 0);
        RTE_ASSERT(end_ >= 0);
        RTE_ASSERT(begin_ <= end_);
        RTE_ASSERT(end_ <= 2);
        /* if size of the spin range is 2, this is a full-spinor case */
        if (this->size() == 2) {
            spinor_index_ = 0;
        } else {
            spinor_index_ = begin_;
        }
    }
    spin_range(int ispn__)
        : begin_{ispn__}
        , end_{ispn__ + 1}
        , spinor_index_{ispn__}
    {
        RTE_ASSERT(begin_ >= 0);
        RTE_ASSERT(end_ >= 0);
        RTE_ASSERT(begin_ <= end_);
        RTE_ASSERT(end_ <= 2);
    }
    inline auto begin() const
    {
        return spin_index(begin_);
    }
    inline auto end() const
    {
        return spin_index(end_);
    }
    inline int size() const
    {
        return end_ - begin_;
    }
    inline int spinor_index() const
    {
        return spinor_index_;
    }
};

/* PW and LAPW wave-functions
 *
 * Wave_functions wf(gkvec_factory(..), 10);
 *
 *
 * Local coefficients consit of two parts: PW and MT
 * +-------+
 * |       |
 * |  G+k  |   -> swap only PW part
 * |       |
 * +-------+
 * | atom1 |
 * +-------+
 * | atom2 |
 * +-------+
 * | ....  |
 * +-------+
 *
 * wf_fft = remap_to_fft(gkvec_partition, wf, N, n);
 *
 * hpsi_fft = wf_fft_factory(gkvec_partition, n);
 *
 * remap_from_fft(gkvec_partition, wf_fft, wf, N, n)
 *
 * consider Wave_functions_fft class
 *
 *
 * Wave_functions wf(...);
 * memory_guard mem_guard(wf, memory_t::device);
 *
 *
 *
 */

enum class copy_to : unsigned int
{
    none   = 0b0000,
    device = 0b0001,
    host   = 0b0010
};
inline copy_to operator|(copy_to a__, copy_to b__)
{
    return static_cast<copy_to>(static_cast<unsigned int>(a__) | static_cast<unsigned int>(b__));
}

template <typename T>
class device_memory_guard
{
  private:
    T& obj_;
    device_memory_guard(device_memory_guard const&) = delete;
    device_memory_guard& operator=(device_memory_guard const&) = delete;
    sddk::memory_t mem_;
    copy_to copy_to_;
  public:
    device_memory_guard(T& obj__, sddk::memory_t mem__, copy_to copy_to__)
        : obj_{obj__}
        , mem_{mem__}
        , copy_to_{copy_to__}
    {
        if (is_device_memory(mem_)) {
#ifndef NDEBUG
            std::cout << "allocate " << obj_.data_[0].label() << " on GPU" << std::endl;
#endif
            obj_.allocate(mem_);
            if (static_cast<unsigned int>(copy_to_) & static_cast<unsigned int>(copy_to::device)) {
#ifndef NDEBUG
                std::cout << "copy " << obj_.data_[0].label() << " to GPU" << std::endl;
#endif
                obj_.copy_to(mem_);
            }
        }
    }
    device_memory_guard(device_memory_guard&& src__) = default;
    ~device_memory_guard()
    {
        if (is_device_memory(mem_)) {
            if (static_cast<unsigned int>(copy_to_) & static_cast<unsigned int>(copy_to::host)) {
#ifndef NDEBUG
                std::cout << "copy " << obj_.data_[0].label() << " to host" << std::endl;
#endif
                obj_.copy_to(sddk::memory_t::host);
            }
#ifndef NDEBUG
            std::cout << "deallocate " << obj_.data_[0].label() << " on GPU" << std::endl;
#endif
            obj_.deallocate(mem_);
        }
    }
};

template <typename T>
class Wave_functions_base
{
  protected:
    int num_pw_{0};
    int num_mt_{0};
    num_mag_dims num_md_{0};
    num_bands num_wf_{0};
    num_spins num_sc_{0};

    inline void allocate(sddk::memory_t mem__)
    {
        for (int s = 0; s < num_sc_.get(); s++) {
            data_[s].allocate(mem__);
        }
    }

    inline void deallocate(sddk::memory_t mem__)
    {
        for (int s = 0; s < num_sc_.get(); s++) {
            data_[s].deallocate(mem__);
        }
    }

    inline void copy_to(sddk::memory_t mem__)
    {
        for (int s = 0; s < num_sc_.get(); s++) {
            data_[s].copy_to(mem__);
        }
    }

    friend class device_memory_guard<Wave_functions_base<T>>;

    std::array<sddk::mdarray<std::complex<T>, 2>, 2> data_;
  public:
    Wave_functions_base()
    {
    }
    Wave_functions_base(int num_pw__, int num_mt__, num_mag_dims num_md__, num_bands num_wf__, sddk::memory_t default_mem__)
        : num_pw_{num_pw__}
        , num_mt_{num_mt__}
        , num_md_{num_md__}
        , num_wf_{num_wf__}
    {
        if (!(num_md_.get() == 0 || num_md_.get() == 1 || num_md_.get() == 3)) {
            RTE_THROW("wrong number of magnetic dimensions");
        }

        if (num_md_.get() == 0) {
            num_sc_ = num_spins(1);
        } else {
            num_sc_ = num_spins(2);
        }
        for (int is = 0; is < num_sc_.get(); is++) {
            data_[is] = sddk::mdarray<std::complex<T>, 2>(num_pw_ + num_mt_, num_wf_.get(), default_mem__,
                "Wave_functions_base::data_");
        }
    }

    auto memory_guard(sddk::memory_t mem__, wf::copy_to copy_to__ = copy_to::none)
    {
        return std::move(device_memory_guard<Wave_functions_base<T>>(*this, mem__, copy_to__));
    }

    inline auto num_sc() const
    {
        return num_sc_;
    }

    inline auto num_md() const
    {
        return num_md_;
    }

    inline auto ld() const
    {
        return num_pw_ + num_mt_;
    }

    inline auto actual_spin_index(spin_index s__) const
    {
        if (num_sc_.get() == 2) {
            return s__;
        } else {
            return spin_index(0);
        }
    }

    inline void zero(sddk::memory_t mem__, spin_index s__, band_range br__)
    {
        if (is_host_memory(mem__)) {
            for (int ib = br__.begin(); ib < br__.end(); ib++) {
                auto ptr = data_[s__.get()].at(mem__, 0, ib);
                std::fill(ptr, ptr + this->ld(), 0);
            }
        }
        if (is_device_memory(mem__)) {
            acc::zero(data_[s__.get()].at(mem__, 0, br__.begin()), this->ld(), this->ld(), br__.size());
        }
    }

    auto const data_ptr(sddk::memory_t mem__, int i__, spin_index s__, band_index b__) const
    {
        return data_[s__.get()].at(mem__, i__, b__.get());
    }

    auto data_ptr(sddk::memory_t mem__, int i__, spin_index s__, band_index b__) 
    {
        return data_[s__.get()].at(mem__, i__, b__.get());
    }
};

template <typename T>
class Wave_functions_fft : public Wave_functions_base<T>
{
  private:
    std::shared_ptr<sddk::Gvec_partition> gkvec_fft_;
    sddk::splindex<sddk::splindex_t::block> spl_num_wf_;

  public:
    Wave_functions_fft(std::shared_ptr<sddk::Gvec_partition> gkvec_fft__, num_bands num_wf_max__, sddk::memory_t default_mem__)
        : Wave_functions_base<T>(gkvec_fft__->gvec_count_fft(), 0,
                num_mag_dims(0), num_bands(sddk::splindex<sddk::splindex_t::block>(num_wf_max__.get(), gkvec_fft__->comm_ortho_fft().size(),
                    gkvec_fft__->comm_ortho_fft().rank()).local_size()), default_mem__)
        , gkvec_fft_(gkvec_fft__)
    {
    }

    auto grid_layout(int n__)
    {
        auto& comm_row = gkvec_fft_->comm_fft();
        auto& comm_col = gkvec_fft_->comm_ortho_fft();

        spl_num_wf_ = sddk::splindex<sddk::splindex_t::block>(n__, comm_col.size(), comm_col.rank());

        std::vector<int> rowsplit(comm_row.size() + 1);
        rowsplit[0] = 0;
        for (int i = 0; i < comm_row.size(); i++) {
            rowsplit[i + 1] = rowsplit[i] + gkvec_fft_->gvec_count_fft(i);
        }

        std::vector<int> colsplit(comm_col.size() + 1);
        colsplit[0] = 0;
        for (int i = 0; i < comm_col.size(); i++) {
            colsplit[i + 1] = colsplit[i] + spl_num_wf_.local_size(i);
        }

        std::vector<int> owners(gkvec_fft_->gvec().comm().size());
        for (int i = 0; i < gkvec_fft_->gvec().comm().size(); i++) {
            owners[i] = i;
        }
        costa::block_t localblock;
        localblock.data = this->data_[0].at(sddk::memory_t::host);
        localblock.ld = this->ld();
        localblock.row = gkvec_fft_->comm_fft().rank();
        localblock.col = comm_col.rank();

        return costa::custom_layout<std::complex<T>>(comm_row.size(), comm_col.size(), rowsplit.data(),
                colsplit.data(), owners.data(), 1, &localblock, 'C');
    }

    int num_wf_local() const
    {
        return spl_num_wf_.local_size();
    }

    auto spl_num_wf() const
    {
        return spl_num_wf_;
    }

    inline std::complex<T>& pw_coeffs(int ig__, band_index b__)
    {
        return this->data_[0](ig__, b__.get());
    }

    inline T* pw_coeffs(sddk::memory_t mem__, band_index b__)
    {
        return reinterpret_cast<T*>(this->data_[0].at(mem__, 0, b__.get()));
    }
};

template <typename T>
class Wave_functions_mt : public Wave_functions_base<T>
{
  protected:
    sddk::Communicator const& comm_;
    int num_atoms_{0};
    sddk::splindex<sddk::splindex_t::block> spl_num_atoms_;
    /// Local size of muffin-tin coefficients for each rank.
    /** Each rank stores local fraction of atoms. Each atom has a set of MT coefficients. */
    sddk::block_data_descriptor mt_coeffs_distr_;
    /// Local offset in the block of MT coefficients for current rank.
    /** The size of the vector is equal to the local number of atoms for the current rank. */
    std::vector<int> offset_in_local_mt_coeffs_;
    /// Numbef of muffin-tin coefficients for each atom.
    std::vector<int> num_mt_coeffs_;

    static int get_local_num_mt_coeffs(std::vector<int> num_mt_coeffs__, sddk::Communicator const& comm__)
    {
        int num_atoms = static_cast<int>(num_mt_coeffs__.size());
        sddk::splindex<sddk::splindex_t::block> spl_atoms(num_atoms, comm__.size(), comm__.rank());
        auto it_begin = num_mt_coeffs__.begin() + spl_atoms.global_offset();
        auto it_end = it_begin + spl_atoms.local_size();
        return std::accumulate(it_begin, it_end, 0);
    }

    Wave_functions_mt(sddk::Communicator const& comm__, num_mag_dims num_md__, num_bands num_wf__,
            sddk::memory_t default_mem__, int num_pw__)
        : Wave_functions_base<T>(num_pw__, 0, num_md__, num_wf__, default_mem__)
        , comm_{comm__}
    {
    }

  public:
    Wave_functions_mt()
    {
    }

    Wave_functions_mt(sddk::Communicator const& comm__, std::vector<int> num_mt_coeffs__, num_mag_dims num_md__,
            num_bands num_wf__, sddk::memory_t default_mem__, int num_pw__ = 0)
        : Wave_functions_base<T>(num_pw__, get_local_num_mt_coeffs(num_mt_coeffs__, comm__), num_md__, num_wf__,
                                 default_mem__)
        , comm_{comm__}
        , num_atoms_{static_cast<int>(num_mt_coeffs__.size())}
        , spl_num_atoms_{sddk::splindex<sddk::splindex_t::block>(num_atoms_, comm_.size(), comm_.rank())}
        , num_mt_coeffs_{num_mt_coeffs__}
    {
        mt_coeffs_distr_ = sddk::block_data_descriptor(comm_.size());

        for (int ia = 0; ia < num_atoms_; ia++) {
            int rank = spl_num_atoms_.local_rank(ia);
            if (rank == comm_.rank()) {
                offset_in_local_mt_coeffs_.push_back(mt_coeffs_distr_.counts[rank]);
            }
            /* increment local number of MT coeffs. for a given rank */
            mt_coeffs_distr_.counts[rank] += num_mt_coeffs__[ia];
        }
        mt_coeffs_distr_.calc_offsets();
    }

    inline auto&
    mt_coeffs(sddk::memory_t mem__, int xi__, atom_index ia__, spin_index ispn__, band_index i__)
    {
        return *this->data_[ispn__.get()].at(mem__, this->num_pw_ + xi__ + offset_in_local_mt_coeffs_[ia__.get()],
                               i__.get());
    }

    inline auto const&
    mt_coeffs(sddk::memory_t mem__, int xi__, atom_index ia__, spin_index ispn__, band_index i__) const
    {
        return *this->data_[ispn__.get()].at(mem__, this->num_pw_ + xi__ + offset_in_local_mt_coeffs_[ia__.get()],
                               i__.get());
    }

    inline auto const& spl_num_atoms() const
    {
        return spl_num_atoms_;
    }

    auto grid_layout_mt(spin_index ispn__, band_range b__)
    {
        std::vector<int> rowsplit(comm_.size() + 1);
        rowsplit[0] = 0;
        for (int i = 0; i < comm_.size(); i++) {
            rowsplit[i + 1] = rowsplit[i] + mt_coeffs_distr_.counts[i];
        }
        std::vector<int> colsplit({0, b__.size()});
        std::vector<int> owners(comm_.size());
        for (int i = 0; i < comm_.size(); i++) {
            owners[i] = i;
        }
        costa::block_t localblock;
        localblock.data = this->data_[ispn__.get()].at(sddk::memory_t::host, this->num_pw_, b__.begin());
        localblock.ld = this->ld();
        localblock.row = comm_.rank();
        localblock.col = 0;

        return costa::custom_layout<std::complex<T>>(comm_.size(), 1, rowsplit.data(), colsplit.data(),
                owners.data(), 1, &localblock, 'C');
    }

    inline auto checksum_mt(sddk::memory_t mem__, spin_index s__, band_range br__) const
    {
        std::complex<T> cs{0};
        if (is_host_memory(mem__)) {
            for (int ib = br__.begin(); ib < br__.end(); ib++) {
                auto ptr = this->data_[s__.get()].at(mem__, this->num_pw_, ib);
                cs = std::accumulate(ptr, ptr + this->num_mt_, cs);
            }
        }
        comm_.allreduce(&cs, 1);
        return cs;
    }

    auto num_mt_coeffs() const
    {
        return num_mt_coeffs_;
    }

    auto const& comm() const
    {
        return comm_;
    }
};

template <typename T>
class Wave_functions : public Wave_functions_mt<T>
{
  private:
    std::shared_ptr<sddk::Gvec> gkvec_;
  public:
    Wave_functions(std::shared_ptr<sddk::Gvec> gkvec__, num_mag_dims num_md__, num_bands num_wf__, sddk::memory_t default_mem__)
        : Wave_functions_mt<T>(gkvec__->comm(), num_md__, num_wf__, default_mem__, gkvec__->count())
        , gkvec_{gkvec__}
    {
    }

    Wave_functions(std::shared_ptr<sddk::Gvec> gkvec__, std::vector<int> num_mt_coeffs__, num_mag_dims num_md__,
            num_bands num_wf__, sddk::memory_t default_mem__)
        : Wave_functions_mt<T>(gkvec__->comm(), num_mt_coeffs__, num_md__, num_wf__, default_mem__, gkvec__->count())
        , gkvec_{gkvec__}
    {
    }

    inline auto& pw_coeffs(sddk::memory_t mem__, int ig__, spin_index ispn__, band_index i__)
    {
        return *this->data_[ispn__.get()].at(mem__, ig__, i__.get());
    }

    auto grid_layout_pw(spin_index ispn__, band_range b__)
    {
        std::vector<int> rowsplit(this->comm_.size() + 1);
        rowsplit[0] = 0;
        for (int i = 0; i < this->comm_.size(); i++) {
            rowsplit[i + 1] = rowsplit[i] + gkvec_->gvec_count(i);
        }
        std::vector<int> colsplit({0, b__.size()});
        std::vector<int> owners(this->comm_.size());
        for (int i = 0; i < this->comm_.size(); i++) {
            owners[i] = i;
        }
        costa::block_t localblock;
        localblock.data = this->data_[ispn__.get()].at(sddk::memory_t::host, 0, b__.begin());
        localblock.ld = this->ld();
        localblock.row = this->comm_.rank();
        localblock.col = 0;

        return costa::custom_layout<std::complex<T>>(this->comm_.size(), 1, rowsplit.data(), colsplit.data(),
                owners.data(), 1, &localblock, 'C');
    }

    auto const& gkvec() const
    {
        RTE_ASSERT(gkvec_ != nullptr);
        return *gkvec_;
    }

    inline auto checksum_pw(sddk::memory_t mem__, spin_index s__, band_range b__) const
    {
        std::complex<T> cs{0};
        if (is_host_memory(mem__)) {
            for (int ib = b__.begin(); ib < b__.end(); ib++) {
                auto ptr = this->data_[s__.get()].at(mem__, 0, ib);
                cs = std::accumulate(ptr, ptr + this->num_pw_, cs);
            }
        }
        this->comm_.allreduce(&cs, 1);
        return cs;
    }

    inline auto checksum(sddk::memory_t mem__, spin_index s__, band_range b__) const
    {
        return this->checksum_pw(mem__, s__, b__) + this->checksum_mt(mem__, s__, b__);
    }
};

template <typename T>
void transform_to_fft_layout(Wave_functions<T>& wf_in__, Wave_functions_fft<T>& wf_fft_out__,
        spin_index ispn__, band_range b__)
{
    auto layout_in  = wf_in__.grid_layout_pw(ispn__, b__);
    auto layout_out = wf_fft_out__.grid_layout(b__.size());

    costa::transform(layout_in, layout_out, 'N', sddk::linalg_const<std::complex<T>>::one(),
            sddk::linalg_const<std::complex<T>>::zero(), wf_in__.gkvec().comm().mpi_comm());
}

template <typename T>
void transform_from_fft_layout(Wave_functions_fft<T>& wf_fft_in__, Wave_functions<T>& wf_out__,
        spin_index ispn__, band_range b__)
{
    auto layout_in  = wf_fft_in__.grid_layout(b__.size());
    auto layout_out = wf_out__.grid_layout_pw(ispn__, b__);

    costa::transform(layout_in, layout_out, 'N', sddk::linalg_const<std::complex<T>>::one(),
            sddk::linalg_const<std::complex<T>>::zero(), wf_out__.gkvec().comm().mpi_comm());
}

template <typename T>
void check_wf_diff(std::string label__, sddk::Wave_functions<T>& wf_old__, wf::Wave_functions<T>& wf_new__)
{
    RTE_ASSERT(wf_old__.num_sc() == wf_new__.num_sc().get());

    double diff_g{0};
    double diff_mt{0};
    auto num_mt_coeffs = wf_new__.num_mt_coeffs();
    for (int is = 0; is < wf_old__.num_sc(); is++) {
        for (int i = 0; i < wf_old__.num_wf(); i++) {
            for (int ig = 0; ig < wf_old__.gkvec().count(); ig++) {
                diff_g += std::abs(wf_old__.pw_coeffs(is).prime(ig, i) -
                        wf_new__.pw_coeffs(sddk::memory_t::host, ig, wf::spin_index(is), wf::band_index(i)));
            }
            for (int ialoc = 0; ialoc < wf_old__.spl_num_atoms().local_size(); ialoc++) {
                int ia = wf_old__.spl_num_atoms()[ialoc];
                for (int xi = 0; xi < num_mt_coeffs[ia]; xi++) {
                    int j = wf_old__.offset_mt_coeffs(ialoc) + xi;
                    diff_mt += std::abs(wf_old__.mt_coeffs(is).prime(j, i) -
                        wf_new__.mt_coeffs(sddk::memory_t::host, xi, wf::atom_index(ialoc), wf::spin_index(is), wf::band_index(i)));
                }
            }
        }
    }
    if (diff_g + diff_mt > 1e-10) {
        std::stringstream s;
        s << label__ << ": wave functions are different: " << diff_g << " " << diff_mt;
        RTE_THROW(s);
    }
    std::cout << label__ << " OK" << std::endl;
}

template <typename T>
static std::vector<T>
inner_diag_local(sddk::memory_t mem__, wf::Wave_functions<T> const& lhs__, wf::Wave_functions_base<T> const& rhs__,
        wf::spin_range spins__, wf::num_bands num_wf__)
{
    RTE_ASSERT(lhs__.ld() == rhs__.ld());

    std::vector<T> result(num_wf__.get(), 0);

    if (is_host_memory(mem__)) {
        for (auto s = spins__.begin(); s != spins__.end(); s++) {
            for (int i = 0; i < num_wf__.get(); i++) {
                auto ptr1 = lhs__.data_ptr(sddk::memory_t::host, 0, s, wf::band_index(i));
                auto ptr2 = rhs__.data_ptr(sddk::memory_t::host, 0, s, wf::band_index(i));
                for (int j = 0; j < lhs__.ld(); j++) {
                    result[i] += std::real(std::conj(ptr1[j]) * ptr2[j]);
                }
            }
        }
    } else {
        RTE_THROW("implement inner_diag_local on GPUs");
    }
    return result;
}

//template <typename T>
//std::vector<T>
//inner_diag(sddk::memory_t mem__, wf::Wave_functions<T> const& lhs__, wf::Wave_functions_base<T> const& rhs__, wf::num_bands num_wf__)
//{
//    auto result = inner_diag_local(mem__, lhs__, rhs__, wf::spin_range(0), num_wf__);
//    lhs__.comm().allreduce(result);
//    return result;
//}

template <typename T>
std::vector<T>
inner_diag(sddk::memory_t mem__, wf::Wave_functions<T> const& lhs__, wf::Wave_functions_base<T> const& rhs__,
        wf::spin_range spins__, wf::num_bands num_wf__)
{
    auto result = inner_diag_local(mem__, lhs__, rhs__, spins__, num_wf__);
    lhs__.comm().allreduce(result);
    return result;
}

template <typename T>
void scale(sddk::memory_t mem__, wf::Wave_functions<T>& wf__, wf::spin_range spins__, wf::num_bands num_wf__, std::vector<T> scale__)
{
    if (is_host_memory(mem__)) {
        for (auto s = spins__.begin(); s != spins__.end(); s++) {
            for (int i = 0; i < num_wf__.get(); i++) {
                auto ptr = wf__.data_ptr(sddk::memory_t::host, 0, s, wf::band_index(i));
                for (int j = 0; j < wf__.ld(); j++) {
                    ptr[j] = ptr[j] * scale__[i];
                }
            }
        }
    } else {
        RTE_THROW("implement scale on GPUs");
    }
}

template <typename T>
void copy(Wave_functions<T> const& in__, wf::spin_index s_in__, wf::band_range br_in__,
          Wave_functions<T>& out__, wf::spin_index s_out__, wf::band_range br_out__)
{
    RTE_ASSERT(br_in__.size() == br_out__.size());
    RTE_ASSERT(in__.ld() == out__.ld());

    for (int i = 0; i < br_in__.size(); i++) {
        auto in_ptr = in__.data_ptr(sddk::memory_t::host, 0, s_in__, wf::band_index(br_in__.begin() + i));
        auto out_ptr = out__.data_ptr(sddk::memory_t::host, 0, s_out__, wf::band_index(br_out__.begin() + i));
        std::copy(in_ptr, in_ptr + in__.ld(), out_ptr);
    }
}

/// Apply linear transformation to the wave-functions.
/**
 * \tparam T Precision type of the wave-functions (float or double).
 * \tparam F Type of the subspace (float or double for Gamma-point calculation,
 *           complex<float> or complex<double> otherwise).
 */
template <typename T, typename F>
inline std::enable_if_t<std::is_same<T, real_type<F>>::value, void>
transform(::spla::Context& spla_ctx__, sddk::memory_t mem__, sddk::dmatrix<F> const& M__, int irow0__, int jcol0__,
        real_type<F> alpha__, Wave_functions<T> const& wf_in__, spin_index s_in__, band_range br_in__,
        real_type<F> beta__, Wave_functions<T>& wf_out__, spin_index s_out__, band_range br_out__)
{
    PROFILE("wf::transform");

    RTE_ASSERT(wf_in__.ld() == wf_out__.ld());

    /* spla manages the resources through the context which can be updated during the call;
     * that's why the const must be removed here */
    auto& spla_mat_dist = const_cast<sddk::dmatrix<F>&>(M__).spla_distribution();

    /* for Gamma point case (transformation matrix is real) we treat complex wave-function coefficients as
     * a doubled list of real values */
    int ld = wf_in__.ld();
    if (std::is_same<F, real_type<F>>::value) {
        ld *= 2;
    }

    F const* mtrx_ptr = M__.size_local() ? M__.at(sddk::memory_t::host, 0, 0) : nullptr;

    F const* in_ptr = reinterpret_cast<F const*>(wf_in__.data_ptr(mem__, 0, s_in__, band_index(br_in__.begin())));

    F* out_ptr = reinterpret_cast<F*>(wf_out__.data_ptr(mem__, 0, s_out__, band_index(br_out__.begin())));

    spla::pgemm_sbs(ld, br_out__.size(), br_in__.size(), alpha__, in_ptr, ld, mtrx_ptr, M__.ld(), irow0__, jcol0__,
            spla_mat_dist, beta__, out_ptr, ld, spla_ctx__);
}

template <typename T, typename F>
inline std::enable_if_t<!std::is_same<T, real_type<F>>::value, void>
transform(::spla::Context& spla_ctx__, sddk::memory_t mem__, sddk::dmatrix<F> const& M__, int irow0__, int jcol0__,
        real_type<F> alpha__, Wave_functions<T> const& wf_in__, spin_index s_in__, band_range br_in__,
        real_type<F> beta__, Wave_functions<T>& wf_out__, spin_index s_out__, band_range br_out__)
{
    RTE_THROW("implement this");
}

/// Compute inner product between the two sets of wave-functions.
/**
 * \tparam T Precision type of the wave-functions (float or double).
 * \tparam F Type of the subspace (float or double for Gamma-point calculation,
 *           complex<float> or complex<double> otherwise).
 *
 * \param [in] spla_ctx   Context of the SPLA library.
 * \param [in] mem        Location of the input wave-functions (host or device).
 * \param [in] spins      Spin range of the wave-functions.
 * \param [in] wf_i       Left hand side of <wf_i | wf_j> product.
 * \param [in] br_i       Band range of the <wf_i| wave-functions.
 * \param [in] wf_j       Right hand side of <wf_i | wf_j> product.
 * \param [in] br_j       Band range of the |wf_j> wave-functions.
 * \param [out] result    Resulting inner product matrix.
 * \param [in] irow0      Starting row of the output sub-block.
 * \param [in] jcol0      Starting column of the output sub-block.
 */
template <typename T, typename F>
inline std::enable_if_t<std::is_same<T, real_type<F>>::value, void>
inner(::spla::Context& spla_ctx__, sddk::memory_t mem__, spin_range spins__, Wave_functions<T> const& wf_i__,
        band_range br_i__, Wave_functions<T> const& wf_j__, band_range br_j__, sddk::dmatrix<F>& result__,
        int irow0__, int jcol0__)
{
    PROFILE("wf::inner");

    RTE_ASSERT(wf_i__.ld() == wf_j__.ld());
    RTE_ASSERT((wf_i__.gkvec().reduced() == std::is_same<F, real_type<F>>::value));
    RTE_ASSERT((wf_j__.gkvec().reduced() == std::is_same<F, real_type<F>>::value));

    if (spins__.size() == 2) {
        if (wf_i__.num_md() != wf::num_mag_dims(3)) {
            RTE_THROW("input wave-functions are not 2-component spinors");
        }
        if (wf_j__.num_md() != wf::num_mag_dims(3)) {
            RTE_THROW("input wave-functions are not 2-component spinors");
        }
    }

    spla::MatrixDistribution spla_mat_dist = wf_i__.comm().size() > result__.comm().size()
                                           ? spla::MatrixDistribution::create_mirror(wf_i__.comm().mpi_comm())
                                           : result__.spla_distribution();

    auto ld = wf_i__.ld();

    F alpha = 1.0;
    /* inner product matrix is real */
    if (std::is_same<F, real_type<F>>::value) {
        alpha = 2.0;
        ld   *= 2;
    }

    T scale_half(0.5);
    T scale_two(2.0);

    auto scale_gamma_wf = [&ld, &mem__, &spins__, &br_i__, &wf_i__](T* scale__)
    {
        RTE_ASSERT(spins__.size() == 1);

        auto& wf = const_cast<Wave_functions<T>&>(wf_i__);
        RTE_ASSERT(wf.num_sc() == wf::num_spins(1));

        auto sp = wf.actual_spin_index(spins__.begin());

        auto ptr = wf.data_ptr(mem__, 0, sp, wf::band_index(br_i__.begin()));
        auto m = br_i__.size();

        if (mem__ == sddk::memory_t::device) {
#if defined(SIRIUS_GPU)
            if (std::is_same<T, double>::value) {
                accblas::dscal(m, reinterpret_cast<double*>(scale__), reinterpret_cast<double*>(ptr), ld);
            } else if (std::is_same<T, float>::value) {
                accblas::sscal(m, reinterpret_cast<float*>(scale__), reinterpret_cast<float*>(ptr), ld);
            }
#else
            RTE_THROW("not compiled with GPU support!");
#endif
        } else {
            if (std::is_same<T, double>::value) {
                FORTRAN(dscal)(&m, reinterpret_cast<double*>(scale__), reinterpret_cast<double*>(ptr), &ld);
            } else if (std::is_same<T, float>::value) {
                FORTRAN(sscal)(&m, reinterpret_cast<float*>(scale__), reinterpret_cast<float*>(ptr), &ld);
            }
        }
    };

    /* for Gamma case, contribution of G = 0 vector must not be counted double -> multiply by 0.5 */
    if (std::is_same<F, real_type<F>>::value && wf_i__.comm().rank() == 0) {
        scale_gamma_wf(&scale_half);
    }

    F beta = 0.0;

    F* result_ptr = result__.size_local() ? result__.at(sddk::memory_t::host, 0, 0) : nullptr;

    for (auto s = spins__.begin(); s != spins__.end(); s++) {
        auto s_i = wf_i__.actual_spin_index(s);
        auto s_j = wf_j__.actual_spin_index(s);
        auto wf_i_ptr = wf_i__.data_ptr(mem__, 0, s_i, wf::band_index(br_i__.begin()));
        auto wf_j_ptr = wf_j__.data_ptr(mem__, 0, s_j, wf::band_index(br_j__.begin()));

        spla::pgemm_ssb(br_i__.size(), br_j__.size(), ld, SPLA_OP_CONJ_TRANSPOSE,
                        alpha,
                        reinterpret_cast<F const*>(wf_i_ptr), ld,
                        reinterpret_cast<F const*>(wf_j_ptr), ld,
                        beta,
                        result_ptr, result__.ld(), irow0__, jcol0__, spla_mat_dist, spla_ctx__);
        beta = 1.0;
    }

    /* for gamma case, G = 0 vector is rescaled back */
    if (std::is_same<F, real_type<F>>::value && wf_i__.comm().rank() == 0) {
        scale_gamma_wf(&scale_two);
    }

    /* make sure result is updated on device as well */
    if (result__.on_device()) {
        result__.copy_to(sddk::memory_t::device);
    }
}

template <typename T, typename F>
int
orthogonalize(::spla::Context& spla_ctx__, sddk::memory_t mem__, spin_range spins__, band_range br_old__,
        band_range br_new__, Wave_functions<T> const& wf_i__, Wave_functions<T> const& wf_j__,
        sddk::dmatrix<F>& o__, std::vector<Wave_functions<T>*> wfs__, Wave_functions<T>& tmp__, bool project_out__)
{
    PROFILE("wf::orthogonalize");

    /* number of new states */
    int n = br_new__.size();

//    //const char* sddk_pp_raw = std::getenv("SDDK_PRINT_PERFORMANCE");
//    //int sddk_pp             = (sddk_pp_raw == NULL) ? 0 : std::atoi(sddk_pp_raw);
//
//    auto& comm = wfs__[0]->comm();
//
//    int K{0};
//
//    if (sddk_pp) {
//        K = wfs__[0]->gkvec().num_gvec() + wfs__[0]->num_mt_coeffs();
//        if (std::is_same<T, real_type<T>>::value) {
//            K *= 2;
//        }
//    }
//
//    //auto sddk_debug_ptr = utils::get_env<int>("SDDK_DEBUG");
//    //int sddk_debug      = (sddk_debug_ptr) ? (*sddk_debug_ptr) : 0;
//
//    /* prefactor for the matrix multiplication in complex or double arithmetic (in Giga-operations) */
//    double ngop{8e-9}; // default value for complex type
//    if (std::is_same<T, real_type<T>>::value) { // change it if it is real type
//        ngop = 2e-9;
//    }
//
//    if (sddk_pp) {
//        comm.barrier();
//    }
//    // double time = -omp_get_wtime();
//
//    double gflops{0};
//
    /* project out the old subspace:
     * |\tilda phi_new> = |phi_new> - |phi_old><phi_old|phi_new> 
     * H|\tilda phi_new> = H|phi_new> - H|phi_old><phi_old|phi_new> 
     * S|\tilda phi_new> = S|phi_new> - S|phi_old><phi_old|phi_new> */
    if (br_old__.size() > 0 && project_out__) {
        inner(spla_ctx__, mem__, spins__, wf_i__, br_old__, wf_j__, br_new__, o__, 0, 0);
        for (auto s = spins__.begin(); s != spins__.end(); s++) {
            for (auto wf: wfs__) {
                auto sp = wf->actual_spin_index(s);
                transform(spla_ctx__, o__, 0, 0, -1.0, *wf, sp, br_old__, 1.0, *wf, sp, br_new__);
            }
        }
        //if (sddk_pp) {
        //    /* inner and transform have the same number of flops */
        //    gflops += static_cast<int>(1 + wfs__.size()) * ngop * N__ * n__ * K;
        //}
    }

//    if (sddk_debug >= 2) {
//        if (o__.comm().rank() == 0) {
//            RTE_OUT(std::cout) << "check QR decomposition, matrix size : " << n__ << std::endl;
//        }
//        inner(spla_ctx__, spins__, *wfs__[idx_bra__], N__, n__, *wfs__[idx_ket__], N__, n__, o__, 0, 0);
//
//        linalg(linalg_t::scalapack).geqrf(n__, n__, o__, 0, 0);
//        auto diag = o__.get_diag(n__);
//        if (o__.comm().rank() == 0) {
//            for (int i = 0; i < n__; i++) {
//                if (std::abs(diag[i]) < std::numeric_limits<real_type<T>>::epsilon() * 10) {
//                    RTE_OUT(std::cout) << "small norm: " << i << " " << diag[i] << std::endl;
//                }
//            }
//        }
//
//        if (o__.comm().rank() == 0) {
//            RTE_OUT(std::cout) << "check eigen-values, matrix size : " << n__ << std::endl;
//        }
//        inner(spla_ctx__, spins__, *wfs__[idx_bra__], N__, n__, *wfs__[idx_ket__], N__, n__, o__, 0, 0);
//
//        // if (sddk_debug >= 3) {
//        //    save_to_hdf5("nxn_overlap.h5", o__, n__);
//        //}
//
//        std::vector<real_type<F>> eo(n__);
//        dmatrix<F> evec(o__.num_rows(), o__.num_cols(), o__.blacs_grid(), o__.bs_row(), o__.bs_col());
//
//        auto solver = (o__.comm().size() == 1) ? Eigensolver_factory("lapack", nullptr) :
//                                                 Eigensolver_factory("scalapack", nullptr);
//        solver->solve(n__, o__, eo.data(), evec);
//
//        if (o__.comm().rank() == 0) {
//            for (int i = 0; i < n__; i++) {
//                if (eo[i] < 1e-6) {
//                    RTE_OUT(std::cout) << "small eigen-value " << i << " " << eo[i] << std::endl;
//                }
//            }
//        }
//    }
//
    /* orthogonalize new n x n block */
    inner(spla_ctx__, spins__, wf_i__, br_new__, wf_j__, br_new__, o__, 0, 0);

    // TODO: test magma and cuSolver
    sddk::linalg_t la = sddk::linalg_t::lapack;
    sddk::linalg_t la1 = sddk::linalg_t::blas;
    sddk::memory_t mem = sddk::memory_t::host;
    if (o__.comm().size() > 1) {
        la = sddk::linalg_t::scalapack;
    }

    /* compute the transformation matrix (inverse of the Cholesky factor) */
    PROFILE_START("wf::orthogonalize|tmtrx");
    auto o_ptr = (o__.size_local() == 0) ? nullptr : o__.at(mem);
    if (la == sddk::linalg_t::scalapack) {
        o__.make_real_diag(n);
    }
    /* Cholesky factorization */
    if (int info = sddk::linalg(la).potrf(n, o_ptr, o__.ld(), o__.descriptor())) {
        std::stringstream s;
        s << "error in Cholesky factorization, info = " << info << std::endl
          << "number of existing states: " << br_old__.size() << std::endl
          << "number of new states: " << br_new__.size();
        RTE_THROW(s);
    }
    /* inversion of triangular matrix */
    if (sddk::linalg(la).trtri(n, o_ptr, o__.ld(), o__.descriptor())) {
        RTE_THROW("error in inversion");
    }
    //if (is_device_memory(mem)) {
    //    acc::copyin(o__.at(memory_t::device), o__.ld(), o__.at(memory_t::host), o__.ld(), n, n);
    //}
    PROFILE_STOP("wf::orthogonalize|tmtrx");

    /* single MPI rank and precision types of wave-functions and transformation matrices match */
    if (o__.comm().size() == 1 && std::is_same<T, real_type<F>>::value) {
        PROFILE_START("wf::orthogonalize|trans");
        int sid{0};
        for (auto s = spins__.begin(); s != spins__.end(); s++) {
            /* multiplication by triangular matrix */
            for (auto& wf : wfs__) {
                auto sp = wf->actual_spin_index(s);
                auto ptr = reinterpret_cast<F*>(wf->data_ptr(mem__, 0, sp, wf::band_index(br_new__.begin())));
                int ld = wf->ld();
                /* Gamma-point case */
                if (std::is_same<F, real_type<F>>::value) {
                    ld *= 2;
                }

                sddk::linalg(la1).trmm('R', 'U', 'N', ld, n, &sddk::linalg_const<F>::one(),
                        o__.at(mem__), o__.ld(), ptr, ld, stream_id(sid++));
            }
        }
        if (la == sddk::linalg_t::gpublas || la == sddk::linalg_t::cublasxt || la == sddk::linalg_t::magma) {
            /* sync stream only if processing unit is GPU */
            for (int i = 0; i < sid; i++) {
                acc::sync_stream(stream_id(i));
            }
        }
        PROFILE_STOP("wf::orthogonalize|trans");
    } else {
        /* o is upper triangular matrix */
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                o__.set(j, i, 0);
            }
        }

        /* phi is transformed into phi, so we can't use it as the output buffer;
         * use tmp instead and then overwrite phi */
        for (auto s = spins__.begin(); s != spins__.end(); s++) {
            for (auto wf: wfs__) {
                auto sp = wf->actual_spin_index(s);
                auto sp1 = tmp__.actual_spin_index(s);
                auto br1 = wf::band_range(0, br_new__.size());
                transform(spla_ctx__, o__, 0, 0, 1.0, *wf, sp, br_new__, 0.0, tmp__, sp1, br1);
                copy(tmp__, sp1, br1, *wf, sp, br_new__);
            }
        }
    }
        //if (tmp__.num_wf() < br_new__.size()) {
        //    RTE_THROW("not enough workspace");
        //}

//== //
//== //    if (sddk_debug >= 1) {
//== //        //auto cs = o__.checksum(n__, n__);
//== //        //if (o__.comm().rank() == 0) {
//== //        //    //utils::print_checksum("n x n overlap", cs);
//== //        //}
//== //        if (o__.comm().rank() == 0) {
//== //            RTE_OUT(std::cout) << "check diagonal" << std::endl;
//== //        }
//== //        auto diag = o__.get_diag(n__);
//== //        for (int i = 0; i < n__; i++) {
//== //            if (std::real(diag[i]) <= 0 || std::imag(diag[i]) > 1e-12) {
//== //                RTE_OUT(std::cout) << "wrong diagonal: " << i << " " << diag[i] << std::endl;
//== //            }
//== //        }
//== //        if (o__.comm().rank() == 0) {
//== //            RTE_OUT(std::cout) << "check hermitian" << std::endl;
//== //        }
//== //        auto d = check_hermitian(o__, n__);
//== //        if (o__.comm().rank() == 0) {
//== //            if (d > 1e-12) {
//== //                std::stringstream s;
//== //                s << "matrix is not hermitian, max diff = " << d;
//== //                WARNING(s);
//== //            } else {
//== //                RTE_OUT(std::cout) << "OK! n x n overlap matrix is hermitian" << std::endl;
//== //            }
//== //        }
//== //
//== //    }
//== //
//== //    if (sddk_pp) {
//== //        gflops += ngop * n__ * n__ * K;
//== //    }
//== //
//==     /* single MPI rank and precision types of wave-functions and transformation matrices match */
//==     if (o__.comm().size() == 1 && std::is_same<T, real_type<F>>::value) {
//==         bool use_magma{false};
//== //
//== //        // MAGMA performance for Cholesky and inversion is not good enough; use lapack for the moment
//== //        //#if defined(SIRIUS_GPU) && defined(SIRIUS_MAGMA)
//== //        //        if (pu__ == GPU) {
//== //        //            use_magma = true;
//== //        //        }
//== //        //#endif
//== //
//==         PROFILE_START("wf::orthogonalize|trm");
//==         if (use_magma) {
//== //            /* Cholesky factorization */
//== //            if (int info = linalg(linalg_t::magma).potrf(n__, o__.at(memory_t::device), o__.ld())) {
//== //                std::stringstream s;
//== //                s << "error in GPU factorization, info = " << info;
//== //                RTE_THROW(s);
//== //            }
//== //            /* inversion of triangular matrix */
//== //            if (linalg(linalg_t::magma).trtri(n__, o__.at(memory_t::device), o__.ld())) {
//== //                RTE_THROW("error in inversion");
//== //            }
//==         } else { /* CPU version */
//==             /* Cholesky factorization */
//==             if (int info = sddk::linalg(sddk::linalg_t::lapack).potrf(br_new__.size(), &o__(0, 0), o__.ld())) {
//==                 std::stringstream s;
//==                 s << "error in Cholesky factorization, info = " << info << std::endl
//==                   << "number of existing states: " << br_old__.size() << std::endl
//==                   << "number of new states: " << br_new__.size();
//==                 RTE_THROW(s);
//==             }
//==             /* inversion of triangular matrix */
//==             if (sddk::linalg(sddk::linalg_t::lapack).trtri(br_new__.size(), &o__(0, 0), o__.ld())) {
//==                 RTE_THROW("error in inversion");
//==             }
//== //            if (is_device_memory(mem__)) {
//== //                acc::copyin(o__.at(memory_t::device), o__.ld(), o__.at(memory_t::host), o__.ld(), n__, n__);
//== //            }
//==         }
//==         PROFILE_STOP("wf::orthogonalize|trm");
//== 
//==         PROFILE_START("wf::orthogonalize|transform");
//==         int sid{0};
//==         for (auto s = spins__.begin(); s != spins__.end(); s++) {
//==             /* multiplication by triangular matrix */
//==             for (auto& wf : wfs__) {
//==                 auto sp = wf->actual_spin_index(s);
//==                 auto ptr = reinterpret_cast<F*>(wf->data_ptr(mem__, 0, sp, wf::band_index(br_new__.begin())));
//==                 int ld = wf->ld();
//==                 /* Gamma-point case */
//==                 if (std::is_same<F, real_type<F>>::value) {
//==                     ld *= 2;
//==                 }
//== 
//==                 sddk::linalg(la__).trmm('R', 'U', 'N', ld, br_new__.size(), &sddk::linalg_const<F>::one(),
//==                         o__.at(mem__), o__.ld(), ptr, ld, stream_id(sid++));
//== 
//==             }
//==         }
//==         if (la__ == sddk::linalg_t::gpublas || la__ == sddk::linalg_t::cublasxt || la__ == sddk::linalg_t::magma) {
//==             /* sync stream only if processing unit is GPU */
//==             for (int i = 0; i < sid; i++) {
//==                 acc::sync_stream(stream_id(i));
//==             }
//==         }
//==         PROFILE_STOP("wf::orthogonalize|transform");
//==     } else { /* parallel transformation */
//==         PROFILE_START("wf::orthogonalize|trm");
//== //        sddk::mdarray<F, 1> diag;
//==         o__.make_real_diag(br_new__.size());
//== //        if (sddk_debug >= 1) {
//== //            diag = o__.get_diag(n__);
//== //        }
//==         auto o_ptr = (o__.size_local() == 0) ? nullptr : o__.at(sddk::memory_t::host);
//== //        if (sddk_debug >= 2 && n__ <= 20) {
//== //            auto s1 = o__.serialize("wf_ortho:o_nn", n__, n__);
//== //            if (o__.comm().rank() == 0) {
//== //                std::cout << s1.str() << std::endl;
//== //            }
//== //        }
//==         if (int info = sddk::linalg(sddk::linalg_t::scalapack).potrf(br_new__.size(), o_ptr, o__.ld(), o__.descriptor())) {
//==             std::stringstream s;
//==             s << "error in Cholesky factorization, info = " << info << std::endl
//==               << "number of existing states: " << br_old__.size() << std::endl
//==               << "number of new states: " << br_new__.size();
//==             RTE_THROW(s);
//==         }
//== 
//==         if (sddk::linalg(sddk::linalg_t::scalapack).trtri(br_new__.size(), o_ptr, o__.ld(), o__.descriptor())) {
//==             RTE_THROW("error in inversion");
//==         }
//==         PROFILE_STOP("wf::orthogonalize|trm");
//== 
//==         /* o is upper triangular matrix */
//==         for (int i = 0; i < br_new__.size(); i++) {
//==             for (int j = i + 1; j < br_new__.size(); j++) {
//==                 o__.set(j, i, 0);
//==             }
//==         }
//== 
//==         //if (tmp__.num_wf() < br_new__.size()) {
//==         //    RTE_THROW("not enough workspace");
//==         //}
//== 
//==         /* phi is transformed into phi, so we can't use it as the output buffer; use tmp instead and then overwrite phi
//==          */
//==         for (auto s = spins__.begin(); s != spins__.end(); s++) {
//==             for (auto wf: wfs__) {
//==                 auto sp = wf->actual_spin_index(s);
//==                 auto sp1 = tmp__.actual_spin_index(s);
//==                 auto br1 = wf::band_range(0, br_new__.size());
//==                 transform(spla_ctx__, o__, 0, 0, 1.0, *wf, sp, br_new__, 0.0, tmp__, sp1, br1);
//==                 copy(tmp__, sp1, br1, *wf, sp, br_new__);
//==             }
//==         }
//==     }
//== //    if (sddk_debug >= 1) {
//== //        inner(spla_ctx__, spins__, *wfs__[idx_bra__], N__, n__, *wfs__[idx_ket__], N__, n__, o__, 0, 0);
//== //        auto err = check_identity(o__, n__);
//== //        if (o__.comm().rank() == 0) {
//== //            RTE_OUT(std::cout) << "orthogonalization error : " << err << std::endl;
//== //        }
//== //    }
//== //
//== //    // TODO: remove this?
//== //    (void) gflops;
//== //
    return 0;
}


} // namespace wf

#endif
