/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file wave_functions.hpp
 *
 *  \brief Contains declaration and implementation of Wave_functions class.
 */

#ifndef __WAVE_FUNCTIONS_HPP__
#define __WAVE_FUNCTIONS_HPP__

#include <costa/grid2grid/grid_layout.hpp>
#include <cstdlib>
#include <iostream>
#include <costa/layout.hpp>
#include <costa/grid2grid/transformer.hpp>
#include "core/la/linalg.hpp"
#include "core/strong_type.hpp"
#include "core/hdf5_tree.hpp"
#include "core/fft/gvec.hpp"
#include "core/env/env.hpp"
#include "core/rte/rte.hpp"
#include "core/time_tools.hpp"

namespace sirius {

#if defined(SIRIUS_GPU)
extern "C" {

void
add_square_sum_gpu_double(std::complex<double> const* wf__, int num_rows_loc__, int nwf__, int reduced__,
                          int mpi_rank__, double* result__);

void
add_square_sum_gpu_float(std::complex<float> const* wf__, int num_rows_loc__, int nwf__, int reduced__, int mpi_rank__,
                         float* result__);

void
scale_matrix_columns_gpu_double(int nrow__, int ncol__, std::complex<double>* mtrx__, double* a__);

void
scale_matrix_columns_gpu_float(int nrow__, int ncol__, std::complex<float>* mtrx__, float* a__);

void
add_checksum_gpu_double(void const* wf__, int ld__, int num_rows_loc__, int nwf__, void* result__);

void
add_checksum_gpu_float(void const* wf__, int ld__, int num_rows_loc__, int nwf__, void* result__);

void
inner_diag_local_gpu_double_complex_double(void const* wf1__, int ld1__, void const* wf2__, int ld2__, int ngv_loc__,
                                           int nwf__, void* result__);

void
inner_diag_local_gpu_double_double(void const* wf1__, int ld1__, void const* wf2__, int ld2__, int ngv_loc__, int nwf__,
                                   int reduced__, void* result__);

void
axpby_gpu_double_complex_double(int nwf__, void const* alpha__, void const* x__, int ld1__, void const* beta__,
                                void* y__, int ld2__, int ngv_loc__);

void
axpby_gpu_double_double(int nwf__, void const* alpha__, void const* x__, int ld1__, void const* beta__, void* y__,
                        int ld2__, int ngv_loc__);

void
axpy_scatter_gpu_double_complex_double(int nwf__, void const* alpha__, void const* x__, int ld1__, int const* idx__,
                                       void* y__, int ld2__, int ngv_loc__);

void
axpy_scatter_gpu_double_double(int nwf__, void const* alpha__, void const* x__, int ld1__, int const* idx__, void* y__,
                               int ld2__, int ngv_loc__);
}
#endif

/// Add checksum for the arrays on GPUs.
template <typename T>
auto
checksum_gpu(std::complex<T> const* wf__, int ld__, int num_rows_loc__, int nwf__)
{
    std::complex<T> cs{0};
#if defined(SIRIUS_GPU)
    mdarray<std::complex<T>, 1> cs1({nwf__}, mdarray_label("checksum"));
    cs1.allocate(memory_t::device).zero(memory_t::device);

    if (std::is_same<T, float>::value) {
        add_checksum_gpu_float(wf__, ld__, num_rows_loc__, nwf__, cs1.at(memory_t::device));
    } else if (std::is_same<T, double>::value) {
        add_checksum_gpu_double(wf__, ld__, num_rows_loc__, nwf__, cs1.at(memory_t::device));
    } else {
        std::stringstream s;
        s << "Precision type not yet implemented";
        RTE_THROW(s);
    }
    cs1.copy_to(memory_t::host);
    cs = cs1.checksum();
#endif
    return cs;
}

/// Namespace for the wave-functions.
namespace wf {

using spin_index = strong_type<int, struct __spin_index_tag>;
// using atom_index = strong_type<int, struct __atom_index_tag>;
using band_index = strong_type<int, struct __band_index_tag>;

using num_bands    = strong_type<int, struct __num_bands_tag>;
using num_spins    = strong_type<int, struct __num_spins_tag>;
using num_mag_dims = strong_type<int, struct __num_mag_dims_tag>;

/// Describe a range of bands.
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
        RTE_ASSERT(size__ >= 0);
    }
    inline auto
    begin() const
    {
        return begin_;
    }
    inline auto
    end() const
    {
        return end_;
    }
    inline auto
    size() const
    {
        return end_ - begin_;
    }
};

/// Describe a range of spins.
/** Only 3 combinations of spin range are allowed:
    - [0, 1)
    - [1, 2)
    - [0, 2)
*/
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
    inline auto
    begin() const
    {
        return spin_index(begin_);
    }
    inline auto
    end() const
    {
        return spin_index(end_);
    }
    inline int
    size() const
    {
        return end_ - begin_;
    }
    inline int
    spinor_index() const
    {
        return spinor_index_;
    }
};

enum class copy_to : unsigned int
{
    none   = 0b0000,
    device = 0b0001,
    host   = 0b0010
};
inline copy_to
operator|(copy_to a__, copy_to b__)
{
    return static_cast<copy_to>(static_cast<unsigned int>(a__) | static_cast<unsigned int>(b__));
}

/// Helper class to allocate and copy wave-functions to/from device.
class device_memory_guard
{
  private:
    void* obj_{nullptr};
    memory_t mem_{memory_t::host};
    copy_to copy_to_{wf::copy_to::none};
    std::function<void(void*, memory_t, wf::copy_to)> handler_;

    device_memory_guard(device_memory_guard const&) = delete;
    device_memory_guard&
    operator=(device_memory_guard const&) = delete;

  public:
    device_memory_guard()
    {
    }

    template <typename T>
    device_memory_guard(T const& obj__, memory_t mem__, copy_to copy_to__)
        : obj_{const_cast<T*>(&obj__)}
        , mem_{mem__}
        , copy_to_{copy_to__}
    {
        if (is_device_memory(mem_)) {
            auto obj = static_cast<T*>(obj_);
            obj->allocate(mem_);
            if (static_cast<unsigned int>(copy_to_) & static_cast<unsigned int>(copy_to::device)) {
                obj->copy_to(mem_);
            }
        }
        handler_ = [](void* p__, memory_t mem__, wf::copy_to copy_to__) {
            if (p__) {
                auto obj = static_cast<T*>(p__);
                if (is_device_memory(mem__)) {
                    if (static_cast<unsigned int>(copy_to__) & static_cast<unsigned int>(copy_to::host)) {
                        obj->copy_to(memory_t::host);
                    }
                    obj->deallocate(mem__);
                }
            }
        };
    }
    device_memory_guard(device_memory_guard&& src__)
    {
        this->obj_     = src__.obj_;
        src__.obj_     = nullptr;
        this->handler_ = src__.handler_;
        this->mem_     = src__.mem_;
        this->copy_to_ = src__.copy_to_;
    }
    device_memory_guard&
    operator=(device_memory_guard&& src__)
    {
        if (this != &src__) {
            this->obj_     = src__.obj_;
            src__.obj_     = nullptr;
            this->handler_ = src__.handler_;
            this->mem_     = src__.mem_;
            this->copy_to_ = src__.copy_to_;
        }
        return *this;
    }

    ~device_memory_guard()
    {
        handler_(obj_, mem_, copy_to_);
    }
};

/* forward declaration */
template <typename T>
class Wave_functions_fft;

/// Base class for the wave-functions.
/** Wave-functions are represented by a set of plane-wave and muffin-tin coefficients stored consecutively in a 2D
 array.
 *  The leading dimensions of this array is a sum of the number of plane-waves and the number of muffin-tin
 coefficients. \verbatim

         band index
       +-----------+
       |           |
       |           |
    ig |  PW part  |
       |           |
       |           |
       +-----------+
       |           |
    xi |  MT part  |
       |           |
       +-----------+


    \endverbatim
  */
template <typename T>
class Wave_functions_base
{
  protected:
    /// Local number of plane-wave coefficients.
    int num_pw_{0};
    /// Local number of muffin-tin coefficients.
    int num_mt_{0};
    /// Number of magnetic dimensions (0, 1, or 3).
    /** This helps to distinguish between non-magnetic, collinear and full spinor wave-functions. */
    num_mag_dims num_md_{0};
    /// Total number of wave-functions.
    num_bands num_wf_{0};
    /// Number of spin components (1 or 2).
    num_spins num_sc_{0};
    /// Friend class declaration.
    /** Wave_functions_fft needs access to data to alias the pointers and avoid copy in trivial cases. */
    friend class Wave_functions_fft<T>;
    /// Data storage for the wave-functions.
    /** Wave-functions are stored as two independent arrays for spin-up and spin-dn. The planewave and muffin-tin
        coefficients are stored consecutively. */
    std::array<mdarray<std::complex<T>, 2>, 2> data_;

  public:
    /// Constructor.
    Wave_functions_base()
    {
    }
    /// Constructor.
    Wave_functions_base(int num_pw__, int num_mt__, num_mag_dims num_md__, num_bands num_wf__, memory_t default_mem__)
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
            data_[is] = mdarray<std::complex<T>, 2>({num_pw_ + num_mt_, num_wf_.get()}, get_memory_pool(default_mem__),
                                                    mdarray_label("Wave_functions_base::data_"));
        }
    }

    /// Return an instance of the memory guard.
    /** When the instance is created, it allocates the GPU memory and optionally copies data to the GPU. When the
        instance is destroyed, the data is optionally copied to host and GPU memory is deallocated. */
    auto
    memory_guard(memory_t mem__, wf::copy_to copy_to__ = copy_to::none) const
    {
        return device_memory_guard(*this, mem__, copy_to__);
    }

    /// Return number of spin components.
    inline auto
    num_sc() const
    {
        return num_sc_;
    }

    /// Return number of magnetic dimensions.
    inline auto
    num_md() const
    {
        return num_md_;
    }

    /// Return number of wave-functions.
    inline auto
    num_wf() const
    {
        return num_wf_;
    }

    /// Return leading dimensions of the wave-functions coefficients array.
    inline auto
    ld() const
    {
        return num_pw_ + num_mt_;
    }

    /// Return the actual spin index of the wave-functions.
    /** Return 0 if the wave-functions are non-magnetic, otherwise return the input spin index. */
    inline auto
    actual_spin_index(spin_index s__) const
    {
        if (num_sc_.get() == 2) {
            return s__;
        } else {
            return spin_index(0);
        }
    }

    /// Zero a spin component of the wave-functions in a band range.
    inline void
    zero(memory_t mem__, spin_index s__, band_range br__)
    {
        if (this->ld()) {
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
    }

    /// Zero all wave-functions.
    inline void
    zero(memory_t mem__)
    {
        if (this->ld()) {
            for (int is = 0; is < num_sc_.get(); is++) {
                data_[is].zero(mem__);
            }
        }
    }

    /// Return const pointer to the wave-function coefficient at a given index, spin and band
    inline std::complex<T> const*
    at(memory_t mem__, int i__, spin_index s__, band_index b__) const
    {
        return data_[s__.get()].at(mem__, i__, b__.get());
    }

    /// Return pointer to the wave-function coefficient at a given index, spin and band
    inline auto
    at(memory_t mem__, int i__, spin_index s__, band_index b__)
    {
        return data_[s__.get()].at(mem__, i__, b__.get());
    }

    /// Allocate wave-functions.
    /** This function is primarily called by a memory_guard to allocate GPU memory. */
    inline void
    allocate(memory_t mem__)
    {
        for (int s = 0; s < num_sc_.get(); s++) {
            data_[s].allocate(get_memory_pool(mem__));
        }
    }

    /// Deallocate wave-functions.
    /** This function is primarily called by a memory_guard to deallocate GPU memory. */
    inline void
    deallocate(memory_t mem__)
    {
        for (int s = 0; s < num_sc_.get(); s++) {
            data_[s].deallocate(mem__);
        }
    }

    /// Copy date to host or device.
    inline void
    copy_to(memory_t mem__)
    {
        for (int s = 0; s < num_sc_.get(); s++) {
            data_[s].copy_to(mem__);
        }
    }
};

/// Wave-functions for the muffin-tin part of LAPW.
template <typename T>
class Wave_functions_mt : public Wave_functions_base<T>
{
  protected:
    /// Communicator that is used to split atoms between MPI ranks.
    mpi::Communicator const& comm_;
    /// Total number of atoms.
    int num_atoms_{0};
    /// Distribution of atoms between MPI ranks.
    splindex_block<atom_index_t> spl_num_atoms_;
    /// Local size of muffin-tin coefficients for each rank.
    /** Each rank stores local fraction of atoms. Each atom has a set of MT coefficients. */
    mpi::block_data_descriptor mt_coeffs_distr_;
    /// Local offset in the block of MT coefficients for current rank.
    /** The size of the vector is equal to the local number of atoms for the current rank. */
    std::vector<int> offset_in_local_mt_coeffs_;
    /// Numbef of muffin-tin coefficients for each atom.
    std::vector<int> num_mt_coeffs_;

    /// Calculate the local number of muffin-tin coefficients.
    /** Compute the local fraction of atoms and then sum the muffin-tin coefficients for this fraction. */
    static int
    get_local_num_mt_coeffs(std::vector<int> num_mt_coeffs__, mpi::Communicator const& comm__)
    {
        int num_atoms = static_cast<int>(num_mt_coeffs__.size());
        splindex_block<atom_index_t> spl_atoms(num_atoms, n_blocks(comm__.size()), block_id(comm__.rank()));
        int result{0};
        for (auto it : spl_atoms) {
            result += num_mt_coeffs__[it.i];
        }
        return result;
    }

    /// Construct without muffin-tin part.
    Wave_functions_mt(mpi::Communicator const& comm__, num_mag_dims num_md__, num_bands num_wf__,
                      memory_t default_mem__, int num_pw__)
        : Wave_functions_base<T>(num_pw__, 0, num_md__, num_wf__, default_mem__)
        , comm_{comm__}
        , spl_num_atoms_{splindex_block<atom_index_t>(num_atoms_, n_blocks(comm_.size()), block_id(comm_.rank()))}
    {
    }

  public:
    /// Constructor.
    Wave_functions_mt()
    {
    }

    /// Constructor.
    Wave_functions_mt(mpi::Communicator const& comm__, std::vector<int> num_mt_coeffs__, num_mag_dims num_md__,
                      num_bands num_wf__, memory_t default_mem__, int num_pw__ = 0)
        : Wave_functions_base<T>(num_pw__, get_local_num_mt_coeffs(num_mt_coeffs__, comm__), num_md__, num_wf__,
                                 default_mem__)
        , comm_{comm__}
        , num_atoms_{static_cast<int>(num_mt_coeffs__.size())}
        , spl_num_atoms_{splindex_block<atom_index_t>(num_atoms_, n_blocks(comm_.size()), block_id(comm_.rank()))}
        , num_mt_coeffs_{num_mt_coeffs__}
    {
        mt_coeffs_distr_ = mpi::block_data_descriptor(comm_.size());

        for (int ia = 0; ia < num_atoms_; ia++) {
            auto rank = spl_num_atoms_.location(atom_index_t::global(ia)).ib;
            if (rank == comm_.rank()) {
                offset_in_local_mt_coeffs_.push_back(mt_coeffs_distr_.counts[rank]);
            }
            /* increment local number of MT coeffs. for a given rank */
            mt_coeffs_distr_.counts[rank] += num_mt_coeffs__[ia];
        }
        mt_coeffs_distr_.calc_offsets();
    }

    /// Return reference to the coefficient by atomic orbital index, atom, spin and band indices.
    inline auto&
    mt_coeffs(int xi__, atom_index_t::local ia__, spin_index ispn__, band_index i__)
    {
        return this->data_[ispn__.get()](this->num_pw_ + xi__ + offset_in_local_mt_coeffs_[ia__.get()], i__.get());
    }

    /// Return const reference to the coefficient by atomic orbital index, atom, spin and band indices.
    inline auto const&
    mt_coeffs(int xi__, atom_index_t::local ia__, spin_index ispn__, band_index i__) const
    {
        return this->data_[ispn__.get()](this->num_pw_ + xi__ + offset_in_local_mt_coeffs_[ia__.get()], i__.get());
    }

    using Wave_functions_base<T>::at;

    /// Return const pointer to the coefficient by atomic orbital index, atom, spin and band indices.
    inline std::complex<T> const*
    at(memory_t mem__, int xi__, atom_index_t::local ia__, spin_index s__, band_index b__) const
    {
        return this->data_[s__.get()].at(mem__, this->num_pw_ + xi__ + offset_in_local_mt_coeffs_[ia__.get()],
                                         b__.get());
    }

    /// Return pointer to the coefficient by atomic orbital index, atom, spin and band indices.
    inline auto
    at(memory_t mem__, int xi__, atom_index_t::local ia__, spin_index s__, band_index b__)
    {
        return this->data_[s__.get()].at(mem__, this->num_pw_ + xi__ + offset_in_local_mt_coeffs_[ia__.get()],
                                         b__.get());
    }

    /// Return a split index for the number of atoms.
    inline auto const&
    spl_num_atoms() const
    {
        return spl_num_atoms_;
    }

    /// Copy muffin-tin coefficients to host or GPU memory.
    /** This functionality is required for the application of LAPW overlap operator to the wave-functions, which
     *  is always done on the CPU. */
    inline void
    copy_mt_to(memory_t mem__, spin_index s__, band_range br__)
    {
        if (this->ld() && this->num_mt_) {
            auto ptr     = this->data_[s__.get()].at(memory_t::host, this->num_pw_, br__.begin());
            auto ptr_gpu = this->data_[s__.get()].at(memory_t::device, this->num_pw_, br__.begin());
            if (is_device_memory(mem__)) {
                acc::copyin(ptr_gpu, this->ld(), ptr, this->ld(), this->num_mt_, br__.size());
            }
            if (is_host_memory(mem__)) {
                acc::copyout(ptr, this->ld(), ptr_gpu, this->ld(), this->num_mt_, br__.size());
            }
        }
    }

    /// Return COSTA layout for the muffin-tin part for a given spin index and band range.
    auto
    grid_layout_mt(spin_index ispn__, band_range b__) -> costa::grid_layout<std::complex<T>>
    {
        std::vector<int> rowsplit({0});
        std::vector<int> owners;
        for (int i = 0; i < comm_.size(); i++) {
            if (mt_coeffs_distr_.counts[i] > 0) {
                rowsplit.push_back(rowsplit.back() + mt_coeffs_distr_.counts[i]);
                owners.push_back(i);
            }
        }
        std::vector<int> colsplit({0, b__.size()});
        costa::block_t localblock;
        localblock.data =
                this->num_mt_ ? this->data_[ispn__.get()].at(memory_t::host, this->num_pw_, b__.begin()) : nullptr;
        localblock.ld  = this->ld();
        localblock.row = comm_.rank();
        localblock.col = 0;

        int nlocal_blocks = this->num_mt_ ? 1 : 0;

        return costa::custom_layout<std::complex<T>>(rowsplit.size() - 1, 1, rowsplit.data(), colsplit.data(),
                                                     owners.data(), nlocal_blocks, &localblock, 'C');
    }

    /// Compute checksum of the muffin-tin coefficients.
    inline auto
    checksum_mt(memory_t mem__, spin_index s__, band_range br__) const
    {
        std::complex<T> cs{0};
        if (this->num_mt_ && br__.size()) {
            if (is_host_memory(mem__)) {
                for (int ib = br__.begin(); ib < br__.end(); ib++) {
                    auto ptr = this->data_[s__.get()].at(mem__, this->num_pw_, ib);
                    cs       = std::accumulate(ptr, ptr + this->num_mt_, cs);
                }
            }
            if (is_device_memory(mem__)) {
                auto ptr = this->data_[s__.get()].at(mem__, this->num_pw_, br__.begin());
                cs       = checksum_gpu<T>(ptr, this->ld(), this->num_mt_, br__.size());
            }
        }
        comm_.allreduce(&cs, 1);
        return cs;
    }

    /// Return vector of muffin-tin coefficients for all atoms.
    auto
    num_mt_coeffs() const
    {
        return num_mt_coeffs_;
    }

    /// Return const reference to the communicator.
    auto const&
    comm() const
    {
        return comm_;
    }

    auto const&
    mt_coeffs_distr() const
    {
        return mt_coeffs_distr_;
    }
};

/// Wave-functions representation.
/** Wave-functions consist of two parts: plane-wave part and mufin-tin part. Wave-functions have one or two spin
 *  components. In case of collinear magnetism each component represents a pure (up- or dn-) spinor state and they
 *  are independent. In non-collinear case the two components represent a full spinor state.
 *
 *  \tparam T  Precision type of the wave-functions (double or float).
 */
template <typename T>
class Wave_functions : public Wave_functions_mt<T>
{
  private:
    /// Pointer to G+k- vectors object.
    std::shared_ptr<fft::Gvec> gkvec_;

  public:
    /// Constructor for pure plane-wave functions.
    Wave_functions(std::shared_ptr<fft::Gvec> gkvec__, num_mag_dims num_md__, num_bands num_wf__,
                   memory_t default_mem__)
        : Wave_functions_mt<T>(gkvec__->comm(), num_md__, num_wf__, default_mem__, gkvec__->count())
        , gkvec_{gkvec__}
    {
    }

    /// Constructor for wave-functions with plane-wave and muffin-tin parts (LAPW case).
    Wave_functions(std::shared_ptr<fft::Gvec> gkvec__, std::vector<int> num_mt_coeffs__, num_mag_dims num_md__,
                   num_bands num_wf__, memory_t default_mem__)
        : Wave_functions_mt<T>(gkvec__->comm(), num_mt_coeffs__, num_md__, num_wf__, default_mem__, gkvec__->count())
        , gkvec_{gkvec__}
    {
    }

    /// Return reference to the plane-wave coefficient for a given plane-wave, spin and band indices.
    inline auto&
    pw_coeffs(int ig__, spin_index ispn__, band_index i__)
    {
        return this->data_[ispn__.get()](ig__, i__.get());
    }

    inline auto&
    pw_coeffs(spin_index ispn__)
    {
        return this->data_[ispn__.get()];
    }

    inline const auto&
    pw_coeffs(spin_index ispn__) const
    {
        return this->data_[ispn__.get()];
    }

    /// Return COSTA layout for the plane-wave part for a given spin index and band range.
    auto
    grid_layout_pw(spin_index ispn__, band_range b__) const
    {
        PROFILE("sirius::wf::Wave_functions_fft::grid_layout_pw");

        std::vector<int> rowsplit(this->comm_.size() + 1);
        rowsplit[0] = 0;
        for (int i = 0; i < this->comm_.size(); i++) {
            rowsplit[i + 1] = rowsplit[i] + gkvec_->count(i);
        }
        std::vector<int> colsplit({0, b__.size()});
        std::vector<int> owners(this->comm_.size());
        for (int i = 0; i < this->comm_.size(); i++) {
            owners[i] = i;
        }
        costa::block_t localblock;
        localblock.data = const_cast<std::complex<T>*>(this->data_[ispn__.get()].at(memory_t::host, 0, b__.begin()));
        localblock.ld   = this->ld();
        localblock.row  = this->comm_.rank();
        localblock.col  = 0;

        return costa::custom_layout<std::complex<T>>(this->comm_.size(), 1, rowsplit.data(), colsplit.data(),
                                                     owners.data(), 1, &localblock, 'C');
    }

    auto const&
    gkvec() const
    {
        RTE_ASSERT(gkvec_ != nullptr);
        return *gkvec_;
    }

    auto
    gkvec_sptr() const
    {
        return gkvec_;
    }

    inline auto
    checksum_pw(memory_t mem__, spin_index s__, band_range b__) const
    {
        std::complex<T> cs{0};
        if (b__.size()) {
            if (is_host_memory(mem__)) {
                for (int ib = b__.begin(); ib < b__.end(); ib++) {
                    auto ptr = this->data_[s__.get()].at(mem__, 0, ib);
                    cs       = std::accumulate(ptr, ptr + this->num_pw_, cs);
                }
            }
            if (is_device_memory(mem__)) {
                auto ptr = this->data_[s__.get()].at(mem__, 0, b__.begin());
                cs       = checksum_gpu<T>(ptr, this->ld(), this->num_pw_, b__.size());
            }
            this->comm_.allreduce(&cs, 1);
        }
        return cs;
    }

    inline auto
    checksum(memory_t mem__, spin_index s__, band_range b__) const
    {
        return this->checksum_pw(mem__, s__, b__) + this->checksum_mt(mem__, s__, b__);
    }

    inline auto
    checksum(memory_t mem__, band_range b__) const
    {
        std::complex<T> cs{0};
        for (int is = 0; is < this->num_sc().get(); is++) {
            cs += this->checksum(mem__, wf::spin_index(is), b__);
        }
        return cs;
    }
};

struct shuffle_to
{
    /// Do nothing.
    static const unsigned int none = 0b0000;
    /// Shuffle to FFT distribution.
    static const unsigned int fft_layout = 0b0001;
    /// Shuffle to back to default slab distribution.
    static const unsigned int wf_layout = 0b0010;
};

/// Wave-fucntions in the FFT-friendly distribution.
/** To reduce the FFT MPI communication volume, it is often beneficial to redistribute wave-functions from
 *  a default slab layout to a FFT-friendly layout. Often this is a full swap from G-vector to band distribution.
 *  In general this is a redistribution of data from [N x 1] to [M x K] MPI grids.
   \verbatim
                  band index               band index               band index
               ┌──────────────┐          ┌───────┬──────┐          ┌───┬───┬───┬──┐
               │              │          │       │      │          │   │   │   │  │
               │              │          │       │      │          │   │   │   │  │
               ├──────────────┤          │       │      │          │   │   │   │  │
               │              │          │       │      │          │   │   │   │  │
               │              │ partial  │       │      │   full   │   │   │   │  │
               │              │  swap    │       │      │   swap   │   │   │   │  │
   G+k index   ├──────────────┤    ->    ├───────┼──────┤    ->    ├───┼───┼───┼──┤
 (distributed) │              │          │       │      │          │   │   │   │  │
               │              │          │       │      │          │   │   │   │  │
               │              │          │       │      │          │   │   │   │  │
               ├──────────────┤          │       │      │          │   │   │   │  │
               │              │          │       │      │          │   │   │   │  │
               │              │          │       │      │          │   │   │   │  │
               └──────────────┘          └───────┴──────┘          └───┴───┴───┴──┘

    \endverbatim

    Wave-functions in FFT distribution are scalar with only one spin component.

 *  \tparam T  Precision type of the wave-functions (double or float).
 */
template <typename T>
class Wave_functions_fft : public Wave_functions_base<T>
{
  private:
    /// Pointer to FFT-friendly G+k vector deistribution.
    std::shared_ptr<fft::Gvec_fft> gkvec_fft_;
    /// Split number of wave-functions between column communicator.
    splindex_block<> spl_num_wf_;
    /// Pointer to the original wave-functions.
    Wave_functions<T>* wf_{nullptr};
    /// Spin-index of the wave-function component
    spin_index s_{0};
    /// Range of bands in the input wave-functions to be swapped.
    band_range br_{0};
    /// Direction of the reshuffling: to FFT layout or back to WF layout or both.
    unsigned int shuffle_flag_{0};
    /// True if the FFT wave-functions are also available on the device.
    bool on_device_{false};

    /// Return COSTA grd layout description.
    auto
    grid_layout(int n__)
    {
        PROFILE("sirius::wf::Wave_functions_fft::grid_layout");

        auto& comm_row = gkvec_fft_->comm_fft();
        auto& comm_col = gkvec_fft_->comm_ortho_fft();

        std::vector<int> rowsplit(comm_row.size() + 1);
        rowsplit[0] = 0;
        for (int i = 0; i < comm_row.size(); i++) {
            rowsplit[i + 1] = rowsplit[i] + gkvec_fft_->count(i);
        }

        std::vector<int> colsplit(comm_col.size() + 1);
        colsplit[0] = 0;
        for (int i = 0; i < comm_col.size(); i++) {
            colsplit[i + 1] = colsplit[i] + spl_num_wf_.local_size(block_id(i));
        }

        std::vector<int> owners(gkvec_fft_->gvec().comm().size());
        for (int i = 0; i < gkvec_fft_->gvec().comm().size(); i++) {
            owners[i] = i;
        }
        costa::block_t localblock;
        localblock.data = this->data_[0].at(memory_t::host);
        localblock.ld   = this->ld();
        localblock.row  = gkvec_fft_->comm_fft().rank();
        localblock.col  = comm_col.rank();

        return costa::custom_layout<std::complex<T>>(comm_row.size(), comm_col.size(), rowsplit.data(), colsplit.data(),
                                                     owners.data(), 1, &localblock, 'C');
    }

    /// Shuffle wave-function to the FFT distribution.
    void
    shuffle_to_fft_layout(spin_index ispn__, band_range b__)
    {
        PROFILE("shuffle_to_fft_layout");

        auto sp = wf_->actual_spin_index(ispn__);
        auto t0 = ::sirius::time_now();
        if (false) {
            auto layout_in  = wf_->grid_layout_pw(sp, b__);
            auto layout_out = this->grid_layout(b__.size());

            costa::transform(layout_in, layout_out, 'N', la::constant<std::complex<T>>::one(),
                             la::constant<std::complex<T>>::zero(), wf_->gkvec().comm().native());
        } else {
            /*
             * old implementation (to be removed when performance of COSTA is understood)
             */
            auto& comm_col = gkvec_fft_->comm_ortho_fft();

            /* in full-potential case leading dimenstion is larger than the number of plane-wave
             * coefficients, so we have to copy data into temporary storage with necessary leading
             * dimension */
            mdarray<std::complex<T>, 2> wf_tmp;
            if (wf_->ld() == wf_->num_pw_) { /* pure plane-wave coeffs */
                auto ptr = (wf_->num_pw_ == 0) ? nullptr : wf_->data_[sp.get()].at(memory_t::host, 0, b__.begin());
                wf_tmp   = mdarray<std::complex<T>, 2>({wf_->num_pw_, b__.size()}, ptr);
            } else {
                wf_tmp = mdarray<std::complex<T>, 2>({wf_->num_pw_, b__.size()}, get_memory_pool(memory_t::host));
                for (int i = 0; i < b__.size(); i++) {
                    auto in_ptr = wf_->data_[sp.get()].at(memory_t::host, 0, b__.begin() + i);
                    std::copy(in_ptr, in_ptr + wf_->num_pw_, wf_tmp.at(memory_t::host, 0, i));
                }
            }

            auto* send_buf = (wf_tmp.ld() == 0) ? nullptr : wf_tmp.at(memory_t::host);

            /* local number of columns */
            int n_loc = spl_num_wf_.local_size();

            mdarray<std::complex<T>, 1> recv_buf({gkvec_fft_->count() * n_loc}, get_memory_pool(memory_t::host),
                                                 mdarray_label("recv_buf"));

            auto& row_distr = gkvec_fft_->gvec_slab();

            /* send and receive dimensions */
            mpi::block_data_descriptor sd(comm_col.size()), rd(comm_col.size());
            for (int j = 0; j < comm_col.size(); j++) {
                sd.counts[j] = spl_num_wf_.local_size(block_id(j)) * row_distr.counts[comm_col.rank()];
                rd.counts[j] = spl_num_wf_.local_size(block_id(comm_col.rank())) * row_distr.counts[j];
            }
            sd.calc_offsets();
            rd.calc_offsets();

            comm_col.alltoall(send_buf, sd.counts.data(), sd.offsets.data(), recv_buf.at(memory_t::host),
                              rd.counts.data(), rd.offsets.data());

            /* reorder received blocks */
            #pragma omp parallel for
            for (int i = 0; i < n_loc; i++) {
                for (int j = 0; j < comm_col.size(); j++) {
                    int offset = row_distr.offsets[j];
                    int count  = row_distr.counts[j];
                    if (count) {
                        auto from = &recv_buf[offset * n_loc + count * i];
                        std::copy(from, from + count, this->data_[0].at(memory_t::host, offset, i));
                    }
                }
            }
        }

        if (env::print_performance() && wf_->gkvec().comm().rank() == 0) {
            auto t = ::sirius::time_interval(t0);
            std::cout << "[transform_to_fft_layout] throughput: "
                      << 2 * sizeof(T) * wf_->gkvec().num_gvec() * b__.size() / std::pow(2.0, 30) / t << " Gb/sec"
                      << std::endl;
        }
    }

    /// Shuffle wave-function to the original slab layout.
    void
    shuffle_to_wf_layout(spin_index ispn__, band_range b__)
    {
        PROFILE("shuffle_to_wf_layout");

        auto sp = wf_->actual_spin_index(ispn__);
        auto pp = env::print_performance();

        auto t0 = ::sirius::time_now();
        if (false) {
            auto layout_in  = this->grid_layout(b__.size());
            auto layout_out = wf_->grid_layout_pw(sp, b__);

            costa::transform(layout_in, layout_out, 'N', la::constant<std::complex<T>>::one(),
                             la::constant<std::complex<T>>::zero(), wf_->gkvec().comm().native());
        } else {

            auto& comm_col = gkvec_fft_->comm_ortho_fft();

            /* local number of columns */
            int n_loc = spl_num_wf_.local_size();

            /* send buffer */
            mdarray<std::complex<T>, 1> send_buf({gkvec_fft_->count() * n_loc}, get_memory_pool(memory_t::host),
                                                 mdarray_label("send_buf"));

            auto& row_distr = gkvec_fft_->gvec_slab();

            /* reorder sending blocks */
            #pragma omp parallel for
            for (int i = 0; i < n_loc; i++) {
                for (int j = 0; j < comm_col.size(); j++) {
                    int offset = row_distr.offsets[j];
                    int count  = row_distr.counts[j];
                    if (count) {
                        auto from = this->data_[0].at(memory_t::host, offset, i);
                        std::copy(from, from + count, &send_buf[offset * n_loc + count * i]);
                    }
                }
            }
            /* send and receive dimensions */
            mpi::block_data_descriptor sd(comm_col.size()), rd(comm_col.size());
            for (int j = 0; j < comm_col.size(); j++) {
                sd.counts[j] = spl_num_wf_.local_size(block_id(comm_col.rank())) * row_distr.counts[j];
                rd.counts[j] = spl_num_wf_.local_size(block_id(j)) * row_distr.counts[comm_col.rank()];
            }
            sd.calc_offsets();
            rd.calc_offsets();

#if !defined(NDEBUG)
            for (int i = 0; i < n_loc; i++) {
                for (int j = 0; j < comm_col.size(); j++) {
                    int offset = row_distr.offsets[j];
                    int count  = row_distr.counts[j];
                    for (int igg = 0; igg < count; igg++) {
                        if (send_buf[offset * n_loc + count * i + igg] != this->data_[0](offset + igg, i)) {
                            RTE_THROW("wrong packing of send buffer");
                        }
                    }
                }
            }
#endif
            /* full potential wave-functions have extra muffin-tin part;
             * that makes the PW block of data not consecutive and thus we need to copy to a consecutive buffer
             * for alltoall */
            mdarray<std::complex<T>, 2> wf_tmp;
            if (wf_->ld() == wf_->num_pw_) { /* pure plane-wave coeffs */
                auto ptr = (wf_->num_pw_ == 0) ? nullptr : wf_->data_[sp.get()].at(memory_t::host, 0, b__.begin());
                wf_tmp   = mdarray<std::complex<T>, 2>({wf_->num_pw_, b__.size()}, ptr);
            } else {
                wf_tmp = mdarray<std::complex<T>, 2>({wf_->num_pw_, b__.size()}, get_memory_pool(memory_t::host));
            }

            auto* recv_buf = (wf_tmp.ld() == 0) ? nullptr : wf_tmp.at(memory_t::host);

            comm_col.alltoall(send_buf.at(memory_t::host), sd.counts.data(), sd.offsets.data(), recv_buf,
                              rd.counts.data(), rd.offsets.data());

            if (wf_->ld() != wf_->num_pw_) {
                for (int i = 0; i < b__.size(); i++) {
                    auto out_ptr = wf_->data_[sp.get()].at(memory_t::host, 0, b__.begin() + i);
                    std::copy(wf_tmp.at(memory_t::host, 0, i), wf_tmp.at(memory_t::host, 0, i) + wf_->num_pw_, out_ptr);
                }
            }
        }
        if (pp && wf_->gkvec().comm().rank() == 0) {
            auto t = ::sirius::time_interval(t0);
            std::cout << "[transform_from_fft_layout] throughput: "
                      << 2 * sizeof(T) * wf_->gkvec().num_gvec() * b__.size() / std::pow(2.0, 30) / t << " Gb/sec"
                      << std::endl;
        }
    }

  public:
    /// Constructor.
    Wave_functions_fft()
    {
    }

    /// Constructor.
    Wave_functions_fft(std::shared_ptr<fft::Gvec_fft> gkvec_fft__, Wave_functions<T>& wf__, spin_index s__,
                       band_range br__, unsigned int shuffle_flag___)
        : gkvec_fft_{gkvec_fft__}
        , wf_{&wf__}
        , s_{s__}
        , br_{br__}
        , shuffle_flag_{shuffle_flag___}
    {
        auto& comm_col = gkvec_fft_->comm_ortho_fft();
        spl_num_wf_    = splindex_block<>(br__.size(), n_blocks(comm_col.size()), block_id(comm_col.rank()));
        this->num_mt_  = 0;
        this->num_md_  = wf::num_mag_dims(0);
        this->num_sc_  = wf::num_spins(1);
        this->num_wf_  = wf::num_bands(spl_num_wf_.local_size());

        auto sp = wf_->actual_spin_index(s__);

        /* special case when wave-functions are not redistributed */
        if (comm_col.size() == 1) {
            auto i       = wf::band_index(br__.begin());
            auto ptr     = wf__.at(memory_t::host, 0, sp, i);
            auto ptr_gpu = wf__.data_[sp.get()].on_device() ? wf__.at(memory_t::device, 0, sp, i) : nullptr;
            if (ptr_gpu) {
                on_device_ = true;
            }
            /* make alias to the fraction of the wave-functions */
            this->data_[0] = mdarray<std::complex<T>, 2>({wf__.ld(), this->num_wf_.get()}, ptr, ptr_gpu);
            this->num_pw_  = wf_->num_pw_;
        } else {
            /* do wave-functions swap */
            this->data_[0] = mdarray<std::complex<T>, 2>({gkvec_fft__->count(), this->num_wf_.get()},
                                                         get_memory_pool(memory_t::host),
                                                         mdarray_label("Wave_functions_fft.data"));
            this->num_pw_  = gkvec_fft__->count();

            if (shuffle_flag_ & shuffle_to::fft_layout) {
                if (wf__.data_[sp.get()].on_device()) {
                    /* copy block of wave-functions to host memory before calling COSTA */
                    auto ptr     = wf__.at(memory_t::host, 0, sp, wf::band_index(br__.begin()));
                    auto ptr_gpu = wf__.at(memory_t::device, 0, sp, wf::band_index(br__.begin()));
                    acc::copyout(ptr, wf__.ld(), ptr_gpu, wf__.ld(), wf__.num_pw_, br__.size());
                }
                shuffle_to_fft_layout(s__, br__);
            }
        }
    }

    /// Move assignment operator.
    Wave_functions_fft&
    operator=(Wave_functions_fft&& src__)
    {
        if (this != &src__) {
            gkvec_fft_    = src__.gkvec_fft_;
            spl_num_wf_   = src__.spl_num_wf_;
            wf_           = src__.wf_;
            src__.wf_     = nullptr;
            s_            = src__.s_;
            br_           = src__.br_;
            shuffle_flag_ = src__.shuffle_flag_;
            on_device_    = src__.on_device_;
            this->num_pw_ = src__.num_pw_;
            this->num_mt_ = src__.num_mt_;
            this->num_md_ = src__.num_md_;
            this->num_wf_ = src__.num_wf_;
            this->num_sc_ = src__.num_sc_;
            for (int is = 0; is < this->num_sc_.get(); is++) {
                this->data_[is] = std::move(src__.data_[is]);
            }
        }
        return *this;
    }

    /// Destructor.
    ~Wave_functions_fft()
    {
        if (wf_) {
            auto& comm_col = gkvec_fft_->comm_ortho_fft();
            if ((comm_col.size() != 1) && (shuffle_flag_ & shuffle_to::wf_layout)) {
                shuffle_to_wf_layout(s_, br_);
                auto sp = wf_->actual_spin_index(s_);
                if (wf_->data_[sp.get()].on_device()) {
                    /* copy block of wave-functions to device memory after calling COSTA */
                    auto ptr     = wf_->at(memory_t::host, 0, sp, wf::band_index(br_.begin()));
                    auto ptr_gpu = wf_->at(memory_t::device, 0, sp, wf::band_index(br_.begin()));
                    acc::copyin(ptr_gpu, wf_->ld(), ptr, wf_->ld(), wf_->num_pw_, br_.size());
                }
            }
        }
    }

    /// Return local number of wave-functions.
    /** Wave-function band index is distributed over the columns of MPI grid. Each group of FFT communiators
     * is working on its local set of wave-functions. */
    int
    num_wf_local() const
    {
        return spl_num_wf_.local_size();
    }

    /// Return the split index for the number of wave-functions.
    auto
    spl_num_wf() const
    {
        return spl_num_wf_;
    }

    /// Return reference to the plane-wave coefficient.
    inline std::complex<T>&
    pw_coeffs(int ig__, band_index b__)
    {
        return this->data_[0](ig__, b__.get());
    }

    /// Return pointer to the beginning of wave-functions casted to real type as required by the SpFFT library.
    inline T*
    pw_coeffs_spfft(memory_t mem__, band_index b__)
    {
        return reinterpret_cast<T*>(this->data_[0].at(mem__, 0, b__.get()));
    }

    /// Return true if data is avaliable on the device memory.
    inline auto
    on_device() const
    {
        return on_device_;
    }

    /// Return const pointer to the data for a given plane-wave and band indices.
    inline std::complex<T> const*
    at(memory_t mem__, int i__, band_index b__) const
    {
        return this->data_[0].at(mem__, i__, b__.get());
    }

    /// Return pointer to the data for a given plane-wave and band indices.
    inline auto
    at(memory_t mem__, int i__, band_index b__)
    {
        return this->data_[0].at(mem__, i__, b__.get());
    }
};

/// For real-type F (double or float).
template <typename T, typename F>
static inline std::enable_if_t<std::is_scalar<F>::value, F>
inner_diag_local_aux(std::complex<T> z1__, std::complex<T> z2__)
{
    return z1__.real() * z2__.real() + z1__.imag() * z2__.imag();
}

/// For complex-type F (complex<double> or complex<float>).
template <typename T, typename F>
static inline std::enable_if_t<!std::is_scalar<F>::value, F>
inner_diag_local_aux(std::complex<T> z1__, std::complex<T> z2__)
{
    return std::conj(z1__) * z2__;
}

template <typename T, typename F>
static auto
inner_diag_local(memory_t mem__, wf::Wave_functions<T> const& lhs__, wf::Wave_functions_base<T> const& rhs__,
                 wf::spin_range spins__, wf::num_bands num_wf__)
{
    RTE_ASSERT(lhs__.ld() == rhs__.ld());
    if (spins__.size() == 2) {
        if (lhs__.num_md() != wf::num_mag_dims(3)) {
            RTE_THROW("Wave-functions are not spinors");
        }
        if (rhs__.num_md() != wf::num_mag_dims(3)) {
            RTE_THROW("Wave-functions are not spinors");
        }
    }

    std::vector<F> result(num_wf__.get(), 0);

    if (is_host_memory(mem__)) {
        for (auto s = spins__.begin(); s != spins__.end(); s++) {
            auto s1 = lhs__.actual_spin_index(s);
            auto s2 = rhs__.actual_spin_index(s);
            for (int i = 0; i < num_wf__.get(); i++) {
                auto ptr1 = lhs__.at(mem__, 0, s1, wf::band_index(i));
                auto ptr2 = rhs__.at(mem__, 0, s2, wf::band_index(i));
                for (int j = 0; j < lhs__.ld(); j++) {
                    result[i] += inner_diag_local_aux<T, F>(ptr1[j], ptr2[j]);
                }
                /* gamma-point case */
                if (std::is_same<F, real_type<F>>::value) {
                    if (lhs__.comm().rank() == 0) {
                        result[i] = F(2.0) * result[i] - F(std::real(std::conj(ptr1[0]) * ptr2[0]));
                    } else {
                        result[i] *= F(2.0);
                    }
                }
            }
        }
    } else {
#if defined(SIRIUS_GPU)
        int reduced{0};
        /* gamma-point case */
        if (std::is_same<F, real_type<F>>::value) {
            reduced = lhs__.comm().rank() + 1;
        }
        mdarray<F, 1> result_gpu({num_wf__.get()});
        result_gpu.allocate(mem__).zero(mem__);

        for (auto s = spins__.begin(); s != spins__.end(); s++) {
            auto s1   = lhs__.actual_spin_index(s);
            auto s2   = rhs__.actual_spin_index(s);
            auto ptr1 = lhs__.at(mem__, 0, s1, wf::band_index(0));
            auto ptr2 = rhs__.at(mem__, 0, s2, wf::band_index(0));
            if (std::is_same<T, double>::value) {

                if (std::is_same<F, double>::value) {
                    inner_diag_local_gpu_double_double(ptr1, lhs__.ld(), ptr2, rhs__.ld(), lhs__.ld(), num_wf__.get(),
                                                       reduced, result_gpu.at(mem__));
                }
                if (std::is_same<F, std::complex<double>>::value) {
                    inner_diag_local_gpu_double_complex_double(ptr1, lhs__.ld(), ptr2, rhs__.ld(), lhs__.ld(),
                                                               num_wf__.get(), result_gpu.at(mem__));
                }
            }
        }
        result_gpu.copy_to(memory_t::host);
        for (int i = 0; i < num_wf__.get(); i++) {
            result[i] = result_gpu[i];
        }
#endif
    }
    return result;
}

template <typename T, typename F>
auto
inner_diag(memory_t mem__, wf::Wave_functions<T> const& lhs__, wf::Wave_functions_base<T> const& rhs__,
           wf::spin_range spins__, wf::num_bands num_wf__)
{
    PROFILE("wf::inner_diag");
    auto result = inner_diag_local<T, F>(mem__, lhs__, rhs__, spins__, num_wf__);
    lhs__.comm().allreduce(result);
    return result;
}

/// For real-type F (double or float).
template <typename T, typename F>
static inline std::enable_if_t<std::is_scalar<F>::value, std::complex<T>>
axpby_aux(F a__, std::complex<T> x__, F b__, std::complex<T> y__)
{
    return std::complex<T>(a__ * x__.real() + b__ * y__.real(), a__ * x__.imag() + b__ * y__.imag());
}

/// For complex-type F (double or float).
template <typename T, typename F>
static inline std::enable_if_t<!std::is_scalar<F>::value, std::complex<T>>
axpby_aux(F a__, std::complex<T> x__, F b__, std::complex<T> y__)
{
    auto z1 = F(x__.real(), x__.imag());
    auto z2 = F(y__.real(), y__.imag());
    auto z3 = a__ * z1 + b__ * z2;
    return std::complex<T>(z3.real(), z3.imag());
}

/// Perform y <- a * x + b * y type of operation.
template <typename T, typename F>
void
axpby(memory_t mem__, wf::spin_range spins__, wf::band_range br__, F const* alpha__, wf::Wave_functions<T> const* x__,
      F const* beta__, wf::Wave_functions<T>* y__)
{
    PROFILE("wf::axpby");
    if (x__) {
        RTE_ASSERT(x__->ld() == y__->ld());
    }
    if (is_host_memory(mem__)) {
        for (auto s = spins__.begin(); s != spins__.end(); s++) {
            auto spy = y__->actual_spin_index(s);
            auto spx = x__ ? x__->actual_spin_index(s) : spy;
            #pragma omp parallel for
            for (int i = 0; i < br__.size(); i++) {
                auto ptr_y = y__->at(memory_t::host, 0, spy, wf::band_index(br__.begin() + i));
                if (x__) {
                    auto ptr_x = x__->at(memory_t::host, 0, spx, wf::band_index(br__.begin() + i));
                    if (beta__[i] == F(0)) {
                        for (int j = 0; j < y__->ld(); j++) {
                            ptr_y[j] = axpby_aux<T, F>(alpha__[i], ptr_x[j], 0.0, 0.0);
                        }
                    } else if (alpha__[i] == F(0)) {
                        for (int j = 0; j < y__->ld(); j++) {
                            ptr_y[j] = axpby_aux<T, F>(0.0, 0.0, beta__[i], ptr_y[j]);
                        }
                    } else {
                        for (int j = 0; j < y__->ld(); j++) {
                            ptr_y[j] = axpby_aux<T, F>(alpha__[i], ptr_x[j], beta__[i], ptr_y[j]);
                        }
                    }
                } else {
                    for (int j = 0; j < y__->ld(); j++) {
                        ptr_y[j] = axpby_aux<T, F>(0.0, 0.0, beta__[i], ptr_y[j]);
                    }
                }
            }
        }
    } else {
#if defined(SIRIUS_GPU)
        for (auto s = spins__.begin(); s != spins__.end(); s++) {
            auto spy   = y__->actual_spin_index(s);
            auto spx   = x__ ? x__->actual_spin_index(s) : spy;
            auto ptr_y = y__->at(mem__, 0, spy, wf::band_index(br__.begin()));
            auto ptr_x = x__ ? x__->at(mem__, 0, spx, wf::band_index(br__.begin())) : nullptr;

            mdarray<F, 1> alpha;
            if (x__) {
                alpha = mdarray<F, 1>({br__.size()}, const_cast<F*>(alpha__));
                alpha.allocate(mem__).copy_to(mem__);
            }
            mdarray<F, 1> beta({br__.size()}, const_cast<F*>(beta__));
            beta.allocate(mem__).copy_to(mem__);

            auto ldx       = x__ ? x__->ld() : 0;
            auto ptr_alpha = x__ ? alpha.at(mem__) : nullptr;

            if (std::is_same<T, double>::value) {

                if (std::is_same<F, double>::value) {
                    axpby_gpu_double_double(br__.size(), ptr_alpha, ptr_x, ldx, beta.at(mem__), ptr_y, y__->ld(),
                                            y__->ld());
                }
                if (std::is_same<F, std::complex<double>>::value) {
                    axpby_gpu_double_complex_double(br__.size(), ptr_alpha, ptr_x, ldx, beta.at(mem__), ptr_y,
                                                    y__->ld(), y__->ld());
                }
            }
            if (std::is_same<T, float>::value) {
                RTE_THROW("[wf::axpby] implement GPU kernel for float");
            }
        }
#endif
    }
}

template <typename T, typename F>
void
axpy_scatter(memory_t mem__, wf::spin_range spins__, F const* alphas__, Wave_functions<T> const* x__, int const* idx__,
             Wave_functions<T>* y__, int n__)
{
    PROFILE("wf::axpy_scatter");
    if (is_host_memory(mem__)) {
        for (auto s = spins__.begin(); s != spins__.end(); s++) {
            auto spy = y__->actual_spin_index(s);
            auto spx = x__ ? x__->actual_spin_index(s) : spy;
            #pragma omp parallel for
            for (int i = 0; i < n__; i++) {
                auto ii    = idx__[i];
                auto alpha = alphas__[i];

                auto ptr_y = y__->at(memory_t::host, 0, spy, wf::band_index(ii));
                auto ptr_x = x__->at(memory_t::host, 0, spx, wf::band_index(i));
                for (int j = 0; j < y__->ld(); j++) {
                    ptr_y[j] += alpha * ptr_x[j];
                }
            }
        }
    } else {
#if defined(SIRIUS_GPU)
        for (auto s = spins__.begin(); s != spins__.end(); s++) {
            auto spy = y__->actual_spin_index(s);
            auto spx = x__ ? x__->actual_spin_index(s) : spy;

            auto ptr_y = y__->at(mem__, 0, spy, wf::band_index(0));
            auto ptr_x = x__->at(mem__, 0, spx, wf::band_index(0));

            mdarray<F, 1> alpha({n__}, const_cast<F*>(alphas__));
            alpha.allocate(mem__).copy_to(mem__);

            mdarray<int, 1> idx({n__}, const_cast<int*>(idx__));
            idx.allocate(mem__).copy_to(mem__);

            if (std::is_same<T, double>::value) {
                if (std::is_same<F, double>::value) {
                    axpy_scatter_gpu_double_double(n__, alpha.at(mem__), ptr_x, x__->ld(), idx.at(mem__), ptr_y,
                                                   y__->ld(), y__->ld());
                }
                if (std::is_same<F, std::complex<double>>::value) {
                    axpy_scatter_gpu_double_complex_double(n__, alpha.at(mem__), ptr_x, x__->ld(), idx.at(mem__), ptr_y,
                                                           y__->ld(), y__->ld());
                }
            }
            if (std::is_same<T, float>::value) {
                RTE_THROW("[wf::axpy_scatter] implement GPU kernel for float");
            }
        }
#endif
    }
}

/// Copy wave-functions.
template <typename T, typename F = T>
void
copy(memory_t mem__, Wave_functions<T> const& in__, wf::spin_index s_in__, wf::band_range br_in__,
     Wave_functions<F>& out__, wf::spin_index s_out__, wf::band_range br_out__)
{
    // PROFILE("wf::copy");
    RTE_ASSERT(br_in__.size() == br_out__.size());
    if (in__.ld() != out__.ld()) {
        std::stringstream s;
        s << "Leading dimensions of wave-functions do not match" << std::endl
          << "  in__.ld() = " << in__.ld() << std::endl
          << "  out__.ld() = " << out__.ld() << std::endl;
        RTE_THROW(s);
    }

    auto in_ptr  = in__.at(mem__, 0, s_in__, wf::band_index(br_in__.begin()));
    auto out_ptr = out__.at(mem__, 0, s_out__, wf::band_index(br_out__.begin()));

    if (is_host_memory(mem__)) {
        std::copy(in_ptr, in_ptr + in__.ld() * br_in__.size(), out_ptr);
    } else {
        if (!std::is_same<T, F>::value) {
            RTE_THROW("copy of different types on GPU is not implemented");
        }
        acc::copy(reinterpret_cast<std::complex<T>*>(out_ptr), in_ptr, in__.ld() * br_in__.size());
    }
}

/// Apply linear transformation to the wave-functions.
/**
 * \tparam T Precision type of the wave-functions (float or double).
 * \tparam F Type of the subspace and transformation matrix (float or double for Gamma-point calculation,
 *           complex<float> or complex<double> otherwise).
 * \param [in] spla_ctx   Context of the SPLA library.
 * \param [in] mem        Location of the input wave-functions (host or device).
 * \param [in] M          The whole transformation matrix.
 * \param [in] irow0      Location of the 1st row of the transfoormation sub-matrix.
 * \param [in] jcol0      Location of the 1st column of the transfoormation sub-matrix.
 */
template <typename T, typename F>
inline std::enable_if_t<std::is_same<T, real_type<F>>::value, void>
transform(::spla::Context& spla_ctx__, memory_t mem__, la::dmatrix<F> const& M__, int irow0__, int jcol0__,
          real_type<F> alpha__, Wave_functions<T> const& wf_in__, spin_index s_in__, band_range br_in__,
          real_type<F> beta__, Wave_functions<T>& wf_out__, spin_index s_out__, band_range br_out__)
{
    PROFILE("wf::transform");

    RTE_ASSERT(wf_in__.ld() == wf_out__.ld());

    /* spla manages the resources through the context which can be updated during the call;
     * that's why the const must be removed here */
    auto& spla_mat_dist = const_cast<la::dmatrix<F>&>(M__).spla_distribution();

    /* for Gamma point case (transformation matrix is real) we treat complex wave-function coefficients as
     * a doubled list of real values */
    int ld = wf_in__.ld();
    if (std::is_same<F, real_type<F>>::value) {
        ld *= 2;
    }

    F const* mtrx_ptr = M__.size_local() ? M__.at(memory_t::host, 0, 0) : nullptr;

    F const* in_ptr = reinterpret_cast<F const*>(wf_in__.at(mem__, 0, s_in__, band_index(br_in__.begin())));

    F* out_ptr = reinterpret_cast<F*>(wf_out__.at(mem__, 0, s_out__, band_index(br_out__.begin())));

    spla::pgemm_sbs(ld, br_out__.size(), br_in__.size(), alpha__, in_ptr, ld, mtrx_ptr, M__.ld(), irow0__, jcol0__,
                    spla_mat_dist, beta__, out_ptr, ld, spla_ctx__);
}

template <typename T, typename F>
inline std::enable_if_t<!std::is_same<T, real_type<F>>::value, void>
transform(::spla::Context& spla_ctx__, memory_t mem__, la::dmatrix<F> const& M__, int irow0__, int jcol0__,
          real_type<F> alpha__, Wave_functions<T> const& wf_in__, spin_index s_in__, band_range br_in__,
          real_type<F> beta__, Wave_functions<T>& wf_out__, spin_index s_out__, band_range br_out__)
{
    if (is_device_memory(mem__)) {
        RTE_THROW("wf::transform(): mixed FP32/FP64 precision is implemented only for CPU");
    }
    RTE_ASSERT(wf_in__.ld() == wf_out__.ld());
    for (int j = 0; j < br_out__.size(); j++) {
        for (int k = 0; k < wf_in__.ld(); k++) {
            auto wf_out_ptr = wf_out__.at(memory_t::host, k, s_out__, wf::band_index(j + br_out__.begin()));
            std::complex<real_type<F>> z(0, 0);
            ;
            for (int i = 0; i < br_in__.size(); i++) {
                auto wf_in_ptr = wf_in__.at(memory_t::host, k, s_in__, wf::band_index(i + br_in__.begin()));

                z += static_cast<std::complex<real_type<F>>>(*wf_in_ptr) * M__(irow0__ + i, jcol0__ + j);
            }
            if (beta__ == 0) {
                *wf_out_ptr = alpha__ * z;
            } else {
                *wf_out_ptr = alpha__ * z + static_cast<std::complex<real_type<F>>>(*wf_out_ptr) * beta__;
            }
        }
    }
}

/// Scale G=0 component of the wave-functions.
/** This is needed for the Gamma-point calculation to exclude the double-counting of G=0 term.
 */
template <typename T>
inline void
scale_gamma_wf(memory_t mem__, wf::Wave_functions<T> const& wf__, wf::spin_range spins__, wf::band_range br__,
               T* scale__)
{
    RTE_ASSERT(spins__.size() == 1);

    auto& wf = const_cast<Wave_functions<T>&>(wf__);
    RTE_ASSERT(wf.num_sc() == wf::num_spins(1)); // TODO: might be too strong check

    /* rank 0 stores the G=0 component */
    if (wf.comm().rank() != 0) {
        return;
    }

    auto ld = wf.ld() * 2;

    auto sp = wf.actual_spin_index(spins__.begin());

    auto ptr = wf.at(mem__, 0, sp, wf::band_index(br__.begin()));
    auto m   = br__.size();

    if (is_device_memory(mem__)) {
#if defined(SIRIUS_GPU)
        if (std::is_same<T, double>::value) {
            acc::blas::dscal(m, reinterpret_cast<double*>(scale__), reinterpret_cast<double*>(ptr), ld);
        } else if (std::is_same<T, float>::value) {
            acc::blas::sscal(m, reinterpret_cast<float*>(scale__), reinterpret_cast<float*>(ptr), ld);
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
 * \return None
 *
 * Depending on the spin range this functions computes the inner product between individaul spin components
 * or between full spinor wave functions:
 * \f[
 *    M_{irow0+i,jcol0+j} = \sum_{\sigma=s0}^{s1} \langle \phi_{i0 + i}^{\sigma} | \phi_{j0 + j}^{\sigma} \rangle
 * \f]
 * where i0 and j0 and the dimensions of the resulting inner product matrix are determined by the band ranges for
 * bra- and ket- states.
 *
 * The location of the wave-functions data is determined by the mem parameter. The result is always returned in the
 * CPU memory. If resulting matrix is allocated on the GPU memory, the result is copied to GPU as well.
 */
template <typename F, typename W, typename T>
inline std::enable_if_t<std::is_same<T, real_type<F>>::value, void>
inner(::spla::Context& spla_ctx__, memory_t mem__, spin_range spins__, W const& wf_i__, band_range br_i__,
      Wave_functions<T> const& wf_j__, band_range br_j__, la::dmatrix<F>& result__, int irow0__, int jcol0__)
{
    PROFILE("wf::inner");

    RTE_ASSERT(wf_i__.ld() == wf_j__.ld());
    // RTE_ASSERT((wf_i__.gkvec().reduced() == std::is_same<F, real_type<F>>::value));
    RTE_ASSERT((wf_j__.gkvec().reduced() == std::is_same<F, real_type<F>>::value));

    if (spins__.size() == 2) {
        if (wf_i__.num_md() != wf::num_mag_dims(3)) {
            RTE_THROW("input wave-functions are not 2-component spinors");
        }
        if (wf_j__.num_md() != wf::num_mag_dims(3)) {
            RTE_THROW("input wave-functions are not 2-component spinors");
        }
    }

    auto spla_mat_dist = wf_i__.comm().size() > result__.comm().size()
                                 ? spla::MatrixDistribution::create_mirror(wf_i__.comm().native())
                                 : result__.spla_distribution();

    auto ld = wf_i__.ld();

    F alpha = 1.0;
    /* inner product matrix is real */
    if (std::is_same<F, real_type<F>>::value) {
        alpha = 2.0;
        ld *= 2;
    }

    T scale_half(0.5);
    T scale_two(2.0);

    /* for Gamma case, contribution of G = 0 vector must not be counted double -> multiply by 0.5 */
    if (is_real_v<F>) {
        scale_gamma_wf(mem__, wf_j__, spins__, br_j__, &scale_half);
    }

    F beta = 0.0;

    F* result_ptr = result__.size_local() ? result__.at(memory_t::host, 0, 0) : nullptr;

    for (auto s = spins__.begin(); s != spins__.end(); s++) {
        auto s_i      = wf_i__.actual_spin_index(s);
        auto s_j      = wf_j__.actual_spin_index(s);
        auto wf_i_ptr = wf_i__.at(mem__, 0, s_i, wf::band_index(br_i__.begin()));
        auto wf_j_ptr = wf_j__.at(mem__, 0, s_j, wf::band_index(br_j__.begin()));

        spla::pgemm_ssb(br_i__.size(), br_j__.size(), ld, SPLA_OP_CONJ_TRANSPOSE, alpha,
                        reinterpret_cast<F const*>(wf_i_ptr), ld, reinterpret_cast<F const*>(wf_j_ptr), ld, beta,
                        result_ptr, result__.ld(), irow0__, jcol0__, spla_mat_dist, spla_ctx__);
        beta = 1.0;
    }

    /* for Gamma case, G = 0 vector is rescaled back */
    if (is_real_v<F>) {
        scale_gamma_wf(mem__, wf_j__, spins__, br_j__, &scale_two);
    }

    /* make sure result is updated on device as well */
    if (result__.on_device()) {
        result__.copy_to(memory_t::device, irow0__, jcol0__, br_i__.size(), br_j__.size());
    }
}

template <typename T, typename F>
inline std::enable_if_t<!std::is_same<T, real_type<F>>::value, void>
inner(::spla::Context& spla_ctx__, memory_t mem__, spin_range spins__, Wave_functions<T> const& wf_i__,
      band_range br_i__, Wave_functions<T> const& wf_j__, band_range br_j__, la::dmatrix<F>& result__, int irow0__,
      int jcol0__)
{
    if (is_device_memory(mem__)) {
        RTE_THROW("wf::inner(): mixed FP32/FP64 precision is implemented only for CPU");
    }
    RTE_ASSERT(wf_i__.ld() == wf_j__.ld());
    RTE_ASSERT((wf_i__.gkvec().reduced() == std::is_same<F, real_type<F>>::value));
    RTE_ASSERT((wf_j__.gkvec().reduced() == std::is_same<F, real_type<F>>::value));
    for (int i = 0; i < br_i__.size(); i++) {
        for (int j = 0; j < br_j__.size(); j++) {
            result__(irow0__ + i, jcol0__ + j) = 0.0;
        }
    }
    for (auto s = spins__.begin(); s != spins__.end(); s++) {
        auto s_i = wf_i__.actual_spin_index(s);
        auto s_j = wf_j__.actual_spin_index(s);
        int nk   = wf_i__.ld();

        for (int i = 0; i < br_i__.size(); i++) {
            for (int j = 0; j < br_j__.size(); j++) {
                auto wf_i_ptr = wf_i__.at(memory_t::host, 0, s_i, wf::band_index(br_i__.begin() + i));
                auto wf_j_ptr = wf_j__.at(memory_t::host, 0, s_j, wf::band_index(br_j__.begin() + j));
                F z           = 0.0;

                for (int k = 0; k < nk; k++) {
                    z += inner_diag_local_aux<T, F>(wf_i_ptr[k], wf_j_ptr[k]);
                }
                result__(irow0__ + i, jcol0__ + j) += z;
            }
        }
    }
}

/// Orthogonalize n new wave-functions to the N old wave-functions
/** Orthogonalize sets of wave-fuctionsfuctions.
\tparam T                  Precision of the wave-functions (float or double).
\tparam F                  Type of the inner-product matrix (float, double or complex).
\param [in]  spla_ctx      SPLA library context.
\param [in]  mem           Location of the wave-functions data.
\param [in]  spins         Spin index range.
\param [in]  br_old        Band range of the functions that are alredy orthogonal and that will be peojected out.
\param [in]  br_new        Band range of the functions that needed to be orthogonalized.
\param [in]  wf_i          The <wf_i| states used to compute overlap matrix O_{ij}.
\param [in]  wf_j          The |wf_j> states used to compute overlap matrix O_{ij}.
\param [out  wfs           List of wave-functions sets (typically phi, hphi and sphi).
\param [out] o             Work matrix to compute overlap <wf_i|wf_j>
\param [out] tmp           Temporary wave-functions to store intermediate results.
\param [in]  project_out   Project out old subspace (if this was not done before).
\return                    Number of linearly independent wave-functions found.
*/
template <typename T, typename F>
int
orthogonalize(::spla::Context& spla_ctx__, memory_t mem__, spin_range spins__, band_range br_old__, band_range br_new__,
              Wave_functions<T> const& wf_i__, Wave_functions<T> const& wf_j__, std::vector<Wave_functions<T>*> wfs__,
              la::dmatrix<F>& o__, Wave_functions<T>& tmp__, bool project_out__)
{
    PROFILE("wf::orthogonalize");

    /* number of new states */
    int n = br_new__.size();

    auto pp = env::print_performance();

    auto& comm = wf_i__.gkvec().comm();

    int K{0};

    if (pp) {
        K = wf_i__.ld();
        if (is_real_v<F>) {
            K *= 2;
        }
    }

    //    //auto sddk_debug_ptr = utils::get_env<int>("SDDK_DEBUG");
    //    //int sddk_debug      = (sddk_debug_ptr) ? (*sddk_debug_ptr) : 0;
    //
    /* prefactor for the matrix multiplication in complex or double arithmetic (in Giga-operations) */
    double ngop{8e-9};  // default value for complex type
    if (is_real_v<F>) { // change it if it is real type
        ngop = 2e-9;
    }

    if (pp) {
        comm.barrier();
    }
    auto t0 = ::sirius::time_now();

    double gflops{0};

    /* project out the old subspace:
     * |\tilda phi_new> = |phi_new> - |phi_old><phi_old|phi_new>
     * H|\tilda phi_new> = H|phi_new> - H|phi_old><phi_old|phi_new>
     * S|\tilda phi_new> = S|phi_new> - S|phi_old><phi_old|phi_new> */
    if (br_old__.size() > 0 && project_out__) {
        inner(spla_ctx__, mem__, spins__, wf_i__, br_old__, wf_j__, br_new__, o__, 0, 0);
        for (auto s = spins__.begin(); s != spins__.end(); s++) {
            for (auto wf : wfs__) {
                auto sp = wf->actual_spin_index(s);
                transform(spla_ctx__, mem__, o__, 0, 0, -1.0, *wf, sp, br_old__, 1.0, *wf, sp, br_new__);
            }
        }
        if (pp) {
            /* inner and transform have the same number of flops */
            gflops += spins__.size() * static_cast<int>(1 + wfs__.size()) * ngop * br_old__.size() * n * K;
        }
    }

    //    if (sddk_debug >= 2) {
    //        if (o__.comm().rank() == 0) {
    //            RTE_OUT(std::cout) << "check QR decomposition, matrix size : " << n__ << std::endl;
    //        }
    //        inner(spla_ctx__, spins__, *wfs__[idx_bra__], N__, n__, *wfs__[idx_ket__], N__, n__, o__, 0, 0);
    //
    //        linalg(lib_t::scalapack).geqrf(n__, n__, o__, 0, 0);
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
    inner(spla_ctx__, mem__, spins__, wf_i__, br_new__, wf_j__, br_new__, o__, 0, 0);
    if (pp) {
        gflops += spins__.size() * ngop * n * n * K;
    }

    /* At this point overlap matrix is computed for the new block and stored on the CPU. We
     * now have this choices
     *   - mem: CPU
     *     - o is not distributed
     *       - potrf is computed on CPU with lapack
     *       - trtri is computed on CPU with lapack
     *       - trmm is computed on CPU with blas
     *     - o is distributed
     *       - potrf is computed on CPU with scalapack
     *       - trtri is computed on CPU with scalapack
     *       - trmm is computed on CPU with wf::transform
     *
     *   - mem: GPU
     *     - o is not distributed
     *       - potrf is computed on CPU with lapack; later with cuSolver
     *       - trtri is computed on CPU with lapack; later with cuSolver
     *       - trmm is computed on GPU with cublas
     *
     *     - o is distributed
     *       - potrf is computed on CPU with scalapack
     *       - trtri is computed on CPU with scalapack
     *       - trmm is computed on GPU with wf::transform
     */
    // TODO: test magma and cuSolver
    /*
     * potrf from cuSolver works in a standalone test, but not here; here it returns -1;
     *   disbled for further investigation
     *
     */
    auto la  = la::lib_t::lapack;
    auto la1 = la::lib_t::blas;
    auto mem = memory_t::host;
    /* if matrix is distributed, we use ScaLAPACK for Cholesky factorization */
    if (o__.comm().size() > 1) {
        la = la::lib_t::scalapack;
    }
    if (is_device_memory(mem__)) {
        /* this is for trmm */
        la1 = la::lib_t::gpublas;
        /* this is for potrf */
        // if (o__.comm().size() == 1) {
        //     mem = mem__;
        //     la = sddk::linalg_t::gpublas;
        // }
    }

    /* compute the transformation matrix (inverse of the Cholesky factor) */
    PROFILE_START("wf::orthogonalize|tmtrx");
    auto o_ptr = (o__.size_local() == 0) ? nullptr : o__.at(mem);
    if (la == la::lib_t::scalapack) {
        o__.make_real_diag(n);
    }
    /* Cholesky factorization */
    if (int info = la::wrap(la).potrf(n, o_ptr, o__.ld(), o__.descriptor())) {
        std::stringstream s;
        s << "error in Cholesky factorization, info = " << info << std::endl
          << "number of existing states: " << br_old__.size() << std::endl
          << "number of new states: " << br_new__.size();
        RTE_THROW(s);
    }
    /* inversion of triangular matrix */
    if (la::wrap(la).trtri(n, o_ptr, o__.ld(), o__.descriptor())) {
        RTE_THROW("error in inversion");
    }
    PROFILE_STOP("wf::orthogonalize|tmtrx");

    /* single MPI rank and precision types of wave-functions and transformation matrices match */
    if (o__.comm().size() == 1 && std::is_same<T, real_type<F>>::value) {
        PROFILE_START("wf::orthogonalize|trans");
        if (is_device_memory(mem__)) {
            o__.copy_to(mem__, 0, 0, n, n);
        }
        int sid{0};
        for (auto s = spins__.begin(); s != spins__.end(); s++) {
            /* multiplication by triangular matrix */
            for (auto& wf : wfs__) {
                auto sp  = wf->actual_spin_index(s);
                auto ptr = reinterpret_cast<F*>(wf->at(mem__, 0, sp, wf::band_index(br_new__.begin())));
                int ld   = wf->ld();
                /* Gamma-point case */
                if (is_real_v<F>) {
                    ld *= 2;
                }

                la::wrap(la1).trmm('R', 'U', 'N', ld, n, &la::constant<F>::one(), o__.at(mem__), o__.ld(), ptr, ld,
                                   acc::stream_id(sid++));
            }
        }
        if (la1 == la::lib_t::gpublas || la1 == la::lib_t::cublasxt || la1 == la::lib_t::magma) {
            /* sync stream only if processing unit is GPU */
            for (int i = 0; i < sid; i++) {
                acc::sync_stream(acc::stream_id(i));
            }
        }
        if (pp) {
            gflops += spins__.size() * wfs__.size() * ngop * 0.5 * n * n * K;
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
            for (auto wf : wfs__) {
                auto sp  = wf->actual_spin_index(s);
                auto sp1 = tmp__.actual_spin_index(s);
                auto br1 = wf::band_range(0, br_new__.size());
                transform(spla_ctx__, mem__, o__, 0, 0, 1.0, *wf, sp, br_new__, 0.0, tmp__, sp1, br1);
                copy(mem__, tmp__, sp1, br1, *wf, sp, br_new__);
            }
        }
    }

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
    //==
    //== //    if (sddk_debug >= 1) {
    //== //        inner(spla_ctx__, spins__, *wfs__[idx_bra__], N__, n__, *wfs__[idx_ket__], N__, n__, o__, 0, 0);
    //== //        auto err = check_identity(o__, n__);
    //== //        if (o__.comm().rank() == 0) {
    //== //            RTE_OUT(std::cout) << "orthogonalization error : " << err << std::endl;
    //== //        }
    //== //    }
    if (pp) {
        comm.barrier();
        auto t = ::sirius::time_interval(t0);
        if (comm.rank() == 0) {
            RTE_OUT(std::cout) << "effective performance : " << gflops / t << " GFlop/s/rank, "
                               << gflops * comm.size() / t << " GFlop/s" << std::endl;
        }
    }

    return 0;
}

} // namespace wf

} // namespace sirius

#endif
