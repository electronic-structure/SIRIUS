/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __SPHERIC_FUNCTION_SET_HPP__
#define __SPHERIC_FUNCTION_SET_HPP__

#include "unit_cell/unit_cell.hpp"

namespace sirius {

using lmax_t = strong_type<int, struct __lmax_t_tag>;

template <typename T, typename I>
class Spheric_function_set
{
  private:
    /// Pointer to the unit cell
    Unit_cell const* unit_cell_{nullptr};
    /// Text label of the function set.
    std::string label_;
    /// List of atoms for which the spherical expansion is defined.
    std::vector<int> atoms_;
    /// Split the number of atoms between MPI ranks.
    /** If the pointer is null, spheric functions set is treated as global, without MPI distribution */
    splindex_block<I> const* spl_atoms_{nullptr};
    /// List of spheric functions.
    std::vector<Spheric_function<function_domain_t::spectral, T>> func_;

    bool all_atoms_{false};

    void
    init(std::function<lmax_t(int)> lmax__, spheric_function_set_ptr_t<T> const* sptr__ = nullptr)
    {
        func_.resize(unit_cell_->num_atoms());

        auto set_func = [&](int ia) {
            if (sptr__) {
                func_[ia] = Spheric_function<function_domain_t::spectral, T>(
                        sptr__->ptr + sptr__->lmmax * sptr__->nrmtmax * ia, sptr__->lmmax,
                        unit_cell_->atom(ia).radial_grid());
            } else {
                func_[ia] = Spheric_function<function_domain_t::spectral, T>(sf::lmmax(lmax__(ia)),
                                                                             unit_cell_->atom(ia).radial_grid());
            }
        };

        if (spl_atoms_) {
            for (auto it : (*spl_atoms_)) {
                set_func(atoms_[it.i]);
            }
        } else {
            for (int ia : atoms_) {
                set_func(ia);
            }
        }
    }

  public:
    Spheric_function_set()
    {
    }

    /// Constructor for all atoms.
    Spheric_function_set(std::string label__, Unit_cell const& unit_cell__, std::function<lmax_t(int)> lmax__,
                         splindex_block<I> const* spl_atoms__        = nullptr,
                         spheric_function_set_ptr_t<T> const* sptr__ = nullptr)
        : unit_cell_{&unit_cell__}
        , label_{label__}
        , spl_atoms_{spl_atoms__}
        , all_atoms_{true}
    {
        atoms_.resize(unit_cell__.num_atoms());
        std::iota(atoms_.begin(), atoms_.end(), 0);
        if (spl_atoms_) {
            if (spl_atoms_->size() != unit_cell__.num_atoms()) {
                RTE_THROW("wrong split atom index");
            }
        }
        init(lmax__, sptr__);
    }

    /// Constructor for a subset of atoms.
    Spheric_function_set(std::string label__, Unit_cell const& unit_cell__, std::vector<int> atoms__,
                         std::function<lmax_t(int)> lmax__, splindex_block<I> const* spl_atoms__ = nullptr)
        : unit_cell_{&unit_cell__}
        , label_{label__}
        , atoms_{atoms__}
        , spl_atoms_{spl_atoms__}
        , all_atoms_{false}
    {
        if (spl_atoms_) {
            if (spl_atoms_->size() != static_cast<int>(atoms__.size())) {
                RTE_THROW("wrong split atom index");
            }
        }
        init(lmax__);
    }

    auto const&
    atoms() const
    {
        return atoms_;
    }

    auto&
    operator[](int ia__)
    {
        return func_[ia__];
    }

    auto const&
    operator[](int ia__) const
    {
        return func_[ia__];
    }

    inline auto const&
    unit_cell() const
    {
        return *unit_cell_;
    }

    inline void
    zero()
    {
        if (unit_cell_) {
            for (int ia = 0; ia < unit_cell_->num_atoms(); ia++) {
                if (func_[ia].size()) {
                    func_[ia].zero();
                }
            }
        }
    }

    /// Synchronize global function.
    /** Assuming that each MPI rank was handling part of the global spherical function, broadcast data
     *  from each rank. As a result, each rank stores a full and identical copy of global spherical function. */
    inline void
    sync(splindex_block<I> const& spl_atoms__)
    {
        for (int i = 0; i < spl_atoms__.size(); i++) {
            auto loc = spl_atoms__.location(typename I::global(i));
            int ia   = atoms_[i];
            unit_cell_->comm().bcast(func_[ia].at(memory_t::host), static_cast<int>(func_[ia].size()), loc.ib);
        }
    }

    Spheric_function_set<T, I>&
    operator+=(Spheric_function_set<T, I> const& rhs__)
    {
        for (int ia = 0; ia < unit_cell_->num_atoms(); ia++) {
            if (func_[ia].size() && rhs__[ia].size()) {
                func_[ia] += rhs__[ia];
            }
        }
        return *this;
    }

    template <typename T_, typename I_>
    friend T_
    inner(Spheric_function_set<T_, I_> const& f1__, Spheric_function_set<T_, I_> const& f2__);

    template <typename T_, typename I_>
    friend void
    copy(Spheric_function_set<T_, I_> const& src__, Spheric_function_set<T_, I_>& dest__);

    template <typename T_, typename I_>
    friend void
    copy(Spheric_function_set<T_, I_> const& src__, spheric_function_set_ptr_t<T_> dest__);

    template <typename T_, typename I_>
    friend void
    copy(spheric_function_set_ptr_t<T_> src__, Spheric_function_set<T_, I_> const& dest__);

    template <typename T_, typename I_>
    friend void
    scale(T_ alpha__, Spheric_function_set<T_, I_>& x__);

    template <typename T_, typename I_>
    friend void
    axpy(T_ alpha__, Spheric_function_set<T_, I_> const& x__, Spheric_function_set<T_, I_>& y__);
};

template <typename T, typename I>
inline T
inner(Spheric_function_set<T, I> const& f1__, Spheric_function_set<T, I> const& f2__)
{
    auto ptr = (f1__.spl_atoms_) ? f1__.spl_atoms_ : f2__.spl_atoms_;

    /* if both functions are split then the split index must match */
    if (f1__.spl_atoms_ && f2__.spl_atoms_) {
        RTE_ASSERT(f1__.spl_atoms_ == f2__.spl_atoms_);
    }

    T result{0};

    auto const& comm = f1__.unit_cell_->comm();

    if (ptr) {
        for (int i = 0; i < ptr->local_size(); i++) {
            int ia = f1__.atoms_[(*ptr).global_index(typename I::local(i))];
            result += inner(f1__[ia], f2__[ia]);
        }
    } else {
        splindex_block<I> spl_atoms(f1__.atoms_.size(), n_blocks(comm.size()), block_id(comm.rank()));
        for (int i = 0; i < spl_atoms.local_size(); i++) {
            int ia = f1__.atoms_[spl_atoms.global_index(typename I::local(i))];
            result += inner(f1__[ia], f2__[ia]);
        }
    }
    comm.allreduce(&result, 1);
    return result;
}

/// Copy from Spheric_function_set to external pointer.
/** External pointer is assumed to be global. */
template <typename T, typename I>
inline void
copy(Spheric_function_set<T, I> const& src__, spheric_function_set_ptr_t<T> dest__)
{
    auto p = dest__.ptr;
    for (auto ia : src__.atoms()) {
        if (src__[ia].size()) {
            if (src__[ia].angular_domain_size() > dest__.lmmax) {
                RTE_THROW("wrong angular_domain_size");
            }
            mdarray<T, 2> rlm({dest__.lmmax, dest__.nrmtmax}, p);
            for (int ir = 0; ir < src__[ia].radial_grid().num_points(); ir++) {
                for (int lm = 0; lm < src__[ia].angular_domain_size(); lm++) {
                    rlm(lm, ir) = src__[ia](lm, ir);
                }
            }
        }
        p += dest__.lmmax * dest__.nrmtmax;
    }
    if (src__.spl_atoms_) {
        int ld = dest__.lmmax * dest__.nrmtmax;
        src__.unit_cell_->comm().allgather(dest__.ptr, ld * src__.spl_atoms_->local_size(),
                                           ld * src__.spl_atoms_->global_offset());
    }
}

/// Copy from external pointer to Spheric_function_set.
/** External pointer is assumed to be global. */
template <typename T, typename I>
inline void
copy(spheric_function_set_ptr_t<T> const src__, Spheric_function_set<T, I>& dest__)
{
    auto p = src__.ptr;
    for (auto ia : dest__.atoms()) {
        if (dest__[ia].size()) {
            if (dest__[ia].angular_domain_size() > src__.lmmax) {
                RTE_THROW("wrong angular_domain_size");
            }
            mdarray<T, 2> rlm({src__.lmmax, src__.nrmtmax}, p);
            for (int ir = 0; ir < dest__[ia].radial_grid().num_points(); ir++) {
                for (int lm = 0; lm < dest__[ia].angular_domain_size(); lm++) {
                    dest__[ia](lm, ir) = rlm(lm, ir);
                }
            }
        }
        p += src__.lmmax * src__.nrmtmax;
    }
}

template <typename T, typename I>
inline void
copy(Spheric_function_set<T, I> const& src__, Spheric_function_set<T, I>& dest__)
{
    for (int ia = 0; ia < src__.unit_cell_->num_atoms(); ia++) {
        if (src__.func_[ia].size()) {
            copy(src__.func_[ia], dest__.func_[ia]);
        }
    }
}

template <typename T, typename I>
inline void
scale(T alpha__, Spheric_function_set<T, I>& x__)
{
    for (int ia = 0; ia < x__.unit_cell_->num_atoms(); ia++) {
        if (x__.func_[ia].size()) {
            x__.func_[ia] *= alpha__;
        }
    }
}

template <typename T, typename I>
inline void
axpy(T alpha__, Spheric_function_set<T, I> const& x__, Spheric_function_set<T, I>& y__)
{
    for (int ia = 0; ia < x__.unit_cell_->num_atoms(); ia++) {
        if (x__.func_[ia].size()) {
            y__.func_[ia] += x__.func_[ia] * alpha__;
        }
    }
}

} // namespace sirius

#endif
