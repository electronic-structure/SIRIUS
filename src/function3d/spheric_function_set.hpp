#ifndef __SPHERIC_FUNCTION_SET_HPP__
#define __SPHERIC_FUNCTION_SET_HPP__

#include "unit_cell/unit_cell.hpp"
#include "strong_type.hpp"

namespace sirius {

using lmax_t = strong_type<int, struct __lmax_t_tag>;

template <typename T>
class Spheric_function_set
{
  private:
    /// Pointer to the unit cell
    Unit_cell const* unit_cell_{nullptr};
    /// List of atoms for which the spherical expansion is defined.
    std::vector<int> atoms_;
    /// Split the number of atoms between MPI ranks.
    /** If the pointer is null, spheric functions set is treated as global, without MPI distribution */
    sddk::splindex<sddk::splindex_t::block> const* spl_atoms_{nullptr};
    /// List of spheric functions.
    std::vector<Spheric_function<function_domain_t::spectral, T>> func_;

    bool all_atoms_{false};

    void init(std::function<lmax_t(int)> lmax__, spheric_function_set_ptr_t<T> const* sptr__ = nullptr)
    {
        func_.resize(unit_cell_->num_atoms());

        auto set_func = [&](int ia)
        {
            if (sptr__) {
                func_[ia] = Spheric_function<function_domain_t::spectral, T>(
                            sptr__->ptr + sptr__->lmmax * sptr__->nrmtmax * ia,
                            sptr__->lmmax, unit_cell_->atom(ia).radial_grid());
            } else {
                func_[ia] = Spheric_function<function_domain_t::spectral, T>(utils::lmmax(lmax__(ia)),
                            unit_cell_->atom(ia).radial_grid());
            }
        };

        if (spl_atoms_) {
            for (int i = 0; i < spl_atoms_->local_size(); i++) {
                int ia = atoms_[(*spl_atoms_)[i]];
                set_func(ia);
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
    Spheric_function_set(Unit_cell const& unit_cell__, std::function<lmax_t(int)> lmax__,
            sddk::splindex<sddk::splindex_t::block> const* spl_atoms__ = nullptr,
            spheric_function_set_ptr_t<T> const* sptr__ = nullptr)
        : unit_cell_{&unit_cell__}
        , spl_atoms_{spl_atoms__}
        , all_atoms_{true}
    {
        atoms_.resize(unit_cell__.num_atoms());
        std::iota(atoms_.begin(), atoms_.end(), 0);
        if (spl_atoms_) {
            if (spl_atoms_->global_index_size() != unit_cell__.num_atoms()) {
                RTE_THROW("wrong split atom index");
            }
        }
        init(lmax__, sptr__);
    }

    /// Constructor for a subset of atoms.
    Spheric_function_set(Unit_cell const& unit_cell__, std::vector<int> atoms__, std::function<lmax_t(int)> lmax__,
            sddk::splindex<sddk::splindex_t::block> const* spl_atoms__ = nullptr)
        : unit_cell_{&unit_cell__}
        , atoms_{atoms__}
        , spl_atoms_{spl_atoms__}
        , all_atoms_{false}
    {
        if (spl_atoms_) {
            if (spl_atoms_->global_index_size() != static_cast<int>(atoms__.size())) {
                RTE_THROW("wrong split atom index");
            }
        }
        init(lmax__);
    }

    auto const& atoms() const
    {
        return atoms_;
    }

    auto& operator[](int ia__)
    {
        return func_[ia__];
    }

    auto const& operator[](int ia__) const
    {
        return func_[ia__];
    }

    inline auto const& unit_cell() const
    {
        return *unit_cell_;
    }

    inline void zero()
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
    inline void sync(sddk::splindex<sddk::splindex_t::block> const& spl_atoms__)
    {
        for (int i = 0; i < spl_atoms__.global_index_size(); i++) {
            auto loc = spl_atoms__.location(i);
            int ia = atoms_[i];
            unit_cell_->comm().bcast(func_[ia].at(sddk::memory_t::host), static_cast<int>(func_[ia].size()), loc.rank);
        }
    }

    Spheric_function_set<T>& operator+=(Spheric_function_set<T> const& rhs__)
    {
        for (int ia = 0; ia < unit_cell_->num_atoms(); ia++) {
            if (func_[ia].size() && rhs__[ia].size()) {
                func_[ia] += rhs__[ia];
            }
        }
        return *this;
    }

    template <typename F>
    friend F
    inner(Spheric_function_set<F> const& f1__, Spheric_function_set<F> const& f2__);

    template <typename F>
    friend void
    copy(Spheric_function_set<F> const& src__, Spheric_function_set<F>& dest__);

    template <typename F>
    friend void
    copy(Spheric_function_set<F> const& src__, spheric_function_set_ptr_t<F> dest__);

    template <typename F>
    friend void
    copy(spheric_function_set_ptr_t<F> src__, Spheric_function_set<F> const& dest__);

    template <typename F>
    friend void
    scale(F alpha__, Spheric_function_set<F>& x__);

    template <typename F>
    friend void
    axpy(F alpha__, Spheric_function_set<F> const& x__, Spheric_function_set<F>& y__);
};

template <typename T>
inline T inner(Spheric_function_set<T> const& f1__, Spheric_function_set<T> const& f2__)
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
            int ia = f1__.atoms_[(*ptr)[i]];
            result += inner(f1__[ia], f2__[ia]);
        }
    } else {
        sddk::splindex<sddk::splindex_t::block> spl_atoms(f1__.atoms_.size(), comm.size(), comm.rank());
        for (int i = 0; i < spl_atoms.local_size(); i++) {
            int ia = f1__.atoms_[spl_atoms[i]];
            result += inner(f1__[ia], f2__[ia]);
        }
    }
    comm.allreduce(&result, 1);
    return result;
}

/// Copy from Spheric_function_set to external pointer.
/** External pointer is assumed to be global. */
template <typename T>
inline void
copy(Spheric_function_set<T> const& src__, spheric_function_set_ptr_t<T> dest__)
{
    auto p = dest__.ptr;
    for (auto ia : src__.atoms()) {
        if (src__[ia].size()) {
            if (src__[ia].angular_domain_size() > dest__.lmmax) {
                RTE_THROW("wrong angular_domain_size");
            }
            sddk::mdarray<T, 2> rlm(p, dest__.lmmax, dest__.nrmtmax);
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
template <typename T>
inline void
copy(spheric_function_set_ptr_t<T> const src__, Spheric_function_set<T>& dest__)
{
    auto p = src__.ptr;
    for (auto ia : dest__.atoms()) {
        if (dest__[ia].size()) {
            if (dest__[ia].angular_domain_size() > src__.lmmax) {
                RTE_THROW("wrong angular_domain_size");
            }
            sddk::mdarray<T, 2> rlm(p, src__.lmmax, src__.nrmtmax);
            for (int ir = 0; ir < dest__[ia].radial_grid().num_points(); ir++) {
                for (int lm = 0; lm < dest__[ia].angular_domain_size(); lm++) {
                    dest__[ia](lm, ir) = rlm(lm, ir);
                }
            }
        }
        p += src__.lmmax * src__.nrmtmax;
    }
}

template <typename T>
inline void
copy(Spheric_function_set<T> const& src__, Spheric_function_set<T>& dest__)
{
    for (int ia = 0; ia < src__.unit_cell_->num_atoms(); ia++) {
        if (src__.func_[ia].size()) {
            copy(src__.func_[ia], dest__.func_[ia]);
        }
    }
}

template <typename T>
inline void
scale(T alpha__, Spheric_function_set<T>& x__)
{
    for (int ia = 0; ia < x__.unit_cell_->num_atoms(); ia++) {
        if (x__.func_[ia].size()) {
            x__.func_[ia] *= alpha__;
        }
    }
}

template <typename T>
inline void
axpy(T alpha__, Spheric_function_set<T> const& x__, Spheric_function_set<T>& y__)
{
    for (int ia = 0; ia < x__.unit_cell_->num_atoms(); ia++) {
        if (x__.func_[ia].size()) {
            y__.func_[ia] += x__.func_[ia] * alpha__;
        }
    }
}

}

#endif
