#ifndef __SPHERIC_FUNCTION_SET_HPP__
#define __SPHERIC_FUNCTION_SET_HPP__

#include "unit_cell/unit_cell.hpp"

namespace sirius {

template <typename T>
class Spheric_function_set
{
  private:
    /// Pointer to the unit cell
    Unit_cell const* unit_cell_{nullptr};
    /// List of atoms for which the spherical expansion is defined.
    std::vector<int> atoms_;
    /// Split the number of atoms between MPI ranks.
    /** If the pointer is not set, set of spheric functions is treated as global, without MPI distribution */
    sddk::splindex<sddk::splindex_t::block> const* spl_atoms_{nullptr};
    /// List of spheric functions.
    std::vector<Spheric_function<function_domain_t::spectral, T>> func_;

    bool all_atoms_{false};

    template <typename F>
    friend F inner(Spheric_function_set<F> const& f1__, Spheric_function_set<F> const& f2__);

    void init(std::function<int(int)> lmax__)
    {
        func_.resize(unit_cell_->num_atoms());
        if (spl_atoms_) {
            for (int i = 0; i < spl_atoms_->local_size(); i++) {
                int ia = atoms_[(*spl_atoms_)[i]];
                func_[ia] = Spheric_function<function_domain_t::spectral, T>(utils::lmmax(lmax__(ia)),
                        unit_cell_->atom(ia).radial_grid());
            }
        } else {
            for (int ia : atoms_) {
                func_[ia] = Spheric_function<function_domain_t::spectral, T>(utils::lmmax(lmax__(ia)),
                        unit_cell_->atom(ia).radial_grid());
            }
        }
    }

  public:
    Spheric_function_set()
    {
    }

    /// Constructor for all atoms.
    Spheric_function_set(Unit_cell const& unit_cell__, std::function<int(int)> lmax__,
            sddk::splindex<sddk::splindex_t::block> const* spl_atoms__ = nullptr)
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
        init(lmax__);
    }

    /// Constructor for a subset of atoms.
    Spheric_function_set(Unit_cell const& unit_cell__, std::vector<int> atoms__, std::function<int(int)> lmax__,
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

    auto const& unit_cell() const
    {
        return *unit_cell_;
    }

    void zero()
    {
        for (int ia = 0; ia < unit_cell_->num_atoms(); ia++) {
            if (func_[ia].size()) {
                func_[ia].zero();
            }
        }
    }

    /// Synchronize global function.
    /** Assuming that each MPI rank was handling part of the global spherical function, broadcast data
     *  from each rank. As a result, each rank stores a full and identical copy of global spherical function. */
    void sync(sddk::splindex<sddk::splindex_t::block> const& spl_atoms__)
    {
        for (int i = 0; i < spl_atoms__.global_index_size(); i++) {
            auto loc = spl_atoms__.location(i);
            int ia = atoms_[i];
            unit_cell_->comm().bcast(func_[ia].at(sddk::memory_t::host), static_cast<int>(func_[ia].size()), loc.rank);
        }
    }
};

template <typename T>
inline T inner(Spheric_function_set<T> const& f1__, Spheric_function_set<T> const& f2__)
{
    RTE_ASSERT(f1__.spl_atoms_ == f2__.spl_atoms_);

    T result{0};

    auto const& comm = f1__.unit_cell_->comm();

    if (f1__.spl_atoms_) {
        for (int i = 0; i < f1__.spl_atoms_->local_size(); i++) {
            int ia = f1__.atoms_[(*f1__.spl_atoms_)[i]];
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

}

#endif
