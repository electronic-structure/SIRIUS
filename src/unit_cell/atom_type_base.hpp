/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file atom_type_base.hpp
 *
 *  \brief Contains declaration and implementation of sirius::Atom_type_base class.
 */

#ifndef __ATOM_TYPE_BASE_HPP__
#define __ATOM_TYPE_BASE_HPP__

#include "atomic_data.hpp"
#include "radial/radial_grid.hpp"

namespace sirius {

/// Base class for sirius::Atom_type and sirius::Free_atom classes.
class Atom_type_base
{
  protected:
    /// Nucleus charge or pseudocharge, treated as positive(!) integer.
    int zn_{0};

    /// Chemical element symbol.
    std::string symbol_;

    /// Chemical element name.
    std::string name_;

    /// Atom mass.
    double mass_{0};

    /// List of atomic levels.
    std::vector<atomic_level_descriptor> atomic_levels_;

    /* forbid copy constructor */
    Atom_type_base(Atom_type_base const& src) = delete;

    /* forbid assignment operator */
    Atom_type_base&
    operator=(Atom_type_base& src) = delete;

    /// Density of a free atom.
    Spline<double> free_atom_density_spline_;

    /// Density of a free atom as read from the input file.
    /** Does not contain 4 Pi and r^2 prefactors. */
    std::vector<double> free_atom_density_;

    /// Radial grid of a free atom.
    Radial_grid<double> free_atom_radial_grid_;

  private:
    /// Initialize atomic levels of neutral atom and radial grid.
    void
    init()
    {
        /* add valence levels to the list of atom's levels */
        for (auto& e : atomic_conf[zn_ - 1]) {
            /* check if this level is already in the list */
            bool in_list{false};
            for (auto& c : atomic_levels_) {
                if (c.n == e.n && c.l == e.l && c.k == e.k) {
                    in_list = true;
                    break;
                }
            }
            if (!in_list) {
                auto level = e;
                level.core = false;
                atomic_levels_.push_back(level);
            }
        }

        free_atom_radial_grid_ = Radial_grid_exp<double>(2000 + 150 * zn(), 1e-7, 20.0 + 0.25 * zn(), 1.0);
    }

  public:
    /// Constructor.
    Atom_type_base(int zn__)
        : zn_(zn__)
        , symbol_(atomic_symb[zn_ - 1])
        , name_(atomic_name[zn_ - 1])
    {
        init();
    }

    /// Constructor.
    Atom_type_base(std::string symbol__)
        : zn_(atomic_zn.at(symbol__))
        , symbol_(symbol__)
        , name_(atomic_name[zn_ - 1])
    {
        init();
    }

    /// Get atomic charge.
    inline int
    zn() const
    {
        assert(zn_ > 0);
        return zn_;
    }

    /// Set atomic charge.
    inline int
    zn(int zn__)
    {
        zn_ = zn__;
        return zn_;
    }

    /// Get atomic symbol.
    inline std::string const&
    symbol() const
    {
        return symbol_;
    }

    /// Get name of the element.
    inline std::string const&
    name() const
    {
        return name_;
    }

    /// Get atomic mass.
    inline double
    mass() const
    {
        return mass_;
    }

    /// Get the whole radial grid.
    inline Radial_grid<double> const&
    free_atom_radial_grid() const
    {
        return free_atom_radial_grid_;
    }

    /// Get the radial point at a given index.
    inline double
    free_atom_radial_grid(int ir) const
    {
        return free_atom_radial_grid_[ir];
    }

    /// Return number of the atomic levels.
    inline int
    num_atomic_levels() const
    {
        return static_cast<int>(atomic_levels_.size());
    }

    inline atomic_level_descriptor const&
    atomic_level(int idx) const
    {
        return atomic_levels_[idx];
    }
};

} // namespace sirius

#endif // __ATOM_TYPE_BASE_HPP__
