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

/** \file atom_type_base.hpp
 *
 *  \brief Contains declaration and implementation of sirius::Atom_type_base class.
 */

#ifndef __ATOM_TYPE_BASE_HPP__
#define __ATOM_TYPE_BASE_HPP__

#include "atomic_data.hpp"
#include "radial_grid.hpp"

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

    ///// Number of core electrons.
    //double num_core_electrons_{0};

    ///// Number of valence electrons.
    //double num_valence_electrons_{0};

    /* forbid copy constructor */
    Atom_type_base(Atom_type_base const& src) = delete;

    /* forbid assignment operator */
    Atom_type_base& operator=(Atom_type_base& src) = delete;

    /// Density of a free atom.
    Spline<double> free_atom_density_spline_;

    /// Density of a free atom as read from the input file.
    /** Does not contain 4 Pi and r^2 prefactors. */
    std::vector<double> free_atom_density_;

    /// Radial grid of a free atom.
    Radial_grid<double> free_atom_radial_grid_;

  private:
    /// Initialize atomic levels of neutral atom and radial grid.
    void init()
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
        ///* get the number of core electrons */
        //for (auto& e : atomic_levels_) {
        //    if (e.core) {
        //        num_core_electrons_ += e.occupancy;
        //    }
        //}

        ///* get number of valence electrons */
        //num_valence_electrons_ = zn_ - num_core_electrons_;

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
    inline int zn() const
    {
        assert(zn_ > 0);
        return zn_;
    }

    /// Set atomic charge.
    inline int zn(int zn__)
    {
        zn_ = zn__;
        return zn_;
    }

    /// Get atomic symbol.
    inline std::string const& symbol() const
    {
        return symbol_;
    }

    /// Get name of the element.
    inline std::string const& name() const
    {
        return name_;
    }

    /// Get atomic mass.
    inline double mass() const
    {
        return mass_;
    }

    /// Get the whole radial grid.
    inline Radial_grid<double> const& free_atom_radial_grid() const
    {
        return free_atom_radial_grid_;
    }

    /// Get the radial point at a given index.
    inline double free_atom_radial_grid(int ir) const
    {
        return free_atom_radial_grid_[ir];
    }

    /// Return number of the atomic levels.
    inline int num_atomic_levels() const
    {
        return static_cast<int>(atomic_levels_.size());
    }

    inline atomic_level_descriptor const& atomic_level(int idx) const
    {
        return atomic_levels_[idx];
    }

    //inline double num_core_electrons() const
    //{
    //    return num_core_electrons_;
    //}

    //inline double num_valence_electrons() const
    //{
    //    return num_valence_electrons_;
    //}
};

} // namespace

#endif // __ATOM_TYPE_BASE_HPP__

