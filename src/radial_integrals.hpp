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

/** \file radial_integrals.hpp
 *
 *  \brief Representation of various radial integrals.
 */

#ifndef __RADIAL_INTEGRALS_HPP__
#define __RADIAL_INTEGRALS_HPP__

#include "Unit_cell/unit_cell.hpp"
#include "sbessel.hpp"

namespace sirius {

/// Base class for all kinds of radial integrals.
template <int N>
class Radial_integrals_base
{
  protected:
    /// Unit cell.
    Unit_cell const& unit_cell_;

    /// Linear grid of q-points on which the interpolation of radial integrals is done.
    Radial_grid<double> grid_q_;

    /// Split index of q-points.
    splindex<splindex_t::block> spl_q_;

    /// Array with integrals.
    mdarray<Spline<double>, N> values_;

  public:
    /// Constructor.
    Radial_integrals_base(Unit_cell const& unit_cell__, double qmax__, int np__)
        : unit_cell_(unit_cell__)
    {
        /* Add extra length to the cutoffs in order to interpolate radial integrals for q > cutoff.
           This is needed for the variable cell relaxation when lattice changes and the G-vectors in
           Cartiesin coordinates exceed the initial cutoff length. Do not remove this extra delta! */

        /* add extra length in [a.u.^-1] */
        qmax__ += std::max(5.0, qmax__ * 0.1);

        grid_q_ = Radial_grid_lin<double>(static_cast<int>(np__ * qmax__), 0, qmax__);
        spl_q_  = splindex<splindex_t::block>(grid_q_.num_points(), unit_cell_.comm().size(), unit_cell_.comm().rank());
    }

    /// Get starting index iq and delta dq for the q-point on the linear grid.
    /** The following condition is satisfied: q = grid_q[iq] + dq */
    inline std::pair<int, double> iqdq(double q__) const
    {
        if (q__ > grid_q_.last()) {
            std::stringstream s;
            s << "[sirius::Radial_integrals_base::iqdq] q-point is out of range" << std::endl
              << "  q : " << q__ << std::endl
              << "  last point of the q-grid : " << grid_q_.last();
            TERMINATE(s);
        }
        std::pair<int, double> result;
        /* find index of q-point */
        result.first = static_cast<int>((grid_q_.num_points() - 1) * q__ / grid_q_.last());
        /* delta q = q - q_i */
        result.second = q__ - grid_q_[result.first];
        return result;
    }

    template <typename... Args>
    inline double value(Args... args, double q__) const
    {
        auto idx = iqdq(q__);
        return values_(args...)(idx.first, idx.second);
    }

    inline int nq() const
    {
        return grid_q_.num_points();
    }
};

/// Radial integrals of the atomic centered orbitals.
/** Used in initialize_subspace and in the hubbard correction. */
template<bool jl_deriv>
class Radial_integrals_atomic_wf : public Radial_integrals_base<2>
{
  private:
    void generate();

  public:
    Radial_integrals_atomic_wf(Unit_cell const& unit_cell__, double qmax__, int np__)
        : Radial_integrals_base<2>(unit_cell__, qmax__, np__)
    {
        int no_max{0};
        for (int iat = 0; iat < unit_cell__.num_atom_types(); iat++) {
            no_max = std::max(no_max, unit_cell__.atom_type(iat).num_ps_atomic_wf());
        }

        values_ = mdarray<Spline<double>, 2>(no_max, unit_cell_.num_atom_types());

        generate();
    }

    /// retrieve a given orbital from an atom type
    inline Spline<double> const& values(int iwf__, int iat__) const
    {
        return values_(iwf__, iat__);
    }

    /// Get all values for a given atom type and q-point.
    inline mdarray<double, 1> values(int iat__, double q__) const
    {
        auto idx        = iqdq(q__);
        auto& atom_type = unit_cell_.atom_type(iat__);
        mdarray<double, 1> val(atom_type.num_ps_atomic_wf());
        for (int i = 0; i < atom_type.num_ps_atomic_wf(); i++) {
            val(i) = values_(i, iat__)(idx.first, idx.second);
        }
        return val;
    }
};

/// Radial integrals of the augmentation operator.
template <bool jl_deriv>
class Radial_integrals_aug : public Radial_integrals_base<3>
{
  private:
    void generate();

  public:
    Radial_integrals_aug(Unit_cell const& unit_cell__, double qmax__, int np__)
        : Radial_integrals_base<3>(unit_cell__, qmax__, np__)
    {
        int nmax = unit_cell_.max_mt_radial_basis_size();
        int lmax = unit_cell_.lmax();

        values_ = mdarray<Spline<double>, 3>(nmax * (nmax + 1) / 2, 2 * lmax + 1, unit_cell_.num_atom_types());

        generate();
    }

    inline mdarray<double, 2> values(int iat__, double q__) const
    {
        auto idx = iqdq(q__);

        auto& atom_type = unit_cell_.atom_type(iat__);
        int lmax        = atom_type.indexr().lmax();
        int nbrf        = atom_type.mt_radial_basis_size();

        mdarray<double, 2> val(nbrf * (nbrf + 1) / 2, 2 * lmax + 1);
        for (int l = 0; l <= 2 * lmax; l++) {
            for (int i = 0; i < nbrf * (nbrf + 1) / 2; i++) {
                val(i, l) = values_(i, l, iat__)(idx.first, idx.second);
            }
        }
        return val;
    }
};


class Radial_integrals_rho_pseudo : public Radial_integrals_base<1>
{
  private:
    void generate();

  public:
    Radial_integrals_rho_pseudo(Unit_cell const& unit_cell__, double qmax__, int np__)
        : Radial_integrals_base<1>(unit_cell__, qmax__, np__)
    {
        values_ = mdarray<Spline<double>, 1>(unit_cell_.num_atom_types());
        generate();

        if (unit_cell_.parameters().control().print_checksum_ && unit_cell_.comm().rank() == 0) {
            double cs{0};
            for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
                for (int iq = 0; iq < grid_q_.num_points(); iq++) {
                    cs += values_(iat)(iq);
                }
            }
            utils::print_checksum("Radial_integrals_rho_pseudo", cs);
        }
    }
};


template <bool jl_deriv>
class Radial_integrals_rho_core_pseudo : public Radial_integrals_base<1>
{
  private:
    void generate();

  public:
    Radial_integrals_rho_core_pseudo(Unit_cell const& unit_cell__, double qmax__, int np__)
        : Radial_integrals_base<1>(unit_cell__, qmax__, np__)
    {
        values_ = mdarray<Spline<double>, 1>(unit_cell_.num_atom_types());
        generate();
    }
};


template <bool jl_deriv>
class Radial_integrals_beta : public Radial_integrals_base<2>
{
  private:
    void generate();

  public:
    Radial_integrals_beta(Unit_cell const& unit_cell__, double qmax__, int np__)
        : Radial_integrals_base<2>(unit_cell__, qmax__, np__)
    {
        /* create space for <j_l(qr)|beta> or <d j_l(qr) / dq|beta> radial integrals */
        values_ = mdarray<Spline<double>, 2>(unit_cell_.max_mt_radial_basis_size(), unit_cell_.num_atom_types());
        generate();
    }

    /// Get all values for a given atom type and q-point.
    inline mdarray<double, 1> values(int iat__, double q__) const
    {
        auto idx        = iqdq(q__);
        auto& atom_type = unit_cell_.atom_type(iat__);
        mdarray<double, 1> val(atom_type.mt_radial_basis_size());
        for (int i = 0; i < atom_type.mt_radial_basis_size(); i++) {
            val(i) = values_(i, iat__)(idx.first, idx.second);
        }
        return val;
    }
};


class Radial_integrals_beta_jl : public Radial_integrals_base<3>
{
  private:
    int lmax_;

    void generate();

  public:
    Radial_integrals_beta_jl(Unit_cell const& unit_cell__, double qmax__, int np__)
        : Radial_integrals_base<3>(unit_cell__, qmax__, np__)
    {
        lmax_ = unit_cell__.lmax() + 2;
        /* create space for <j_l(qr)|beta> radial integrals */
        values_ = mdarray<Spline<double>, 3>(unit_cell_.max_mt_radial_basis_size(), lmax_ + 1,
                                             unit_cell_.num_atom_types());
        generate();
    }
};


template <bool jl_deriv>
class Radial_integrals_vloc : public Radial_integrals_base<1>
{
  private:
    void generate();

  public:
    Radial_integrals_vloc(Unit_cell const& unit_cell__, double qmax__, int np__)
        : Radial_integrals_base<1>(unit_cell__, qmax__, np__)
    {
        values_ = mdarray<Spline<double>, 1>(unit_cell_.num_atom_types());
        generate();
    }

    /// Special implementation to recover the true radial integral value.
    inline double value(int iat__, double q__) const
    {
        auto idx = iqdq(q__);
        if (std::abs(q__) < 1e-12) {
            if (jl_deriv) {
                return 0;
            } else {
                return values_(iat__)(0);
            }
        } else {
            auto& atom_type = unit_cell_.atom_type(iat__);

            auto q2 = std::pow(q__, 2);
            if (jl_deriv) {
                if (!unit_cell_.parameters().parameters_input().enable_esm_ ||
                    unit_cell_.parameters().parameters_input().esm_bc_ == "pbc") {
                    return values_(iat__)(idx.first, idx.second) / q2 / q__ -
                           atom_type.zn() * std::exp(-q2 / 4) * (4 + q2) / 2 / q2 / q2;
                } else {
                    return values_(iat__)(idx.first, idx.second) / q2 / q__;
                }
            } else {
                if (!unit_cell_.parameters().parameters_input().enable_esm_ ||
                    unit_cell_.parameters().parameters_input().esm_bc_ == "pbc") {
                    return values_(iat__)(idx.first, idx.second) / q__ - atom_type.zn() * std::exp(-q2 / 4) / q2;
                } else {
                    return values_(iat__)(idx.first, idx.second) / q__;
                }
            }
        }
    }
};


class Radial_integrals_rho_free_atom : public Radial_integrals_base<1>
{
  private:
    void generate();

  public:
    Radial_integrals_rho_free_atom(Unit_cell const& unit_cell__, double qmax__, int np__)
        : Radial_integrals_base<1>(unit_cell__, qmax__, np__)
    {
        values_ = mdarray<Spline<double>, 1>(unit_cell_.num_atom_types());
        generate();
    }

    /// Special implementation to recover the true radial integral value.
    inline double value(int iat__, double q__) const
    {
        auto idx = iqdq(q__);
        if (std::abs(q__) < 1e-12) {
            return values_(iat__)(0);
        } else {
            return values_(iat__)(idx.first, idx.second) / q__;
        }
    }
};

} // namespace sirius

#endif // __RADIAL_INTEGRALS_H__
