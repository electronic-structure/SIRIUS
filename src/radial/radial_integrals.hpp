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

#include "unit_cell/unit_cell.hpp"
#include "specfunc/sbessel.hpp"
#include "utils/rte.hpp"

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
    sddk::mdarray<Spline<double>, N> values_;

    /// Maximum length of the reciprocal wave-vector.
    double qmax_{0};

  public:
    /// Constructor.
    Radial_integrals_base(Unit_cell const& unit_cell__, double const qmax__, int const np__)
        : unit_cell_(unit_cell__)
    {
        /* Add extra length to the cutoffs in order to interpolate radial integrals for q > cutoff.
           This is needed for the variable cell relaxation when lattice changes and the G-vectors in
           Cartiesin coordinates exceed the initial cutoff length. Do not remove this extra delta! */

        /* add extra length in [a.u.^-1] */
        qmax_ = qmax__ + std::max(10.0, qmax__ * 0.1);

        grid_q_ = Radial_grid_lin<double>(static_cast<int>(np__ * qmax_), 0, qmax_);
        spl_q_  = splindex<splindex_t::block>(grid_q_.num_points(), unit_cell_.comm().size(), unit_cell_.comm().rank());
    }

    /// Get starting index iq and delta dq for the q-point on the linear grid.
    /** The following condition is satisfied: q = grid_q[iq] + dq */
    inline std::pair<int, double> iqdq(double q__) const
    {
        if (q__ > grid_q_.last()) {
            std::stringstream s;
            s << "q-point is out of range" << std::endl
              << "  q : " << q__ << std::endl
              << "  last point of the q-grid : " << grid_q_.last() << std::endl;
            auto uc = unit_cell_.serialize();
            s << "unit cell: " << uc;
            RTE_THROW(s);
        }
        std::pair<int, double> result;
        /* find index of q-point */
        result.first = static_cast<int>((grid_q_.num_points() - 1) * q__ / grid_q_.last());
        /* delta q = q - q_i */
        result.second = q__ - grid_q_[result.first];
        return result;
    }

    /// Return value of the radial integral with specific indices.
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

    inline double qmax() const
    {
        return qmax_;
    }
};

/// Radial integrals of the atomic centered orbitals.
/** Used in initialize_subspace and in the hubbard correction. */
template<bool jl_deriv>
class Radial_integrals_atomic_wf : public Radial_integrals_base<2>
{
  private:
    /// Return radial basis index for a given atom type.
    std::function<sirius::experimental::radial_functions_index const&(int)> indexr_;
    /// Generate radial integrals.
    void generate(std::function<Spline<double> const&(int, int)> fl__);

  public:
    /// Constructor.
    Radial_integrals_atomic_wf(Unit_cell const& unit_cell__, double qmax__, int np__,
        std::function<sirius::experimental::radial_functions_index const&(int)> indexr__,
        std::function<Spline<double> const&(int, int)> fl__)
        : Radial_integrals_base<2>(unit_cell__, qmax__, np__)
        , indexr_(indexr__)
    {
        int nrf_max{0};
        for (int iat = 0; iat < unit_cell__.num_atom_types(); iat++) {
            nrf_max = std::max(nrf_max, static_cast<int>(indexr_(iat).size()));
        }

        values_ = sddk::mdarray<Spline<double>, 2>(nrf_max, unit_cell_.num_atom_types());

        generate(fl__);
    }

    /// Retrieve a value for a given orbital of an atom type.
    inline Spline<double> const& values(int iwf__, int iat__) const
    {
        return values_(iwf__, iat__);
    }

    /// Get all values for a given atom type and q-point.
    inline sddk::mdarray<double, 1> values(int iat__, double q__) const
    {
        auto idx        = iqdq(q__);
        auto& atom_type = unit_cell_.atom_type(iat__);
        int nrf         = indexr_(atom_type.id()).size();

        sddk::mdarray<double, 1> val(nrf);
        for (int i = 0; i < nrf; i++) {
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
    std::function<void(int, double, double*, int, int)> ri_callback_{nullptr};

    void generate();

  public:
    Radial_integrals_aug(Unit_cell const& unit_cell__, double qmax__, int np__,
        std::function<void(int, double, double*, int, int)> ri_callback__)
        : Radial_integrals_base<3>(unit_cell__, qmax__, np__)
        , ri_callback_(ri_callback__)
    {
        if (ri_callback_ == nullptr) {
            int nmax = unit_cell_.max_mt_radial_basis_size();
            int lmax = unit_cell_.lmax();

            values_ = sddk::mdarray<Spline<double>, 3>(nmax * (nmax + 1) / 2, 2 * lmax + 1, unit_cell_.num_atom_types());

            generate();
        }
    }

    inline sddk::mdarray<double, 2> values(int iat__, double q__) const
    {
        auto& atom_type = unit_cell_.atom_type(iat__);
        int lmax        = atom_type.indexr().lmax();
        int nbrf        = atom_type.mt_radial_basis_size();

        sddk::mdarray<double, 2> val(nbrf * (nbrf + 1) / 2, 2 * lmax + 1);

        if (ri_callback_ == nullptr) {
            auto idx = iqdq(q__);

            for (int l = 0; l <= 2 * lmax; l++) {
                for (int i = 0; i < nbrf * (nbrf + 1) / 2; i++) {
                    val(i, l) = values_(i, l, iat__)(idx.first, idx.second);
                }
            }
        } else {
            ri_callback_(iat__ + 1, q__, &val[0], nbrf * (nbrf + 1) / 2, 2 * lmax + 1);
        }
        return val;
    }
};


class Radial_integrals_rho_pseudo : public Radial_integrals_base<1>
{
  private:
    std::function<void(int, int, double*, double*)> ri_callback_{nullptr};
    void generate();

  public:
    Radial_integrals_rho_pseudo(Unit_cell const& unit_cell__, double qmax__, int np__,
        std::function<void(int, int, double*, double*)> ri_callback__)
        : Radial_integrals_base<1>(unit_cell__, qmax__, np__)
        , ri_callback_(ri_callback__)
    {
        if (ri_callback__ == nullptr) {
            values_ = mdarray<Spline<double>, 1>(unit_cell_.num_atom_types());
            generate();

            if (unit_cell_.parameters().cfg().control().print_checksum() && unit_cell_.comm().rank() == 0) {
                double cs{0};
                for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
                    for (int iq = 0; iq < grid_q_.num_points(); iq++) {
                        cs += values_(iat)(iq);
                    }
                }
                utils::print_checksum("Radial_integrals_rho_pseudo", cs);
            }
        }
    }

    /// Compute all values of the raial integrals.
    inline sddk::mdarray<double, 2> values(std::vector<double>& q__, sddk::Communicator const& comm__) const
    {
        int nq = static_cast<int>(q__.size());
        splindex<splindex_t::block> splq(nq, comm__.size(), comm__.rank());
        sddk::mdarray<double, 2> result(nq, unit_cell_.num_atom_types());
        result.zero();
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            if (!unit_cell_.atom_type(iat).ps_total_charge_density().empty()) {
                #pragma omp parallel for
                for (int iqloc = 0; iqloc < splq.local_size(); iqloc++) {
                    int iq = splq[iqloc];
                    if (ri_callback_) {
                        ri_callback_(iat + 1, 1, &q__[iq], &result(iq, iat));
                    } else {
                        result(iq, iat) = this->value<int>(iat, q__[iq]);
                    }
                }
                comm__.allgather(&result(0, iat), splq.local_size(), splq.global_offset());
            }
        }
        return result;
    }
};


template <bool jl_deriv>
class Radial_integrals_rho_core_pseudo : public Radial_integrals_base<1>
{
  private:
    std::function<void(int, int, double*, double*)> ri_callback_{nullptr};
    void generate();

  public:
    Radial_integrals_rho_core_pseudo(Unit_cell const& unit_cell__, double qmax__, int np__,
        std::function<void(int, int, double*, double*)> ri_callback__)
        : Radial_integrals_base<1>(unit_cell__, qmax__, np__)
        , ri_callback_(ri_callback__)
    {
        if (ri_callback_ == nullptr) {
            values_ = mdarray<Spline<double>, 1>(unit_cell_.num_atom_types());
            generate();
        }
    }

    /// Compute all values of the raial integrals.
    inline sddk::mdarray<double, 2> values(std::vector<double>& q__, sddk::Communicator const& comm__) const
    {
        int nq = static_cast<int>(q__.size());
        splindex<splindex_t::block> splq(nq, comm__.size(), comm__.rank());
        sddk::mdarray<double, 2> result(nq, unit_cell_.num_atom_types());
        result.zero();
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            if (!unit_cell_.atom_type(iat).ps_core_charge_density().empty()) {
                #pragma omp parallel for
                for (int iqloc = 0; iqloc < splq.local_size(); iqloc++) {
                    int iq = splq[iqloc];
                    if (ri_callback_) {
                        ri_callback_(iat + 1, 1, &q__[iq], &result(iq, iat));
                    } else {
                        result(iq, iat) = this->value<int>(iat, q__[iq]);
                    }
                }
                comm__.allgather(&result(0, iat), splq.local_size(), splq.global_offset());
            }
        }
        return result;
    }
};

/// Radial integrals of beta projectors.
template <bool jl_deriv>
class Radial_integrals_beta : public Radial_integrals_base<2>
{
  private:
    /// Callback function to compute radial integrals using the host code.
    std::function<void(int, double, double*, int)> ri_callback_{nullptr};

    /// Generate radial integrals on the q-grid.
    void generate();

  public:
    Radial_integrals_beta(Unit_cell const& unit_cell__, double qmax__, int np__,
        std::function<void(int, double, double*, int)> ri_callback__)
        : Radial_integrals_base<2>(unit_cell__, qmax__, np__)
        , ri_callback_(ri_callback__)
    {
        if (ri_callback_ == nullptr) {
            /* create space for <j_l(qr)|beta> or <d j_l(qr) / dq|beta> radial integrals */
            values_ = mdarray<Spline<double>, 2>(unit_cell_.max_mt_radial_basis_size(), unit_cell_.num_atom_types());
            generate();
        }
    }

    /// Get all values for a given atom type and q-point.
    inline sddk::mdarray<double, 1> values(int iat__, double q__) const
    {
        auto& atom_type = unit_cell_.atom_type(iat__);
        sddk::mdarray<double, 1> val(atom_type.mt_radial_basis_size());
        if (ri_callback_ == nullptr) {
            auto idx = iqdq(q__);
            for (int i = 0; i < atom_type.mt_radial_basis_size(); i++) {
                val(i) = values_(i, iat__)(idx.first, idx.second);
            }
        } else {
            ri_callback_(iat__ + 1, q__, &val[0], atom_type.mt_radial_basis_size());
        }
        return val;
    }
};

template <bool jl_deriv>
class Radial_integrals_vloc : public Radial_integrals_base<1>
{
  private:
    std::function<void(int, int, double*, double*)> ri_callback_{nullptr};
    void generate();

  public:
    Radial_integrals_vloc(Unit_cell const& unit_cell__, double qmax__, int np__,
        std::function<void(int, int, double*, double*)> ri_callback__)
        : Radial_integrals_base<1>(unit_cell__, qmax__, np__)
        , ri_callback_(ri_callback__)
    {
        if (ri_callback_ == nullptr) {
            values_ = mdarray<Spline<double>, 1>(unit_cell_.num_atom_types());
            generate();
        }
    }

    /// Special implementation to recover the true radial integral value.
    inline double value(int iat__, double q__) const
    {
        if (unit_cell_.atom_type(iat__).local_potential().empty()) {
            return 0;
        }
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
                return values_(iat__)(idx.first, idx.second) / q2 / q__ -
                    atom_type.zn() * std::exp(-q2 / 4) * (4 + q2) / 2 / q2 / q2;
            } else {
                return values_(iat__)(idx.first, idx.second) / q__ - atom_type.zn() * std::exp(-q2 / 4) / q2;
            }
        }
    }

    /// Compute all values of the raial integrals.
    inline sddk::mdarray<double, 2> values(std::vector<double>& q__, sddk::Communicator const& comm__) const
    {
        int nq = static_cast<int>(q__.size());
        splindex<splindex_t::block> splq(nq, comm__.size(), comm__.rank());
        sddk::mdarray<double, 2> result(nq, unit_cell_.num_atom_types());
        result.zero();
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            if (!unit_cell_.atom_type(iat).local_potential().empty()) {
                #pragma omp parallel for
                for (int iqloc = 0; iqloc < splq.local_size(); iqloc++) {
                    int iq = splq[iqloc];
                    if (ri_callback_) {
                        ri_callback_(iat + 1, 1, &q__[iq], &result(iq, iat));
                    } else {
                        result(iq, iat) = this->value(iat, q__[iq]);
                    }
                }
                comm__.allgather(&result(0, iat), splq.local_size(), splq.global_offset());
            }
        }
        return result;
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
