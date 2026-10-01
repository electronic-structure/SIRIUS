/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __PAW_LOCAL_FIELDS_HPP__
#define __PAW_LOCAL_FIELDS_HPP__

#include "function3d/kinetic_density.hpp"
#include "unit_cell/atom_type.hpp"

namespace sirius {

/// One AE or pseudo partial-wave representation and its projector-matrix adjoint.
/** Pseudo charge may include Q, whereas positive tau has no Q contribution.
 *  The two representations may be evaluated separately for semilocal PAW XC.
 *  A nonlocal functional instead requires their joint reconstruction first.
 */
class PAW_local_fields
{
  private:
    Atom_type const& type_;
    bool ae_;
    bool include_q_;
    int lmax_;
    Gaunt_coefficients<double> gaunt_;
    Spheric_kinetic_density_operator<double> tau_;

    static Atom_type const&
    checked_type(Atom_type const& type__)
    {
        if (!type__.is_paw() || type__.spin_orbit_coupling() || type__.indexb().size() == 0 ||
            type__.radial_grid().first() <= 0) {
            RTE_THROW("local PAW fields require scalar partial waves on a positive radial grid");
        }
        return type__;
    }

    static std::vector<int>
    radial_sizes(Atom_type const& type__, bool ae__)
    {
        std::vector<int> result;
        for (auto const& rf : type__.indexr()) {
            result.push_back(
                    static_cast<int>(ae__ ? type__.ae_paw_wf(rf.idxrf).size() : type__.ps_paw_wf(rf.idxrf).size()));
        }
        return result;
    }

    double
    rho_product(int i, int j, int l, int ir) const
    {
        int a = type_.indexb(i).idxrf, b = type_.indexb(j).idxrf;
        auto const& wf = ae_ ? type_.ae_paw_wfs_array() : type_.ps_paw_wfs_array();
        double q       = ae_ || !include_q_ ? 0 : type_.q_radial_function(a, b, l)(ir);
        return (wf(ir, a) * wf(ir, b) + q) / std::pow(type_.radial_grid(ir), 2);
    }

  public:
    PAW_local_fields(Atom_type const& type__, bool ae__, bool include_q__ = true)
        : type_(checked_type(type__))
        , ae_(ae__)
        , include_q_(include_q__)
        , lmax_(2 * type_.indexr().lmax())
        , gaunt_(type_.indexr().lmax(), lmax_, type_.indexr().lmax(), SHT::gaunt_rrr)
        , tau_(type_.radial_grid(), type_.indexb(), ae_ ? type_.ae_paw_wfs_array() : type_.ps_paw_wfs_array(), lmax_,
               radial_wave_function_t::r_times_R, radial_sizes(type_, ae_))
    {
    }

    /// Return {rho, tau} in one total or two canonical spin channels.
    std::array<std::vector<Flm>, 2>
    fields(mdarray<std::complex<double>, 3> const& dm__, bool include_core__ = true) const
    {
        int n = type_.indexb().size(), ns = static_cast<int>(dm__.size(2));
        if (dm__.size(0) != n || dm__.size(1) != n || (ns != 1 && ns != 2)) {
            RTE_THROW("local PAW fields: incompatible density matrix");
        }
        auto const& grid = type_.radial_grid();
        std::array<std::vector<Flm>, 2> result;
        mdarray<std::complex<double>, 2> spin_dm({n, n});
        for (int s = 0; s < ns; s++) {
            result[0].emplace_back(sf::lmmax(lmax_), grid);
            result[1].emplace_back(sf::lmmax(lmax_), grid);
            auto& rho = result[0].back();
            auto& tau = result[1].back();
            rho.zero();
            tau.zero();
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++) {
                    spin_dm(i, j) = dm__(j, i, s);
                }
                for (int i = 0; i <= j; i++) {
                    for (auto const& g : gaunt_.gaunt_vector(type_.indexb(i).lm, type_.indexb(j).lm)) {
                        double w = (i == j ? 1 : 2) * std::real(dm__(i, j, s)) * g.coef;
                        for (int ir = 0; ir < grid.num_points(); ir++) {
                            rho(g.lm3, ir) += w * rho_product(i, j, g.l3, ir);
                        }
                    }
                }
            }
            tau_.add_density(spin_dm, tau);
            if (include_core__) {
                auto const& core = ae_ ? type_.paw_ae_core_charge_density() : type_.ps_core_charge_density();
                if (ae_) {
                    if (core.size() != static_cast<size_t>(grid.num_points()) ||
                        std::any_of(core.begin(), core.end(), [](double v) { return !std::isfinite(v) || v < 0; })) {
                        RTE_THROW("local PAW fields require a finite AE core density on the radial grid");
                    }
                    type_.paw_ae_core_kinetic_density();
                } else {
                    type_.check_ps_core_kinetic_density();
                }
                for (int ir = 0; ir < grid.num_points(); ir++) {
                    rho(0, ir) += (core.empty() ? 0 : core[ir]) / (ns * y00);
                    double t = ae_ ? type_.paw_ae_core_kinetic_density()[ir]
                                   : (type_.has_ps_core_kinetic_density() ? type_.ps_core_kinetic_density()[ir] : 0);
                    tau(0, ir) += t / (ns * y00);
                }
            }
        }
        return result;
    }

    /// Derivative with respect to one spin block of D, with all quadrature weights already included.
    mdarray<double, 2>
    matrix_elements(Flm const& d_rho__, Flm const& d_tau__) const
    {
        if (d_rho__.angular_domain_size() != sf::lmmax(lmax_) ||
            d_rho__.radial_grid().hash() != type_.radial_grid().hash()) {
            RTE_THROW("local PAW fields: incompatible charge covectors");
        }
        auto result = tau_.matrix_elements_from_derivatives(d_tau__);
        int n       = type_.indexb().size();
        for (int j = 0; j < n; j++) {
            for (int i = 0; i <= j; i++) {
                for (auto const& g : gaunt_.gaunt_vector(type_.indexb(i).lm, type_.indexb(j).lm)) {
                    for (int ir = 0; ir < type_.num_mt_points(); ir++) {
                        result(i, j) += d_rho__(g.lm3, ir) * g.coef * rho_product(i, j, g.l3, ir);
                    }
                }
                result(j, i) = result(i, j);
            }
        }
        return result;
    }
};

} // namespace sirius
#endif
