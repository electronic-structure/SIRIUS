/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __PAW_XC_CORRECTION_HPP__
#define __PAW_XC_CORRECTION_HPP__

#include "paw_local_fields.hpp"

namespace sirius {

/// Linear atom-local PAW reconstruction and its discrete projector-matrix adjoint.
/** The smooth charge input is the pseudo-orbital density, without SIRIUS's Q
 *  augmentation or pseudo-core density. Add rho_AE - rho_PS + rho_core locally.
 *  Smooth tau is the pseudo-orbital gradient density; its local correction is
 *  tau_AE - tau_PS + tau_core, with no charge-compensation term. No independent
 *  AE/PS functional evaluations are performed here. All reconstructed fields
 *  must be combined before evaluating a nonlinear, nonlocal functional.
 */
class PAW_xc_correction
{
  private:
    Atom_type const& type_;
    PAW_local_fields ae_;
    PAW_local_fields ps_;

    static Atom_type const&
    checked_type(Atom_type const& type__)
    {
        if (!type__.is_paw() || type__.spin_orbit_coupling() || type__.indexb().size() == 0 ||
            type__.radial_grid().first() <= 0) {
            RTE_THROW("PAW XC correction requires scalar PAW partial waves on a positive radial grid");
        }
        auto const& rho = type__.paw_ae_core_charge_density();
        if (rho.size() != static_cast<size_t>(type__.num_mt_points()) ||
            std::any_of(rho.begin(), rho.end(), [](double x) { return !std::isfinite(x) || x < 0; })) {
            RTE_THROW("PAW XC correction requires a finite nonnegative AE core density on the radial grid");
        }
        type__.paw_ae_core_kinetic_density(); // Missing is not an implicit zero core.
        return type__;
    }

  public:
    explicit PAW_xc_correction(Atom_type const& type__)
        : type_(checked_type(type__))
        , ae_(type_, true)
        , ps_(type_, false, false)
    {
    }

    /// Source support, including explicit core fields, independent of the current density matrix.
    /** A PAW table can extend far beyond its truncated partial waves. Do not
     *  propagate the spline's artificial zero-padding tail to periodic images.
     *  Include the first zero sample after the last supported entry, when one
     *  exists. No amplitude threshold is applied to the supplied core data.
     */
    double
    support_radius() const
    {
        int last{0};
        for (auto const& rf : type_.indexr()) {
            last = std::max(
                    last,
                    static_cast<int>(std::max(type_.ae_paw_wf(rf.idxrf).size(), type_.ps_paw_wf(rf.idxrf).size())) - 1);
        }
        int nr = type_.num_mt_points();
        for (int ir = last + 1; ir < nr; ir++) {
            if (type_.paw_ae_core_charge_density()[ir] != 0 || type_.paw_ae_core_kinetic_density()[ir] != 0) {
                last = ir;
            }
        }
        return type_.radial_grid(std::min(last + 1, nr - 1));
    }

    /// Return {rho corrections, tau corrections}, each with one total or two spin fields.
    /** dm__ uses SIRIUS's projector convention D_ij=sum_n f_n c_ni conj(c_nj).
     *  The closed-shell frozen core is split equally between the two collinear
     *  channels. Fields are real-harmonic coefficients, without radial weights.
     *  A sampler may request valence-only corrections and interpolate the fixed
     *  core separately to avoid state-dependent cancellation in radial slopes.
     */
    std::array<std::vector<Flm>, 2>
    fields(mdarray<std::complex<double>, 3> const& dm__, bool include_core__ = true) const
    {
        int ns           = static_cast<int>(dm__.size(2));
        auto result      = ae_.fields(dm__, false);
        auto pseudo      = ps_.fields(dm__, false);
        auto const& grid = type_.radial_grid();
        for (int s = 0; s < ns; s++) {
            auto& rho = result[0][s];
            auto& tau = result[1][s];
            rho -= pseudo[0][s];
            tau -= pseudo[1][s];
            if (include_core__) {
                for (int ir = 0; ir < grid.num_points(); ir++) {
                    rho(0, ir) += type_.paw_ae_core_charge_density()[ir] / (ns * y00);
                    tau(0, ir) += type_.paw_ae_core_kinetic_density()[ir] / (ns * y00);
                }
            }
        }
        return result;
    }

    /// One spin channel's matrix derivative from complete discrete field covectors.
    /** Cartesian-gradient derivatives must first be returned through the same
     *  sampling transpose as rho. Covectors already contain all quadrature and
     *  energy-partition weights. The frozen core has no projector derivative.
     *  Neither the forward reconstruction nor this derivative includes Q.
     */
    mdarray<double, 2>
    matrix_elements(Flm const& d_rho__, Flm const& d_tau__) const
    {
        auto result = ae_.matrix_elements(d_rho__, d_tau__);
        auto pseudo = ps_.matrix_elements(d_rho__, d_tau__);
        for (size_t i = 0; i < result.size(); i++) {
            result[i] -= pseudo[i];
        }
        return result;
    }
};

} // namespace sirius

#endif
