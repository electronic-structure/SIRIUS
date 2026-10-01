/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "potential.hpp"

namespace sirius {

void
Potential::set_lapw_xc_derivatives(lapw_xc_field_adjoint_t d__, std::optional<double> energy__)
{
    int present_min = energy__.has_value(), present_max = present_min;
    ctx_.comm().allreduce<int, mpi::op_t::min>(&present_min, 1);
    ctx_.comm().allreduce<int, mpi::op_t::max>(&present_max, 1);
    if (present_min != present_max) {
        RTE_THROW("LAPW XC energy presence differs between context ranks");
    }
    bool failed{false};
    try {
        failed = energy__ && !std::isfinite(*energy__);
        int ns = ctx_.num_spins();
        if (!ctx_.full_potential() || ctx_.processing_unit() != device_t::CPU ||
            ctx_.spfft<double>().processing_unit() != SPFFT_PU_HOST ||
            ctx_.spfft_coarse<double>().processing_unit() != SPFFT_PU_HOST || ctx_.num_mag_dims() == 3 ||
            ctx_.so_correction() || ctx_.valence_relativity() != relativity_t::none || d__.rho.size() != ns ||
            d__.tau.size() != ns || d__.local.size() != unit_cell_.num_atoms() ||
            d__.core.size() != unit_cell_.num_atoms()) {
            RTE_THROW("incompatible LAPW XC fields or unsupported relativistic/spin/device mode");
        }
        for (auto const* fields : {&d__.rho, &d__.tau}) {
            for (auto const& field : *fields) {
                if (&field.gvec() != &ctx_.gvec()) {
                    RTE_THROW("incompatible LAPW XC reciprocal grid");
                }
                for (auto v : field.f_pw_local()) {
                    failed |= !std::isfinite(v.real()) || !std::isfinite(v.imag());
                }
            }
        }
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto const& grid = unit_cell_.atom(ia).type().radial_grid();
            for (int part = 0; part < 2; part++) {
                if (d__.local[ia][part].size() != ns || d__.core[ia][part].size() != grid.num_points()) {
                    RTE_THROW("incompatible LAPW XC atom fields");
                }
                for (auto const& field : d__.local[ia][part]) {
                    if (field.angular_domain_size() != sf::lmmax(ctx_.lmax_rho()) ||
                        field.radial_grid().hash() != grid.hash()) {
                        RTE_THROW("incompatible LAPW XC radial grid or angular extent");
                    }
                    for (size_t i = 0; i < field.size(); i++) {
                        failed |= !std::isfinite(field[i]);
                    }
                }
                for (double v : d__.core[ia][part]) {
                    failed |= !std::isfinite(v);
                }
                for (int ir = 0; ir < grid.num_points(); ir++) {
                    double spherical{0};
                    for (int s = 0; s < ns; s++) {
                        spherical += d__.local[ia][part][s](0, ir) / (ns * y00);
                    }
                    double core = d__.core[ia][part][ir];
                    failed |= std::abs(core - spherical) > 1e-12 * std::max({1.0, std::abs(core), std::abs(spherical)});
                }
            }
        }
    } catch (std::exception const&) {
        failed = true;
    }
    int invalid = failed;
    ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("invalid or unsupported LAPW XC derivatives on a context rank");
    }
    if (energy__) {
        double emin = *energy__, emax = *energy__;
        ctx_.comm().allreduce<double, mpi::op_t::min>(&emin, 1);
        ctx_.comm().allreduce<double, mpi::op_t::max>(&emax, 1);
        if (emax - emin > 1e-12 * std::max(1.0, std::max(std::abs(emin), std::abs(emax)))) {
            RTE_THROW("LAPW XC must have the same total energy on every context rank");
        }
    }
    lapw_xc_derivatives_ = std::make_unique<lapw_xc_field_adjoint_t>(std::move(d__));
    lapw_xc_energy_      = energy__;
}

std::vector<std::array<std::vector<double>, 2>>
Potential::get_core_xc_derivatives() const
{
    if (!lapw_xc_derivatives_) {
        return {};
    }
    std::vector<std::array<std::vector<double>, 2>> result(unit_cell_.num_atom_symmetry_classes());
    for (int ic = 0; ic < unit_cell_.num_atom_symmetry_classes(); ic++) {
        auto const& cls = unit_cell_.atom_symmetry_class(ic);
        int nr          = cls.atom_type().num_mt_points();
        for (int part = 0; part < 2; part++) {
            result[ic][part].resize(nr, 0);
            for (int i = 0; i < cls.num_atoms(); i++) {
                auto const& d = lapw_xc_derivatives_->core[cls.atom_id(i)][part];
                for (int ir = 0; ir < nr; ir++) {
                    result[ic][part][ir] += d[ir] / cls.num_atoms();
                }
            }
        }
    }
    return result;
}

lapw_xc_contractions_t
Potential::lapw_xc_contractions(Density const& density__) const
{
    check_xc_energy_available();
    if (!lapw_xc_derivatives_) {
        return {};
    }
    int invalid = &density__.ctx() != &ctx_ || !density__.has_kinetic_density();
    ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("LAPW XC contractions require physical rho/tau from the same context");
    }
    auto const& d = *lapw_xc_derivatives_;
    int ns        = ctx_.num_spins();
    std::array<double, 3> result{0, 0, 0};
    for (int s = 0; s < ns; s++) {
        double sign = 1 - 2 * s;
        for (int ig = 0; ig < ctx_.gvec().count(); ig++) {
            double multiplicity = ctx_.gvec().reduced() && ig >= ctx_.gvec().skip_g0() ? 2 : 1;
            double weight       = ctx_.gvec().omega() * multiplicity / ns;
            auto r              = density__.rho().rg().f_pw_local(ig);
            auto t              = density__.kinetic_density().scalar().rg().f_pw_local(ig);
            result[0] += weight * std::real(std::conj(d.rho[s].f_pw_local(ig)) * r);
            if (ns == 2) {
                result[1] += weight * sign *
                             std::real(std::conj(d.rho[s].f_pw_local(ig)) * density__.mag(0).rg().f_pw_local(ig));
                t += sign * density__.kinetic_density().vector(0).rg().f_pw_local(ig);
            }
            result[2] += weight * std::real(std::conj(d.tau[s].f_pw_local(ig)) * t);
        }
        // MT covectors are replicated, but each physical sphere contributes once.
        // Stored fields already contain the core, so no separate core term is added.
        for (auto it : unit_cell_.spl_num_atoms()) {
            auto const& r = density__.rho().mt()[it.i];
            auto const& t = density__.kinetic_density().scalar().mt()[it.i];
            for (size_t i = 0; i < d.local[it.i][0][s].size(); i++) {
                double v = d.local[it.i][0][s][i] / ns;
                double w = d.local[it.i][1][s][i] / ns;
                result[0] += v * r[i];
                result[2] += w * t[i];
                if (ns == 2) {
                    result[1] += sign * v * density__.mag(0).mt()[it.i][i];
                    result[2] += sign * w * density__.kinetic_density().vector(0).mt()[it.i][i];
                }
            }
        }
    }
    ctx_.comm().allreduce(result.data(), result.size());
    for (double v : result) {
        if (!std::isfinite(v)) {
            RTE_THROW("nonfinite LAPW XC energy contraction");
        }
    }
    return {result[0], result[1], result[2]};
}

} // namespace sirius
