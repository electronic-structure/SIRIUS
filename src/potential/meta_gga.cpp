/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "potential.hpp"
#include "xc_mt_meta.hpp"

namespace sirius {

void
Potential::xc_rg_meta(Density const& density__, lapw_xc_field_adjoint_t* lapw__)
{
    PROFILE("sirius::Potential::xc_rg_meta");
    int ns      = ctx_.num_spins();
    int nr      = ctx_.spfft<double>().local_slice_size();
    int ng      = ns == 1 ? 1 : 3;
    int invalid = !density__.has_kinetic_density() || add_delta_rho_xc_ != 0 || add_delta_mag_xc_ != 0 ||
                  ctx_.full_potential() != bool(lapw__);
    ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("meta-GGA requires stored kinetic density and unscaled physical fields");
    }

    std::vector<Smooth_periodic_function<double>> spin_rho;
    std::vector<Smooth_periodic_vector_function<double>> gradient_rho;
    std::vector<Smooth_periodic_function<double>> compensation;
    if (!lapw__ && unit_cell_.num_paw_atoms()) {
        // Bloechl PAW XC uses orbital valence + pseudo core, with no compensation
        // charge in either the grid or the matching local pseudo contribution.
        auto q = density__.generate_rho_aug();
        for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
            compensation.emplace_back(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
            for (int ig = 0; ig < ctx_.gvec().count(); ig++) {
                compensation.back().f_pw_local(ig) = q(ig, j);
            }
            compensation.back().fft_transform(1);
        }
    }
    std::vector<double> rho(nr * ns), tau(nr * ns), sigma(nr * ng);
    for (int s = 0; s < ns; s++) {
        spin_rho.emplace_back(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
        for (int ir = 0; ir < nr; ir++) {
            double n = density__.rho().rg().value(ir);
            double t = density__.kinetic_density().scalar().rg().value(ir);
            if (!lapw__) {
                n += density__.rho_pseudo_core().value(ir);
                t += density__.tau_pseudo_core().value(ir);
            }
            if (ns == 2) {
                n = 0.5 * (n + (1 - 2 * s) * density__.mag(0).rg().value(ir));
                t = 0.5 * (t + (1 - 2 * s) * density__.kinetic_density().vector(0).rg().value(ir));
            }
            if (!compensation.empty()) {
                n -= (compensation[0].value(ir) + (ns == 1 ? 0 : (1 - 2 * s) * compensation[1].value(ir))) / ns;
            }
            invalid |= !std::isfinite(n) || !std::isfinite(t) || n < 0 || t < 0;
            spin_rho.back().value(ir) = rho[ns * ir + s] = n;
            tau[ns * ir + s]                             = t;
        }
    }
    ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        double minimum_rho = std::numeric_limits<double>::infinity();
        double minimum_tau = minimum_rho;
        for (double n : rho) {
            minimum_rho = std::min(minimum_rho, n);
        }
        for (double t : tau) {
            minimum_tau = std::min(minimum_tau, t);
        }
        ctx_.comm().allreduce<double, mpi::op_t::min>(&minimum_rho, 1);
        ctx_.comm().allreduce<double, mpi::op_t::min>(&minimum_tau, 1);
        std::stringstream message;
        message << "meta-GGA grid contains nonfinite or negative spin density/tau; check mixing and grid resolution"
                << "; minimum rho=" << minimum_rho << ", tau=" << minimum_tau;
        RTE_THROW(message);
    }
    for (auto& n : spin_rho) {
        n.fft_transform(-1);
        gradient_rho.push_back(to_rg(gradient(n)));
    }
    for (int ir = 0; ir < nr; ir++) {
        for (int x = 0; x < 3; x++) {
            double up = gradient_rho[0][x].value(ir);
            sigma[ng * ir] += up * up;
            if (ns == 2) {
                double dn = gradient_rho[1][x].value(ir);
                sigma[ng * ir + 1] += up * dn;
                sigma[ng * ir + 2] += dn * dn;
            }
        }
    }

    std::vector<double> vrho(nr * ns), vtau(nr * ns), vsigma(nr * ng), exc(nr);
    std::vector<double> total_rho(nr * ns), total_tau(nr * ns), total_sigma(nr * ng);
    vdw_energy_ = 0;
    // Libxc uses interleaved spin components. Stored fields use total/difference.
    try {
        for (auto& functional : xc_func_) {
            std::fill(vsigma.begin(), vsigma.end(), 0);
            std::fill(vtau.begin(), vtau.end(), 0);
            if (nr) {
                if (functional.is_meta_gga()) {
                    functional.get_meta(nr, rho.data(), sigma.data(), tau.data(), vrho.data(), vsigma.data(),
                                        vtau.data(), exc.data());
                } else if (functional.is_gga()) {
                    xc_gga_exc_vxc(functional.handler(), nr, rho.data(), sigma.data(), exc.data(), vrho.data(),
                                   vsigma.data());
                } else {
                    xc_lda_exc_vxc(functional.handler(), nr, rho.data(), exc.data(), vrho.data());
                }
            }
            double weight = functional.weight();
            for (int i = 0; i < nr * ns; i++) {
                total_rho[i] += weight * vrho[i];
                total_tau[i] += weight * vtau[i];
            }
            for (int i = 0; i < nr * ng; i++) {
                total_sigma[i] += weight * vsigma[i];
            }
            for (int ir = 0; ir < nr; ir++) {
                xc_energy_density_->rg().value(ir) += weight * exc[ir];
            }
        }
    } catch (std::exception const&) {
        invalid = 1;
    }
    for (auto const* values : {&total_rho, &total_tau, &total_sigma}) {
        for (double v : *values) {
            invalid |= !std::isfinite(v);
        }
    }
    for (int ir = 0; ir < nr; ir++) {
        invalid |= !std::isfinite(xc_energy_density_->rg().value(ir));
    }
    ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("Libxc meta-GGA evaluation failed or returned nonfinite fields");
    }

    if (!compensation.empty()) {
        double energy{0};
        for (int ir = 0; ir < nr; ir++) {
            for (int s = 0; s < ns; s++) {
                energy += rho[ns * ir + s] * xc_energy_density_->rg().value(ir);
            }
        }
        energy *= unit_cell_.omega() / ctx_.fft_grid().num_points();
        ctx_.comm_fft().allreduce(&energy, 1);
        paw_meta_grid_energy_ = energy;
    }

    for (int s = 0; s < ns; s++) {
        if (lapw__) {
            lapw__->rho.emplace_back(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
            lapw__->tau.emplace_back(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
        }
        Smooth_periodic_vector_function<double> flux(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
        for (int x = 0; x < 3; x++) {
            for (int ir = 0; ir < nr; ir++) {
                flux[x].value(ir) = 2 * total_sigma[ng * ir + 2 * s] * gradient_rho[s][x].value(ir);
                if (ns == 2) {
                    flux[x].value(ir) += total_sigma[ng * ir + 1] * gradient_rho[1 - s][x].value(ir);
                }
                // Differentiate the interstitial quadrature, including its boundary.
                if (lapw__) {
                    flux[x].value(ir) *= ctx_.theta(ir);
                }
            }
            flux[x].fft_transform(-1);
        }
        auto div = to_rg(divergence(flux));
        for (int ir = 0; ir < nr; ir++) {
            if (lapw__) {
                lapw__->rho[s].value(ir) = ctx_.theta(ir) * total_rho[ns * ir + s] - div.value(ir);
                lapw__->tau[s].value(ir) = ctx_.theta(ir) * total_tau[ns * ir + s];
                continue;
            }
            double v = (total_rho[ns * ir + s] - div.value(ir)) / ns;
            double t = total_tau[ns * ir + s] / ns;
            xc_potential_->rg().value(ir) += v;
            kinetic_potential_->scalar().rg().value(ir) += t;
            if (ns == 2) {
                effective_magnetic_field(0).rg().value(ir) += (1 - 2 * s) * v;
                kinetic_potential_->vector(0).rg().value(ir) += (1 - 2 * s) * t;
            }
        }
        if (lapw__) {
            lapw__->rho[s].fft_transform(-1);
            lapw__->tau[s].fft_transform(-1);
        }
    }
    for (int i = 0; i < ng; i++) {
        for (int ir = 0; ir < nr; ir++) {
            vsigma_[i]->value(ir) = total_sigma[ng * ir + i];
        }
    }
    kinetic_potential_->fft_transform(-1);
}

void
Potential::xc_lapw_meta(Density const& density__)
{
    lapw_xc_field_adjoint_t d;
    xc_rg_meta(density__, &d);
    double energy{0};
    for (int ir = 0; ir < ctx_.spfft<double>().local_slice_size(); ir++) {
        energy += ctx_.theta(ir) * density__.rho().rg().value(ir) * xc_energy_density_->rg().value(ir);
    }
    energy *= unit_cell_.omega() / ctx_.fft_grid().num_points();
    ctx_.comm_fft().allreduce(&energy, 1);

    int ns = ctx_.num_spins(), lmmax = sf::lmmax(ctx_.lmax_rho());
    d.local.resize(unit_cell_.num_atoms());
    d.core.resize(unit_cell_.num_atoms());
    double mt_energy{0};
    int failed{0};
    std::string error;
    try {
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto const& grid = unit_cell_.atom(ia).type().radial_grid();
            for (int part = 0; part < 2; part++) {
                for (int s = 0; s < ns; s++) {
                    d.local[ia][part].emplace_back(lmmax, grid);
                    d.local[ia][part].back().zero();
                }
                d.core[ia][part].resize(grid.num_points());
            }
        }
        for (auto it : unit_cell_.spl_num_atoms()) {
            auto& fields = d.local[it.i];
            for (int s = 0; s < ns; s++) {
                for (size_t i = 0; i < fields[0][s].size(); i++) {
                    fields[0][s][i] = density__.rho().mt()[it.i][i];
                    fields[1][s][i] = density__.kinetic_density().scalar().mt()[it.i][i];
                    if (ns == 2) {
                        fields[0][s][i] = 0.5 * (fields[0][s][i] + (1 - 2 * s) * density__.mag(0).mt()[it.i][i]);
                        fields[1][s][i] = 0.5 * (fields[1][s][i] +
                                                 (1 - 2 * s) * density__.kinetic_density().vector(0).mt()[it.i][i]);
                    }
                }
            }
            auto result = xc_mt_meta(*sht_, xc_func_, fields[0], fields[1]);
            mt_energy += result.energy;
            fields = std::move(result.fields);
        }
    } catch (std::exception const& e) {
        failed = 1;
        error  = e.what();
    }
    ctx_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("LAPW meta-GGA sphere evaluation failed on a context rank: " + error);
    }
    ctx_.comm().allreduce(&mt_energy, 1);
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        int owner = unit_cell_.spl_num_atoms().location(atom_index_t::global(ia)).ib;
        for (int part = 0; part < 2; part++) {
            for (int s = 0; s < ns; s++) {
                auto& field = d.local[ia][part][s];
                ctx_.comm().bcast(&field[0], field.size(), owner);
                for (int ir = 0; ir < field.radial_grid().num_points(); ir++) {
                    d.core[ia][part][ir] += field(0, ir) / (ns * y00);
                }
            }
        }
    }
    set_lapw_xc_derivatives(std::move(d), energy + mt_energy);
}

} // namespace sirius
