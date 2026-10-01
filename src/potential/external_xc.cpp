/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "potential.hpp"
#include "density/lapw_xc_field_map.hpp"
#include "density/paw_xc_field_map.hpp"

namespace sirius {

xc_point_set_t::xc_point_set_t(Simulation_context const& ctx__)
{
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            lattice_[i + 3 * j] = ctx__.unit_cell().lattice_vectors()(i, j);
        }
    }
    for (int ia = 0; ia < ctx__.unit_cell().num_atoms(); ia++) {
        auto r = ctx__.unit_cell().atom(ia).position();
        atoms_.push_back({r[0], r[1], r[2]});
    }
}

void
xc_point_set_t::validate(Simulation_context const& ctx__) const
{
    size_t nr  = coordinates.size();
    int failed = nr > static_cast<size_t>(std::numeric_limits<int>::max()) || energy_weights.size() != nr ||
                 descriptor_weights.size() != nr || block_sizes.size() != block_centres.size() ||
                 atoms_.size() != static_cast<size_t>(ctx__.unit_cell().num_atoms());
    int64_t count{0};
    for (int n : block_sizes) {
        failed |= n <= 0;
        count += n;
    }
    failed |= count != static_cast<int64_t>(nr);
    for (auto const* a : {&coordinates, &block_centres}) {
        for (auto const& r : *a) {
            for (double x : r) {
                failed |= !std::isfinite(x);
            }
        }
    }
    for (auto const* a : {&energy_weights, &descriptor_weights}) {
        for (double w : *a) {
            failed |= !std::isfinite(w);
        }
    }
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            failed |= std::abs(lattice_[i + 3 * j] - ctx__.unit_cell().lattice_vectors()(i, j)) > 1e-12;
        }
    }
    if (atoms_.size() == static_cast<size_t>(ctx__.unit_cell().num_atoms())) {
        for (int ia = 0; ia < ctx__.unit_cell().num_atoms(); ia++) {
            for (int x = 0; x < 3; x++) {
                failed |= std::abs(atoms_[ia][x] - ctx__.unit_cell().atom(ia).position()[x]) > 1e-12;
            }
        }
    }
    int any_points = nr > 0;
    ctx__.comm().allreduce<int, mpi::op_t::max>(&any_points, 1);
    failed |= !any_points;
    ctx__.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("external XC: invalid point quadrature or geometry changed since its construction");
    }
}

namespace {

void
evaluate_external_xc(Simulation_context const& ctx, xc_callback_input_t const& input, xc_callback_output_t& output)
{
    size_t n    = static_cast<size_t>(input.num_points) * input.num_spins;
    auto finite = [](double const* p, size_t count) {
        for (size_t i = 0; i < count; i++) {
            if (!std::isfinite(p[i])) {
                return false;
            }
        }
        return true;
    };
    int failed = !finite(input.rho, n) || !finite(input.gradient, 3 * n) || !finite(input.tau, n);
    ctx.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("external XC received nonfinite reconstructed fields on a context rank");
    }
    std::string message;
    try {
        ctx.cb().external_xc_(ctx, input, output);
    } catch (std::exception const& e) {
        failed  = 1;
        message = e.what();
    }
    failed |= !std::isfinite(output.energy) || !finite(output.vrho, n) || !finite(output.vgradient, 3 * n) ||
              !finite(output.vtau, n);
    ctx.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("external XC failed or returned nonfinite output on a context rank: " + message);
    }
    double emin = output.energy, emax = output.energy;
    ctx.comm().allreduce<double, mpi::op_t::min>(&emin, 1);
    ctx.comm().allreduce<double, mpi::op_t::max>(&emax, 1);
    if (emax - emin > 1e-12 * std::max(1.0, std::max(std::abs(emin), std::abs(emax)))) {
        RTE_THROW("external XC must return the same total energy on every context rank");
    }
}

} // namespace

void
validate_external_xc_context(Simulation_context const& ctx__)
{
    if (!ctx__.cb().external_xc_) {
        return;
    }
    auto const& points = ctx__.cb().external_xc_points_;
    int point_min = bool(points), point_max = point_min;
    ctx__.comm().allreduce<int, mpi::op_t::min>(&point_min, 1);
    ctx__.comm().allreduce<int, mpi::op_t::max>(&point_max, 1);
    if (point_min != point_max) {
        RTE_THROW("external XC point/FFT quadrature mode differs between context ranks");
    }
    if ((ctx__.full_potential() || ctx__.unit_cell().augment()) && !points) {
        RTE_THROW("external XC: PAW and full potential require reconstructed point quadrature");
    }
    if (ctx__.processing_unit() != device_t::CPU || ctx__.spfft<double>().processing_unit() != SPFFT_PU_HOST ||
        ctx__.spfft_coarse<double>().processing_unit() != SPFFT_PU_HOST || ctx__.num_mag_dims() == 3 ||
        ctx__.so_correction()) {
        RTE_THROW("external XC currently requires CPU FFTs and nonmagnetic or collinear scalar orbitals");
    }
    if (!ctx__.xc_functionals().empty() || ctx__.cb().veff_) {
        RTE_THROW("external XC cannot be combined with built-in XC or an effective-potential callback");
    }
    if (ctx__.cfg().parameters().extra_charge() != 0) {
        RTE_THROW("external XC: charged-cell background must be separated from the physical density first");
    }
    if (ctx__.cfg().parameters().veff_pw_cutoff() > 0) {
        RTE_THROW("external XC does not support an additional potential-only Fourier filter");
    }
    if (points) {
        points->validate(ctx__);
        if (ctx__.cfg().parameters().use_symmetry()) {
            RTE_THROW("external XC point quadrature requires symmetry disabled until its symmetry map is supplied");
        }
    }
    if (ctx__.full_potential()) {
        if (ctx__.valence_relativity() != relativity_t::none || ctx__.cfg().parameters().precision_wf() != "fp64") {
            RTE_THROW("external LAPW XC currently requires nonrelativistic fp64 valence orbitals");
        }
        if (ctx__.unit_cell().num_core_electrons() != 0 && ctx__.core_relativity() != relativity_t::none) {
            RTE_THROW("external LAPW XC: tau-dependent radial core response requires nonrelativistic core states");
        }
        return;
    }
    for (int iat = 0; iat < ctx__.unit_cell().num_atom_types(); iat++) {
        auto const& type = ctx__.unit_cell().atom_type(iat);
        auto const& core = type.ps_core_charge_density();
        if ((type.augment() && !type.is_paw()) || type.spin_orbit_coupling() ||
            (!type.is_paw() && std::any_of(core.begin(), core.end(), [](double x) { return x != 0; }))) {
            RTE_THROW("external XC: this atomic setup needs core or spinor reconstruction");
        }
        if (type.is_paw() && (!points || !type.has_paw_core_kinetic_density())) {
            RTE_THROW("external XC: PAW requires a point quadrature and explicitly supplied positive core tau");
        }
    }
}

void
Potential::xc_external(Density const& density__)
{
    PROFILE("sirius::Potential::xc_external");
    validate_external_xc_context(ctx_);
    if (ctx_.cb().external_xc_points_) {
        xc_external_points(density__);
        return;
    }
    auto& fft = ctx_.spfft<double>();
    int nr    = fft.local_slice_size();
    int ns    = ctx_.num_spins();
    mdarray<double, 2> rho({nr, ns}), tau({nr, ns}), vrho({nr, ns}), vtau({nr, ns});
    mdarray<double, 3> grad({nr, 3, ns}), vgrad({nr, 3, ns});
    vrho.zero();
    vtau.zero();
    vgrad.zero();

    for (int s = 0; s < ns; s++) {
        Smooth_periodic_function<double> r(fft, ctx_.gvec_fft_sptr());
        for (int ir = 0; ir < nr; ir++) {
            double n = (1 + add_delta_rho_xc_) * density__.rho().rg().value(ir);
            double t = density__.kinetic_density().scalar().rg().value(ir);
            if (ns == 2) {
                n = 0.5 * (n + (1 - 2 * s) * (1 + add_delta_mag_xc_) * density__.mag(0).rg().value(ir));
                t = 0.5 * (t + (1 - 2 * s) * density__.kinetic_density().vector(0).rg().value(ir));
            }
            rho(ir, s) = r.value(ir) = n;
            tau(ir, s)               = t;
        }
        r.fft_transform(-1);
        auto g = to_rg(gradient(r));
        for (int x = 0; x < 3; x++) {
            for (int ir = 0; ir < nr; ir++) {
                grad(ir, x, s) = g[x].value(ir);
            }
        }
    }

    xc_callback_input_t input{nr,
                              ns,
                              unit_cell_.omega() / ctx_.fft_grid().num_points(),
                              rho.at(memory_t::host),
                              grad.at(memory_t::host),
                              tau.at(memory_t::host)};
    xc_callback_output_t output{0, vrho.at(memory_t::host), vgrad.at(memory_t::host), vtau.at(memory_t::host)};
    evaluate_external_xc(ctx_, input, output);
    external_xc_energy_ = output.energy;
    xc_potential_->zero();
    kinetic_potential_->zero();

    for (int s = 0; s < ns; s++) {
        Smooth_periodic_vector_function<double> g(fft, ctx_.gvec_fft_sptr());
        for (int x = 0; x < 3; x++) {
            for (int ir = 0; ir < nr; ir++) {
                g[x].value(ir) = vgrad(ir, x, s);
            }
            g[x].fft_transform(-1);
        }
        auto div = to_rg(divergence(g));
        for (int ir = 0; ir < nr; ir++) {
            double v      = vrho(ir, s) - div.value(ir);
            double factor = ns == 1 ? 1.0 : 0.5;
            xc_potential_->rg().value(ir) += factor * v;
            kinetic_potential_->scalar().rg().value(ir) += factor * vtau(ir, s);
            if (ns == 2) {
                effective_magnetic_field(0).rg().value(ir) += (1 - 2 * s) * factor * v;
                kinetic_potential_->vector(0).rg().value(ir) += (1 - 2 * s) * factor * vtau(ir, s);
            }
        }
    }
    kinetic_potential_->fft_transform(-1);
}

void
Potential::xc_external_points(Density const& density__)
{
    PROFILE("sirius::Potential::xc_external_points");
    if (add_delta_rho_xc_ != 0 || add_delta_mag_xc_ != 0) {
        RTE_THROW("external XC point reconstruction does not support scaled-density diagnostic perturbations");
    }
    auto const& quadrature = *ctx_.cb().external_xc_points_;
    std::vector<r3::vector<double>> points;
    for (auto const& r : quadrature.coordinates) {
        points.push_back({r[0], r[1], r[2]});
    }
    int nr = static_cast<int>(points.size());
    int ns = ctx_.num_spins();
    mdarray<std::complex<double>, 2> augmentation;
    if (unit_cell_.num_paw_atoms()) {
        // Cancel Q in its original Fourier representation, not on a radial grid.
        augmentation = density__.generate_rho_aug();
    }
    std::vector<Smooth_periodic_function<double>> rho, tau;
    for (int s = 0; s < ns; s++) {
        rho.emplace_back(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
        tau.emplace_back(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
        for (int ig = 0; ig < ctx_.gvec().count(); ig++) {
            auto r = density__.rho().rg().f_pw_local(ig);
            auto t = density__.kinetic_density().scalar().rg().f_pw_local(ig);
            if (unit_cell_.num_paw_atoms()) {
                r -= augmentation(ig, 0);
            }
            if (ns == 2) {
                auto m = density__.mag(0).rg().f_pw_local(ig);
                if (unit_cell_.num_paw_atoms()) {
                    m -= augmentation(ig, 1);
                }
                r = 0.5 * (r + (1.0 - 2 * s) * m);
                t = 0.5 * (t + (1.0 - 2 * s) * density__.kinetic_density().vector(0).rg().f_pw_local(ig));
            }
            rho.back().f_pw_local(ig) = r;
            tau.back().f_pw_local(ig) = t;
        }
    }
    xc_point_fields_t derivatives(nr, ns);
    auto evaluate = [&](auto const& map) {
        auto fields = map.sample(rho, tau);
        derivatives.rho.zero();
        derivatives.gradient.zero();
        derivatives.tau.zero();
        xc_callback_input_t input{nr,
                                  ns,
                                  0,
                                  fields.rho.at(memory_t::host),
                                  fields.gradient.at(memory_t::host),
                                  fields.tau.at(memory_t::host),
                                  &quadrature};
        xc_callback_output_t output{0, derivatives.rho.at(memory_t::host), derivatives.gradient.at(memory_t::host),
                                    derivatives.tau.at(memory_t::host)};
        evaluate_external_xc(ctx_, input, output);
        return output.energy;
    };
    xc_potential_->zero();
    kinetic_potential_->zero();
    if (ctx_.full_potential()) {
        LAPW_xc_field_map map(density__, points);
        double energy = evaluate(map);
        set_lapw_xc_derivatives(map.adjoint(derivatives), energy);
        return;
    }
    PAW_xc_field_map map(density__, points);
    double energy        = evaluate(map);
    auto adjoint         = map.adjoint(derivatives);
    external_xc_energy_  = energy;
    external_paw_d_mtrx_ = std::move(adjoint.local_matrices);
    for (int s = 0; s < ns; s++) {
        for (int ig = 0; ig < ctx_.gvec().count(); ig++) {
            auto v = adjoint.rho[s].f_pw_local(ig) / double(ns);
            auto t = adjoint.tau[s].f_pw_local(ig) / double(ns);
            xc_potential_->rg().f_pw_local(ig) += v;
            kinetic_potential_->scalar().rg().f_pw_local(ig) += t;
            if (ns == 2) {
                effective_magnetic_field(0).rg().f_pw_local(ig) += (1.0 - 2 * s) * v;
                kinetic_potential_->vector(0).rg().f_pw_local(ig) += (1.0 - 2 * s) * t;
            }
        }
    }
    xc_potential_->rg().fft_transform(1);
    kinetic_potential_->fft_transform(1);
    if (ns == 2) {
        effective_magnetic_field(0).rg().fft_transform(1);
    }
}

} // namespace sirius
