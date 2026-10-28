/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "potential.hpp"
#include "function3d/smooth_periodic_function_sampler.hpp"
#include <limits>
#include <numeric>

namespace sirius {

xc_point_set_t::xc_point_set_t(Simulation_context const& ctx__)
{
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            lattice_[i + 3 * j] = ctx__.unit_cell().lattice_vectors()(i, j);
        }
    }
    for (int ia = 0; ia < ctx__.unit_cell().num_atoms(); ia++) {
        auto p = ctx__.unit_cell().atom(ia).position();
        atoms_.push_back({p[0], p[1], p[2]});
    }
}

void
xc_point_set_t::validate(Simulation_context const& ctx__) const
{
    size_t n   = coordinates.size();
    int failed = n > static_cast<size_t>(std::numeric_limits<int>::max()) || energy_weights.size() != n ||
                 descriptor_weights.size() != n || block_sizes.size() != block_centres.size() ||
                 atoms_.size() != static_cast<size_t>(ctx__.unit_cell().num_atoms());
    int64_t count{0};
    for (int b : block_sizes) {
        failed |= b <= 0;
        count += b;
    }
    failed |= count != static_cast<int64_t>(n);
    for (auto const* a : {&coordinates, &block_centres}) {
        for (auto const& p : *a) {
            for (double x : p) {
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
    int any = n != 0;
    ctx__.comm().allreduce<int, mpi::op_t::max>(&any, 1);
    failed |= !any;
    ctx__.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("external XC requires a finite, fixed point quadrature with complete atom blocks");
    }
}

void
validate_external_xc_context(Simulation_context const& ctx__)
{
    int failed = ctx__.full_potential() || ctx__.processing_unit() != device_t::CPU ||
                 ctx__.spfft<double>().processing_unit() != SPFFT_PU_HOST ||
                 ctx__.spfft_coarse<double>().processing_unit() != SPFFT_PU_HOST || ctx__.num_mag_dims() == 3 ||
                 ctx__.so_correction() || ctx__.veff_callback() || ctx__.cfg().parameters().precision_wf() != "fp64" ||
                 ctx__.cfg().parameters().veff_pw_cutoff() > 0 || !ctx__.xc_functionals().empty() ||
                 (ctx__.cb().external_xc_points_ && ctx__.use_symmetry());
    for (int iat = 0; iat < ctx__.unit_cell().num_atom_types(); iat++) {
        auto const& type = ctx__.unit_cell().atom_type(iat);
        failed |= type.augment() || type.spin_orbit_coupling();
        for (double c : type.ps_core_charge_density()) {
            failed |= c != 0;
        }
    }
    ctx__.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("external XC requires CPU scalar/collinear fp64 norm-conserving PW calculations without core "
                  "charge, other XC functionals or potential callbacks; point quadratures require disabled symmetry");
    }
    if (ctx__.cb().external_xc_points_) {
        ctx__.cb().external_xc_points_->validate(ctx__);
    }
}

namespace {

void
evaluate(Simulation_context const& ctx, xc_callback_input_t const& in, xc_callback_output_t& out)
{
    auto finite = [](double const* p, size_t n) {
        if (n == 0) {
            return true;
        }
        return std::all_of(p, p + n, [](double v) { return std::isfinite(v); });
    };
    size_t n   = static_cast<size_t>(in.num_points) * in.num_spins;
    int failed = !finite(in.rho, n) || !finite(in.gradient, 3 * n) || !finite(in.tau, n);
    ctx.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("external XC received nonfinite fields");
    }
    std::string error;
    try {
        ctx.cb().external_xc_(ctx, in, out);
    } catch (std::exception const& e) {
        failed = 1;
        error  = e.what();
    }
    failed |=
            !std::isfinite(out.energy) || !finite(out.vrho, n) || !finite(out.vgradient, 3 * n) || !finite(out.vtau, n);
    ctx.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("external XC evaluation failed on a context rank: " + error);
    }
    double lo = out.energy, hi = out.energy;
    ctx.comm().allreduce<double, mpi::op_t::min>(&lo, 1);
    ctx.comm().allreduce<double, mpi::op_t::max>(&hi, 1);
    if (hi - lo > 1e-12 * std::max(1.0, std::max(std::abs(lo), std::abs(hi)))) {
        RTE_THROW("external XC must return the same globally summed energy on every context rank");
    }
}

} // namespace

/// Evaluate the external functional and its variational orbital contribution.
/** FFT derivatives give v_sigma=dE/drho_sigma-div(dE/dgrad rho_sigma) and
 *  v_tau,sigma=dE/dtau_sigma after division by the grid weight. For arbitrary
 *  points the Fourier sampler applies the adjoint to raw discrete derivatives.
 *  Since tau_sigma=1/2 sum_nk w_k f_nk,sigma |grad psi_nk,sigma|^2,
 *  the orbital Hamiltonian receives -1/2 div(v_tau,sigma grad psi).
 *  Store total/up-minus-down potentials with the inverse spin transformation
 *  so SIRIUS applies the same spin-channel operator in exact and Davidson solves.
 */
void
Potential::xc_external(Density const& density__)
{
    validate_external_xc_context(ctx_);
    if (add_delta_rho_xc_ != 0 || add_delta_mag_xc_ != 0) {
        RTE_THROW("external XC does not support scaled-density diagnostic perturbations");
    }
    auto const& points = ctx_.cb().external_xc_points_;
    std::vector<r3::vector<double>> coordinates;
    if (points) {
        for (auto const& p : points->coordinates) {
            coordinates.emplace_back(p);
        }
    }
    std::unique_ptr<Smooth_periodic_function_sampler> sampler;
    if (points) {
        sampler = std::make_unique<Smooth_periodic_function_sampler>(ctx_.gvec(), coordinates);
    }
    auto& fft = ctx_.spfft<double>();
    int nr    = points ? static_cast<int>(coordinates.size()) : fft.local_slice_size();
    int ns    = ctx_.num_spins();
    mdarray<double, 2> rho({nr, ns}), tau({nr, ns}), vrho({nr, ns}), vtau({nr, ns});
    mdarray<double, 3> grad({nr, 3, ns}), vgrad({nr, 3, ns});
    vrho.zero();
    vtau.zero();
    vgrad.zero();
    for (int s = 0; s < ns; s++) {
        Smooth_periodic_function<double> r(fft, ctx_.gvec_fft_sptr()), t(fft, ctx_.gvec_fft_sptr());
        for (int ig = 0; ig < ctx_.gvec().count(); ig++) {
            auto n = density__.rho().rg().f_pw_local(ig);
            auto k = density__.kinetic_density().scalar().rg().f_pw_local(ig);
            if (ctx_.gvec().offset() + ig == 0) {
                n -= ctx_.cfg().parameters().extra_charge() / unit_cell_.omega();
            }
            if (ns == 2) {
                n = 0.5 * (n + double(1 - 2 * s) * density__.mag(0).rg().f_pw_local(ig));
                k = 0.5 * (k + double(1 - 2 * s) * density__.kinetic_density().vector(0).rg().f_pw_local(ig));
            }
            r.f_pw_local(ig) = n;
            t.f_pw_local(ig) = k;
        }
        if (sampler) {
            auto sampled_r = sampler->sample(r), sampled_t = sampler->sample(t);
            for (int p = 0; p < nr; p++) {
                rho(p, s) = sampled_r[p][0];
                tau(p, s) = sampled_t[p][0];
                for (int x = 0; x < 3; x++) {
                    grad(p, x, s) = sampled_r[p][x + 1];
                }
            }
        } else {
            auto g = to_rg(gradient(r));
            r.fft_transform(1);
            t.fft_transform(1);
            for (int p = 0; p < nr; p++) {
                rho(p, s) = r.value(p);
                tau(p, s) = t.value(p);
                for (int x = 0; x < 3; x++) {
                    grad(p, x, s) = g[x].value(p);
                }
            }
        }
    }
    xc_callback_input_t in{nr,
                           ns,
                           unit_cell_.omega() / ctx_.fft_grid().num_points(),
                           rho.at(memory_t::host),
                           grad.at(memory_t::host),
                           tau.at(memory_t::host),
                           points.get()};
    xc_callback_output_t out{0, vrho.at(memory_t::host), vgrad.at(memory_t::host), vtau.at(memory_t::host)};
    evaluate(ctx_, in, out);
    external_xc_energy_ = out.energy;
    kinetic_potential_->zero();
    for (int s = 0; s < ns; s++) {
        Smooth_periodic_function<double> v(fft, ctx_.gvec_fft_sptr()), t(fft, ctx_.gvec_fft_sptr());
        if (sampler) {
            std::vector<std::array<double, 4>> dr(nr), dt(nr);
            for (int p = 0; p < nr; p++) {
                dr[p] = {vrho(p, s), vgrad(p, 0, s), vgrad(p, 1, s), vgrad(p, 2, s)};
                dt[p] = {vtau(p, s), 0, 0, 0};
            }
            sampler->adjoint(dr, v);
            sampler->adjoint(dt, t);
            v.fft_transform(1);
            t.fft_transform(1);
        } else {
            Smooth_periodic_vector_function<double> g(fft, ctx_.gvec_fft_sptr());
            for (int x = 0; x < 3; x++) {
                for (int p = 0; p < nr; p++) {
                    g[x].value(p) = vgrad(p, x, s);
                }
                g[x].fft_transform(-1);
            }
            auto div = to_rg(divergence(g));
            for (int p = 0; p < nr; p++) {
                v.value(p) = vrho(p, s) - div.value(p);
                t.value(p) = vtau(p, s);
            }
        }
        for (int p = 0; p < fft.local_slice_size(); p++) {
            double weight = ns == 1 ? 1 : 0.5;
            xc_potential_->rg().value(p) += weight * v.value(p);
            kinetic_potential_->scalar().rg().value(p) += weight * t.value(p);
            if (ns == 2) {
                effective_magnetic_field(0).rg().value(p) += double(1 - 2 * s) * weight * v.value(p);
                kinetic_potential_->vector(0).rg().value(p) += double(1 - 2 * s) * weight * t.value(p);
            }
        }
    }
    kinetic_potential_->fft_transform(-1);
}

} // namespace sirius
