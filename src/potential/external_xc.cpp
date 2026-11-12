/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "potential.hpp"
#include "function3d/smooth_periodic_function_sampler.hpp"
#include "radial/spline.hpp"
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
    int invalid_core{0};
    for (int iat = 0; iat < ctx__.unit_cell().num_atom_types(); iat++) {
        auto const& type = ctx__.unit_cell().atom_type(iat);
        failed |= type.augment() || type.spin_orbit_coupling();
        auto const& core = type.ps_core_charge_density();
        auto const& tau  = type.ps_core_kinetic_density();
        bool has_core    = std::any_of(core.begin(), core.end(), [](double n) { return n != 0; });
        if (has_core || !tau.empty()) {
            auto n = static_cast<size_t>(type.num_mt_points());
            invalid_core |= !ctx__.cb().external_xc_points_ || !has_core || n < 4 || core.size() > n ||
                            tau.size() != n ||
                            !std::all_of(core.begin(), core.end(), [](double v) { return std::isfinite(v); });
        }
    }
    ctx__.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    ctx__.comm().allreduce<int, mpi::op_t::max>(&invalid_core, 1);
    if (failed) {
        RTE_THROW("external XC requires CPU scalar/collinear fp64 norm-conserving PW calculations without "
                  "other XC functionals or potential callbacks; point quadratures require disabled symmetry");
    }
    if (invalid_core) {
        RTE_THROW("external XC with core charge requires a point quadrature and matching pseudo-core kinetic density");
    }
    if (ctx__.cb().external_xc_points_) {
        ctx__.cb().external_xc_points_->validate(ctx__);
    }
}

namespace {

/// Value and first two radial derivatives of the same spline.
std::array<double, 3>
core_radial_sample(Spline<double> const& spline, double r)
{
    if (r < spline.first()) {
        // Continue the innermost segment to second order, as for the spherical l=0 field.
        double dr = r - spline.first();
        double d1 = spline.deriv(1, 0), d2 = spline.deriv(2, 0);
        return {spline(0) + dr * (d1 + 0.5 * dr * d2), d1 + dr * d2, d2};
    }
    int i     = spline.index_of(r);
    double dr = r - spline.x(i);
    return {spline(i, dr), spline.deriv(1, i, dr), spline.deriv(2, i, dr)};
}

template <typename F>
void
for_each_core_sample(Simulation_context const& ctx, std::vector<r3::vector<double>> const& points, F&& f)
{
    auto const& cell = ctx.unit_cell();
    auto const& a    = cell.lattice_vectors();
    int failed{0};
    try {
        for (int iat = 0; iat < cell.num_atom_types(); iat++) {
            auto const& type = cell.atom_type(iat);
            if (type.ps_core_kinetic_density().empty() || points.empty()) {
                continue;
            }
            auto const& grid = type.radial_grid();
            Spline<double> rho(grid, type.ps_core_charge_density()), tau(grid, type.ps_core_kinetic_density());
            auto limits = r3::find_translations(grid.last(), a);
            for (int i = 0; i < type.num_atoms(); i++) {
                int ia    = type.atom_id(i);
                auto atom = r3::dot(a, cell.atom(ia).position());
                for (int p = 0; p < static_cast<int>(points.size()); p++) {
                    auto displacement = points[p] - atom;
                    auto image        = r3::dot(cell.inverse_lattice_vectors(), displacement);
                    for (int x = 0; x < 3; x++) {
                        image[x] = std::nearbyint(image[x]);
                    }
                    displacement -= r3::dot(a, image);
                    for (int z = -limits[2]; z <= limits[2]; z++) {
                        for (int y = -limits[1]; y <= limits[1]; y++) {
                            for (int x = -limits[0]; x <= limits[0]; x++) {
                                r3::vector<double> shift{double(x), double(y), double(z)};
                                auto dr  = displacement - r3::dot(a, shift);
                                double r = dr.length();
                                if (r > grid.last()) {
                                    continue;
                                }
                                if (!std::isfinite(r) || r < 1e-12) {
                                    RTE_THROW("invalid pseudo-core sampling point at or near a nucleus");
                                }
                                f(p, ia, image + shift, r, (1 / r) * dr, core_radial_sample(rho, r),
                                  core_radial_sample(tau, r));
                            }
                        }
                    }
                }
            }
        }
    } catch (std::exception const&) {
        failed = 1;
    }
    ctx.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("pseudo-core sampling failed on a context rank");
    }
}

void
add_core_fields(Simulation_context const& ctx, std::vector<r3::vector<double>> const& points, mdarray<double, 2>& rho,
                mdarray<double, 3>& gradient, mdarray<double, 2>& tau)
{
    int ns = ctx.num_spins();
    for_each_core_sample(ctx, points,
                         [&](auto p, int, auto const&, double, auto const& u, auto const& n, auto const& t) {
                             for (int s = 0; s < ns; s++) {
                                 rho(p, s) += n[0] / ns;
                                 tau(p, s) += t[0] / ns;
                                 for (int x = 0; x < 3; x++) {
                                     gradient(p, x, s) += n[1] * u[x] / ns;
                                 }
                             }
                         });
}

std::array<std::vector<Smooth_periodic_function<double>>, 2>
orbital_fields(Simulation_context const& ctx, Density const& density)
{
    std::array<std::vector<Smooth_periodic_function<double>>, 2> fields;
    int ns = ctx.num_spins();
    for (auto& component : fields) {
        for (int s = 0; s < ns; s++) {
            component.emplace_back(ctx.spfft<double>(), ctx.gvec_fft_sptr());
        }
    }
    for (int ig = 0; ig < ctx.gvec().count(); ig++) {
        auto n = density.rho().rg().f_pw_local(ig);
        auto t = density.kinetic_density().scalar().rg().f_pw_local(ig);
        if (ctx.gvec().offset() + ig == 0) {
            n -= ctx.cfg().parameters().extra_charge() / ctx.unit_cell().omega();
        }
        if (ns == 1) {
            fields[0][0].f_pw_local(ig) = n;
            fields[1][0].f_pw_local(ig) = t;
        } else { // ns == 2: spin-up and spin-down channels
            auto mz                     = density.mag(0).rg().f_pw_local(ig);
            auto tz                     = density.kinetic_density().vector(0).rg().f_pw_local(ig);
            fields[0][0].f_pw_local(ig) = 0.5 * (n + mz);
            fields[0][1].f_pw_local(ig) = 0.5 * (n - mz);
            fields[1][0].f_pw_local(ig) = 0.5 * (t + tz);
            fields[1][1].f_pw_local(ig) = 0.5 * (t - tz);
        }
    }
    return fields;
}

void
evaluate(Simulation_context const& ctx, xc_callback_input_t const& in, xc_callback_output_t& out,
         xc_callback_t const& callback)
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
        callback(ctx, in, out);
    } catch (std::exception const& e) {
        failed = 1;
        error  = e.what();
    }
    failed |=
            !std::isfinite(out.energy) || !finite(out.vrho, n) || !finite(out.vgradient, 3 * n) || !finite(out.vtau, n);
    if (out.geometry) {
        auto const& d = *out.geometry;
        failed |= !in.points || d.coordinates.size() != static_cast<size_t>(in.num_points) ||
                  d.energy_weights.size() != static_cast<size_t>(in.num_points) ||
                  d.descriptor_weights.size() != static_cast<size_t>(in.num_points) ||
                  (in.points && d.block_centres.size() != in.points->block_sizes.size());
        failed |= !finite(d.energy_weights.data(), d.energy_weights.size()) ||
                  !finite(d.descriptor_weights.data(), d.descriptor_weights.size());
        for (auto const* values : {&d.coordinates, &d.block_centres}) {
            for (auto const& row : *values) {
                failed |= !finite(row.data(), row.size());
            }
        }
    }
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
    auto fields = orbital_fields(ctx_, density__);
    for (int s = 0; s < ns; s++) {
        auto& r = fields[0][s];
        auto& t = fields[1][s];
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
    if (points) {
        add_core_fields(ctx_, coordinates, rho, grad, tau);
    }
    xc_callback_input_t in{nr,
                           ns,
                           unit_cell_.omega() / ctx_.fft_grid().num_points(),
                           rho.at(memory_t::host),
                           grad.at(memory_t::host),
                           tau.at(memory_t::host),
                           points.get()};
    xc_callback_output_t out{0, vrho.at(memory_t::host), vgrad.at(memory_t::host), vtau.at(memory_t::host)};
    evaluate(ctx_, in, out, ctx_.cb().external_xc_);
    external_xc_energy_ = out.energy;
    kinetic_potential_->zero();
    // Reuse sampled-field buffers for the spin-channel potentials.
    auto potentials = std::move(fields);
    for (int s = 0; s < ns; s++) {
        auto& v = potentials[0][s];
        auto& t = potentials[1][s];
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
    }
    for (int p = 0; p < fft.local_slice_size(); p++) {
        if (ns == 1) {
            xc_potential_->rg().value(p) += potentials[0][0].value(p);
            kinetic_potential_->scalar().rg().value(p) += potentials[1][0].value(p);
        } else { // ns == 2: transform spin channels to scalar and magnetic potentials
            auto v_up = potentials[0][0].value(p), v_dn = potentials[0][1].value(p);
            auto t_up = potentials[1][0].value(p), t_dn = potentials[1][1].value(p);
            xc_potential_->rg().value(p) += 0.5 * (v_up + v_dn);
            effective_magnetic_field(0).rg().value(p) += 0.5 * (v_up - v_dn);
            kinetic_potential_->scalar().rg().value(p) += 0.5 * (t_up + t_dn);
            kinetic_potential_->vector(0).rg().value(p) += 0.5 * (t_up - t_dn);
        }
    }
    kinetic_potential_->fft_transform(-1);
}

xc_geometry_derivatives_t
Potential::external_xc_geometry(Density const& density__) const
{
    int failed = !ctx_.cb().external_xc_ || !ctx_.cb().external_xc_points_ || !ctx_.cb().external_xc_geometry_ ||
                 !ctx_.cb().external_xc_pullback_ || !density__.has_kinetic_density() || add_delta_rho_xc_ != 0 ||
                 add_delta_mag_xc_ != 0;
    ctx_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("external XC geometry requires point, field-geometry and quadrature-pullback callbacks");
    }
    validate_external_xc_context(ctx_);
    auto const& q = *ctx_.cb().external_xc_points_;
    std::vector<r3::vector<double>> points;
    for (auto const& r : q.coordinates) {
        points.emplace_back(r);
    }
    Smooth_periodic_function_sampler sampler(ctx_.gvec(), points);
    auto fields = orbital_fields(ctx_, density__);
    int nr = static_cast<int>(points.size()), ns = ctx_.num_spins();
    mdarray<double, 2> rho({nr, ns}), tau({nr, ns}), vrho({nr, ns}), vtau({nr, ns});
    mdarray<double, 3> grad({nr, 3, ns}), vgrad({nr, 3, ns});
    vrho.zero();
    vtau.zero();
    vgrad.zero();
    for (int s = 0; s < ns; s++) {
        auto n = sampler.sample(fields[0][s]), t = sampler.sample(fields[1][s]);
        for (int p = 0; p < nr; p++) {
            rho(p, s) = n[p][0];
            tau(p, s) = t[p][0];
            for (int x = 0; x < 3; x++) {
                grad(p, x, s) = n[p][x + 1];
            }
        }
    }
    add_core_fields(ctx_, points, rho, grad, tau);
    xc_point_derivatives_t d(q);
    xc_callback_input_t in{nr, ns, 0, rho.at(memory_t::host), grad.at(memory_t::host), tau.at(memory_t::host), &q};
    xc_callback_output_t out{0, vrho.at(memory_t::host), vgrad.at(memory_t::host), vtau.at(memory_t::host), &d};
    evaluate(ctx_, in, out, ctx_.cb().external_xc_geometry_);
    r3::matrix<double> lattice;
    for (int s = 0; s < ns; s++) {
        std::vector<std::array<double, 4>> dn(nr), dt(nr);
        for (int p = 0; p < nr; p++) {
            dn[p] = {vrho(p, s), vgrad(p, 0, s), vgrad(p, 1, s), vgrad(p, 2, s)};
            dt[p] = {vtau(p, s), 0, 0, 0};
        }
        auto n = sampler.geometry_adjoint(fields[0][s], dn), t = sampler.geometry_adjoint(fields[1][s], dt);
        lattice += n.lattice + t.lattice;
        for (int p = 0; p < nr; p++) {
            for (int x = 0; x < 3; x++) {
                d.coordinates[p][x] += n.points[p][x] + t.points[p][x];
            }
        }
    }
    int nat = unit_cell_.num_atoms();
    mdarray<double, 2> core_atoms({3, nat});
    core_atoms.zero();
    r3::matrix<double> core_lattice;
    // For d = r_p - R_a - A n, the point derivative is the radial gradient
    // contraction plus H(rho_core) times the gradient covector. The atom
    // derivative has the opposite sign, and the cell derivative is -dE/dd * n^T.
    for_each_core_sample(ctx_, points,
                         [&](auto p, int ia, auto const& image, double r, auto const& u, auto const& n, auto const& t) {
                             r3::vector<double> v{0, 0, 0};
                             double transverse = n[1] / r;
                             for (int s = 0; s < ns; s++) {
                                 r3::vector<double> g{vgrad(p, 0, s), vgrad(p, 1, s), vgrad(p, 2, s)};
                                 double radial =
                                         vrho(p, s) * n[1] + vtau(p, s) * t[1] + (n[2] - transverse) * r3::dot(g, u);
                                 v += (1.0 / ns) * (radial * u + transverse * g);
                             }
                             for (int x = 0; x < 3; x++) {
                                 d.coordinates[p][x] += v[x];
                                 core_atoms(x, ia) -= v[x];
                                 for (int y = 0; y < 3; y++) {
                                     core_lattice(x, y) -= v[x] * image[y];
                                 }
                             }
                         });
    ctx_.comm().allreduce(core_atoms.at(memory_t::host), core_atoms.size());
    ctx_.comm().allreduce(&core_lattice(0, 0), 9);
    lattice += core_lattice;
    xc_geometry_derivatives_t result(nat);
    try {
        ctx_.cb().external_xc_pullback_(ctx_, q, d, result);
    } catch (std::exception const&) {
        failed = 1;
    }
    failed |= result.atoms.size() != static_cast<size_t>(unit_cell_.num_atoms());
    std::vector<double> values(result.lattice.begin(), result.lattice.end());
    for (auto const& atom : result.atoms) {
        values.insert(values.end(), atom.begin(), atom.end());
    }
    for (double x : values) {
        failed |= !std::isfinite(x);
    }
    ctx_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("external XC quadrature pullback returned invalid geometry derivatives");
    }
    auto minimum = values, maximum = values;
    ctx_.comm().allreduce<double, mpi::op_t::min>(minimum.data(), minimum.size());
    ctx_.comm().allreduce<double, mpi::op_t::max>(maximum.data(), maximum.size());
    for (size_t i = 0; i < values.size(); i++) {
        if (maximum[i] - minimum[i] > 1e-12 * std::max(1.0, std::abs(minimum[i]))) {
            RTE_THROW("external XC quadrature pullback must return globally summed derivatives on every rank");
        }
    }
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            result.lattice[i + 3 * j] += lattice(i, j);
            failed |= !std::isfinite(result.lattice[i + 3 * j]);
        }
    }
    for (int ia = 0; ia < nat; ia++) {
        for (int x = 0; x < 3; x++) {
            result.atoms[ia][x] += core_atoms(x, ia);
            failed |= !std::isfinite(result.atoms[ia][x]);
        }
    }
    ctx_.comm().allreduce<int, mpi::op_t::max>(&failed, 1);
    if (failed) {
        RTE_THROW("external XC geometry returned nonfinite derivatives");
    }
    return result;
}

} // namespace sirius
