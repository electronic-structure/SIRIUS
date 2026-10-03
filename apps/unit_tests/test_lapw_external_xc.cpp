/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include "dft/dft_ground_state.hpp"
#include "hamiltonian/diagonalize_fp.hpp"

using namespace sirius;

namespace {

double
functional_scale(Simulation_context const& ctx)
{
    return ctx.unit_cell().num_core_electrons() != 0 ? 1e-5 : 1.0;
}

void
functional(Simulation_context const& ctx, xc_callback_input_t const& in, xc_callback_output_t& out)
{
    if (!in.points) {
        RTE_THROW("LAPW fixture requires reconstructed point fields");
    }
    double moment{0}, energy{0};
    for (int s = 0; s < in.num_spins; s++) {
        for (int p = 0; p < in.num_points; p++) {
            int j = p + in.num_points * s;
            moment += in.points->descriptor_weights[p] * (in.rho[j] + 0.3 * in.tau[j]);
        }
    }
    ctx.comm().allreduce(&moment, 1);
    double nonlocal = 0.01 * moment / ctx.unit_cell().omega();
    for (int s = 0; s < in.num_spins; s++) {
        for (int p = 0; p < in.num_points; p++) {
            int j    = p + in.num_points * s;
            double w = in.points->energy_weights[p], d = in.points->descriptor_weights[p];
            double r = in.rho[j], t = in.tau[j];
            energy += w * (0.02 * r * r + 0.01 * t * t + 0.005 * r * t);
            out.vrho[j] = w * (0.04 * r + 0.005 * t) + d * nonlocal;
            out.vtau[j] = w * (0.02 * t + 0.005 * r) + 0.3 * d * nonlocal;
            for (int x = 0; x < 3; x++) {
                int k = p + in.num_points * (x + 3 * s);
                energy += 0.002 * w * in.gradient[k] * in.gradient[k];
                out.vgradient[k] = 0.004 * w * in.gradient[k];
            }
        }
    }
    ctx.comm().allreduce(&energy, 1);
    out.energy = energy + 0.5 * nonlocal * moment;
    if (ctx.unit_cell().num_core_electrons() != 0) {
        // Keep the quadratic fixture's near-nucleus fields in the weak-response regime.
        double scale = functional_scale(ctx);
        out.energy *= scale;
        for (int j = 0; j < in.num_points * in.num_spins; j++) {
            out.vrho[j] *= scale;
            out.vtau[j] *= scale;
        }
        for (int j = 0; j < 3 * in.num_points * in.num_spins; j++) {
            out.vgradient[j] *= scale;
        }
    }
}

auto
context(int mag, bool distributed_fft, bool second_variation, bool core = false, bool relativistic_core = false)
{
    auto config                               = R"({"parameters": {
        "electronic_structure_method": "full_potential_lapwlo", "use_symmetry": false,
        "pw_cutoff": 5.0, "gk_cutoff": 1.5, "num_bands": 4, "xc_functionals": [],
        "core_relativity": "none", "valence_relativity": "none", "auto_rmt": 0,
        "lmax_apw": 3, "lmax_rho": 6, "lmax_pot": 6, "smearing_width": 0.01
    }, "control": {"processing_unit": "CPU", "verbosity": 0, "use_second_variation": true},
    "iterative_solver": {"type": "exact", "min_tolerance": 1e-11},
    "mixer": {"type": "linear", "beta": 0.4, "use_hartree": false}})"_json;
    config["parameters"]["num_mag_dims"]      = mag;
    config["parameters"]["fixed_mag"]         = mag ? 0.4 : 0.0;
    config["control"]["use_second_variation"] = second_variation;
    config["control"]["mpi_grid_dims"]        = {distributed_fft ? mpi::Communicator::world().size() : 1, 1};
    config["parameters"]["core_relativity"]   = relativistic_core ? "dirac" : "none";
    config["parameters"]["num_bands"]         = core ? 8 : 4;
    auto ctx                                  = std::make_unique<Simulation_context>(config);
    ctx->unit_cell().set_lattice_vectors({{6, 0, 0}, {0, 7, 0}, {0, 0, 8}});
    std::string label = core ? "Ne" : "He";
    auto& type        = ctx->unit_cell().add_atom_type(label);
    type.zn(core ? 10 : 2);
    type.set_radial_grid(radial_grid_t::power, core ? 3201 : 801, 1e-6, 1.8, 2.0);
    auto grid = Radial_grid_factory<double>(radial_grid_t::power, 3000, 1e-7, 20.0, 3.0);
    std::vector<double> rho(grid.num_points());
    for (int ir = 0; ir < grid.num_points(); ir++) {
        rho[ir] = core ? 2 * std::pow(9.0, 3) / pi * std::exp(-18 * grid[ir]) +
                                  8 * std::pow(1.6, 3) / pi * std::exp(-3.2 * grid[ir])
                       : 16 / pi * std::exp(-4 * grid[ir]);
    }
    type.free_atom_density(rho);
    type.set_free_atom_radial_grid(std::move(grid));
    for (int l = 0; l <= 3; l++) {
        type.add_aw_descriptor(std::max(core ? 2 : 1, l + 1), l, -0.3, 0, 0);
        type.add_aw_descriptor(std::max(core ? 2 : 1, l + 1), l, -0.3, 1, 0);
    }
    type.add_lo_descriptor(0, core ? 2 : 1, 0, -0.7, 0, 0);
    type.add_lo_descriptor(0, core ? 2 : 1, 0, -0.7, 1, 0);
    if (core) {
        type.set_configuration(1, 0, 1, 2.0, true);
    }
    ctx->unit_cell().add_atom(label, {0.23, 0.31, 0.43}, {0, 0, mag ? 0.2 : 0.0});
    auto points     = std::make_shared<xc_point_set_t>(*ctx);
    constexpr int n = 10;
    // All points on one rank exercise empty model ranks and replicated FFTs.
    if (ctx->comm().rank() == 0) {
        for (int z = 0; z < n; z++) {
            for (int y = 0; y < n; y++) {
                for (int x = 0; x < n; x++) {
                    std::array<double, 3> p{6 * (x + 0.37) / n, 7 * (y + 0.29) / n, 8 * (z + 0.41) / n};
                    double r2 = std::pow(p[0] - 1.38, 2) + std::pow(p[1] - 2.17, 2) + std::pow(p[2] - 3.44, 2);
                    if (core && r2 <= 1.8 * 1.8) {
                        continue;
                    }
                    points->coordinates.push_back(p);
                    points->energy_weights.push_back(336.0 / (n * n * n));
                    points->descriptor_weights.push_back(336.0 / (n * n * n));
                }
            }
        }
        points->energy_weights[0] = 0;
        if (core) {
            // Radially resolved sphere quadrature is essential for core feedback.
            auto const& radial = type.radial_grid();
            Spline<double> prototype(radial);
            mdarray<double, 2> integral({radial.num_points(), 4});
            integral.zero();
            for (int ir = 0; ir < radial.num_points() - 1; ir++) {
                double h = radial[ir + 1] - radial[ir];
                for (int j = 0; j < 4; j++) {
                    integral(ir, j) = std::pow(h, j + 1) / (j + 1);
                }
            }
            auto weights = prototype.interpolation_adjoint(integral);
            for (int ir = 0; ir < radial.num_points(); ir++) {
                for (int axis = 0; axis < 3; axis++) {
                    for (int sign : {-1, 1}) {
                        std::array<double, 3> p{1.38, 2.17, 3.44};
                        p[axis] += sign * radial[ir];
                        points->coordinates.push_back(p);
                        double w = fourpi / 6 * radial[ir] * radial[ir] * weights[ir];
                        points->energy_weights.push_back(w);
                        points->descriptor_weights.push_back(w);
                    }
                }
            }
        }
        points->block_sizes   = {static_cast<int>(points->coordinates.size())};
        points->block_centres = {{1.38, 2.17, 3.44}};
    }
    ctx->cb().external_xc_points_ = std::move(points);
    ctx->cb().external_xc_        = functional;
    ctx->initialize();
    return ctx;
}

int
test_lapw_external(int mag, bool distributed_fft, bool second_variation, bool core = false)
{
    auto ctx = context(mag, distributed_fft, second_variation, core);
    int failures{0};
    Test_check check(failures, ctx->comm(),
                     "external LAPW core=" + std::to_string(core) + " spins=" + std::to_string(ctx->num_spins()) +
                             " distributed FFT=" + std::to_string(distributed_fft) +
                             " second variation=" + std::to_string(second_variation));
    K_point_set ks(*ctx);
    ks.add_kpoint({0.07, -0.03, 0.02}, 1);
    ks.initialize();
    DFT_ground_state gs(ks);
    gs.initial_state();
    auto& density   = gs.density();
    auto& potential = gs.potential();
    Density regenerated(*ctx);
    copy(density, regenerated);
    regenerated.generate<double>(ks, false, true, true);
    double initial_field_error{0};
    for (int component = 0; component <= ctx->num_mag_dims(); ++component) {
        for (auto fields : {std::make_pair(&density.component(component), &regenerated.component(component)),
                            std::make_pair(&density.kinetic_density().component(component),
                                           &regenerated.kinetic_density().component(component))}) {
            for (size_t i = 0; i < fields.first->rg().f_pw_local().size(); ++i) {
                initial_field_error = std::max(initial_field_error, std::abs(fields.first->rg().f_pw_local(i) -
                                                                             fields.second->rg().f_pw_local(i)));
            }
            for (auto it : ctx->unit_cell().spl_num_atoms()) {
                for (size_t i = 0; i < fields.first->mt()[it.i].size(); ++i) {
                    initial_field_error = std::max(
                            initial_field_error, std::abs(fields.first->mt()[it.i][i] - fields.second->mt()[it.i][i]));
                }
            }
        }
    }
    check("initial fields regenerated from occupied orbitals", initial_field_error, 1e-12);
    // Isolate the tau guess from angular truncation of the matched atomic density.
    Density initial(*ctx);
    initial.zero();
    for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
        auto const& grid = ctx->unit_cell().atom(ia).type().radial_grid();
        for (int ir = 0; ir < grid.num_points(); ir++) {
            initial.rho().mt()[ia](0, ir) = std::exp(-2 * grid[ir]) / y00;
            if (mag) {
                initial.mag(0).mt()[ia](0, ir) = 0.2 * initial.rho().mt()[ia](0, ir);
            }
        }
    }
    initial.initial_kinetic_density();
    SHT sht(device_t::CPU, ctx->lmax_rho());
    double initial_violation{0};
    for (auto it : ctx->unit_cell().spl_num_atoms()) {
        for (int s = 0; s < ctx->num_spins(); s++) {
            auto n = 1.0 * initial.rho().mt()[it.i];
            auto t = 1.0 * initial.kinetic_density().scalar().mt()[it.i];
            if (mag) {
                for (size_t i = 0; i < n.size(); i++) {
                    n[i] = 0.5 * (n[i] + (1 - 2 * s) * initial.mag(0).mt()[it.i][i]);
                    t[i] = 0.5 * (t[i] + (1 - 2 * s) * initial.kinetic_density().vector(0).mt()[it.i][i]);
                }
            }
            auto real_n = transform(sht, n), real_t = transform(sht, t);
            auto grad = gradient(n);
            auto gx = transform(sht, grad[0]), gy = transform(sht, grad[1]), gz = transform(sht, grad[2]);
            for (size_t i = 0; i < real_n.size(); i++) {
                double sigma      = gx[i] * gx[i] + gy[i] * gy[i] + gz[i] * gz[i];
                double product    = 8 * real_n[i] * real_t[i];
                initial_violation = std::max(initial_violation, (sigma - product) / std::max({1.0, sigma, product}));
            }
        }
    }
    ctx->comm().allreduce<double, mpi::op_t::max>(&initial_violation, 1);
    check("initial density/tau Cauchy inequality", initial_violation, 1e-12);
    if (failures) {
        return failures;
    }
    auto check_energy = [&] {
        double energy = potential.energy_exc(density);
        auto c        = potential.lapw_xc_contractions(density);
        check("callback Euler identity", std::abs(c.rho + c.magnetic + c.tau - 2 * energy), 2e-11);
        check("nonzero tau contraction", std::abs(c.tau) > 1e-6 ? 0 : 1, 0);
        if (mag) {
            check("nonzero magnetic contraction", std::abs(c.magnetic) / functional_scale(*ctx) > 1e-6 ? 0 : 1, 0);
        }
        check("XC operator is staged", potential.lapw_xc_derivatives() ? 0 : 1, 0);
    };
    check_energy();
    check.reject("point potential cannot be symmetrized independently",
                 [&] { potential.generate(density, true, true); });

    auto result = gs.find(1e-9, 1e-10, 1e-9, 100, false);
    check("SCF convergence", result.at("converged").get<bool>() ? 0 : 1, 0);
    check_energy();
    density.generate<double>(ks, false, true, true);
    potential.generate(density, false, true);
    check_energy();

    auto orbital_snapshot = [&] {
        std::vector<std::complex<double>> values;
        for (auto it : ks.spl_num_kpoints()) {
            auto kp = ks.get<double>(it.i);
            for (int s = 0; s < ctx->num_spins(); s++) {
                for (int b = 0; b < ctx->num_bands(); b++) {
                    auto const& wf = kp->spinor_wave_functions();
                    auto data      = wf.at(memory_t::host, 0, wf::spin_index(s), wf::band_index(b));
                    values.insert(values.end(), data, data + wf.ld());
                }
            }
        }
        return values;
    };
    auto basis_snapshot = [&] {
        std::vector<double> values;
        for (int ia = 0; ia < ctx->unit_cell().num_atoms(); ia++) {
            auto const& atom = ctx->unit_cell().atom(ia);
            for (auto rf : atom.type().indexr()) {
                for (int ir = 0; ir < atom.num_mt_points(); ir++) {
                    values.push_back(atom.symmetry_class().radial_function(ir, rf.idxrf));
                }
            }
        }
        return values;
    };
    auto saved_orbitals = orbital_snapshot();
    auto saved_basis    = basis_snapshot();
    double saved_energy = gs.total_energy(), saved_exc = potential.energy_exc(density);
    auto verify_restored = [&] {
        check("orbital diagnostic restores PW/AW/LO coefficients", orbital_snapshot() == saved_orbitals ? 0 : 1, 0);
        check("orbital diagnostic holds radial basis fixed", basis_snapshot() == saved_basis ? 0 : 1, 0);
        check("orbital diagnostic preserves ground-state energy", std::abs(gs.total_energy() - saved_energy), 0);
        check("orbital diagnostic preserves ground-state XC", std::abs(potential.energy_exc(density) - saved_exc), 0);
    };
    auto derivatives = gs.check_external_xc_derivative(1e-4);
    for (int direction = 0; direction < 2; direction++) {
        auto const& values = derivatives[direction];
        check(direction == 0 ? "matched PW/AW orbital derivative" : "local-orbital derivative",
              std::abs(values[0] - values[1]), 2e-8);
        check("orbital derivative at halved step", std::abs(values[0] - values[2]), 6e-9);
        check("nonzero orbital derivative", std::abs(values[0]) / functional_scale(*ctx) > 1e-7 ? 0 : 1, 0);
    }
    verify_restored();
    int evaluations{0};
    ctx->cb().external_xc_ = [&](auto const& c, auto const& in, auto& out) {
        functional(c, in, out);
        if (++evaluations == 2 && c.comm().rank() == 0) {
            throw std::runtime_error("deliberate LAPW derivative-check callback failure");
        }
    };
    check.reject("orbital diagnostic propagates callback failure", [&] { gs.check_external_xc_derivative(1e-4); });
    verify_restored();
    potential.generate(density, false, true);
    check("orbital diagnostic restores evaluator", evaluations == 3 ? 0 : 1, 0);
    ctx->cb().external_xc_ = [](auto const& c, auto const& in, auto& out) {
        functional(c, in, out);
        for (int j = 0; j < in.num_points * in.num_spins; j++) {
            out.vtau[j] = 0;
        }
    };
    auto incomplete = gs.check_external_xc_derivative(1e-4);
    double missing_tau{0};
    for (auto const& values : incomplete) {
        missing_tau = std::max(missing_tau, std::abs(values[0] - values[2]));
    }
    check("orbital diagnostic detects missing tau", missing_tau / functional_scale(*ctx) > 1e-7 ? 0 : 1, 0);
    ctx->cb().external_xc_ = functional;
    verify_restored();

    Density displaced(*ctx);
    Potential probe(*ctx);
    auto c = potential.lapw_xc_contractions(density);
    for (int part = 0; part < ctx->num_spins() + 1; part++) {
        auto evaluate = [&](double step) {
            copy(density, displaced);
            if (part == 0) {
                displaced.rho() *= 1 + step;
            } else if (part == 1) {
                for (int j = 0; j < ctx->num_spins(); j++) {
                    displaced.kinetic_density().component(j) *= 1 + step;
                }
            } else {
                displaced.mag(0) *= 1 + step;
            }
            for (int j = 0; j < ctx->num_spins(); j++) {
                displaced.component(j).mt().sync(ctx->unit_cell().spl_num_atoms());
                displaced.kinetic_density().component(j).mt().sync(ctx->unit_cell().spl_num_atoms());
            }
            displaced.fft_transform(-1);
            displaced.kinetic_density().fft_transform(-1);
            probe.xc(displaced);
            return probe.energy_exc(displaced);
        };
        for (double h : {1e-4, 5e-5}) {
            double fd = (evaluate(h) - evaluate(-h)) / (2 * h);
            check("callback physical-field derivative",
                  std::abs(fd - (part == 0   ? c.rho
                                 : part == 1 ? c.tau
                                             : c.magnetic)),
                  2e-10);
        }
    }
    double kinetic = density.kinetic_density().scalar().integrate().total;
    double direct =
            kinetic + potential.energy_exc(density) + 0.5 * potential.energy_vha() + energy_enuc(*ctx, potential);
    double total = ks_energy(*ctx, ks, density, potential);
    check("kinetic energy from physical orbital tau", std::abs(kinetic - energy_kin(*ctx, ks, density, potential)),
          2e-7);
    check("direct total-energy identity", std::abs(total - direct), 2e-7);
    if (ctx->comm().rank() == 0) {
        std::cout << "external LAPW total=" << total << " direct=" << direct << " kinetic=" << kinetic << '\n';
    }
    if (core) {
        auto vs      = potential.get_spherical_potential();
        auto d       = potential.get_core_xc_derivatives();
        auto old_tau = density.core_kinetic_density(0);
        density.generate_core_charge_density(vs, d);
        auto tau    = density.core_kinetic_density(0);
        double peak = *std::max_element(tau.begin(), tau.end()), residual{0}, response{0};
        for (size_t ir = 0; ir < tau.size(); ir++) {
            residual = std::max(residual, std::abs(tau[ir] - old_tau[ir]) / peak);
        }
        check("self-consistent core tau residual", residual, 2e-6);
        check("core leakage", std::abs(density.core_leakage()), 1e-6);
        double derivative{0};
        auto const& cls = ctx->unit_cell().atom(0).symmetry_class();
        for (size_t ir = 0; ir < tau.size(); ir++) {
            derivative += density.core_charge_density(0)[ir] * d[cls.id()][0][ir] + tau[ir] * d[cls.id()][1][ir];
        }
        for (double h : {1e-3, 5e-4}) {
            auto energy = [&](double step) {
                auto shifted = d;
                for (auto& atom : shifted) {
                    for (auto& field : atom) {
                        for (auto& value : field) {
                            value *= 1 + step;
                        }
                    }
                }
                density.generate_core_charge_density(vs, shifted);
                return density.core_eval_sum();
            };
            double fd = (energy(h) - energy(-h)) / (2 * h);
            check("callback-to-core eigenvalue derivative", std::abs(fd - derivative), 2e-7);
        }
        density.generate_core_charge_density(vs);
        for (size_t ir = 0; ir < tau.size(); ir++) {
            response = std::max(response, std::abs(tau[ir] - density.core_kinetic_density(0)[ir]) / peak);
        }
        check("nonzero tau-dependent core response", response > 1e-5 ? 0 : 1, 0);
        density.generate_core_charge_density(vs, d);
    }
    return failures;
}

int
test_core_guard()
{
    auto ctx = context(0, false, true, true, true);
    int failures{0};
    for (bool density : {false, true}) {
        bool rejected{false};
        try {
            if (density) {
                Density d(*ctx);
            } else {
                Potential p(*ctx);
            }
        } catch (std::exception const& e) {
            rejected = std::string(e.what()).find("tau-dependent radial core response") != std::string::npos;
        }
        failures += !rejected;
    }
    return failures;
}

int
test_replicated_eigenvector_remap()
{
    auto ctx = context(0, true, true);
    double error{0};
    for (auto counts : {std::vector<int>{}, std::vector<int>{0, 0, 0, 2, 0, 1}}) {
        int ng  = ctx->gvec().num_gvec();
        int nlo = std::accumulate(counts.begin(), counts.end(), 0);
        la::dmatrix<std::complex<double>> matrix(ng + nlo, 3);
        auto value = [](int row, int band) { return std::complex<double>(row + 0.3 * band, 0.2 * row - band); };
        for (int b = 0; b < 3; b++) {
            for (int i = 0; i < ng + nlo; i++) {
                matrix(i, b) = ctx->comm().rank() == 0 ? value(i, b) : std::complex<double>(-9, -7);
            }
        }
        wf::Wave_functions<double> slab(ctx->gvec_sptr(), counts, wf::num_mag_dims(0), wf::num_bands(3),
                                        memory_t::host);
        remap_lapw_evec_to_slab(ng, nlo, 3, matrix, slab, ctx->comm());
        for (int b = 0; b < 3; b++) {
            for (int ig = 0; ig < ctx->gvec().count(); ig++) {
                error = std::max(error, std::abs(slab.pw_coeffs(ig, wf::spin_index(0), wf::band_index(b)) -
                                                 value(ctx->gvec().offset() + ig, b)));
            }
            for (auto atom : slab.spl_num_atoms()) {
                int offset = ng + std::accumulate(counts.begin(), counts.begin() + atom.i, 0);
                for (int xi = 0; xi < counts[atom.i]; xi++) {
                    error = std::max(error, std::abs(slab.mt_coeffs(xi, atom.li, wf::spin_index(0), wf::band_index(b)) -
                                                     value(offset + xi, b)));
                }
            }
        }
    }
    ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
    return !std::isfinite(error) || error != 0;
}

} // namespace

int
main(int argc, char** argv)
{
    sirius::initialize(true);
    int result{0};
    if (argc > 1 && std::string(argv[1]) == "core") {
        for (int mag : {0, 1}) {
            for (bool second : {false, true}) {
                bool distributed = mpi::Communicator::world().size() > 1;
                result += call_test("external LAPW core SCF", test_lapw_external, mag, distributed, second, true);
            }
        }
        sirius::finalize();
        return std::min(result, 1);
    }
    for (int mag : {0, 1}) {
        for (bool second : {false, true}) {
            bool distributed = mpi::Communicator::world().size() > 1 && mag != second;
            result += call_test("external LAPW SCF", test_lapw_external, mag, distributed, second, false);
        }
    }
    result += call_test("external LAPW core response guard", test_core_guard);
    result += call_test("replicated LAPW eigenvector redistribution", test_replicated_eigenvector_remap);
    sirius::finalize();
    return std::min(result, 1);
}
