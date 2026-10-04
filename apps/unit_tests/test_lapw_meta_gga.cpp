/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include "dft/dft_ground_state.hpp"

using namespace sirius;

namespace {

auto
context(int mag, bool distributed, bool second, bool core, std::string solver = "exact", int states = -1)
{
    auto config                               = R"({"parameters": {
        "electronic_structure_method": "full_potential_lapwlo", "use_symmetry": false,
        "pw_cutoff": 5.0, "gk_cutoff": 1.5, "num_bands": 4,
        "core_relativity": "none", "valence_relativity": "none", "auto_rmt": 0,
        "lmax_apw": 3, "lmax_rho": 6, "lmax_pot": 6, "smearing_width": 0.01,
        "xc_functionals": ["XC_LDA_X", "XC_MGGA_X_TPSS", "XC_MGGA_C_TPSS", "XC_GGA_C_PBE"],
        "xc_functionals_weight": [0.25, 0.75, 0.8, 0.2]
    }, "control": {"processing_unit": "CPU", "verbosity": 0},
    "iterative_solver": {"type": "exact", "min_tolerance": 1e-11},
    "mixer": {"type": "linear", "beta": 0.2, "use_hartree": false}})"_json;
    config["parameters"]["num_mag_dims"]      = mag;
    config["parameters"]["fixed_mag"]         = mag ? 0.4 : 0.0;
    config["parameters"]["num_bands"]         = core ? 8 : 4;
    config["control"]["use_second_variation"] = second;
    config["control"]["mpi_grid_dims"]        = {distributed ? mpi::Communicator::world().size() : 1, 1};
    config["iterative_solver"]["type"]        = solver;
    config["parameters"]["num_fv_states"]     = states;
    auto ctx                                  = std::make_unique<Simulation_context>(config);
    ctx->unit_cell().set_lattice_vectors({{6, 0, 0}, {0, 7, 0}, {0, 0, 8}});
    auto label = core ? "Ne" : "He";
    auto& type = ctx->unit_cell().add_atom_type(label);
    type.zn(core ? 10 : 2);
    type.set_radial_grid(radial_grid_t::power, core ? 3201 : 801, 1e-6, 1.8, 2.0);
    auto grid = Radial_grid_factory<double>(radial_grid_t::power, 3000, 1e-7, 20.0, 3.0);
    std::vector<double> rho(grid.num_points());
    for (int ir = 0; ir < grid.num_points(); ir++) {
        // A diffuse He starting density stabilizes this small-grid SCF fixture.
        rho[ir] = core ? 2 * std::pow(9.0, 3) / pi * std::exp(-18 * grid[ir]) +
                                  8 * std::pow(1.6, 3) / pi * std::exp(-3.2 * grid[ir])
                       : 2 * std::pow(0.25 / pi, 1.5) * std::exp(-0.25 * grid[ir] * grid[ir]);
    }
    type.free_atom_density(rho);
    type.set_free_atom_radial_grid(std::move(grid));
    for (int l = 0; l <= 3; l++) {
        for (int order : {0, 1}) {
            type.add_aw_descriptor(std::max(core ? 2 : 1, l + 1), l, -0.3, order, 0);
        }
    }
    for (int order : {0, 1}) {
        type.add_lo_descriptor(0, core ? 2 : 1, 0, -0.7, order, 0);
    }
    if (core) {
        type.set_configuration(1, 0, 1, 2.0, true);
    }
    ctx->unit_cell().add_atom(label, {0.23, 0.31, 0.43}, {0, 0, mag ? 0.2 : 0.0});
    ctx->initialize();
    return ctx;
}

double
core_overlap(Density const& density)
{
    // This fixture has a single nodeless, doubly occupied 1s core orbital.
    auto const& atom     = density.ctx().unit_cell().atom(0);
    auto const& type     = atom.type();
    auto const& grid     = type.radial_grid();
    auto const& core_rho = density.core_charge_density(0);
    std::vector<double> overlap(type.indexb().size(), 0);
    for (auto const& b : type.indexb()) {
        if (b.lm != 0) {
            continue;
        }
        Spline<double> product(grid);
        for (int ir = 0; ir < grid.num_points(); ir++) {
            product(ir) = std::sqrt(fourpi * core_rho[ir] / 2) * atom.symmetry_class().radial_function(ir, b.idxrf);
        }
        overlap[b.xi] = product.interpolate().integrate(2);
    }
    double result{0};
    for (int s = 0; s < density.ctx().num_spins(); s++) {
        for (int j = 0; j < type.indexb().size(); j++) {
            for (int i = 0; i < type.indexb().size(); i++) {
                result += overlap[i] * overlap[j] * density.density_matrix(0)(i, j, s).real();
            }
        }
    }
    return std::abs(result);
}

int
test_lapw_meta(int mag, bool distributed, bool second, bool core, bool scf)
{
    // Fit the iterative subspace within this small LAPW basis.
    auto ctx = context(mag, distributed, second, core, scf && second ? "davidson" : "exact",
                       scf && second ? (core ? 8 : 4) : -1);
    int failures{0};
    auto check = [&](char const* label, double error, double tolerance) {
        if (ctx->comm().rank() == 0) {
            std::cout << "mag=" << mag << " distributed=" << distributed << " second=" << second << " core=" << core
                      << ' ' << label << ": " << error << '\n';
        }
        failures += !std::isfinite(error) || error > tolerance;
    };
    K_point_set ks(*ctx);
    ks.add_kpoint({0.07, -0.03, 0.02}, 1);
    ks.initialize();
    DFT_ground_state gs(ks);
    gs.initial_state();
    auto& density   = gs.density();
    auto& potential = gs.potential();
    check("stored tau", density.has_kinetic_density() ? 0 : 1, 0);
    check("staged LAPW XC", potential.lapw_xc_derivatives() ? 0 : 1, 0);
    if (scf) {
        auto result = gs.find(1e-9, 1e-10, 1e-9, 150, false);
        check("SCF converged", result.at("converged").get<bool>() ? 0 : 1, 0);
        density.generate<double>(ks, false, true, true);
        potential.generate(density, false, true);
        for (int s = 0; s < ctx->num_spins(); s++) {
            Smooth_periodic_function<double> n(ctx->spfft<double>(), ctx->gvec_fft_sptr());
            for (int ir = 0; ir < ctx->spfft<double>().local_slice_size(); ir++) {
                n.value(ir) = density.rho().rg().value(ir);
                if (mag) {
                    n.value(ir) = 0.5 * (n.value(ir) + (1 - 2 * s) * density.mag(0).rg().value(ir));
                }
            }
            n.fft_transform(-1);
            auto gradient_n = to_rg(gradient(n));
            double violation{0};
            for (int ir = 0; ir < ctx->spfft<double>().local_slice_size(); ir++) {
                double sigma{0};
                for (int x = 0; x < 3; x++) {
                    sigma += std::pow(gradient_n[x].value(ir), 2);
                }
                double tau = density.kinetic_density().scalar().rg().value(ir);
                if (mag) {
                    tau = 0.5 * (tau + (1 - 2 * s) * density.kinetic_density().vector(0).rg().value(ir));
                }
                violation = std::max(violation, sigma - 8 * n.value(ir) * tau);
            }
            ctx->comm().allreduce<double, mpi::op_t::max>(&violation, 1);
            check("orbital density/tau Cauchy inequality", violation, 1e-12);
        }
        double kinetic = density.kinetic_density().scalar().integrate().total;
        double direct =
                kinetic + potential.energy_exc(density) + 0.5 * potential.energy_vha() + energy_enuc(*ctx, potential);
        check("kinetic-energy identity", std::abs(kinetic - energy_kin(*ctx, ks, density, potential)), 2e-7);
        check("total-energy identity", std::abs(direct - ks_energy(*ctx, ks, density, potential)), 2e-7);
        for (auto const& d : gs.check_lapw_xc_derivative(1e-4)) {
            check("PW/LO orbital derivative at h", std::abs(d[0] - d[1]), 2e-7);
            check("PW/LO orbital derivative at h/2", std::abs(d[0] - d[2]), 2e-7);
        }
        if (core) {
            auto old_tau = density.core_kinetic_density(0);
            density.generate_core_charge_density(potential.get_spherical_potential(),
                                                 potential.get_core_xc_derivatives());
            auto const& tau = density.core_kinetic_density(0);
            double peak     = *std::max_element(tau.begin(), tau.end()), error{0};
            for (size_t ir = 0; ir < tau.size(); ir++) {
                error = std::max(error, std::abs(tau[ir] - old_tau[ir]) / peak);
            }
            check("core tau self-consistency", error, 2e-6);
            check("core leakage", std::abs(density.core_leakage()), 1e-6);
            check("valence occupation in the core orbital", core_overlap(density), 1e-6);
        }
        if (second) {
            // Solver-dependent storage is allocated at initialization, so use a
            // fresh context for the independent exact-diagonalization reference.
            auto exact_ctx = context(mag, distributed, second, core, "exact", ctx->num_fv_states());
            K_point_set exact_ks(*exact_ctx);
            exact_ks.add_kpoint({0.07, -0.03, 0.02}, 1);
            exact_ks.initialize();
            DFT_ground_state exact_gs(exact_ks);
            exact_gs.initial_state();
            auto exact = exact_gs.find(1e-9, 1e-10, 1e-9, 150, false);
            check("exact SCF converged", exact.at("converged").get<bool>() ? 0 : 1, 0);
            check("Davidson/exact SCF energy",
                  std::abs(direct - ks_energy(*exact_ctx, exact_ks, exact_gs.density(), exact_gs.potential())), 2e-7);
        }
        return failures;
    }

    // Independent primitive variations need an interior point of Libxc's domain,
    // not the clipped atomic guess or the one-orbital tau = tau_W boundary.
    density.zero();
    density.kinetic_density().zero();
    for (int ig = 0; ig < ctx->gvec().count(); ig++) {
        auto g = ctx->gvec().gvec(gvec_index_t::local(ig));
        if (g.length2() == 0) {
            density.rho().rg().f_pw_local(ig)                      = 0.05;
            density.kinetic_density().scalar().rg().f_pw_local(ig) = 0.1;
        } else if (g[1] == 0 && g[2] == 0 && std::abs(g[0]) == 1) {
            density.rho().rg().f_pw_local(ig) = std::complex<double>(0.002, 0.001 * g[0]);
        }
        if (mag) {
            density.mag(0).rg().f_pw_local(ig) = 0.2 * density.rho().rg().f_pw_local(ig);
            density.kinetic_density().vector(0).rg().f_pw_local(ig) =
                    0.2 * density.kinetic_density().scalar().rg().f_pw_local(ig);
        }
    }
    for (auto it : ctx->unit_cell().spl_num_atoms()) {
        auto const& grid = ctx->unit_cell().atom(it.i).type().radial_grid();
        for (int ir = 0; ir < grid.num_points(); ir++) {
            density.rho().mt()[it.i](0, ir)                      = (0.05 + 0.001 * grid[ir] * grid[ir]) / y00;
            density.rho().mt()[it.i](3, ir)                      = 0.001 * grid[ir];
            density.kinetic_density().scalar().mt()[it.i](0, ir) = 0.1 / y00;
            if (mag) {
                for (int lm = 0; lm < sf::lmmax(ctx->lmax_rho()); lm++) {
                    density.mag(0).mt()[it.i](lm, ir) = 0.2 * density.rho().mt()[it.i](lm, ir);
                    density.kinetic_density().vector(0).mt()[it.i](lm, ir) =
                            0.2 * density.kinetic_density().scalar().mt()[it.i](lm, ir);
                }
            }
        }
    }
    for (int j = 0; j < ctx->num_spins(); j++) {
        density.component(j).mt().sync(ctx->unit_cell().spl_num_atoms());
        density.kinetic_density().component(j).mt().sync(ctx->unit_cell().spl_num_atoms());
    }
    density.fft_transform(1);
    density.kinetic_density().fft_transform(1);
    potential.generate(density, false, true);
    auto contractions = potential.lapw_xc_contractions(density);
    check("nonzero tau response", std::abs(contractions.tau) > 1e-7 ? 0 : 1, 0);
    Density reference(*ctx);
    copy(density, reference);
    for (int part = 0; part < (mag ? 3 : 2); part++) {
        double derivative = part == 0 ? contractions.rho : part == 1 ? contractions.tau : contractions.magnetic;
        auto evaluate     = [&](double step) {
            copy(reference, density);
            if (part == 0) {
                density.rho() *= 1 + step;
            } else if (part == 1) {
                for (int j = 0; j < ctx->num_spins(); j++) {
                    density.kinetic_density().component(j) *= 1 + step;
                }
            } else {
                density.mag(0) *= 1 + step;
            }
            for (int j = 0; j < ctx->num_spins(); j++) {
                density.component(j).mt().sync(ctx->unit_cell().spl_num_atoms());
                density.kinetic_density().component(j).mt().sync(ctx->unit_cell().spl_num_atoms());
            }
            density.fft_transform(-1);
            density.kinetic_density().fft_transform(-1);
            potential.generate(density, false, true);
            return potential.energy_exc(density);
        };
        for (double h : {1e-4, 5e-5}) {
            double finite_difference = (evaluate(h) - evaluate(-h)) / (2 * h);
            if (ctx->comm().rank() == 0) {
                std::cout << "part=" << part << " h=" << h << " finite difference=" << finite_difference
                          << " contraction=" << derivative << '\n';
            }
            check("primitive-field derivative", std::abs(finite_difference - derivative), 2e-7);
        }
    }
    copy(reference, density);
    potential.generate(density, false, true);
    bool rejected{false};
    try {
        potential.generate(density, true, true);
    } catch (std::exception const&) {
        rejected = true;
    }
    check("reject independent potential symmetrization", rejected ? 0 : 1, 0);
    return failures;
}

} // namespace

int
main(int argc, char** argv)
{
    sirius::initialize(true);
    bool scf    = argc > 1 && std::string(argv[1]) == "scf";
    bool core   = argc > 1 && std::string(argv[1]) == "core";
    bool single = argc > 2 && std::string(argv[2]) == "single";
    bool spin   = argc > 2 && std::string(argv[2]) == "spin";
    int result{0};
    for (int mag : {0, 1}) {
        for (bool second : {false, true}) {
            if (single && (mag || second)) {
                continue;
            }
            if (spin && (!mag || second)) {
                continue;
            }
            bool distributed = mpi::Communicator::world().size() > 1;
            result += call_test("LAPW Libxc meta-GGA", test_lapw_meta, mag, distributed, second, core, scf || core);
        }
    }
    sirius::finalize();
    return std::min(result, 1);
}
