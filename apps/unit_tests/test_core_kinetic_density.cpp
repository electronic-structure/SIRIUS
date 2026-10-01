/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include <limits>

using namespace sirius;

int
test_core_kinetic_density()
{
    int failures{0};
    auto check = [&](char const* name, double error, double tolerance) {
        std::cout << name << ": " << error << '\n';
        failures += !std::isfinite(error) || error > tolerance;
    };
    for (int z : {1, 30}) {
        auto grid = Radial_grid_factory<double>(radial_grid_t::power, 6000, 1e-8 / z, 80.0 / z, 3.0);
        std::vector<double> potential(grid.num_points());
        for (int ir = 0; ir < grid.num_points(); ir++) {
            potential[ir] = -z / grid[ir];
        }
        for (auto nl : {std::pair<int, int>{1, 0}, {2, 0}, {2, 1}, {3, 2}, {4, 3}}) {
            int n = nl.first, l = nl.second;
            double energy = -0.5 * z * z / (n * n);
            Bound_state bs(relativity_t::none, z, n, l, 0, grid, potential, energy, 0.33, 4.5, 1e-10, 1e-10);
            double maximum{0}, error{0};
            for (int ir = 0; ir < grid.num_points() - 1; ir++) {
                double r = grid[ir], x = z * r, u, du;
                if (n == 1) {
                    u  = 2 * std::pow(double(z), 1.5) * std::exp(-x);
                    du = -z * u;
                } else if (n == 2) {
                    double a = std::pow(double(z), 1.5) * std::exp(-0.5 * x);
                    u        = l == 0 ? a * (2 - x) / std::sqrt(8.0) : a * x / std::sqrt(24.0);
                    du = l == 0 ? a * z * (0.5 * x - 2) / std::sqrt(8.0) : a * z * (1 - 0.5 * x) / std::sqrt(24.0);
                } else {
                    double a = std::pow(2.0 * z / n, 1.5) / std::sqrt(2.0 * n * std::tgamma(n + l + 1));
                    u        = a * std::pow(2 * x / n, l) * std::exp(-x / n);
                    du       = (l / r - double(z) / n) * u;
                }
                double expected = 0.5 * (du * du + l * (l + 1) * std::pow(u / r, 2));
                maximum         = std::max(maximum, expected);
                error           = std::max(error, std::abs(bs.positive_kinetic_density()(ir) - expected));
            }
            std::cout << "Schroedinger Z=" << z << " n=" << n << " l=" << l << '\n';
            check("analytic radial tau, peak-scaled error", error / maximum, 2e-6);
            check("integrated tau versus Coulomb kinetic energy",
                  std::abs(bs.positive_kinetic_density().integrate(2) / (-energy) - 1), 2e-7);
            check("energy", std::abs(bs.enu() / energy - 1), 2e-8);
        }

        double gamma  = std::sqrt(1 - std::pow(z / speed_of_light, 2));
        double energy = -z * z / (1 + gamma);
        Bound_state bs(relativity_t::dirac, z, 1, 0, 1, grid, potential, energy, 0.33, 4.5, 1e-10, 1e-10);
        double ratio         = -z / (speed_of_light * (1 + gamma));
        double normalization = std::pow(2.0 * z, 2 * gamma + 1) / ((1 + ratio * ratio) * std::tgamma(2 * gamma + 1));
        double maximum{0}, error{0};
        for (int ir = 0; ir < grid.num_points() - 1; ir++) {
            double r                      = grid[ir];
            double u2                     = normalization * std::pow(r, 2 * gamma - 2) * std::exp(-2 * z * r);
            double logarithmic_derivative = (gamma - 1) / r - z;
            double expected               = 0.5 * u2 *
                              ((1 + ratio * ratio) * std::pow(logarithmic_derivative, 2) + 2 * ratio * ratio / (r * r));
            maximum = std::max(maximum, expected);
            error   = std::max(error, std::abs(bs.positive_kinetic_density()(ir) - expected));
        }
        std::cout << "Dirac 1s Z=" << z << '\n';
        check("analytic spinor tau, peak-scaled error", error / maximum, 2e-6);
        check("spinor normalization", std::abs(inner(bs.p(), bs.p(), 0) + inner(bs.q(), bs.q(), 0) - 1), 1e-10);

        // Differentiate interpolated spinor components independently of the radial ODE.
        // Both p angular branches exercise the lower-component l-1 and l+1 terms.
        for (int k : {1, 2}) {
            double g = std::sqrt(k * k - std::pow(z / speed_of_light, 2));
            double e = speed_of_light * speed_of_light *
                       (1 / std::sqrt(1 + std::pow(z / speed_of_light / (2 - k + g), 2)) - 1);
            Bound_state pstate(relativity_t::dirac, z, 2, 1, k, grid, potential, e, 0.33, 4.5, 1e-10, 1e-10);
            int small_l = 2 * k - 2;
            double integral_error{0}, integral_reference{0};
            for (int ir = 3; ir < grid.num_points() - 3; ir++) {
                double r = grid[ir];
                double p = pstate.p()(ir), q = pstate.q()(ir);
                double dp       = (pstate.p().deriv(1, ir) - p / r) / r;
                double dq       = (pstate.q().deriv(1, ir) - q / r) / r;
                double expected = 0.5 * (dp * dp + 2 * std::pow(p / (r * r), 2) + dq * dq +
                                         small_l * (small_l + 1) * std::pow(q / (r * r), 2));
                double weight   = r * r * grid.dx(ir);
                integral_error += weight * std::abs(expected - pstate.positive_kinetic_density()(ir));
                integral_reference += weight * expected;
            }
            std::cout << "Dirac 2p Z=" << z << " k=" << k << '\n';
            check("independent radial derivative", integral_error / integral_reference, 2e-6);
        }
    }
    return failures;
}

int
test_lapw_core_storage(std::string const& relativity__)
{
    auto config                             = R"({"parameters": {
        "electronic_structure_method": "full_potential_lapwlo",
        "use_symmetry": false, "pw_cutoff": 3.0, "gk_cutoff": 1.0,
        "num_bands": 4, "xc_functionals": ["XC_LDA_X"],
        "auto_rmt": 0, "lmax_apw": 1, "lmax_rho": 2, "lmax_pot": 2
    }, "control": {"mpi_grid_dims": [1, 1], "processing_unit": "CPU", "verbosity": 0}})"_json;
    config["parameters"]["core_relativity"] = relativity__;
    Simulation_context ctx(config);
    ctx.unit_cell().set_lattice_vectors({{9, 0, 0}, {0, 10, 0}, {0, 0, 11}});
    for (auto label : {"Li", "H"}) {
        auto& type = ctx.unit_cell().add_atom_type(label);
        type.zn(std::string(label) == "Li" ? 3 : 1);
        type.set_radial_grid(radial_grid_t::power, 2000, 1e-7, 2.0, 3.0);
        auto free_grid = Radial_grid_factory<double>(radial_grid_t::power, 4000, 1e-7, 20.0, 3.0);
        type.free_atom_density(std::vector<double>(free_grid.num_points(), 0));
        type.set_free_atom_radial_grid(std::move(free_grid));
        for (int l = 0; l <= 1; l++) {
            type.add_aw_descriptor(l + 1, l, -0.5, 0, 0);
        }
        if (std::string(label) == "Li") {
            type.set_configuration(1, 0, 1, 2.0, true);
        }
    }
    ctx.unit_cell().add_atom("Li", {0, 0, 0});
    ctx.unit_cell().add_atom("H", {0.5, 0.5, 0.5});
    ctx.initialize();
    Density density(ctx);
    int failures{0};
    bool rejected{false};
    try {
        density.core_kinetic_density(0);
    } catch (std::exception const&) {
        rejected = true;
    }
    failures += !rejected;
    std::vector<std::vector<double>> potential(ctx.unit_cell().num_atom_symmetry_classes());
    for (int ic = 0; ic < ctx.unit_cell().num_atom_symmetry_classes(); ic++) {
        auto const& type = ctx.unit_cell().atom_symmetry_class(ic).atom_type();
        for (int ir = 0; ir < type.num_mt_points(); ir++) {
            potential[ic].push_back(-type.zn() / type.radial_grid(ir));
        }
    }
    density.generate_core_charge_density(potential);
    auto const& type     = ctx.unit_cell().atom(0).type();
    auto const& tau      = density.core_kinetic_density(0);
    double gamma         = relativity__ == "dirac" ? std::sqrt(1 - std::pow(3.0 / speed_of_light, 2)) : 1;
    double ratio         = relativity__ == "dirac" ? -3.0 / (speed_of_light * (1 + gamma)) : 0;
    double normalization = std::pow(6.0, 2 * gamma + 1) / ((1 + ratio * ratio) * std::tgamma(2 * gamma + 1));
    double maximum{0}, error{0};
    for (int ir = 0; ir < type.num_mt_points(); ir++) {
        double r  = type.radial_grid(ir);
        double u2 = normalization * std::pow(r, 2 * gamma - 2) * std::exp(-6 * r);
        double expected =
                u2 / fourpi * ((1 + ratio * ratio) * std::pow((gamma - 1) / r - 3, 2) + 2 * ratio * ratio / (r * r));
        maximum = std::max(maximum, expected);
        error   = std::max(error, std::abs(tau[ir] - expected));
    }
    error /= maximum;
    ctx.comm().allreduce<double, mpi::op_t::max>(&error, 1);
    std::cout << "LAPW core storage " << relativity__ << ", occupation and MPI replication: " << error << '\n';
    failures += !std::isfinite(error) || error > 2e-6;
    for (double value : density.core_kinetic_density(1)) {
        failures += value != 0;
    }
    if (relativity__ == "none") {
        auto original_tau      = density.core_kinetic_density(0);
        double original_energy = density.core_eval_sum();
        std::vector<std::array<std::vector<double>, 2>> fields(potential.size());
        for (int ic = 0; ic < int(fields.size()); ic++) {
            for (auto& f : fields[ic]) {
                f.resize(potential[ic].size(), 0);
            }
        }
        density.generate_core_charge_density(potential, fields);
        double maximum = *std::max_element(original_tau.begin(), original_tau.end());
        double error{0};
        for (int ir = 0; ir < type.num_mt_points(); ir++) {
            error = std::max(error, std::abs(density.core_kinetic_density(0)[ir] - original_tau[ir]) / maximum);
        }
        std::cout << "variational versus ODE core tau: " << error << '\n';
        failures += error > 2e-6 || !std::isfinite(error);
        failures += std::abs(density.core_eval_sum() - original_energy) > 2e-7;

        int ic           = ctx.unit_cell().atom(0).symmetry_class().id();
        auto const& grid = type.radial_grid();
        Spline<double> prototype(grid);
        mdarray<double, 2> integral({grid.num_points(), 4});
        integral.zero();
        for (int ir = 0; ir < grid.num_points() - 1; ir++) {
            double h = grid[ir + 1] - grid[ir];
            for (int j = 0; j < 4; j++) {
                integral(ir, j) = std::pow(h, j + 1) / (j + 1);
            }
        }
        auto weights = prototype.interpolation_adjoint(integral);
        for (int ir = 0; ir < type.num_mt_points(); ir++) {
            double r          = grid[ir];
            double weight     = fourpi * r * r * weights[ir];
            fields[ic][0][ir] = 0.002 * weight * std::exp(-r);
            fields[ic][1][ir] = 0.1 * weight * std::exp(-r);
        }
        density.generate_core_charge_density(potential, fields);
        double derivative{0}, response{0};
        for (int ir = 0; ir < type.num_mt_points(); ir++) {
            derivative += density.core_charge_density(0)[ir] * fields[ic][0][ir] +
                          density.core_kinetic_density(0)[ir] * fields[ic][1][ir];
            response = std::max(response, std::abs(density.core_kinetic_density(0)[ir] - original_tau[ir]) / maximum);
        }
        std::cout << "tau response to the radial core operator: " << response << '\n';
        failures += response < 1e-5 || !std::isfinite(response);
        for (double h : {1e-3, 5e-4}) {
            auto displaced = fields;
            for (auto& pair : displaced) {
                for (auto& field : pair) {
                    for (auto& v : field)
                        v *= 1 + h;
                }
            }
            density.generate_core_charge_density(potential, displaced);
            double plus = density.core_eval_sum();
            displaced   = fields;
            for (auto& pair : displaced) {
                for (auto& field : pair) {
                    for (auto& v : field)
                        v *= 1 - h;
                }
            }
            density.generate_core_charge_density(potential, displaced);
            double fd = (plus - density.core_eval_sum()) / (2 * h);
            std::cout << "occupied MPI core eigenvalue derivative h=" << h << ": " << std::abs(fd - derivative) << '\n';
            failures += !std::isfinite(fd) || std::abs(fd - derivative) > 2e-7;
        }

        auto invalid = fields;
        if (ctx.comm().rank() == ctx.comm().size() - 1) {
            invalid[ic][1][5] = std::numeric_limits<double>::quiet_NaN();
        }
        rejected = false;
        try {
            density.generate_core_charge_density(potential, invalid);
        } catch (std::exception const&) {
            rejected = true;
        }
        failures += !rejected;
        rejected = false;
        try {
            density.core_kinetic_density(0);
        } catch (std::exception const&) {
            rejected = true;
        }
        failures += !rejected;
        density.generate_core_charge_density(potential, fields);
        failures += !std::isfinite(density.core_eval_sum());
        if (ctx.comm().size() > 1) {
            auto inconsistent = fields;
            if (ctx.comm().rank() == ctx.comm().size() - 1) {
                inconsistent.clear();
            }
            rejected = false;
            try {
                density.generate_core_charge_density(potential, inconsistent);
            } catch (std::exception const&) {
                rejected = true;
            }
            failures += !rejected;
            density.generate_core_charge_density(potential, fields);
        }
    } else {
        std::vector<std::array<std::vector<double>, 2>> fields(potential.size());
        for (int ic = 0; ic < int(fields.size()); ic++) {
            for (auto& f : fields[ic]) {
                f.resize(potential[ic].size(), 0);
            }
        }
        rejected = false;
        try {
            density.generate_core_charge_density(potential, fields);
        } catch (std::exception const& e) {
            rejected = true;
        }
        failures += !rejected;
        density.generate_core_charge_density(potential);
        failures += !std::isfinite(density.core_eval_sum());
    }
    return failures;
}

int
test_paw_core_storage()
{
    Simulation_context ctx(R"({"parameters": {"electronic_structure_method": "pseudopotential"}})"_json);
    ctx.electronic_structure_method("pseudopotential");
    auto grid = Radial_grid_factory<double>(radial_grid_t::power, 200, 1e-7, 3.0, 3.0);
    std::vector<double> tau(grid.num_points()), rho(grid.num_points()), zero(grid.num_points(), 0);
    for (int ir = 0; ir < grid.num_points(); ir++) {
        // A doubly occupied hydrogenic 1s shell with Z=3, without radial Jacobians.
        rho[ir] = 54 * std::exp(-6 * grid[ir]) / pi;
        tau[ir] = 4.5 * rho[ir];
    }
    auto data                                = R"({"pseudo_potential": {
        "header": {"element": "Li", "z_valence": 1, "number_of_proj": 0,
                   "paw_core_energy": -7.4321}, "D_ion": [],
        "paw_data": {"occupations": [], "ae_wfc": [], "ps_wfc": []}
    }})"_json;
    auto& pp                                 = data["pseudo_potential"];
    pp["header"]["mesh_size"]                = grid.num_points();
    pp["radial_grid"]                        = grid.values();
    pp["local_potential"]                    = zero;
    pp["total_charge_density"]               = zero;
    pp["paw_data"]["ae_core_charge_density"] = rho;
    auto& legacy                             = ctx.unit_cell().add_atom_type("legacy");
    legacy.read_input(data);
    int failures = legacy.has_paw_core_kinetic_density();
    bool rejected{false};
    try {
        legacy.paw_ae_core_kinetic_density();
    } catch (std::exception const&) {
        rejected = true;
    }
    failures += !rejected;
    failures += legacy.serialize()["pseudo_potential"]["paw_data"].contains("ae_core_kinetic_density");

    pp["paw_data"]["ae_core_kinetic_density"] = tau;
    std::vector<double> ps_tau(grid.num_points());
    for (int ir = 0; ir < grid.num_points(); ir++) {
        ps_tau[ir] = 0.2 * std::exp(-grid[ir] * grid[ir]);
    }
    pp["core_kinetic_density"] = ps_tau;
    auto& type                 = ctx.unit_cell().add_atom_type("with-core-tau");
    type.read_input(data);
    failures += !type.has_paw_core_kinetic_density() || type.paw_ae_core_kinetic_density() != tau;
    auto& copy = ctx.unit_cell().add_atom_type("round-trip");
    copy.read_input(type.serialize());
    failures += copy.paw_ae_core_kinetic_density() != tau;
    failures += copy.ps_core_kinetic_density() != ps_tau;
    failures += copy.paw_core_energy() != -7.4321;
    failures += !type.serialize()["pseudo_potential"]["paw_data"]["ps_wfc"].is_array();

    for (double bad : {-1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        auto values = tau;
        values[5]   = bad;
        rejected    = false;
        try {
            type.paw_ae_core_kinetic_density(values);
        } catch (std::exception const&) {
            rejected = true;
        }
        failures += !rejected || type.paw_ae_core_kinetic_density() != tau;
    }
    for (int length : {0, grid.num_points() - 1, grid.num_points() + 1}) {
        pp["paw_data"]["ae_core_kinetic_density"] = std::vector<double>(length, 0);
        auto& invalid                             = ctx.unit_cell().add_atom_type("invalid-" + std::to_string(length));
        rejected                                  = false;
        try {
            invalid.read_input(data);
        } catch (std::exception const&) {
            rejected = true;
        }
        failures += !rejected || invalid.has_paw_core_kinetic_density();
    }
    type.paw_ae_core_kinetic_density(zero);
    failures += !type.has_paw_core_kinetic_density() || type.paw_ae_core_kinetic_density() != zero;
    failures += type.ps_core_kinetic_density() != ps_tau;
    std::cout << "PAW core tau import, validation and round-trip failures: " << failures << '\n';
    return failures;
}

int
main(int argc, char** argv)
{
    sirius::initialize(1);
    int result = call_test("test_core_kinetic_density", test_core_kinetic_density);
    result += call_test("LAPW core storage", test_lapw_core_storage, "none");
    result += call_test("LAPW Dirac core storage", test_lapw_core_storage, "dirac");
    result += call_test("PAW core storage", test_paw_core_storage);
    sirius::finalize();
    return result;
}
