/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "radial/radial_core_solver.hpp"
#include <sirius.hpp>
#include <testing.hpp>
#include <limits>

using namespace sirius;

static std::vector<double>
radial_weights(Radial_grid<double> const& grid__)
{
    Spline<double> prototype(grid__);
    mdarray<double, 2> integral({grid__.num_points(), 4});
    integral.zero();
    for (int ir = 0; ir < grid__.num_points() - 1; ir++) {
        double h = grid__[ir + 1] - grid__[ir];
        for (int j = 0; j < 4; j++) {
            integral(ir, j) = std::pow(h, j + 1) / (j + 1);
        }
    }
    auto weights = prototype.interpolation_adjoint(integral);
    for (int ir = 0; ir < grid__.num_points(); ir++) {
        weights[ir] *= fourpi * grid__[ir] * grid__[ir];
    }
    return weights;
}

int
test_radial_core_solver()
{
    int failures{0};
    auto check = [&](char const* name, double error, double tolerance) {
        std::cout << name << ": " << error << " (limit " << tolerance << ")\n";
        failures += !std::isfinite(error) || error > tolerance;
    };
    for (int z : {1, 30}) {
        auto grid = Radial_grid_factory<double>(radial_grid_t::power, 5000, 1e-8 / z, 120.0 / z, 3.0);
        std::vector<double> potential(grid.num_points()), zero(grid.num_points(), 0);
        for (int ir = 0; ir < grid.num_points(); ir++) {
            potential[ir] = -z / grid[ir];
        }
        for (int l = 0; l <= 3; l++) {
            double reference = -0.5 * z * z / std::pow(l + 1.0, 2);
            double coarse_error{0};
            for (int nb : {64, 128}) {
                Radial_core_solver solver(grid, l, z, potential, nb);
                auto states  = solver.solve(l == 0 ? 2 : 1, zero, zero);
                double error = std::abs(states[0].energy / reference - 1);
                std::cout << "Coulomb Z=" << z << " l=" << l << " basis=" << nb << '\n';
                check("Coulomb energy", error, nb == 64 ? 1e-6 : 2e-8);
                if (nb == 64) {
                    coarse_error = error;
                    continue;
                }
                check("basis refinement", error - coarse_error, 2e-9);
                for (size_t j = 0; j < states.size(); j++) {
                    auto const& state = states[j];
                    double expected   = -0.5 * z * z / std::pow(l + 1.0 + j, 2);
                    check("radial eigenvalue ordering", std::abs(state.energy / expected - 1), 2e-8);
                    check("normalization", std::abs(fourpi * Spline<double>(grid, state.rho).integrate(2) - 1), 2e-8);
                    check("Coulomb kinetic energy", std::abs(state.kinetic_energy / (-expected) - 1), 2e-7);
                    check("sampled tau integral",
                          std::abs(fourpi * Spline<double>(grid, state.tau).integrate(2) / state.kinetic_energy - 1),
                          2e-7);
                    if (l == 0 && j == 1) {
                        auto points  = grid.values();
                        auto inside  = std::lower_bound(points.begin(), points.end(), 1.0 / z) - points.begin();
                        auto outside = std::lower_bound(points.begin(), points.end(), 3.0 / z) - points.begin();
                        check("signed 2s core node", state.value[inside] * state.value[outside] < 0 ? 0 : 1, 0);
                    }
                }
                double maximum{0}, error_tau{0};
                double a             = double(z) / (l + 1);
                double normalization = std::pow(2 * a, 2 * l + 3) / std::tgamma(2 * l + 3);
                for (int ir = 0; ir < grid.num_points(); ir++) {
                    double r   = grid[ir];
                    double r2  = normalization * std::pow(r, 2 * l) * std::exp(-2 * a * r);
                    double tau = 0.5 * r2 * (std::pow(l / r - a, 2) + l * (l + 1.0) / (r * r)) / fourpi;
                    maximum    = std::max(maximum, tau);
                    error_tau  = std::max(error_tau, std::abs(states[0].tau[ir] - tau));
                }
                check("analytic positive tau", error_tau / maximum, 2e-5);
                if (z == 1 && l <= 1) {
                    // Include both 1s and the noded 2s core state in the s channel.
                    auto points = grid.values();
                    int nr      = std::upper_bound(points.begin(), points.end(), 12.0) - points.begin();
                    Radial_grid_ext<double> mt_grid(nr, points.data());
                    std::vector<double> mt_potential(potential.begin(), potential.begin() + nr);
                    Radial_solver local(z, mt_potential, mt_grid);
                    double energy     = -0.5 / std::pow(l + 4.0, 2);
                    auto radial       = local.solve(relativity_t::none, 0, l, energy);
                    auto boundary     = radial.uderiv;
                    double boundary_p = radial.p.back(), boundary_d = radial.rdudr.back();
                    orthogonalize_radial_to_core(mt_grid, states, radial);
                    Spline<double> product(mt_grid);
                    for (int ir = 0; ir < nr; ir++) {
                        product(ir) = radial.p[ir] * radial.p[ir];
                    }
                    double norm = std::sqrt(product.interpolate().integrate(0));
                    for (auto const& state : states) {
                        for (int ir = 0; ir < nr; ir++) {
                            product(ir) = state.value[ir] * radial.p[ir];
                        }
                        check("projected core-valence overlap", std::abs(product.interpolate().integrate(1)) / norm,
                              1e-11);
                    }
                    check("projection boundary value", std::abs(radial.p.back() - boundary_p), 0);
                    check("projection boundary slope", std::abs(radial.rdudr.back() - boundary_d), 0);
                    for (int j = 0; j < 3; j++) {
                        check("projection matching data", std::abs(radial.uderiv[j] - boundary[j]), 0);
                    }
                    auto twice = radial;
                    orthogonalize_radial_to_core(mt_grid, states, twice);
                    double idempotence{0};
                    for (int ir = 0; ir < nr; ir++) {
                        idempotence = std::max(idempotence, std::abs(radial.p[ir] - twice.p[ir]) / norm);
                    }
                    check("core projection idempotence", idempotence, 1e-11);
                    // Fixed boundary normalization makes the energy derivative unique.
                    Radial_core_solver auxiliary(mt_grid, l, z, mt_potential, 96, 0, true);
                    std::vector<double> no_xc(nr, 0);
                    auto derivative = auxiliary.solve_at_energy(1, energy, no_xc, no_xc);
                    orthogonalize_radial_to_core(mt_grid, states, derivative);
                    for (double h : {1e-5, 5e-6}) {
                        auto plus  = auxiliary.solve_at_energy(0, energy + h, no_xc, no_xc);
                        auto minus = auxiliary.solve_at_energy(0, energy - h, no_xc, no_xc);
                        orthogonalize_radial_to_core(mt_grid, states, plus);
                        orthogonalize_radial_to_core(mt_grid, states, minus);
                        double error{0}, peak{0};
                        for (int ir = 0; ir < nr; ir++) {
                            for (auto fields : {std::make_tuple(&derivative.p, &plus.p, &minus.p),
                                                std::make_tuple(&derivative.rdudr, &plus.rdudr, &minus.rdudr)}) {
                                double d = (*std::get<0>(fields))[ir];
                                peak     = std::max(peak, std::abs(d));
                                error    = std::max(
                                        error, std::abs(d - ((*std::get<1>(fields))[ir] - (*std::get<2>(fields))[ir]) /
                                                                       (2 * h)));
                            }
                        }
                        check("projected radial energy derivative", error / peak, 2e-6);
                    }
                }
            }
        }
    }

    // A constant v_tau rescales the bare kinetic operator. The effective Coulomb
    // charge of its eigenfunctions is Z/(1+v_tau), not Z.
    for (int l : {0, 1, 2, 3}) {
        int z        = 3;
        auto grid    = Radial_grid_factory<double>(radial_grid_t::power, 6000, 1e-8, 70.0, 3.0);
        auto weights = radial_weights(grid);
        std::vector<double> potential(grid.num_points()), drho(grid.num_points()), dtau(grid.num_points());
        double beta = 0.4, shift = 0.13;
        for (int ir = 0; ir < grid.num_points(); ir++) {
            potential[ir] = -z / grid[ir];
            drho[ir]      = shift * weights[ir];
            dtau[ir]      = beta * weights[ir];
        }
        Radial_core_solver solver(grid, l, z, potential, 144);
        auto state      = solver.solve(1, drho, dtau)[0];
        double expected = -0.5 * z * z / (std::pow(l + 1.0, 2) * (1 + beta)) + shift;
        std::cout << "Constant tau potential l=" << l << '\n';
        check("scaled Coulomb energy", std::abs(state.energy - expected), 2e-8);
        check("scaled Coulomb kinetic energy", std::abs(state.kinetic_energy - (shift - expected) / (1 + beta)), 2e-7);
    }

    // Manufactured variable-mass problem. For v_tau=beta*r^2 and the potential
    // below, R=r^l exp(-a*r) is an exact nodeless eigenstate with E=-a^2/2.
    // This tests the divergence term from grad(v_tau), independently of the matrix assembly.
    for (int l : {0, 1, 2, 3}) {
        double a = 1.5, beta = 0.1 * a * a, z = a * (l + 1);
        auto grid    = Radial_grid_factory<double>(radial_grid_t::power, 6000, 1e-8, 40.0 / a, 3.0);
        auto weights = radial_weights(grid);
        std::vector<double> potential(grid.num_points()), zero(grid.num_points(), 0), dtau(grid.num_points());
        for (int ir = 0; ir < grid.num_points(); ir++) {
            double r      = grid[ir];
            potential[ir] = -z / r + 0.5 * a * a * beta * r * r - a * (l + 2) * beta * r + beta * l;
            dtau[ir]      = weights[ir] * beta * r * r;
        }
        Radial_core_solver solver(grid, l, z, potential, 128);
        auto state = solver.solve(1, zero, dtau)[0];
        std::cout << "Variable tau potential l=" << l << '\n';
        check("manufactured variable-tau energy", std::abs(state.energy + 0.5 * a * a), 2e-8);
        check("manufactured kinetic energy", std::abs(state.kinetic_energy - 0.5 * a * a), 2e-7);

        // Add arbitrary raw radial covectors, without quadrature weights, and
        // differentiate the optimized eigenvalue at two finite-difference steps.
        auto drho = zero;
        std::vector<double> direction_rho(grid.num_points()), direction_tau(grid.num_points());
        for (int ir = 0; ir < grid.num_points(); ir++) {
            double r          = grid[ir];
            double envelope   = std::exp(-r - 1 / r);
            drho[ir]          = 1e-4 * std::sin(r) * envelope;
            direction_rho[ir] = 1e-4 * std::cos(2 * r) * envelope;
            direction_tau[ir] = 2e-4 * std::sin(3 * r) * envelope;
        }
        auto center = solver.solve(1, drho, dtau)[0];
        double derivative{0};
        for (int ir = 0; ir < grid.num_points(); ir++) {
            derivative += center.rho[ir] * direction_rho[ir] + center.tau[ir] * direction_tau[ir];
        }
        for (double h : {1e-3, 5e-4}) {
            auto plus_rho = drho, minus_rho = drho, plus_tau = dtau, minus_tau = dtau;
            for (int ir = 0; ir < grid.num_points(); ir++) {
                plus_rho[ir] += h * direction_rho[ir];
                minus_rho[ir] -= h * direction_rho[ir];
                plus_tau[ir] += h * direction_tau[ir];
                minus_tau[ir] -= h * direction_tau[ir];
            }
            double fd =
                    (solver.solve(1, plus_rho, plus_tau)[0].energy - solver.solve(1, minus_rho, minus_tau)[0].energy) /
                    (2 * h);
            std::cout << "raw-field eigenvalue derivative h=" << h << '\n';
            check("Hellmann-Feynman derivative", std::abs(fd - derivative), 2e-7);
        }
        auto invalid = dtau;
        invalid[2]   = std::numeric_limits<double>::quiet_NaN();
        bool rejected{false};
        try {
            solver.solve(1, drho, invalid);
        } catch (std::exception const&) {
            rejected = true;
        }
        failures += !rejected;
        rejected = false;
        invalid.clear();
        try {
            solver.solve(1, invalid, dtau);
        } catch (std::exception const&) {
            rejected = true;
        }
        failures += !rejected;
    }
    // A localized mass variation checks resolution of a narrow core feature.
    // Construct V from R=r^l exp(-a*r), so both its energy and tau are known.
    for (int l : {0, 1}) {
        double z = 10, a = z / (l + 1);
        auto grid    = Radial_grid_factory<double>(radial_grid_t::power, 6000, 1e-8, 100.0 / a, 3.0);
        auto weights = radial_weights(grid);
        std::vector<double> potential(grid.num_points()), zero(grid.num_points(), 0), dtau(grid.num_points());
        for (int ir = 0; ir < grid.num_points(); ir++) {
            double r = grid[ir], x = a * r, width = 0.2;
            double envelope = 0.12 * std::exp(-std::pow((x - 1.3) / width, 2));
            double v        = x * x * envelope;
            double dv       = a * envelope * (2 * x - 2 * x * x * (x - 1.3) / (width * width));
            potential[ir]   = -z / r + 0.5 * a * a * v - z * v / r + 0.5 * dv * (l / r - a);
            dtau[ir]        = weights[ir] * v;
        }
        double previous_error{0};
        for (int nb : {128, 256, 384}) {
            Radial_core_solver solver(grid, l, z, potential, nb);
            auto state   = solver.solve(1, zero, dtau)[0];
            double error = std::abs(state.energy + 0.5 * a * a);
            std::cout << "Localized tau potential l=" << l << " basis=" << nb << '\n';
            if (nb != 128)
                check("localized basis refinement", error - previous_error, 1e-9);
            previous_error = error;
            if (nb != 384)
                continue;
            check("localized manufactured energy", error, 5e-8);
            double peak{0}, tau_error{0};
            double normalization = std::pow(2 * a, 2 * l + 3) / std::tgamma(2 * l + 3);
            for (int ir = 0; ir < grid.num_points(); ir++) {
                double r   = grid[ir];
                double rho = normalization * std::pow(r, 2 * l) * std::exp(-2 * a * r) / fourpi;
                double tau = 0.5 * rho * (std::pow(l / r - a, 2) + l * (l + 1.0) / (r * r));
                peak       = std::max(peak, tau);
                tau_error  = std::max(tau_error, std::abs(state.tau[ir] - tau));
            }
            check("localized manufactured tau", tau_error / peak, 1e-6);
        }
    }
    auto grid    = Radial_grid_factory<double>(radial_grid_t::power, 2001, 1e-7, 2.0, 2.0);
    auto weights = radial_weights(grid);
    constexpr double q{0.7}, beta{0.2}, shift{0.3};
    std::vector<double> potential(grid.num_points(), shift), rho(grid.num_points(), 0), tau(weights);
    for (auto& w : tau) {
        w *= beta;
    }
    double energy = shift + 0.5 * (1 + beta) * q * q;
    for (int l : {0, 1, 2}) {
        Radial_core_solver solver(grid, l, 0, potential, 96, 0, true);
        auto radial = solver.solve_at_energy(0, energy, rho, tau);
        double error{0}, error_d{0};
        double norm = gsl_sf_bessel_jl(l, q * grid.last());
        for (int ir = 0; ir < grid.num_points(); ir++) {
            double r = grid[ir], u = gsl_sf_bessel_jl(l, q * r) / norm;
            double du = l * u / r - q * gsl_sf_bessel_jl(l + 1, q * r) / norm;
            error     = std::max(error, std::abs(radial.p[ir] - r * u));
            error_d   = std::max(error_d, std::abs(radial.rdudr[ir] - r * du));
        }
        check("regular boundary solution", error, 2e-7);
        check("regular boundary derivative", error_d, 2e-6);
        auto derivative = solver.solve_at_energy(1, energy, rho, tau);
        auto plus       = solver.solve_at_energy(0, energy + 1e-5, rho, tau);
        auto minus      = solver.solve_at_energy(0, energy - 1e-5, rho, tau);
        error           = 0;
        for (int ir = 0; ir < grid.num_points(); ir++) {
            error = std::max(error, std::abs(derivative.p[ir] - (plus.p[ir] - minus.p[ir]) / 2e-5));
        }
        check("radial energy derivative", error, 2e-7);
        auto points = solver.breakpoints();
        std::vector<double> extra;
        for (size_t i = 1; i < points.size(); i++) {
            extra.push_back(0.5 * (points[i - 1] + points[i]));
        }
        Radial_core_solver refined(grid, l, 0, potential, 96, 0, true, extra);
        auto finer = refined.solve_at_energy(0, energy, rho, tau);
        error      = 0;
        for (int ir = 0; ir < grid.num_points(); ir++) {
            error = std::max(error, std::abs(finer.p[ir] - radial.p[ir]));
        }
        check("nested radial basis refinement", error, 2e-7);
    }
    return failures;
}

int
main(int argc, char** argv)
{
    sirius::initialize(1);
    int result = call_test("test_radial_core_solver", test_radial_core_solver);
    sirius::finalize();
    return result;
}
