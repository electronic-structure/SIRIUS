/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "radial/radial_core_solver.hpp"
#include <sirius.hpp>
#include <testing.hpp>
#include <numeric>

using namespace sirius;

int
test_radial_core_xc()
{
    auto grid = Radial_grid_factory<double>(radial_grid_t::power, 4001, 1e-8, 24.0, 3.0);
    std::vector<double> v(grid.num_points()), zero(v.size(), 0), dt(v.size());
    for (int i = 0; i < grid.num_points(); i++) {
        double r = grid[i];
        v[i]     = -2 / r;
        dt[i]    = 4 * pi * r * r * grid.dx(std::min(i, grid.num_points() - 2)) * std::exp(-r - 1 / r);
    }
    constexpr double h = 1e-4;
    auto plus = dt, minus = dt;
    for (size_t i = 0; i < dt.size(); i++) {
        plus[i] *= h;
        minus[i] *= -h;
    }
    for (auto rel : {relativity_t::none, relativity_t::zora}) {
        Radial_core_solver solver(grid, 0, 2, v, 128, 0, {}, rel);
        auto response      = solver.solve_response(1, zero, zero, zero, zero, dt).front();
        auto const& state  = response.value;
        double ep          = solver.solve(1, zero, plus).front().energy;
        double em          = solver.solve(1, zero, minus).front().energy;
        double norm        = fourpi * Spline<double>(grid, state.rho).integrate(2);
        double slope       = (ep - em) / (2 * h);
        double contraction = std::inner_product(dt.begin(), dt.end(), state.tau.begin(), 0.0);
        if (!(std::abs(norm - 1) < 2e-8 && std::abs(response.derivative.energy - slope) < 2e-8 &&
              std::abs(contraction - slope) < 2e-8 &&
              (rel != relativity_t::none || std::abs(state.energy + 2) < 2e-8))) {
            std::cout << "Radial core energy, normalization and kinetic response (analytic, finite difference): "
                      << state.energy << ' ' << norm << ' ' << response.derivative.energy << ' ' << slope << '\n';
            return 1;
        }
    }
    return 0;
}

int
main()
{
    sirius::initialize(1);
    int error = call_test("radial core XC", test_radial_core_xc);
    sirius::finalize();
    return error;
}
