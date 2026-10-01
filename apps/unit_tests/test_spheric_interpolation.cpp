/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include "function3d/spheric_function_interpolator.hpp"

using namespace sirius;

template <typename T>
int
test_spline_transpose(int n__)
{
    std::vector<double> x(n__);
    for (int i = 0; i < n__; i++) {
        x[i] = 0.02 + 3 * std::pow(double(i) / (n__ - 1), 1.7);
    }
    Radial_grid_ext<double> grid(n__, x.data());
    Spline<T> spline(grid);
    mdarray<T, 2> derivative({n__, 4});
    for (int i = 0; i < n__; i++) {
        spline(i) = std::sin(i + 0.3);
        if constexpr (std::is_same_v<T, std::complex<double>>) {
            spline(i) += T(0, std::cos(i + 0.2));
        }
        for (int j = 0; j < 4; j++) {
            derivative(i, j) = std::cos(3 * i + 2 * j + 0.7);
        }
    }
    spline.interpolate();
    auto result = spline.interpolation_adjoint(derivative);
    T forward{0}, reverse{0};
    for (int i = 0; i < n__; i++) {
        reverse += result[i] * spline(i);
        for (int j = 0; j < 4; j++) {
            forward += derivative(i, j) * spline.coeffs()(i, j);
        }
    }
    double error = std::abs(forward - reverse) / std::max(1.0, std::abs(forward));
    std::cout << "spline transpose n=" << n__ << ": " << error << '\n';
    return !std::isfinite(error) || error > 2e-12;
}

int
test_spheric_interpolation()
{
    std::vector<double> x{0.02, 0.09, 0.2, 0.45, 0.75, 1.1, 1.8, 2.4, 3.0};
    Radial_grid_ext<double> grid(static_cast<int>(x.size()), x.data());
    int n = sf::lmmax(3);
    Flm field(n, grid), direction(n, grid);
    for (int ir = 0; ir < grid.num_points(); ir++) {
        for (int lm = 0; lm < n; lm++) {
            field(lm, ir)     = std::sin(0.7 * lm + grid[ir]) / (lm + 1);
            direction(lm, ir) = std::cos(0.2 * lm + 2 * grid[ir]) / (lm + 1);
        }
    }
    Spheric_function_interpolator sampler(field), dsampler(direction);
    std::vector<r3::vector<double>> points{{0.4, 0.3, 0.2},  {0, 0, 0.75},      {0, 0, -1.1}, {0.45, 0, 0},
                                           {0.3, -0.5, 0.7}, {-0.2, 0.4, -0.8}, {0.02, 0, 0}, {0, 3.0, 0}};
    int failures{0};
    auto check = [&](char const* label, double error, double tolerance) {
        std::cout << label << ": " << error << '\n';
        failures += !std::isfinite(error) || error > tolerance;
    };
    Flm polynomial(n, grid);
    polynomial.zero();
    double pnorm = std::sqrt(3 / fourpi);
    for (int ir = 0; ir < grid.num_points(); ir++) {
        double r                      = grid[ir];
        polynomial(0, ir)             = (1 + 0.7 * r * r) / y00;
        polynomial(sf::lm(1, 1), ir)  = -0.2 * r / pnorm;
        polynomial(sf::lm(1, -1), ir) = 0.3 * r / pnorm;
        polynomial(sf::lm(1, 0), ir)  = 0.4 * r / pnorm;
    }
    Spheric_function_interpolator polynomial_sample(polynomial);
    double exact_error{0};
    for (auto const& p : points) {
        auto actual     = polynomial_sample.value_gradient(p);
        double expected = 1 + 0.7 * std::pow(p.length(), 2) + 0.2 * p[0] - 0.3 * p[1] + 0.4 * p[2];
        exact_error     = std::max(exact_error, std::abs(actual[0] - expected));
        for (int a = 0; a < 3; a++) {
            double d    = 1.4 * p[a] + std::array<double, 3>{0.2, -0.3, 0.4}[a];
            exact_error = std::max(exact_error, std::abs(actual[a + 1] - d));
        }
    }
    check("Cartesian polynomial values/gradients including endpoints", exact_error, 2e-12);
    for (double step : {1e-4, 5e-5}) {
        double error{0};
        for (size_t i = 0; i + 2 < points.size(); i++) {
            auto result = sampler.value_gradient(points[i]);
            for (int a = 0; a < 3; a++) {
                auto plus = points[i], minus = points[i];
                plus[a] += step;
                minus[a] -= step;
                double fd = (sampler.value_gradient(plus)[0] - sampler.value_gradient(minus)[0]) / (2 * step);
                error     = std::max(error, std::abs(fd - result[a + 1]));
            }
        }
        check("Cartesian gradient FD including poles and radial knots", error, 2e-6);
    }
    std::vector<std::array<double, 4>> weights(points.size());
    double lhs{0}, rhs{0};
    for (size_t i = 0; i < points.size(); i++) {
        auto values = dsampler.value_gradient(points[i]);
        for (int a = 0; a < 4; a++) {
            weights[i][a] = std::sin(0.5 + 2 * i + a);
            lhs += weights[i][a] * values[a];
        }
    }
    auto back = sampler.adjoint(points, weights);
    for (int ir = 0; ir < grid.num_points(); ir++) {
        for (int lm = 0; lm < n; lm++) {
            rhs += direction(lm, ir) * back(lm, ir);
        }
    }
    check("combined value/gradient adjoint", std::abs(lhs - rhs), 2e-12);

    // A source atom and one of its periodic images both contribute to each target.
    std::vector<r3::vector<double>> displacements;
    for (int i = 0; i < 4; i++) {
        displacements.push_back(points[i]);
        displacements.push_back(points[i] - r3::vector<double>({0.7, 0.2, 0.1}));
    }
    auto evaluate = [&](Flm const& input, std::vector<std::array<double, 4>>* derivative) {
        Spheric_function_interpolator op(input);
        std::array<std::array<double, 4>, 4> values{};
        double sum{0}, energy{0};
        for (int i = 0; i < 4; i++) {
            auto first = op.value_gradient(displacements[2 * i]);
            auto image = op.value_gradient(displacements[2 * i + 1]);
            for (int a = 0; a < 4; a++) {
                values[i][a] = first[a] + image[a];
                energy += 0.5 * i * (a == 0 ? 1 : 0.2) * values[i][a] * values[i][a];
            }
            sum += (1 + 0.1 * i) * values[i][0];
        }
        energy += 0.13 * sum * sum;
        if (derivative) {
            derivative->resize(displacements.size());
            for (int i = 0; i < 4; i++) {
                for (int a = 0; a < 4; a++) {
                    double value = i * (a == 0 ? 1 : 0.2) * values[i][a];
                    if (a == 0) {
                        value += 0.26 * sum * (1 + 0.1 * i);
                    }
                    (*derivative)[2 * i][a] = (*derivative)[2 * i + 1][a] = value;
                }
            }
        }
        return energy;
    };
    std::vector<std::array<double, 4>> derivative;
    evaluate(field, &derivative);
    back = sampler.adjoint(displacements, derivative);
    double analytic{0};
    for (int ir = 0; ir < grid.num_points(); ir++) {
        for (int lm = 0; lm < n; lm++) {
            analytic += direction(lm, ir) * back(lm, ir);
        }
    }
    for (double step : {1e-4, 5e-5}) {
        Flm plus(n, grid), minus(n, grid);
        for (int ir = 0; ir < grid.num_points(); ir++) {
            for (int lm = 0; lm < n; lm++) {
                plus(lm, ir)  = field(lm, ir) + step * direction(lm, ir);
                minus(lm, ir) = field(lm, ir) - step * direction(lm, ir);
            }
        }
        double fd = (evaluate(plus, nullptr) - evaluate(minus, nullptr)) / (2 * step);
        check("nonlocal image-sum energy derivative", std::abs(analytic - fd), 2e-9);
    }
    auto empty = sampler.adjoint({}, {});
    check("empty sample set", std::abs(empty.checksum()), 0);
    for (auto p : {r3::vector<double>({0, 0, 0}), r3::vector<double>({0.01, 0, 0}), r3::vector<double>({4, 0, 0}),
                   r3::vector<double>({std::numeric_limits<double>::quiet_NaN(), 0, 0})}) {
        bool rejected{false};
        try {
            sampler.value_gradient(p);
        } catch (std::exception const&) {
            rejected = true;
        }
        failures += !rejected;
    }
    return failures;
}

int
test_inner_extension(spheric_inner_extension_t mode)
{
    std::vector<double> x{0.02, 0.027, 0.04, 0.065, 0.1, 0.16, 0.3};
    Radial_grid_ext<double> grid(static_cast<int>(x.size()), x.data());
    int n  = sf::lmmax(3);
    auto l = sf::l_by_lm(3);
    Flm field(n, grid), direction(n, grid);
    auto power  = [&](int lm) { return mode == spheric_inner_extension_t::charge ? l[lm] : std::max(0, l[lm] - 1); };
    auto radial = [&](int lm, double r, int derivative) {
        int p = power(lm);
        double value{0};
        // Cubics are reproduced by the spline and also belong to the regular inner ansatz.
        for (int j = 0; j <= std::min(2, 3 - p); j++) {
            int exponent = p + j;
            if (exponent < derivative) {
                continue;
            }
            double c = (1 + 0.13 * j) / (lm + 1);
            for (int k = 0; k < derivative; k++) {
                c *= exponent - k;
            }
            value += c * std::pow(r, exponent - derivative);
        }
        return value;
    };
    for (int ir = 0; ir < grid.num_points(); ir++) {
        for (int lm = 0; lm < n; lm++) {
            field(lm, ir)     = radial(lm, grid[ir], 0);
            direction(lm, ir) = std::cos(lm + 0.2 * ir) * std::pow(grid[ir], power(lm));
        }
    }
    Spheric_function_interpolator sampler(field, mode), dsampler(direction, mode);
    std::vector<r3::vector<double>> points;
    std::vector<std::array<double, 4>> covectors;
    double error{0}, lhs{0}, rhs{0};
    for (double fraction : {1e-6, 0.2, 0.5, 0.99999, 1.0, 1.00001, 1.2}) {
        double r                    = grid.first() * fraction;
        r3::vector<double> position = (r / std::sqrt(14.0)) * r3::vector<double>({1, -2, 3});
        points.push_back(position);
        auto angles = r3::spherical_coordinates(position);
        std::vector<double> y(n);
        mdarray<double, 2> gy({n, 3});
        sf::spherical_harmonics(3, angles[1], angles[2], y.data());
        sf::dRlm_dr(3, position, gy);
        std::array<double, 4> exact{};
        for (int lm = 0; lm < n; lm++) {
            exact[0] += radial(lm, r, 0) * y[lm];
            for (int a = 0; a < 3; a++) {
                exact[a + 1] += radial(lm, r, 1) * position[a] / r * y[lm] + radial(lm, r, 0) * gy(lm, a);
            }
        }
        auto actual = sampler.value_gradient(position);
        auto delta  = dsampler.value_gradient(position);
        covectors.push_back({0.7, 0.2, -0.1, 0.3});
        for (int a = 0; a < 4; a++) {
            error = std::max(error, std::abs(actual[a] - exact[a]) / std::max(1.0, std::abs(exact[a])));
            lhs += covectors.back()[a] * delta[a];
        }
    }
    auto back = sampler.adjoint(points, covectors);
    for (int ir = 0; ir < grid.num_points(); ir++) {
        for (int lm = 0; lm < n; lm++) {
            rhs += back(lm, ir) * direction(lm, ir);
        }
    }
    double adjoint = std::abs(lhs - rhs) / std::max(1.0, std::abs(lhs));
    std::cout << "inner multipole value/gradient and adjoint: " << error << ' ' << adjoint << '\n';
    int failures = !std::isfinite(error) || error > 2e-10 || !std::isfinite(adjoint) || adjoint > 2e-11;

    // Value gradients must also be the spatial derivatives when crossing the join at r_first.
    double gradient_error{0}, curvature_error{0};
    for (double fraction : {0.5, 1.0}) {
        auto position = (grid.first() * fraction / std::sqrt(14.0)) * r3::vector<double>({1, -2, 3});
        auto analytic = sampler.value_gradient(position);
        for (int a = 0; a < 3; a++) {
            auto plus = position, minus = position;
            double step = 1e-7;
            plus[a] += step;
            minus[a] -= step;
            double fd      = (sampler.value_gradient(plus)[0] - sampler.value_gradient(minus)[0]) / (2 * step);
            gradient_error = std::max(gradient_error, std::abs(fd - analytic[a + 1]) / std::max(1.0, std::abs(fd)));
        }
    }
    // C2 continuity for arbitrary samples, not only the polynomial fixture.
    double h   = 1e-6 * grid.first();
    auto at    = dsampler.value_gradient({grid.first(), 0, 0});
    auto left  = dsampler.value_gradient({grid.first() - h, 0, 0});
    auto right = dsampler.value_gradient({grid.first() + h, 0, 0});
    double cl = (at[1] - left[1]) / h, cr = (right[1] - at[1]) / h;
    curvature_error = std::abs(cl - cr) / std::max({1.0, std::abs(cl), std::abs(cr)});
    std::cout << "inner spatial FD and C2 join: " << gradient_error << ' ' << curvature_error << '\n';
    failures += !std::isfinite(gradient_error) || gradient_error > 2e-7;
    failures += !std::isfinite(curvature_error) || curvature_error > 2e-4;
    return failures;
}

int
test_cusp_dipole()
{
    // For psi=exp(-a r)+x exp(-b r), the s/p term in positive tau is
    // (-a+a*b*r)*exp(-(a+b)*r)*x/r. Its l=1 coefficient is not O(r).
    double a = 1.3, b = 0.7;
    auto dipole     = [&](double r) { return (-a + a * b * r) * std::exp(-(a + b) * r); };
    double previous = 1, last_error{0};
    int failures{0};
    for (double r0 : {0.004, 0.002}) {
        std::vector<double> x(100);
        for (int ir = 0; ir < 100; ir++) {
            x[ir] = r0 * std::exp(0.03 * ir);
        }
        Radial_grid_ext<double> grid(100, x.data());
        Flm field(4, grid);
        field.zero();
        for (int ir = 0; ir < 100; ir++) {
            field(sf::lm(1, 0), ir) = dipole(grid[ir]) / std::sqrt(3 / fourpi);
        }
        Spheric_function_interpolator sampler(field, spheric_inner_extension_t::kinetic);
        double error{0};
        for (double fraction : {1e-6, 0.1, 0.5, 0.99}) {
            double r = r0 * fraction;
            error    = std::max(error, std::abs(sampler.value_gradient({0, 0, r})[0] - dipole(r)));
        }
        std::cout << "cusp kinetic dipole r_first=" << r0 << " error=" << error << '\n';
        failures += !std::isfinite(error) || error >= previous / 4;
        previous   = error;
        last_error = error;
        Spheric_function_interpolator wrong(field, spheric_inner_extension_t::charge);
        if (std::abs(wrong.value_gradient({0, 0, r0 * 1e-6})[0] - dipole(r0 * 1e-6)) < 1) {
            failures++;
        }
    }
    return failures + (last_error > 1e-7);
}

int
main(int argc, char** argv)
{
    sirius::initialize(1);
    int result{0};
    for (int n : {4, 9, 31}) {
        result += call_test("real spline transpose", test_spline_transpose<double>, n);
        result += call_test("complex spline transpose", test_spline_transpose<std::complex<double>>, n);
    }
    result += call_test("spherical interpolation", test_spheric_interpolation);
    result += call_test("inner charge continuation", test_inner_extension, spheric_inner_extension_t::charge);
    result += call_test("inner kinetic continuation", test_inner_extension, spheric_inner_extension_t::kinetic);
    result += call_test("cusp kinetic dipole", test_cusp_dipole);
    sirius::finalize();
    return result;
}
