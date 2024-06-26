/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <iomanip>
#include "mixer/anderson_stable_mixer.hpp"
#include "mixer/mixer_factory.hpp"
#include "core/cmd_args.hpp"
#include "testing.hpp"

using namespace sirius;

/**
 * Diagonal matrix operator A(i, i) = 2 + 1 / i where i = 1 ... n
 */
struct Operator
{
    const size_t n; // dimension

    void
    operator()(std::vector<double>& y, std::vector<double> const& x) const
    {
        for (size_t i = 0; i < x.size(); ++i)
            y[i] = (2.0 + 1.0 / (i + 1)) * x[i];
    }
};

/**
 * Return g(x) = Ax - b
 */
std::vector<double>
g(Operator const& A, std::vector<double> const& x, std::vector<double> const& b)
{
    std::vector<double> r(x.size());

    A(r, x);
    for (size_t i = 0; i < x.size(); ++i) {
        r[i] -= b[i];
    }

    return r;
}

double
error_norm(std::vector<double> const& a, std::vector<double> const& b)
{
    double total = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        auto diff = a[i] - b[i];
        total += diff * diff;
    }
    return std::sqrt(total);
}

int
test_mixer(cmd_args const& args)
{
    // Suppose f(x) = g(x) - x where g(x) = Ax - b
    // then f(x) = 0 <=> g(x) = x <=> (A - I)x = b
    // We take A to be a diagonal matrix where A(i, i) = 2 + 1 / i for i = 1 .. n
    // We start out with a guess x = 0.
    // and the mixer input is g(x).
    // it computes a residual, which is f(x) = g(x) - x.
    // TODO: refactor the mixer to just solve for a generic f(x) = 0
    // instead of solving fixed-point problems g(x) = x.

    const auto max_iter    = args.value<size_t>("max_iter", 100);
    const auto max_history = args.value<int>("max_history", 8);
    const auto beta        = args.value<double>("beta", 0.25);
    const auto n           = args.value<size_t>("dim", 100);
    const auto tol         = args.value<double>("tol", 1e-8);

    Operator A{n};

    std::vector<double> b(n);
    std::vector<double> true_x(n, 1.0);

    // We fix the true solution to be x[i] = 1, so b := (A - I)x
    A(b, true_x);
    for (size_t i = 0; i < n; ++i)
        b[i] -= 1;

    auto mixer_function_prop = mixer::FunctionProperties<std::vector<double>>(
            [](const std::vector<double>& x) -> std::size_t { return 1; },
            [](const std::vector<double>& x, const std::vector<double>& y) -> double {
                double result = 0.0;
                for (std::size_t i = 0; i < x.size(); ++i)
                    result += x[i] * y[i];
                return result;
            },
            [](double alpha, std::vector<double>& x) -> void {
                for (auto& val : x)
                    val *= alpha;
            },
            [](const std::vector<double>& x, std::vector<double>& y) -> void {
                std::copy(x.begin(), x.end(), y.begin());
            },
            [](double alpha, const std::vector<double>& x, std::vector<double>& y) -> void {
                for (std::size_t i = 0; i < x.size(); ++i)
                    y[i] += alpha * x[i];
            },
            [](double c, double s, std::vector<double>& x, std::vector<double>& y) -> void {
                for (std::size_t i = 0; i < x.size(); ++i) {
                    auto xi = x[i];
                    auto yi = y[i];

                    x[i] = xi * c + yi * s;
                    y[i] = xi * -s + yi * c;
                }
            });

    nlohmann::json mixer_dict = R"mixer(
    {
      "mixer" : {
        "type" : "linear",
        "beta0" : 0.15,
        "linear_mix_rms_tol" : 1e6,
        "beta_scaling_factor" : 1,
        "use_hartree" : false
      }
    })mixer"_json;

    config_t::mixer_t input(mixer_dict);
    input.beta(beta);
    input.max_history(max_history);

    for (auto const mixer_name : {"anderson", "anderson_stable", "broyden2", "linear"}) {
        input.type(mixer_name);

        std::cout << "max history = " << input.max_history() << ". beta = " << input.beta() << ". dim = " << n
                  << ". mixer = " << input.type() << '\n';

        auto mixer = mixer::Mixer_factory<std::vector<double>>(input);

        std::vector<double> x(n, 0.0);
        mixer->initialize_function<0>(mixer_function_prop, x, n);

        std::cout.precision(std::numeric_limits<double>::digits10);
        std::cout << std::setw(8) << "iter" << std::setw(30) << "||res||" << std::setw(30) << "||err||" << '\n';

        for (size_t step = 0; step < max_iter; ++step) {
            // input = g(output)
            // residual = input - output = g(output) - output = f(output)
            mixer->get_output<0>(x);
            auto g_of_x = g(A, x, b);
            mixer->set_input<0>(g_of_x);

            auto residual_norm = mixer->mix(tol);
            std::cout << std::setw(8) << step << std::setw(30) << residual_norm << std::setw(30)
                      << error_norm(x, true_x) << '\n';

            if (residual_norm < tol)
                break;
        }
    }
    return 0;
}

int
main(int argn, char** argv)
{
    cmd_args args(argn, argv,
                  {{"max_history=", "{int} maximum number of vecs to store"},
                   {"beta=", "{double} first jacobian approximation as diagonal matrix"},
                   {"dim=", "{size_t} problem dimension"},
                   {"max_iter=", "{int} maximum number of iterations"},
                   {"tol=", "{double} tolerance"}});

    return call_test("test_mixer", test_mixer, args);
}
