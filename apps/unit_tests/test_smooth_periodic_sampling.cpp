/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include "function3d/smooth_periodic_function_sampler.hpp"

using namespace sirius;

int
test_smooth_periodic_sampling(bool reduced__, bool replicated_fft__)
{
    auto const& comm     = mpi::Communicator::world();
    auto const& fft_comm = replicated_fft__ ? mpi::Communicator::self() : comm;
    auto const& ortho    = replicated_fft__ ? comm : mpi::Communicator::self();
    r3::matrix<double> reciprocal{{1.1, 0.2, 0.1}, {0, 0.9, 0.15}, {0, 0, 1.2}};
    fft::Gvec gv(reciprocal, 4.5, comm, reduced__);
    auto gvp    = std::make_shared<fft::Gvec_fft>(gv, fft_comm, ortho);
    int const n = 16;
    auto spl_z  = fft::split_z_dimension(n, fft_comm);
    spfft::Grid grid(n, n, n, gvp->zcol_count(), spl_z.local_size(), SPFFT_PU_HOST, 1, fft_comm.native(),
                     SPFFT_EXCH_DEFAULT);
    spfft::Transform transform(grid.create_transform(SPFFT_PU_HOST, reduced__ ? SPFFT_TRANS_R2C : SPFFT_TRANS_C2C, n, n,
                                                     n, spl_z.local_size(), gvp->count(), SPFFT_INDEX_TRIPLETS,
                                                     gvp->gvec_array().at(memory_t::host)));
    Smooth_periodic_function<double> field(transform, gvp), potential(transform, gvp), direction(transform, gvp);
    auto coefficients = [](r3::vector<int> g) -> std::complex<double> {
        if (r3::dot(g, g) == 0) {
            return 0.7;
        }
        if (g[0] == 1 && g[1] == -1 && g[2] == 0) {
            return {0.15, -0.2};
        }
        if (g[0] == -1 && g[1] == 1 && g[2] == 0) {
            return {0.15, 0.2};
        }
        return 0;
    };
    for (int ig = 0; ig < gv.count(); ig++) {
        auto g               = gv.gvec(gvec_index_t::local(ig));
        field.f_pw_local(ig) = coefficients(g);
        direction.f_pw_local(ig) =
                std::complex<double>(0.08 * std::cos(g[0] + 2 * g[1] - g[2]), 0.06 * std::sin(2 * g[0] - g[1] + g[2]));
    }
    int failures{0};
    auto check = [&](char const* name, double error, double tolerance) {
        comm.allreduce<double, mpi::op_t::max>(&error, 1);
        if (comm.rank() == 0) {
            std::cout << "Fourier sampling reduced=" << reduced__ << " replicated FFT=" << replicated_fft__ << " "
                      << name << ": " << error << '\n';
        }
        failures += !std::isfinite(error) || error > tolerance;
    };
    auto contraction = [&](Smooth_periodic_function<double> const& p, Smooth_periodic_function<double> const& f) {
        double value{0};
        for (int ig = 0; ig < gv.count(); ig++) {
            auto g   = gv.gvec(gvec_index_t::local(ig));
            double m = reduced__ && r3::dot(g, g) != 0 ? 2 : 1;
            value += gv.omega() * m * std::real(std::conj(p.f_pw_local(ig)) * f.f_pw_local(ig));
        }
        comm.allreduce(&value, 1);
        return value;
    };
    auto lattice = gv.unit_cell_lattice_vectors();
    auto q       = r3::dot(reciprocal, r3::vector<double>({1, -1, 0}));
    for (int total_points : {0, 1, 7}) {
        std::vector<r3::vector<double>> points;
        for (int p = comm.rank(); p < total_points; p += comm.size()) {
            points.push_back(r3::dot(lattice, r3::vector<double>({0.17 + 0.31 * p, -0.2 + 0.41 * p, 1.3 - 0.2 * p})));
        }
        Smooth_periodic_function_sampler map(gv, points);
        auto sampled = map.sample(field);
        double error{0};
        for (size_t p = 0; p < points.size(); p++) {
            double phase = r3::dot(q, points[p]);
            error = std::max(error, std::abs(sampled[p][0] - (0.7 + 0.3 * std::cos(phase) + 0.4 * std::sin(phase))));
            for (int x = 0; x < 3; x++) {
                error = std::max(error,
                                 std::abs(sampled[p][x + 1] - q[x] * (-0.3 * std::sin(phase) + 0.4 * std::cos(phase))));
            }
        }
        check("analytic values/Cartesian gradients", error, 3e-13);
        auto moved = points;
        for (auto& p : moved) {
            p = p + r3::dot(lattice, r3::vector<double>({3, -2, 1}));
        }
        auto shifted = Smooth_periodic_function_sampler(gv, moved).sample(field);
        error        = 0;
        for (size_t p = 0; p < points.size(); p++) {
            for (int x = 0; x < 4; x++) {
                error = std::max(error, std::abs(shifted[p][x] - sampled[p][x]));
            }
        }
        check("lattice translations", error, 3e-13);
        std::vector<std::array<double, 4>> covectors(points.size());
        auto delta = map.sample(direction);
        double forward{0};
        for (size_t p = 0; p < points.size(); p++) {
            for (int x = 0; x < 4; x++) {
                covectors[p][x] = std::cos(0.3 + comm.rank() + 2 * p + x);
                forward += covectors[p][x] * delta[p][x];
            }
        }
        comm.allreduce(&forward, 1);
        map.adjoint(covectors, potential);
        check("raw-covector/volume-normalized adjoint", std::abs(forward - contraction(potential, direction)), 3e-12);

        // A joint nonlocal energy with different point weights, not separate local calls.
        auto energy = [&](double step, bool reverse) {
            Smooth_periodic_function<double> trial(transform, gvp);
            for (int ig = 0; ig < gv.count(); ig++) {
                trial.f_pw_local(ig) = field.f_pw_local(ig) + step * direction.f_pw_local(ig);
            }
            auto values = map.sample(trial);
            double sum{0}, e{0};
            for (size_t p = 0; p < points.size(); p++) {
                double w = 0.2 + 0.03 * (p * comm.size() + comm.rank());
                sum += w * values[p][0];
                for (int x = 0; x < 4; x++) {
                    e += 0.5 * w * values[p][x] * values[p][x];
                }
            }
            comm.allreduce(&sum, 1);
            comm.allreduce(&e, 1);
            if (reverse) {
                for (size_t p = 0; p < points.size(); p++) {
                    double w = 0.2 + 0.03 * (p * comm.size() + comm.rank());
                    for (int x = 0; x < 4; x++) {
                        covectors[p][x] = w * (values[p][x] + (x == 0 ? 0.26 * sum : 0));
                    }
                }
                map.adjoint(covectors, potential);
            }
            return e + 0.13 * sum * sum;
        };
        energy(0, true);
        double analytic = contraction(potential, direction);
        for (double h : {1e-4, 5e-5}) {
            double fd = (energy(h, false) - energy(-h, false)) / (2 * h);
            check("nonlocal-energy coefficient FD", std::abs(analytic - fd), 2e-9);
        }
    }

    // Dense coefficients and partial tiles must agree with independent direct sums.
    {
        std::vector<r3::vector<double>> points;
        for (int p = 0; p < 269; p++) {
            points.push_back(r3::dot(
                    lattice, r3::vector<double>({0.013 * p + 0.2 * comm.rank(), 0.027 * p - 0.4, 0.037 * p + 1.7})));
        }
        Smooth_periodic_function_sampler map(gv, points);
        auto sampled = map.sample(direction);
        std::vector<std::complex<double>> c(gv.num_gvec());
        comm.allgather(direction.f_pw_local().at(memory_t::host), c.data(), gv.count(), gv.offset());
        std::vector<std::array<double, 4>> covectors(points.size());
        double error{0};
        for (size_t p = 0; p < points.size(); p++) {
            std::array<double, 4> reference{};
            for (int ig = 0; ig < gv.num_gvec(); ig++) {
                auto g   = gv.gvec_cart(gvec_index_t::global(ig));
                auto z   = c[ig] * std::exp(std::complex<double>(0, r3::dot(g, points[p])));
                double m = reduced__ && r3::dot(g, g) != 0 ? 2 : 1;
                reference[0] += m * z.real();
                for (int x = 0; x < 3; x++) {
                    reference[x + 1] -= m * g[x] * z.imag();
                }
            }
            for (int x = 0; x < 4; x++) {
                error           = std::max(error, std::abs(sampled[p][x] - reference[x]));
                covectors[p][x] = std::sin(0.17 * p + x + comm.rank());
            }
        }
        check("tiled sampling/direct sum", error, 3e-12);
        std::vector<std::complex<double>> reference(gv.num_gvec());
        for (int ig = 0; ig < gv.num_gvec(); ig++) {
            auto g = gv.gvec_cart(gvec_index_t::global(ig));
            for (size_t p = 0; p < points.size(); p++) {
                auto const& d = covectors[p];
                double grad   = g[0] * d[1] + g[1] * d[2] + g[2] * d[3];
                reference[ig] += std::complex<double>(d[0], -grad) *
                                 std::exp(std::complex<double>(0, -r3::dot(g, points[p]))) / gv.omega();
            }
        }
        comm.allreduce(reference.data(), reference.size());
        map.adjoint(covectors, potential);
        error = 0;
        for (int ig = 0; ig < gv.count(); ig++) {
            error = std::max(error, std::abs(potential.f_pw_local(ig) - reference[gv.offset() + ig]));
        }
        check("tiled adjoint/direct sum", error, 3e-12);
    }

    // Compare forward and reverse with the existing FFT/gradient implementation.
    std::vector<r3::vector<double>> points;
    std::vector<int> local_indices;
    for (int z = 0; z < transform.local_z_length(); z++) {
        for (int y = 0; y < n; y++) {
            for (int x = 0; x < n; x++) {
                int global_index = x + n * (y + n * (z + transform.local_z_offset()));
                if (replicated_fft__ && global_index % comm.size() != comm.rank()) {
                    continue;
                }
                points.push_back(r3::dot(lattice, r3::vector<double>({double(x) / n, double(y) / n,
                                                                      double(z + transform.local_z_offset()) / n})));
                local_indices.push_back(x + n * (y + n * z));
            }
        }
    }
    Smooth_periodic_function_sampler fft_map(gv, points);
    auto values = fft_map.sample(field);
    field.fft_transform(1);
    auto gradient_field = to_rg(gradient(field));
    std::vector<std::array<double, 4>> covectors(points.size());
    double error{0}, dv = gv.omega() / (n * n * n);
    for (size_t p = 0; p < points.size(); p++) {
        int i           = local_indices[p];
        error           = std::max(error, std::abs(values[p][0] - field.value(i)));
        covectors[p][0] = dv * field.value(i);
        for (int x = 0; x < 3; x++) {
            error               = std::max(error, std::abs(values[p][x + 1] - gradient_field[x].value(i)));
            covectors[p][x + 1] = dv * gradient_field[x].value(i);
        }
    }
    check("FFT value/gradient parity", error, 3e-13);
    fft_map.adjoint(covectors, potential);
    error = 0;
    for (int ig = 0; ig < gv.count(); ig++) {
        auto g         = gv.gvec_cart(gvec_index_t::local(ig));
        auto reference = (1 + r3::dot(g, g)) * field.f_pw_local(ig);
        error          = std::max(error, std::abs(potential.f_pw_local(ig) - reference));
    }
    check("FFT gradient-divergence adjoint parity", error, 3e-13);
    bool rejected{false};
    if (comm.rank() == 0) {
        covectors[0][1] = std::numeric_limits<double>::quiet_NaN();
    }
    try {
        fft_map.adjoint(covectors, potential);
    } catch (std::exception const&) {
        rejected = true;
    }
    check("collective nonfinite-covector rejection", rejected ? 0 : 1, 0);
    return failures;
}

int
main(int argc, char** argv)
{
    sirius::initialize(1);
    int result{0};
    for (bool reduced : {false, true}) {
        for (bool replicated_fft : {false, true}) {
            result += call_test("smooth periodic sampling", test_smooth_periodic_sampling, reduced, replicated_fft);
        }
    }
    sirius::finalize();
    return result;
}
