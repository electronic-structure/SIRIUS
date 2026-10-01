/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include "function3d/kinetic_density.hpp"

using namespace sirius;

template <typename T>
int
test_spheric_kinetic_density(bool origin__)
{
    int const lmax = origin__ ? 2 : 3;
    auto grid      = Radial_grid_factory<double>(radial_grid_t::linear, 1001, origin__ ? 0 : 0.001, 2.0, 1.0);
    radial_functions_index indexr;
    indexr.add(angular_momentum(0));
    for (int l = 0; l <= lmax; l++) {
        indexr.add(angular_momentum(l));
    }
    basis_functions_index indexb(indexr, false);
    int n = indexb.size(), lmmax = sf::lmmax(2 * lmax);
    mdarray<double, 2> radial({grid.num_points(), indexr.size()}), radial_times_r({grid.num_points(), indexr.size()});
    for (auto const& rf : indexr) {
        for (int ir = 0; ir < grid.num_points(); ir++) {
            double r                     = grid[ir];
            radial(ir, rf.idxrf)         = origin__ ? std::pow(r, rf.am.l()) * (1 + 0.2 * rf.idxrf)
                                                    : std::pow(r, rf.am.l()) * std::exp(-(0.4 + 0.1 * rf.idxrf) * r * r);
            radial_times_r(ir, rf.idxrf) = r * radial(ir, rf.idxrf);
        }
    }
    Spheric_kinetic_density_operator<T> op(grid, indexb, radial, 2 * lmax);
    Spheric_kinetic_density_operator<T> paw_op(grid, indexb, radial_times_r, 2 * lmax,
                                               radial_wave_function_t::r_times_R);
    std::vector<std::complex<double>> c(n), direction(n);
    for (int i = 0; i < n; i++) {
        c[i]         = {0.2 * std::cos(i + 0.3), 0.3 * std::sin(2 * i + 0.7)};
        direction[i] = {0.13 * std::sin(i + 0.5), -0.19 * std::cos(3 * i + 0.2)};
    }
    double const occupation = 0.73;
    auto density_matrix     = [&](double step) {
        mdarray<std::complex<double>, 2> dm({n, n});
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                dm(i, j) = occupation * std::conj(c[i] + step * direction[i]) * (c[j] + step * direction[j]);
            }
        }
        return dm;
    };
    auto dm = density_matrix(0);
    Spheric_function<function_domain_t::spectral, double> tau(lmmax, grid), paw_tau(lmmax, grid);
    tau.zero();
    paw_tau.zero();
    op.add_density(dm, tau);
    paw_op.add_density(dm, paw_tau);
    int failures{0};
    auto check = [&](char const* label, double error, double tolerance) {
        std::cout << (std::is_same_v<T, double> ? "Rlm " : "Ylm ") << (origin__ ? "origin " : "positive grid ") << label
                  << ": " << error << '\n';
        failures += !std::isfinite(error) || error > tolerance;
    };
    double error{0};
    for (int ir = 0; ir < grid.num_points(); ir++) {
        for (int lm = 0; lm < lmmax; lm++) {
            double diff = std::abs(tau(lm, ir) - paw_tau(lm, ir));
            if (!std::isfinite(diff)) {
                return 1;
            }
            error = std::max(error, diff);
        }
    }
    check("R and rR conventions", error, 2e-10);

    if (!origin__) {
        // Independent Cartesian-gradient expansion, using Clebsch-Gordan coefficients rather than Gaunt products.
        SHT sht(device_t::CPU, 2 * lmax + 2);
        Spheric_function<function_domain_t::spectral, double> re(sht.lmmax(), grid), im(sht.lmmax(), grid);
        re.zero();
        im.zero();
        for (int i = 0; i < n; i++) {
            for (int ir = 0; ir < grid.num_points(); ir++) {
                re(indexb[i].lm, ir) += c[i].real() * radial(ir, indexb[i].idxrf);
                im(indexb[i].lm, ir) += c[i].imag() * radial(ir, indexb[i].idxrf);
            }
        }
        Spheric_function<function_domain_t::spectral, std::complex<double>> psi(sht.lmmax(), grid);
        if constexpr (std::is_same_v<T, double>) {
            auto zre = convert(re);
            auto zim = convert(im);
            for (int ir = 0; ir < grid.num_points(); ir++) {
                for (int lm = 0; lm < sht.lmmax(); lm++) {
                    psi(lm, ir) = zre(lm, ir) + std::complex<double>(0, 1) * zim(lm, ir);
                }
            }
        } else {
            for (int ir = 0; ir < grid.num_points(); ir++) {
                for (int lm = 0; lm < sht.lmmax(); lm++) {
                    psi(lm, ir) = {re(lm, ir), im(lm, ir)};
                }
            }
        }
        auto grad   = gradient(psi);
        auto actual = transform(sht, tau);
        Spheric_function<function_domain_t::spatial, double> reference(sht.num_points(), grid);
        reference.zero();
        for (int x = 0; x < 3; x++) {
            auto component = transform(sht, grad[x]);
            for (int ir = 0; ir < grid.num_points(); ir++) {
                for (int tp = 0; tp < sht.num_points(); tp++) {
                    reference(tp, ir) += 0.5 * occupation * std::norm(component(tp, ir));
                }
            }
        }
        error = 0;
        for (int ir = 0; ir < grid.num_points(); ir++) {
            for (int tp = 0; tp < sht.num_points(); tp++) {
                error = std::max(error, std::abs(actual(tp, ir) - reference(tp, ir)));
            }
        }
        check("Cartesian gradients", error, 2e-10);
    } else {
        double expected{0};
        for (int i = 0; i < n; i++) {
            if (indexb[i].am.l() == 1) {
                expected += occupation * 1.5 / fourpi * std::norm(c[i]) * std::pow(1 + 0.2 * indexb[i].idxrf, 2);
            }
        }
        check("analytic origin", std::abs(tau(0, 0) * y00 - expected), 2e-11);
        error = 0;
        for (int lm = 1; lm < lmmax; lm++) {
            error = std::max(error, std::abs(tau(lm, 0)));
        }
        check("isotropic p-wave origin", error, 2e-11);
    }

    Spheric_function<function_domain_t::spectral, double> v(lmmax, grid);
    for (int ir = 0; ir < grid.num_points(); ir++) {
        for (int lm = 0; lm < lmmax; lm++) {
            v(lm, ir) = std::sin(0.4 + lm) * std::exp(-0.2 * grid[ir]) / (lm + 1);
        }
    }
    auto matrix = op.matrix_elements(v);
    std::complex<double> expectation{0};
    error = 0;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            expectation += dm(i, j) * matrix(i, j);
            error = std::max(error, std::abs(matrix(i, j) - sirius::conj(matrix(j, i))));
        }
    }
    check("Hermiticity", error, 1e-13);
    check("density/operator normalization", std::abs(expectation - inner(v, tau)), 2e-11);

    // E = 1/2 integral tau^2 gives v_tau = tau and tests the full orbital derivative.
    auto nonlinear_matrix = op.matrix_elements(tau);
    double derivative{0};
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            derivative += 2 * occupation * std::real(std::conj(direction[i]) * nonlinear_matrix(i, j) * c[j]);
        }
    }
    auto energy = [&](double step) {
        auto displaced_dm = density_matrix(step);
        Spheric_function<function_domain_t::spectral, double> displaced_tau(lmmax, grid);
        displaced_tau.zero();
        op.add_density(displaced_dm, displaced_tau);
        return 0.5 * inner(displaced_tau, displaced_tau);
    };
    double step = 1e-5;
    check("nonlinear energy derivative", std::abs(derivative - (energy(step) - energy(-step)) / (2 * step)), 2e-8);

    return failures;
}

int
test_atom_kinetic_density(bool paw__, int num_mag_dims__)
{
    auto conf                                         = R"({"parameters": {
        "use_symmetry": false, "pw_cutoff": 6.0, "gk_cutoff": 1.0,
        "num_bands": 4, "xc_functionals": ["XC_LDA_X"],
        "auto_rmt": 0, "lmax_apw": 2, "lmax_rho": 4, "lmax_pot": 4
    }, "control": {"mpi_grid_dims": [1, 1], "processing_unit": "CPU", "verbosity": 0}})"_json;
    conf["parameters"]["electronic_structure_method"] = paw__ ? "pseudopotential" : "full_potential_lapwlo";
    conf["parameters"]["num_mag_dims"]                = num_mag_dims__;
    Simulation_context ctx(conf);
    ctx.unit_cell().set_lattice_vectors({{8, 0, 0}, {0, 9, 0}, {0, 0, 10}});
    auto& type = ctx.unit_cell().add_atom_type("H");
    type.zn(1);
    type.set_radial_grid(radial_grid_t::linear, 1001, 0.001, 2.0, 1.0);
    int nr                = type.num_mt_points();
    int partial_wave_size = paw__ ? nr / 2 + 1 : nr;
    if (paw__) {
        type.is_paw(true);
        for (int l = 0; l <= 2; l++) {
            std::vector<double> ae(nr), ps(nr), beta(nr);
            for (int ir = 0; ir < nr; ir++) {
                double r = type.radial_grid(ir);
                ae[ir]   = std::pow(r, l + 1) * std::exp(-0.5 * r * r);
                ps[ir]   = std::pow(r, l + 1) * std::exp(-0.3 * r * r);
                beta[ir] = std::pow(r, l + 1) * std::exp(-2.0 * r * r);
            }
            ae.resize(partial_wave_size);
            ps.resize(partial_wave_size);
            type.add_beta_radial_function(angular_momentum(l), beta);
            type.add_ae_paw_wf(ae);
            type.add_ps_paw_wf(ps);
        }
        // Deliberately nonzero charge compensation must not enter the orbital tau.
        std::vector<double> q(nr);
        for (int ir = 0; ir < nr; ir++) {
            double r = type.radial_grid(ir);
            q[ir]    = 0.2 * r * r * std::exp(-r * r);
        }
        type.add_q_radial_function(0, 0, 0, q);
        type.paw_wf_occ({1, 0, 0});
        type.paw_ae_core_charge_density(std::vector<double>(nr, 0));
        type.local_potential(std::vector<double>(nr, 0));
        type.ps_total_charge_density(std::vector<double>(nr, 0));
        matrix<double> dion({3, 3});
        dion.zero();
        type.d_mtrx_ion(dion);
    } else {
        for (int l = 0; l <= 2; l++) {
            type.add_aw_descriptor(l + 1, l, -0.5, 0, 0);
        }
    }
    ctx.unit_cell().add_atom("H", {0, 0, 0}, {0, 0, 1});
    ctx.initialize();
    if (!paw__) {
        for (auto const& rf : type.indexr()) {
            std::vector<double> values(nr), derivatives(nr);
            for (int ir = 0; ir < nr; ir++) {
                double r        = type.radial_grid(ir);
                values[ir]      = std::pow(r, rf.am.l()) * std::exp(-0.5 * r * r);
                derivatives[ir] = (rf.am.l() - r * r) * values[ir];
            }
            ctx.unit_cell().atom(0).symmetry_class().radial_function(rf.idxrf, values);
            ctx.unit_cell().atom(0).symmetry_class().radial_function_derivative(rf.idxrf, derivatives);
        }
    }
    Density density(ctx);
    density.density_matrix(0).zero();
    for (int ispn = 0; ispn < ctx.num_spins(); ispn++) {
        density.density_matrix(0)(0, 0, ispn) = 0.8 / (ispn + 1);
    }
    std::array<std::vector<Spheric_function<function_domain_t::spectral, double>>, 2> tau;
    if (paw__) {
        tau = density.generate_paw_valence_kinetic_density(0);
    } else {
        tau[0] = density.generate_mt_valence_kinetic_density(0);
    }
    double error{0};
    for (int part = 0; part < (paw__ ? 2 : 1); part++) {
        if (static_cast<int>(tau[part].size()) != ctx.num_spins()) {
            return 1;
        }
        double a = part == 0 ? 0.5 : 0.3;
        for (int ispn = 0; ispn < ctx.num_spins(); ispn++) {
            for (int ir = 0; ir < nr; ir++) {
                double r        = type.radial_grid(ir);
                double expected = 2 * a * a * r * r * std::exp(-2 * a * r * r) * 0.8 / (ispn + 1) * y00;
                if (ir >= partial_wave_size) {
                    expected = 0;
                }
                for (int lm = 0; lm < tau[part][ispn].angular_domain_size(); lm++) {
                    double diff = std::abs(tau[part][ispn](lm, ir) - (lm == 0 ? expected : 0));
                    if (!std::isfinite(diff)) {
                        return 1;
                    }
                    error = std::max(error, diff);
                }
            }
        }
    }
    std::cout << (paw__ ? "PAW" : "LAPW") << " density integration, magnetic dimensions=" << num_mag_dims__ << ": "
              << error << '\n';
    int failures = error > 2e-9;

    // A coherent s/p orbital probes the imaginary off-diagonal LAPW density matrix.
    std::complex<double> cs(0.3, 0.2), cp(0.1, -0.4);
    int p = type.indexb().index_by_l_m_order(1, 1, 0);
    density.density_matrix(0).zero();
    for (int ispn = 0; ispn < ctx.num_spins(); ispn++) {
        for (int j : {0, p}) {
            for (int i : {0, p}) {
                auto ci = i == 0 ? cs : cp;
                auto cj = j == 0 ? cs : cp;
                density.density_matrix(0)(i, j, ispn) =
                        0.8 / (ispn + 1) * (paw__ ? ci * std::conj(cj) : std::conj(ci) * cj);
            }
        }
    }
    if (paw__) {
        tau = density.generate_paw_valence_kinetic_density(0);
    } else {
        tau[0] = density.generate_mt_valence_kinetic_density(0);
    }
    SHT sht(device_t::CPU, 4);
    error = 0;
    for (int part = 0; part < (paw__ ? 2 : 1); part++) {
        double a = part == 0 ? 0.5 : 0.3;
        for (int ispn = 0; ispn < ctx.num_spins(); ispn++) {
            auto values    = transform(sht, tau[part][ispn]);
            double angular = -std::sqrt((paw__ ? 3.0 : 1.5) / fourpi);
            for (int ir = 0; ir < nr; ir++) {
                double r = type.radial_grid(ir);
                for (int tp = 0; tp < sht.num_points(); tp++) {
                    auto xyz        = r * sht.coord(tp);
                    auto polynomial = cs * y00 + cp * angular * std::complex<double>(xyz[0], paw__ ? 0 : xyz[1]);
                    std::array<std::complex<double>, 3> grad{
                            cp * angular, paw__ ? std::complex<double>(0) : cp * angular * std::complex<double>(0, 1),
                            0};
                    double expected{0};
                    for (int x = 0; x < 3; x++) {
                        expected += 0.5 * 0.8 / (ispn + 1) * std::exp(-2 * a * r * r) *
                                    std::norm(grad[x] - 2 * a * xyz[x] * polynomial);
                    }
                    if (ir >= partial_wave_size) {
                        expected = 0;
                    }
                    double diff = std::abs(values(tp, ir) - expected);
                    if (!std::isfinite(diff)) {
                        return 1;
                    }
                    error = std::max(error, diff);
                }
            }
        }
    }
    std::cout << (paw__ ? "PAW" : "LAPW") << " coherent s/p orbital: " << error << '\n';
    failures += error > 2e-9;
    return failures;
}

int
main(int argc, char** argv)
{
    sirius::initialize(1);
    int result{0};
    for (bool origin : {false, true}) {
        result += call_test("real-harmonic kinetic density", test_spheric_kinetic_density<double>, origin);
        result += call_test("complex-harmonic kinetic density", test_spheric_kinetic_density<std::complex<double>>,
                            origin);
    }
    for (bool paw : {false, true}) {
        for (int num_mag_dims : {0, 1}) {
            result += call_test("atom kinetic density", test_atom_kinetic_density, paw, num_mag_dims);
        }
    }
    sirius::finalize();
    return result;
}
