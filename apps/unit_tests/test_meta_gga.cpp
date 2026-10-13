/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include <cstdio>
#include "dft/dft_ground_state.hpp"
#include "hamiltonian/hamiltonian.hpp"

using namespace sirius;

auto
meta_context(bool gamma, int mag, bool parallel_fft, bool scf = false, std::string solver = "exact")
{
    auto conf                          = R"({"parameters": {
        "electronic_structure_method": "pseudopotential", "use_symmetry": false,
        "pw_cutoff": 8.0, "gk_cutoff": 2.6, "num_bands": 4, "smearing_width": 0.01,
        "xc_functionals": ["XC_LDA_X", "XC_MGGA_X_TPSS", "XC_MGGA_C_TPSS", "XC_GGA_C_PBE"],
        "xc_functionals_weight": [0.25, 0.75, 0.8, 0.2]
    }, "control": {"mpi_grid_dims": [1, 1], "processing_unit": "CPU", "verbosity": 0},
    "mixer": {"type": "linear", "beta": 0.25, "use_hartree": false},
    "iterative_solver": {"type": "exact", "min_tolerance": 1e-11}})"_json;
    conf["parameters"]["gamma_point"]  = gamma;
    conf["parameters"]["num_mag_dims"] = mag;
    conf["iterative_solver"]["type"]   = solver;
    if (mag) {
        conf["parameters"]["fixed_mag"] = 0.4;
    }
    if (parallel_fft) {
        conf["control"]["mpi_grid_dims"] = {mpi::Communicator::world().size(), 1};
    }
    auto ctx = std::make_unique<Simulation_context>(conf);
    ctx->unit_cell().set_lattice_vectors({{8, 0, 0}, {0, 9, 0}, {0, 0, 10}});
    auto& type = ctx->unit_cell().add_atom_type("He");
    type.zn(2);
    type.set_radial_grid(radial_grid_t::lin_exp, 400, 0.0, 20.0, 6);
    std::vector<double> v(400), rho(400), orbital(400);
    for (int ir = 0; ir < 400; ir++) {
        double r    = type.radial_grid(ir);
        v[ir]       = scf ? -2 / std::sqrt(1 + r * r) : 0;
        rho[ir]     = 4 * std::exp(-r * r) * r;
        orbital[ir] = std::exp(-r);
    }
    type.local_potential(v);
    type.ps_total_charge_density(rho);
    type.ps_core_charge_density(std::vector<double>(400, 0));
    type.add_ps_atomic_wf(1, angular_momentum(0), orbital);
    type.d_mtrx_ion(matrix<double>({0, 0}));
    ctx->unit_cell().add_atom("He", {0.2, 0.3, 0.4});
    ctx->initialize();
    return ctx;
}

int
test_meta_gga(bool gamma, int mag, bool parallel_fft)
{
    auto ctx = meta_context(gamma, mag, parallel_fft);
    K_point_set ks(*ctx);
    if (gamma) {
        ks.add_kpoint({0, 0, 0}, 1);
    } else {
        ks.add_kpoint({0.13, -0.17, 0.07}, 0.3);
        ks.add_kpoint({-0.1, 0.09, 0.21}, 0.7);
    }
    ks.initialize();
    int failures{0};
    auto check = [&](char const* name, double error, double tolerance) {
        if (ctx->comm().rank() == 0) {
            std::cout << "gamma=" << gamma << " mag=" << mag << " parallel_fft=" << parallel_fft << ' ' << name << ": "
                      << error << '\n';
        }
        failures += !std::isfinite(error) || error > tolerance;
    };
    auto coefficient = [&](r3::vector<int> g, double angle, bool derivative) {
        std::complex<double> a{0}, b{0};
        if (g[1] == 0 && g[2] == 0 && (gamma ? std::abs(g[0]) == 2 : g[0] == 2)) {
            a = gamma ? std::sqrt(0.5) : 1;
        }
        if (g[0] == 0 && g[2] == 0 && (gamma ? std::abs(g[1]) == 1 : g[1] == 1)) {
            b = std::polar(gamma ? std::sqrt(0.5) : 1.0, gamma && g[1] < 0 ? -0.4 : 0.4);
        }
        return derivative ? -std::sin(angle) * a + std::cos(angle) * b : std::cos(angle) * a + std::sin(angle) * b;
    };
    auto orbitals = [&](double step) {
        for (auto it : ks.spl_num_kpoints()) {
            auto kp   = ks.get<double>(it.i);
            auto& psi = kp->spinor_wave_functions();
            psi.zero(memory_t::host);
            for (int s = 0; s < ctx->num_spins(); s++) {
                for (int b = 0; b < ctx->num_bands(); b++) {
                    kp->band_occupancy(b, s, b == 0 ? (mag ? (s == 0 ? 1.2 : 0.8) : 2.0) : 0);
                    for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                        auto g = kp->gkvec().gvec(gvec_index_t::local(ig));
                        psi.pw_coeffs(ig, wf::spin_index(s), wf::band_index(b)) =
                                b == 0 ? coefficient(g, 0.37 + 0.1 * s + 0.05 * it.i + step, false) : 0;
                    }
                }
            }
        }
        ks.sync_band<double, sync_band_t::occupancy>();
    };
    Density density(*ctx);
    Potential potential(*ctx);
    check("automatic tau storage", density.has_kinetic_density() ? 0 : 1, 0);
    check("automatic tau operator", potential.has_kinetic_potential() ? 0 : 1, 0);
    auto evaluate = [&](double step) {
        orbitals(step);
        density.generate<double>(ks, false, true, true);
        potential.generate(density, false, true);
        return density.kinetic_density().scalar().integrate().total + 0.5 * potential.energy_vha() +
               energy_vloc(density, potential) + potential.energy_exc(density) + potential.ewald_energy();
    };
    double energy = evaluate(0);
    // Evaluate the same weighted functional directly with Libxc, independently
    // of Potential's spin conversion and XC energy storage.
    int nr = ctx->spfft<double>().local_slice_size(), ns = ctx->num_spins(), ng = ns == 1 ? 1 : 3;
    std::vector<double> rho(nr * ns), tau(nr * ns), sigma(nr * ng), lapl(nr * ns, 0), eps(nr);
    std::vector<Smooth_periodic_vector_function<double>> grad;
    for (int s = 0; s < ns; s++) {
        Smooth_periodic_function<double> n(ctx->spfft<double>(), ctx->gvec_fft_sptr());
        for (int ir = 0; ir < nr; ir++) {
            n.value(ir) =
                    (density.rho().rg().value(ir) + (ns == 1 ? 0 : (1 - 2 * s) * density.mag(0).rg().value(ir))) / ns;
            rho[ns * ir + s] = n.value(ir);
            tau[ns * ir + s] = (density.kinetic_density().scalar().rg().value(ir) +
                                (ns == 1 ? 0 : (1 - 2 * s) * density.kinetic_density().vector(0).rg().value(ir))) /
                               ns;
        }
        n.fft_transform(-1);
        grad.push_back(to_rg(gradient(n)));
    }
    for (int ir = 0; ir < nr; ir++) {
        for (int x = 0; x < 3; x++) {
            sigma[ng * ir] += std::pow(grad[0][x].value(ir), 2);
            if (ns == 2) {
                sigma[ng * ir + 1] += grad[0][x].value(ir) * grad[1][x].value(ir);
                sigma[ng * ir + 2] += std::pow(grad[1][x].value(ir), 2);
            }
        }
    }
    double reference{0};
    int index{0};
    for (int id : {XC_LDA_X, XC_MGGA_X_TPSS, XC_MGGA_C_TPSS, XC_GGA_C_PBE}) {
        xc_func_type functional;
        if (xc_func_init(&functional, id, ns) != 0) {
            RTE_THROW("test Libxc initialization failed");
        }
        xc_func_set_fhc_enforcement(&functional, 0);
        if (nr) {
            if (id == XC_LDA_X) {
                xc_lda_exc(&functional, nr, rho.data(), eps.data());
            } else if (id == XC_GGA_C_PBE) {
                xc_gga_exc(&functional, nr, rho.data(), sigma.data(), eps.data());
            } else {
                xc_mgga_exc(&functional, nr, rho.data(), sigma.data(), lapl.data(), tau.data(), eps.data());
            }
        }
        for (int ir = 0; ir < nr; ir++) {
            for (int s = 0; s < ns; s++) {
                reference += ctx->xc_functionals_weight()[index] * eps[ir] * rho[ns * ir + s];
            }
        }
        xc_func_end(&functional);
        index++;
    }
    mpi::Communicator(ctx->spfft<double>().communicator()).allreduce(&reference, 1);
    reference *= ctx->unit_cell().omega() / ctx->fft_grid().num_points();
    check("weighted Libxc energy", std::abs(reference - potential.energy_exc(density)), 2e-13);
    check("tau contribution nonzero", std::abs(potential.energy_vtau(density)) > 1e-6 ? 0 : 1, 0);

    if (!gamma) {
        auto file = "meta_gga_restart_" + std::to_string(ctx->comm().size()) + ".h5";
        ctx->create_storage_file(file);
        density.save(file);
        Density restored(*ctx);
        restored.load(file);
        double error{0};
        for (int s = 0; s < ns; s++) {
            auto const& original = density.kinetic_density().component(s).rg();
            auto const& loaded   = restored.kinetic_density().component(s).rg();
            for (int ir = 0; ir < nr; ir++) {
                error = std::max(error, std::abs(original.value(ir) - loaded.value(ir)));
            }
            for (int ig = 0; ig < ctx->gvec().count(); ig++) {
                error = std::max(error, std::abs(original.f_pw_local(ig) - loaded.f_pw_local(ig)));
            }
        }
        ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
        check("kinetic-density restart", error, 1e-14);
        if (ctx->comm().rank() == 0) {
            std::remove(file.c_str());
        }
    }

    Hamiltonian0<double> h0(potential);
    double derivative{0};
    for (auto it : ks.spl_num_kpoints()) {
        auto kp = ks.get<double>(it.i);
        auto hk = h0(*kp);
        wf::Wave_functions<double> hp(kp->gkvec_sptr(), wf::num_mag_dims(mag), wf::num_bands(1), memory_t::host);
        for (int s = 0; s < ns; s++) {
            if (gamma) {
                hk.apply_h_s<double>(wf::spin_range(s), wf::band_range(0, 1), kp->spinor_wave_functions(), &hp,
                                     nullptr);
            } else {
                hk.apply_h_s<std::complex<double>>(wf::spin_range(s), wf::band_range(0, 1), kp->spinor_wave_functions(),
                                                   &hp, nullptr);
            }
            double expectation{0};
            for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                auto g   = kp->gkvec().gvec(gvec_index_t::local(ig));
                auto d   = coefficient(g, 0.37 + 0.1 * s + 0.05 * it.i, true);
                auto c   = kp->spinor_wave_functions().pw_coeffs(ig, wf::spin_index(s), wf::band_index(0));
                auto h   = hp.pw_coeffs(ig, wf::spin_index(s), wf::band_index(0));
                double w = gamma && dot(g, g) ? 2 : 1;
                derivative += 2 * kp->weight() * kp->band_occupancy(0, s) * w * std::real(std::conj(d) * h);
                expectation += w * std::real(std::conj(c) * h);
            }
            kp->comm().allreduce(&expectation, 1);
            kp->band_energy(0, s, expectation);
        }
    }
    ctx->comm().allreduce(&derivative, 1);
    ks.sync_band<double, sync_band_t::energy>();
    check("total-energy double counting", std::abs(ks_energy(*ctx, ks, density, potential) - energy), 2e-10);
    for (double step : {1e-4, 5e-5}) {
        double fd = (evaluate(step) - evaluate(-step)) / (2 * step);
        check("orbital energy derivative", std::abs(fd - derivative), 5e-8);
    }
    evaluate(0);
    // Invalid local data must fail on every rank before entering collective FFTs.
    if (ctx->comm().rank() == 0 && nr) {
        density.kinetic_density().scalar().rg().value(0) = -1;
    }
    bool rejected{false};
    try {
        potential.generate(density, false, true);
    } catch (std::exception const&) {
        rejected = true;
    }
    check("collective invalid tau rejection", rejected ? 0 : 1, 0);
    return failures;
}

int
test_meta_scf(int mag, bool parallel_fft)
{
    int failures{0};
    double reference{0};
    for (auto solver : {"exact", "davidson"}) {
        auto ctx = meta_context(false, mag, parallel_fft, true, solver);
        K_point_set ks(*ctx);
        ks.add_kpoint({0.1, -0.1, 0.07}, 1);
        ks.initialize();
        DFT_ground_state ground(ks);
        ground.initial_state();
        auto result = ground.find(1e-8, 1e-9, 1e-9, 200, false);
        failures += !result.at("converged").get<bool>();
        auto const& d = ground.density();
        auto const& p = ground.potential();
        double direct = d.kinetic_density().scalar().integrate().total + 0.5 * p.energy_vha() + energy_vloc(d, p) +
                        p.energy_exc(d) + p.ewald_energy();
        double error  = std::abs(direct - ks_energy(*ctx, ks, d, p));
        failures += !std::isfinite(error) || error > 2e-7;
        if (ctx->cfg().iterative_solver().type() == "exact") {
            reference = direct;
        }
        double solver_error = std::abs(direct - reference);
        failures += !std::isfinite(solver_error) || solver_error > 2e-7;
        if (ctx->comm().rank() == 0) {
            std::cout << "Libxc meta-GGA SCF mag=" << mag << " parallel_fft=" << parallel_fft << " solver=" << solver
                      << " converged=" << result.at("converged") << " direct energy=" << direct
                      << " double-counting error=" << error << " solver difference=" << solver_error << '\n';
        }
    }
    return failures;
}

int
test_meta_capabilities()
{
    int failures{0};
    for (auto name : {"XC_HYB_MGGA_XC_TPSSH", "XC_MGGA_X_BR89", "XC_GGA_X_PBE"}) {
        XC_functional_base functional(name, 1.0, 1);
        bool rejected{false};
        try {
            functional.get_meta(0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        } catch (std::exception const&) {
            rejected = true;
        }
        failures += !rejected;
    }
    for (int ns : {1, 2}) {
        XC_functional_base functional("XC_MGGA_X_TPSS", 0.75, ns);
        functional.get_meta(0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        std::vector<double> n(ns, 0.15), t(ns, 0.12), sigma(ns == 1 ? 1 : 3, 0.02);
        if (ns == 2) {
            n[1]     = 0.10;
            t[1]     = 0.07;
            sigma[1] = -0.005;
        }
        std::vector<double> vn(ns), vt(ns), vs(sigma.size());
        double exc;
        functional.get_meta(1, n.data(), sigma.data(), t.data(), vn.data(), vs.data(), vt.data(), &exc);
        double largest{0};
        for (int variable = 0; variable < 3; variable++) {
            auto& values            = variable == 0 ? n : variable == 1 ? sigma : t;
            auto const& derivatives = variable == 0 ? vn : variable == 1 ? vs : vt;
            for (size_t i = 0; i < values.size(); i++) {
                auto energy = [&](double h) {
                    values[i] += h;
                    double e;
                    std::vector<double> dr(ns), dt(ns), ds(sigma.size());
                    functional.get_meta(1, n.data(), sigma.data(), t.data(), dr.data(), ds.data(), dt.data(), &e);
                    e *= std::accumulate(n.begin(), n.end(), 0.0);
                    values[i] -= h;
                    return e;
                };
                double coarse = (energy(1e-6) - energy(-1e-6)) / 2e-6;
                double fine   = (energy(5e-7) - energy(-5e-7)) / 1e-6;
                double fd     = (4 * fine - coarse) / 3;
                largest       = std::max(largest, std::abs(fd - derivatives[i]));
            }
        }
        std::cout << "Libxc primitive derivatives ns=" << ns << ": " << largest << '\n';
        failures += !std::isfinite(largest) || largest > 2e-9;
        for (double bad : {-0.1, std::numeric_limits<double>::quiet_NaN()}) {
            t[0] = bad;
            bool rejected{false};
            try {
                functional.get_meta(1, n.data(), sigma.data(), t.data(), vn.data(), vs.data(), vt.data(), &exc);
            } catch (std::exception const&) {
                rejected = true;
            }
            failures += !rejected;
        }
        XC_functional_base moved(std::move(functional));
        failures += moved.weight() != 0.75;
    }
    auto ctx = meta_context(false, 0, false);
    if (ctx->comm().rank() == 0) {
        ctx->unit_cell().atom_type(0).ps_core_charge_density(std::vector<double>(400, 0.1));
    }
    bool rejected{false};
    try {
        Potential potential(*ctx);
    } catch (std::exception const&) {
        rejected = true;
    }
    failures += !rejected;
    return failures;
}

int
main(int argc, char** argv)
{
    sirius::initialize(1);
    bool parallel = mpi::Communicator::world().size() > 1;
    int result    = call_test("Libxc meta-GGA capabilities", test_meta_capabilities);
    result += call_test("Libxc meta-GGA orbital derivative, Gamma", test_meta_gga, true, 0, parallel);
    result += call_test("Libxc meta-GGA orbital derivative, collinear", test_meta_gga, false, 1, parallel);
    result += call_test("Libxc meta-GGA SCF", test_meta_scf, parallel ? 1 : 0, parallel);
    sirius::finalize();
    return std::min(result, 1);
}
