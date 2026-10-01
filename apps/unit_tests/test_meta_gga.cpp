/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include "dft/dft_ground_state.hpp"
#include "density/paw_local_fields.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "potential/xc_mt_meta.hpp"

using namespace sirius;

auto
meta_context(bool gamma, int mag, bool parallel_fft, bool scf = false, bool core = false, bool paw = false,
             std::string solver = "exact")
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
    if (paw) {
        conf["settings"]["sht_lmax"] = 4;
    }
    auto ctx = std::make_unique<Simulation_context>(conf);
    ctx->unit_cell().set_lattice_vectors({{8, 0, 0}, {0, 9, 0}, {0, 0, 10}});
    auto& type = ctx->unit_cell().add_atom_type("He");
    type.zn(2);
    type.set_radial_grid(radial_grid_t::lin_exp, 400, paw ? 0.001 : 0.0, 20.0, 6);
    std::vector<double> v(400), rho(400), orbital(400), core_rho(400), core_tau(400);
    for (int ir = 0; ir < 400; ir++) {
        double r = type.radial_grid(ir);
        v[ir]    = scf ? -2 / std::sqrt(1 + r * r) : 0;
        rho[ir]  = 4 * std::exp(-r * r) * r;
        if (core && scf) {
            // Normalized diffuse Gaussian, stored as 4*pi*r^2*n(r), for the initial guess only.
            double a = 0.15;
            rho[ir]  = 8 * std::pow(a, 1.5) / std::sqrt(pi) * r * r * std::exp(-a * r * r);
        }
        orbital[ir]  = std::exp(-r);
        core_rho[ir] = core ? 0.25 * std::exp(-0.5 * r * r) : 0;
        core_tau[ir] = 0.125 * r * r * core_rho[ir];
    }
    type.local_potential(v);
    type.ps_total_charge_density(rho);
    type.ps_core_charge_density(core_rho);
    if (core) {
        type.ps_core_kinetic_density(core_tau);
    }
    type.add_ps_atomic_wf(1, angular_momentum(0), orbital);
    if (paw) {
        type.is_paw(true);
        for (int l : {0, 1}) {
            std::vector<double> ae(400), ps(400), beta(400);
            for (int ir = 0; ir < 400; ir++) {
                double r = type.radial_grid(ir);
                ae[ir]   = std::pow(r, l + 1) * std::exp(-0.9 * r * r);
                ps[ir]   = std::pow(r, l + 1) * std::exp(-0.7 * r * r);
                beta[ir] = 0.12 * std::pow(r, l + 1) * std::exp(-2 * r * r);
            }
            type.add_beta_radial_function(angular_momentum(l), beta);
            type.add_ae_paw_wf(ae);
            type.add_ps_paw_wf(ps);
        }
        for (auto ijl : {std::array<int, 3>{0, 0, 0}, {0, 1, 1}, {1, 1, 0}, {1, 1, 2}}) {
            std::vector<double> q(400);
            for (int ir = 0; ir < 400; ir++) {
                double r = type.radial_grid(ir);
                q[ir]    = 0.005 * std::pow(r, ijl[0] + ijl[1] + 2) * std::exp(-1.6 * r * r);
            }
            type.add_q_radial_function(ijl[0], ijl[1], ijl[2], q);
        }
        type.paw_ae_core_charge_density(core_rho);
        type.paw_ae_core_kinetic_density(core_tau);
        type.paw_wf_occ({2, 0});
        matrix<double> dion({2, 2});
        dion.zero();
        type.d_mtrx_ion(dion);
    } else {
        type.d_mtrx_ion(matrix<double>({0, 0}));
    }
    ctx->unit_cell().add_atom("He", {0.2, 0.3, 0.4});
    ctx->initialize();
    return ctx;
}

int
test_meta_gga(bool gamma, int mag, bool parallel_fft, bool core, bool paw)
{
    auto ctx = meta_context(gamma, mag, parallel_fft, false, core, paw);
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
            std::cout << "gamma=" << gamma << " mag=" << mag << " parallel_fft=" << parallel_fft << " core=" << core
                      << ' ' << name << ": " << error << '\n';
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
               energy_vloc(density, potential) + potential.energy_exc(density) + potential.ewald_energy() +
               potential.PAW_total_energy(density);
    };
    double energy = evaluate(0);
    if (paw) {
        double rho_error{0}, tau_error{0};
        for (auto it : ctx->unit_cell().spl_num_paw_atoms()) {
            auto ia            = ctx->unit_cell().paw_atom_index(it.i);
            auto tau_reference = density.generate_paw_valence_kinetic_density(ia);
            for (int side = 0; side < 2; side++) {
                PAW_local_fields map(ctx->unit_cell().atom(ia).type(), side == 0);
                auto fields        = map.fields(density.density_matrix(ia), false);
                auto rho_reference = side == 0 ? density.paw_ae_density(ia) : density.paw_ps_density(ia);
                for (int s = 0; s < ctx->num_spins(); s++) {
                    for (size_t i = 0; i < fields[0][s].size(); i++) {
                        double reference = ((*rho_reference[0])[i] + (mag ? (1 - 2 * s) * (*rho_reference[1])[i] : 0)) /
                                           ctx->num_spins();
                        rho_error = std::max(rho_error, std::abs(fields[0][s][i] - reference));
                        tau_error = std::max(tau_error, std::abs(fields[1][s][i] - tau_reference[side][s][i]));
                    }
                }
            }
        }
        ctx->comm().allreduce<double, mpi::op_t::max>(&rho_error, 1);
        ctx->comm().allreduce<double, mpi::op_t::max>(&tau_error, 1);
        check("PAW local charge fields", rho_error, 1e-13);
        check("PAW local kinetic fields", tau_error, 1e-13);
        check("PAW local XC energy nonzero", std::abs(potential.PAW_xc_total_energy(density)) > 1e-9 ? 0 : 1, 0);
    }
    // Evaluate the same weighted functional directly with Libxc, independently
    // of Potential's spin conversion and XC energy storage.
    int nr = ctx->spfft<double>().local_slice_size(), ns = ctx->num_spins(), ng = ns == 1 ? 1 : 3;
    std::vector<double> rho(nr * ns), tau(nr * ns), sigma(nr * ng), lapl(nr * ns, 0), eps(nr);
    // The two-mode orbitals give an independent reference for the PAW orbital
    // density. Compensation charge belongs only to its electrostatic density.
    std::vector<double> orbital_rho(nr * ns, 0);
    if (paw) {
        auto const& fft = ctx->spfft<double>();
        for (int z = 0; z < fft.local_z_length(); z++) {
            for (int y = 0; y < fft.dim_y(); y++) {
                for (int x = 0; x < fft.dim_x(); x++) {
                    double gx = 4 * pi * x / fft.dim_x(), gy = 2 * pi * y / fft.dim_y() + 0.4;
                    int ir = ctx->fft_grid().index_by_coord(x, y, z);
                    for (int k = 0; k < (gamma ? 1 : 2); k++) {
                        double weight = gamma ? 1 : (k == 0 ? 0.3 : 0.7);
                        for (int s = 0; s < ns; s++) {
                            double angle = 0.37 + 0.1 * s + 0.05 * k;
                            auto psi = gamma ? std::complex<double>(std::sqrt(2.0) * (std::cos(angle) * std::cos(gx) +
                                                                                      std::sin(angle) * std::cos(gy)),
                                                                    0)
                                             : std::cos(angle) * std::polar(1.0, gx) +
                                                       std::sin(angle) * std::polar(1.0, gy);
                            double occupancy = mag ? (s == 0 ? 1.2 : 0.8) : 2;
                            orbital_rho[ns * ir + s] += weight * occupancy * std::norm(psi) / ctx->unit_cell().omega();
                        }
                    }
                }
            }
        }
    }
    std::vector<Smooth_periodic_vector_function<double>> grad;
    for (int s = 0; s < ns; s++) {
        Smooth_periodic_function<double> n(ctx->spfft<double>(), ctx->gvec_fft_sptr());
        for (int ir = 0; ir < nr; ir++) {
            n.value(ir) = (density.rho().rg().value(ir) + density.rho_pseudo_core().value(ir) +
                           (ns == 1 ? 0 : (1 - 2 * s) * density.mag(0).rg().value(ir))) /
                          ns;
            if (paw) {
                n.value(ir) = orbital_rho[ns * ir + s] + density.rho_pseudo_core().value(ir) / ns;
            }
            rho[ns * ir + s] = n.value(ir);
            tau[ns * ir + s] =
                    (density.kinetic_density().scalar().rg().value(ir) + density.tau_pseudo_core().value(ir) +
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
#if defined(XC_FLAGS_ENFORCE_FHC)
        xc_func_set_fhc_enforcement(&functional, 0);
#endif
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

    Hamiltonian0<double> h0(potential, false);
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
    if (paw) {
        bool rejected{false};
        try {
            potential.generate(density, true, true);
        } catch (std::exception const&) {
            rejected = true;
        }
        check("reject independent PAW potential symmetrization", rejected ? 0 : 1, 0);
    }
    // Invalid local data must fail on every rank before entering collective FFTs.
    int rank = 0;
    if (ctx->comm().rank() == rank && nr) {
        density.kinetic_density().scalar().rg().value(0) = -1;
    }
    bool rejected{false};
    try {
        potential.generate(density, false, true);
    } catch (std::exception const&) {
        rejected = true;
    }
    check("collective invalid tau rejection", rejected ? 0 : 1, 0);
    if (paw) {
        evaluate(0);
        for (auto it : ctx->unit_cell().spl_num_paw_atoms()) {
            auto ia                             = ctx->unit_cell().paw_atom_index(it.i);
            density.density_matrix(ia)(0, 0, 0) = -1e4;
        }
        rejected = false;
        try {
            potential.generate(density, false, true);
        } catch (std::exception const&) {
            rejected = true;
        }
        check("collective invalid local PAW fields", rejected ? 0 : 1, 0);
    }
    return failures;
}

int
test_meta_scf(int mag, bool parallel_fft, bool core, bool paw)
{
    int failures{0};
    double reference{0};
    for (auto solver : {"exact", "davidson"}) {
        auto ctx = meta_context(false, mag, parallel_fft, true, core, paw, solver);
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
                        p.energy_exc(d) + p.ewald_energy() + p.PAW_total_energy(d);
        double error = std::abs(direct - ks_energy(*ctx, ks, d, p));
        failures += !std::isfinite(error) || error > 2e-7;
        if (ctx->cfg().iterative_solver().type() == "exact") {
            reference = direct;
        }
        double solver_error = std::abs(direct - reference);
        failures += !std::isfinite(solver_error) || solver_error > 2e-7;
        if (ctx->comm().rank() == 0) {
            std::cout << "Libxc meta-GGA SCF mag=" << mag << " parallel_fft=" << parallel_fft << " core=" << core
                      << " solver=" << solver << " converged=" << result.at("converged") << " direct energy=" << direct
                      << " double-counting error=" << error << " solver difference=" << solver_error << '\n';
        }
    }
    return failures;
}

int
test_spherical_meta(int ns)
{
    auto ctx = meta_context(false, ns - 1, false);
    SHT sht(device_t::CPU, 4, 0);
    std::vector<double> radii(35);
    for (size_t i = 0; i < radii.size(); i++) {
        radii[i] = 0.01 + 1.8 * std::pow(double(i) / (radii.size() - 1), 1.7);
    }
    Radial_grid_ext<double> grid(static_cast<int>(radii.size()), radii.data());
    std::vector<Flm> rho, tau, direction;
    double pnorm = std::sqrt(3 / fourpi);
    for (int s = 0; s < ns; s++) {
        rho.emplace_back(9, grid);
        tau.emplace_back(9, grid);
        direction.emplace_back(9, grid);
        rho.back().zero();
        tau.back().zero();
        double scale = 1 + 0.2 * s;
        for (int ir = 0; ir < grid.num_points(); ir++) {
            double r                  = grid[ir];
            rho[s](0, ir)             = scale * (0.3 + 0.02 * r * r) / y00;
            rho[s](sf::lm(1, 1), ir)  = -scale * 0.01 * r / pnorm;
            rho[s](sf::lm(1, -1), ir) = scale * 0.015 * r / pnorm;
            rho[s](sf::lm(1, 0), ir)  = scale * 0.02 * r / pnorm;
            tau[s](0, ir)             = scale * (0.2 + 0.01 * r * r) / y00;
            tau[s](sf::lm(1, 1), ir)  = -scale * 0.003 * r / pnorm;
            for (int lm = 0; lm < 9; lm++) {
                direction[s](lm, ir) = 0.03 * std::sin(0.2 + 0.5 * lm + 0.3 * s + r) / (lm + 1);
            }
        }
    }
    std::vector<XC_functional> functionals;
    functionals.emplace_back(ctx->spfft<double>(), ctx->unit_cell().lattice_vectors(), "XC_MGGA_X_TPSS", 0.7, ns);
    auto xc = xc_mt_meta(sht, functionals, rho, tau);
    // Independent Cartesian polynomials and radial integration check the
    // normalization and gradient, not just an energy/adjoint identity.
    int nt = sht.num_points(), ng = ns == 1 ? 1 : 3;
    std::vector<double> n(nt * ns), t(nt * ns), sigma(nt * ng), eps(nt);
    xc_func_type functional;
    if (xc_func_init(&functional, XC_MGGA_X_TPSS, ns) != 0) {
        RTE_THROW("test Libxc initialization failed");
    }
#if defined(XC_FLAGS_ENFORCE_FHC)
    xc_func_set_fhc_enforcement(&functional, 0);
#endif
    Spline<double> radial(grid);
    for (int ir = 0; ir < grid.num_points(); ir++) {
        radial(ir) = 0;
        std::fill(sigma.begin(), sigma.end(), 0);
        for (int p = 0; p < nt; p++) {
            auto position = grid[ir] * sht.coord(p);
            std::array<r3::vector<double>, 2> grad;
            for (int s = 0; s < ns; s++) {
                double scale  = 1 + 0.2 * s;
                n[p * ns + s] = scale * (0.3 + 0.02 * dot(position, position) + 0.01 * position[0] -
                                         0.015 * position[1] + 0.02 * position[2]);
                t[p * ns + s] = scale * (0.2 + 0.01 * dot(position, position) + 0.003 * position[0]);
                for (int x = 0; x < 3; x++) {
                    grad[s][x] = scale * (0.04 * position[x] + std::array<double, 3>{0.01, -0.015, 0.02}[x]);
                }
            }
            sigma[p * ng] = dot(grad[0], grad[0]);
            if (ns == 2) {
                sigma[p * ng + 1] = dot(grad[0], grad[1]);
                sigma[p * ng + 2] = dot(grad[1], grad[1]);
            }
        }
        std::vector<double> lapl(nt * ns, 0);
        xc_mgga_exc(&functional, nt, n.data(), sigma.data(), lapl.data(), t.data(), eps.data());
        for (int p = 0; p < nt; p++) {
            for (int s = 0; s < ns; s++) {
                radial(ir) += 0.7 * fourpi * sht.weight(p) * grid[ir] * grid[ir] * n[p * ns + s] * eps[p];
            }
        }
    }
    xc_func_end(&functional);
    double energy_error = std::abs(xc.energy - radial.interpolate().integrate(0));
    int failures        = !std::isfinite(energy_error) || energy_error > 2e-12;
    std::cout << "spherical analytical-polynomial energy ns=" << ns << ": " << energy_error << '\n';
    for (int field = 0; field < 2; field++) {
        auto& values = field == 0 ? rho : tau;
        double derivative{0};
        for (int s = 0; s < ns; s++) {
            for (size_t i = 0; i < values[s].size(); i++) {
                derivative += direction[s][i] * xc.fields[field][s][i];
            }
        }
        auto energy = [&](double step) {
            for (int s = 0; s < ns; s++) {
                for (size_t i = 0; i < values[s].size(); i++) {
                    values[s][i] += step * direction[s][i];
                }
            }
            double result = xc_mt_meta(sht, functionals, rho, tau).energy;
            for (int s = 0; s < ns; s++) {
                for (size_t i = 0; i < values[s].size(); i++) {
                    values[s][i] -= step * direction[s][i];
                }
            }
            return result;
        };
        for (double step : {1e-4, 5e-5}) {
            double error = std::abs((energy(step) - energy(-step)) / (2 * step) - derivative);
            failures += !std::isfinite(error) || error > 2e-8;
            std::cout << "spherical field derivative ns=" << ns << " field=" << field << ": " << error << '\n';
        }
    }
    return failures;
}

int
test_paw_meta_matrix(int ns)
{
    auto ctx         = meta_context(false, ns - 1, false, false, true, true);
    auto const& type = ctx->unit_cell().atom_type(0);
    SHT sht(device_t::CPU, 4, 0);
    std::vector<XC_functional> functionals;
    for (size_t i = 0; i < ctx->xc_functionals().size(); i++) {
        functionals.emplace_back(ctx->spfft<double>(), ctx->unit_cell().lattice_vectors(), ctx->xc_functionals()[i],
                                 ctx->xc_functionals_weight()[i], ns);
    }
    int n = type.indexb().size();
    mdarray<std::complex<double>, 3> dm({n, n, ns}), direction({n, n, ns});
    for (int s = 0; s < ns; s++) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                auto a             = std::polar(0.4 + 0.1 * i, 0.2 * i + s);
                auto b             = std::polar(0.4 + 0.1 * j, 0.2 * j + s);
                dm(i, j, s)        = a * std::conj(b) + (i == j ? 0.3 : 0);
                direction(i, j, s) = {0.02 * std::cos(i + j + 0.3 * s), 0.01 * std::sin(i - j)};
            }
        }
    }
    int failures{0};
    for (bool ae : {true, false}) {
        PAW_local_fields map(type, ae, false);
        auto fields = map.fields(dm);
        auto xc     = xc_mt_meta(sht, functionals, fields[0], fields[1]);
        double derivative{0};
        for (int s = 0; s < ns; s++) {
            auto matrix = map.matrix_elements(xc.fields[0][s], xc.fields[1][s]);
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++) {
                    derivative += std::real(direction(i, j, s)) * matrix(i, j);
                }
            }
        }
        auto energy = [&](double step) {
            for (size_t i = 0; i < dm.size(); i++) {
                dm[i] += step * direction[i];
            }
            auto fields   = map.fields(dm);
            double result = xc_mt_meta(sht, functionals, fields[0], fields[1]).energy;
            for (size_t i = 0; i < dm.size(); i++) {
                dm[i] -= step * direction[i];
            }
            return result;
        };
        failures += std::abs(derivative) < 1e-4;
        for (double step : {1e-4, 5e-5}) {
            double error = std::abs((energy(step) - energy(-step)) / (2 * step) - derivative);
            failures += !std::isfinite(error) || error > 2e-8;
            std::cout << "PAW meta matrix ns=" << ns << " ae=" << ae << " energy=" << xc.energy
                      << " derivative=" << derivative << " FD error=" << error << '\n';
        }
    }
    return failures;
}

int
test_paw_meta_input()
{
    auto ctx       = meta_context(false, 0, false, false, true, true);
    auto& original = ctx->unit_cell().atom_type(0);
    original.paw_core_energy(-3.125);
    auto ae_tau = original.paw_ae_core_kinetic_density();
    for (auto& value : ae_tau) {
        value *= 1.7;
    }
    original.paw_ae_core_kinetic_density(ae_tau);
    auto data = original.serialize();
    Simulation_context copy_ctx(R"({"parameters": {"electronic_structure_method": "pseudopotential"}})"_json);
    copy_ctx.electronic_structure_method("pseudopotential");
    auto& copy = copy_ctx.unit_cell().add_atom_type("He");
    copy.read_input(data);
    int failures = copy.paw_core_energy() != -3.125 || copy.paw_ae_core_kinetic_density() != ae_tau ||
                   copy.ps_core_kinetic_density() != original.ps_core_kinetic_density();
    for (int i = 0; i < original.num_beta_radial_functions(); i++) {
        failures += copy.ps_paw_wf(i) != original.ps_paw_wf(i) || copy.ae_paw_wf(i) != original.ae_paw_wf(i);
    }
    data["pseudo_potential"]["paw_data"].erase("ae_core_kinetic_density");
    Simulation_context missing_ctx(R"({"parameters": {"electronic_structure_method": "pseudopotential"}})"_json);
    missing_ctx.electronic_structure_method("pseudopotential");
    auto& missing = missing_ctx.unit_cell().add_atom_type("He");
    missing.read_input(data);
    failures += missing.has_paw_core_kinetic_density();
    bool missing_rejected{false};
    try {
        missing.paw_ae_core_kinetic_density();
    } catch (std::exception const&) {
        missing_rejected = true;
    }
    failures += !missing_rejected;
    if (ctx->comm().rank() == 0) {
        ctx->unit_cell().atom_type(0).paw_ae_core_charge_density({});
    }
    bool rejected{false};
    try {
        Potential potential(*ctx);
    } catch (std::exception const&) {
        rejected = true;
    }
    return failures + !rejected;
}

int
test_meta_core_grid()
{
    int failures{0};
    for (double radius : {1.1, 3.7}) {
        auto conf                        = R"({"parameters": {
            "electronic_structure_method": "pseudopotential", "use_symmetry": false,
            "pw_cutoff": 8, "gk_cutoff": 2.6, "num_bands": 4,
            "xc_functionals": ["XC_MGGA_X_TPSS"]},
            "control": {"processing_unit": "CPU", "verbosity": 0}})"_json;
        conf["control"]["mpi_grid_dims"] = {mpi::Communicator::world().size(), 1};
        Simulation_context ctx(conf);
        ctx.unit_cell().set_lattice_vectors({{6, 0, 0}, {2, 5, 0}, {1, 1.5, 4}});
        auto& type = ctx.unit_cell().add_atom_type("He");
        type.zn(2);
        type.set_radial_grid(radial_grid_t::linear, 64, 0, radius, 1);
        std::vector<double> rho(64), tau(64), orbital(64);
        for (int ir = 0; ir < 64; ir++) {
            double r    = type.radial_grid(ir);
            rho[ir]     = 1 - r / radius;
            tau[ir]     = 2 * rho[ir];
            orbital[ir] = std::exp(-r);
        }
        type.ps_core_charge_density(rho);
        type.ps_core_kinetic_density(tau);
        type.ps_total_charge_density(std::vector<double>(64, 0));
        type.local_potential(std::vector<double>(64, 0));
        type.add_ps_atomic_wf(1, angular_momentum(0), orbital);
        type.d_mtrx_ion(matrix<double>({0, 0}));
        std::array<r3::vector<double>, 2> positions{{{0, 0, 0}, {0.93, 0.12, 0.99}}};
        for (auto position : positions) {
            ctx.unit_cell().add_atom("He", position);
        }
        ctx.initialize();
        Density density(ctx);
        auto const& fft = ctx.spfft<double>();
        // Linear radial functions make interpolation exact, isolating periodic image
        // summation, skew-cell bounds, slab ownership and updates of the fixed fields.
        for (auto shift : {r3::vector<double>{0, 0, 0}, {1, -1, 2}, {0.31, 0.23, 0.19}}) {
            for (int ia = 0; ia < 2; ia++) {
                ctx.unit_cell().atom(ia).set_position(positions[ia] + shift);
            }
            density.update();
            double error{0}, integral{0}, minimum{0};
            for (int z = 0; z < fft.local_z_length(); z++) {
                for (int y = 0; y < fft.dim_y(); y++) {
                    for (int x = 0; x < fft.dim_x(); x++) {
                        r3::vector<double> point(double(x) / fft.dim_x(), double(y) / fft.dim_y(),
                                                 double(z + fft.local_z_offset()) / fft.dim_z());
                        double expected{0};
                        for (auto position : positions) {
                            position += shift;
                            for (int j = 0; j < 3; j++) {
                                position[j] -= std::floor(position[j]);
                            }
                            for (int t0 = -2; t0 <= 2; t0++) {
                                for (int t1 = -2; t1 <= 2; t1++) {
                                    for (int t2 = -2; t2 <= 2; t2++) {
                                        auto dr  = point - position + r3::vector<double>(t0, t1, t2);
                                        double r = ctx.unit_cell().get_cartesian_coordinates(dr).length();
                                        if (r < radius) {
                                            expected += 1 - r / radius;
                                        }
                                    }
                                }
                            }
                        }
                        int ir   = ctx.fft_grid().index_by_coord(x, y, z);
                        double n = density.rho_pseudo_core().value(ir), t = density.tau_pseudo_core().value(ir);
                        error   = std::max({error, std::abs(n - expected), std::abs(t - 2 * expected)});
                        minimum = std::min({minimum, n, t});
                        failures += n < 0 || t < 0 || !std::isfinite(n) || !std::isfinite(t);
                        integral += n;
                    }
                }
            }
            ctx.comm().allreduce<double, mpi::op_t::max>(&error, 1);
            ctx.comm().allreduce<double, mpi::op_t::min>(&minimum, 1);
            mpi::Communicator(fft.communicator()).allreduce(&integral, 1);
            integral /= ctx.fft_grid().num_points();
            error = std::max(error, std::abs(integral - density.rho_pseudo_core().f_0().real()));
            error = std::max(error, std::abs(2 * integral - density.tau_pseudo_core().f_0().real()));
            std::cout << "direct pseudo-core grid radius=" << radius << " error=" << error << " minimum=" << minimum
                      << '\n';
            failures += !std::isfinite(error) || error > 2e-13;
        }
    }
    return failures;
}

int
test_meta_core_data()
{
    auto ctx  = meta_context(false, 0, false, false, true);
    auto data = ctx->unit_cell().atom_type(0).serialize();
    Simulation_context copy_ctx(R"({"parameters": {"electronic_structure_method": "pseudopotential"}})"_json);
    copy_ctx.electronic_structure_method("pseudopotential");
    auto& copy = copy_ctx.unit_cell().add_atom_type("He");
    copy.read_input(data);
    int failures = copy.ps_core_kinetic_density() != ctx->unit_cell().atom_type(0).ps_core_kinetic_density();
    copy.check_ps_core_kinetic_density();
    for (double bad : {-1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        auto invalid = copy.ps_core_kinetic_density();
        invalid[0]   = bad;
        bool rejected{false};
        try {
            copy.ps_core_kinetic_density(invalid);
        } catch (std::exception const&) {
            rejected = true;
        }
        failures += !rejected;
    }
    for (auto const& invalid : {std::vector<double>{}, std::vector<double>{0}}) {
        bool rejected{false};
        try {
            copy.ps_core_kinetic_density(invalid);
        } catch (std::exception const&) {
            rejected = true;
        }
        failures += !rejected;
    }
    data["pseudo_potential"].erase("core_kinetic_density");
    Simulation_context missing_ctx(R"({"parameters": {"electronic_structure_method": "pseudopotential"}})"_json);
    missing_ctx.electronic_structure_method("pseudopotential");
    auto& missing = missing_ctx.unit_cell().add_atom_type("He");
    missing.read_input(data);
    failures += missing.has_ps_core_kinetic_density();
    bool rejected{false};
    try {
        missing.check_ps_core_kinetic_density();
    } catch (std::exception const&) {
        rejected = true;
    }
    failures += !rejected;
    missing.ps_core_charge_density(std::vector<double>(missing.num_mt_points(), 0));
    missing.check_ps_core_kinetic_density();
    // A missing pseudo-core field is zero only when there is no pseudo-core charge.
    missing.ps_core_kinetic_density(std::vector<double>(missing.num_mt_points(), 0));
    failures += !missing.has_ps_core_kinetic_density();
    // A one-rank missing input must reject collectively, before the radial transforms.
    auto no_core = meta_context(false, 0, false);
    if (no_core->comm().rank() == 0) {
        no_core->unit_cell().atom_type(0).ps_core_charge_density(std::vector<double>(400, 0.1));
    }
    rejected = false;
    try {
        Density invalid(*no_core);
    } catch (std::exception const&) {
        rejected = true;
    }
    failures += !rejected;
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
    for (int variant = 0; variant < 4; variant++) {
        int ns = 1 + variant % 2;
        // Augmented PAW fields can have sigma > 8*n*tau even for positive n and tau.
        bool augmented = variant >= 2;
        XC_functional_base functional("XC_MGGA_X_TPSS", 0.75, ns);
        functional.get_meta(0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        std::vector<double> n(ns, 0.15), t(ns, augmented ? 0.01 : 0.12), sigma(ns == 1 ? 1 : 3, 0.02);
        if (ns == 2) {
            n[1]     = 0.10;
            t[1]     = augmented ? 0.01 : 0.07;
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
        std::cout << "Libxc primitive derivatives ns=" << ns << " augmented=" << augmented << ": " << largest << '\n';
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
    return failures;
}

int
main(int argc, char** argv)
{
    sirius::initialize(1);
    bool paw   = argc > 1 && std::string(argv[1]) == "--paw";
    int result = call_test("Libxc meta-GGA capabilities", test_meta_capabilities);
    if (!paw) {
        result += call_test("Libxc pseudo-core data", test_meta_core_data);
        result += call_test("positive periodic pseudo-core fields", test_meta_core_grid);
    } else {
        result += call_test("collective PAW meta-GGA core validation", test_paw_meta_input);
        for (int ns : {1, 2}) {
            result += call_test("spherical meta-GGA quadrature and adjoint", test_spherical_meta, ns);
            result += call_test("PAW meta-GGA projector matrices", test_paw_meta_matrix, ns);
        }
    }
    for (bool fft : {false, true}) {
        if (fft && mpi::Communicator::world().size() == 1) {
            continue;
        }
        for (int mag : {0, 1}) {
            for (bool core : {false, true}) {
                if (paw && !core) {
                    continue;
                }
                for (bool gamma : {false, true}) {
                    result += call_test("Libxc meta-GGA orbital derivative", test_meta_gga, gamma, mag, fft, core, paw);
                }
            }
        }
    }
    // The derivative tests cover the combinations above; SCF checks exercise
    // scalar and spin-polarized mixing, with and without pseudo-core fields.
    bool parallel = mpi::Communicator::world().size() > 1;
    if (!parallel) {
        result += call_test("Libxc meta-GGA SCF", test_meta_scf, 0, false, paw, paw);
    }
    result += call_test("Libxc meta-GGA SCF", test_meta_scf, 1, parallel, true, paw);
    sirius::finalize();
    return std::min(result, 1);
}
