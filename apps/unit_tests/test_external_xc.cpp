/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include "dft/energy.hpp"
#include "hamiltonian/hamiltonian.hpp"

using namespace sirius;

// A nonlocal tau moment couples all point rows, including a descriptor-only row.
void
functional(Simulation_context const& ctx, xc_callback_input_t const& in, xc_callback_output_t& out)
{
    int nr = in.num_points;
    mpi::Communicator comm(in.points ? ctx.comm().native() : ctx.spfft<double>().communicator());
    double moment{0}, energy{0};
    for (int s = 0; s < in.num_spins; s++) {
        for (int p = 0; p < nr; p++) {
            moment += (in.points ? in.points->descriptor_weights[p] : in.point_weight) * in.tau[p + nr * s];
        }
    }
    comm.allreduce(&moment, 1);
    for (int s = 0; s < in.num_spins; s++) {
        for (int p = 0; p < nr; p++) {
            int j    = p + nr * s;
            double n = in.rho[j], t = in.tau[j];
            double w     = in.points ? in.points->energy_weights[p] : in.point_weight;
            double d     = in.points ? in.points->descriptor_weights[p] : in.point_weight;
            double scale = in.points ? 1 : 1 / in.point_weight;
            energy += w * (0.7 * n * n + 0.3 * t * t + 0.2 * n * t);
            out.vrho[j] = scale * w * (1.4 * n + 0.2 * t);
            out.vtau[j] = scale * (w * (0.6 * t + 0.2 * n) + d * 0.4 * moment / ctx.unit_cell().omega());
            for (int x = 0; x < 3; x++) {
                int k = p + nr * (x + 3 * s);
                energy += w * 0.1 * in.gradient[k] * in.gradient[k];
                out.vgradient[k] = scale * w * 0.2 * in.gradient[k];
            }
        }
    }
    comm.allreduce(&energy, 1);
    out.energy = energy + 0.2 * moment * moment / ctx.unit_cell().omega();
}

int
test_external_xc(bool gamma, int mag, bool points)
{
    auto conf                          = R"({"parameters": {
        "electronic_structure_method":"pseudopotential", "use_symmetry":false,
        "pw_cutoff":8.0, "gk_cutoff":2.6, "num_bands":4, "xc_functionals":[]
    }, "control":{"mpi_grid_dims":[1,1], "processing_unit":"CPU", "verbosity":0}})"_json;
    conf["parameters"]["gamma_point"]  = gamma;
    conf["parameters"]["num_mag_dims"] = mag;
    double charge                      = points ? 0 : 0.25;
    conf["parameters"]["extra_charge"] = charge;
    // Cover both FFT replication and a distributed FFT with the same test executable.
    if (!points) {
        conf["control"]["mpi_grid_dims"] = {mpi::Communicator::world().size(), 1};
    }
    Simulation_context ctx(conf);
    ctx.unit_cell().set_lattice_vectors({{8, 0, 0}, {0, 9, 0}, {0, 0, 10}});
    auto& type = ctx.unit_cell().add_atom_type("He");
    type.zn(2);
    type.set_radial_grid(radial_grid_t::lin_exp, 400, 0.0, 20.0, 6);
    std::vector<double> rho(400), orbital(400);
    for (int ir = 0; ir < 400; ir++) {
        double r    = type.radial_grid(ir);
        rho[ir]     = 4 * std::exp(-r * r) * r;
        orbital[ir] = std::exp(-r);
    }
    type.local_potential(std::vector<double>(400, 0));
    type.ps_core_charge_density(std::vector<double>(400, 0));
    type.ps_total_charge_density(rho);
    type.add_ps_atomic_wf(1, angular_momentum(0), orbital);
    type.d_mtrx_ion(matrix<double>({0, 0}));
    ctx.unit_cell().add_atom("He", {0.2, 0.3, 0.4});
    ctx.cb().external_xc_ = functional;
    if (points) {
        auto q = std::make_shared<xc_point_set_t>(ctx);
        // A single owner deliberately leaves every other context rank empty.
        if (ctx.comm().rank() == 0) {
            for (int z = 0; z < 4; z++) {
                for (int y = 0; y < 4; y++) {
                    for (int x = 0; x < 4; x++) {
                        q->coordinates.push_back(
                                {8 * (0.037 + x / 4.0), 9 * (0.029 + y / 4.0), 10 * (0.041 + z / 4.0)});
                        q->energy_weights.push_back(720.0 / 64);
                        q->descriptor_weights.push_back(1.1 * 720.0 / 64);
                    }
                }
            }
            q->energy_weights[0] = 0;
            q->block_sizes       = {64};
            q->block_centres     = {{1.6, 2.7, 4.0}};
        }
        ctx.cb().external_xc_points_ = q;
    }
    ctx.initialize();
    K_point_set ks(ctx);
    ks.add_kpoint(gamma ? r3::vector<double>{0, 0, 0} : r3::vector<double>{0.13, -0.17, 0.07}, 1);
    ks.initialize();
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
    Density density(ctx);
    Potential potential(ctx);
    auto evaluate = [&](double step) {
        for (auto it : ks.spl_num_kpoints()) {
            auto kp   = ks.get<double>(it.i);
            auto& psi = kp->spinor_wave_functions();
            psi.zero(memory_t::host);
            for (int s = 0; s < ctx.num_spins(); s++) {
                kp->band_occupancy(0, s, (mag ? (s ? 0.8 : 1.2) : 2.0) * (1 - charge / 2));
                for (int b = 1; b < ctx.num_bands(); b++) {
                    kp->band_occupancy(b, s, 0);
                }
                for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                    auto g = kp->gkvec().gvec(gvec_index_t::local(ig));
                    psi.pw_coeffs(ig, wf::spin_index(s), wf::band_index(0)) =
                            coefficient(g, 0.37 + 0.1 * s + step, false);
                }
            }
        }
        ks.sync_band<double, sync_band_t::occupancy>();
        density.generate<double>(ks, false, true, true);
        potential.generate(density, false, true);
        return density.kinetic_density().scalar().integrate().total + 0.5 * potential.energy_vha() +
               energy_vloc(density, potential) + potential.energy_exc(density) + potential.ewald_energy();
    };
    double center = evaluate(0), derivative{0};
    Hamiltonian0<double> h0(potential);
    for (auto it : ks.spl_num_kpoints()) {
        auto kp = ks.get<double>(it.i);
        auto hk = h0(*kp);
        wf::Wave_functions<double> hphi(kp->gkvec_sptr(), wf::num_mag_dims(mag), wf::num_bands(1), memory_t::host);
        for (int s = 0; s < ctx.num_spins(); s++) {
            if (gamma) {
                hk.apply_h_s<double>(wf::spin_range(s), wf::band_range(0, 1), kp->spinor_wave_functions(), &hphi,
                                     nullptr);
            } else {
                hk.apply_h_s<std::complex<double>>(wf::spin_range(s), wf::band_range(0, 1), kp->spinor_wave_functions(),
                                                   &hphi, nullptr);
            }
            double expectation{0};
            for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                auto g        = kp->gkvec().gvec(gvec_index_t::local(ig));
                double factor = gamma && dot(g, g) ? 2 : 1;
                auto h        = hphi.pw_coeffs(ig, wf::spin_index(s), wf::band_index(0));
                auto c        = coefficient(g, 0.37 + 0.1 * s, false);
                auto d        = coefficient(g, 0.37 + 0.1 * s, true);
                derivative += 2 * kp->weight() * kp->band_occupancy(0, s) * factor * std::real(std::conj(d) * h);
                expectation += factor * std::real(std::conj(c) * h);
            }
            kp->comm().allreduce(&expectation, 1);
            kp->band_energy(0, s, expectation);
        }
    }
    ctx.comm().allreduce(&derivative, 1);
    ks.sync_band<double, sync_band_t::energy>();
    int failures{0};
    auto check = [&](char const* name, double error, double tolerance) {
        if (ctx.comm().rank() == 0) {
            std::cout << name << ": " << error << '\n';
        }
        failures += !std::isfinite(error) || error > tolerance;
    };
    check("XC total-energy accounting", std::abs(ks_energy(ctx, ks, density, potential) - center), 2e-10);
    double h = 5e-5;
    check("nonlocal XC orbital derivative", std::abs((evaluate(h) - evaluate(-h)) / (2 * h) - derivative), 6e-9);
    ctx.cb().external_xc_ = [](auto const&, auto const&, auto&) { throw std::runtime_error("test callback failure"); };
    bool rejected{false};
    try {
        potential.generate(density, false, true);
    } catch (std::exception const&) {
        rejected = true;
    }
    check("collective callback failure", rejected ? 0 : 1, 0);
    return failures;
}

int
main(int argc, char** argv)
{
    sirius::initialize(1);
    int failures = call_test("external XC, FFT collinear", test_external_xc, false, 1, false);
    failures += call_test("external XC, point Gamma", test_external_xc, true, 0, true);
    failures += call_test("external XC, point collinear", test_external_xc, false, 1, true);
    sirius::finalize();
    return std::min(failures, 1);
}
