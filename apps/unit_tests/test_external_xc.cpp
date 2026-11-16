/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include "dft/energy.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "geometry/force.hpp"
#include "geometry/stress.hpp"

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
    // The coupling constant must not hide additional cell dependence from the geometry callback.
    constexpr double moment_scale = 1.0 / 720;
    for (int s = 0; s < in.num_spins; s++) {
        for (int p = 0; p < nr; p++) {
            int j    = p + nr * s;
            double n = in.rho[j], t = in.tau[j];
            double w     = in.points ? in.points->energy_weights[p] : in.point_weight;
            double d     = in.points ? in.points->descriptor_weights[p] : in.point_weight;
            double scale = in.points ? 1 : 1 / in.point_weight;
            double local = 0.7 * n * n + 0.3 * t * t + 0.2 * n * t;
            double position{0};
            if (in.points) {
                for (int x = 0; x < 3; x++) {
                    double r = in.points->coordinates[p][x] - in.points->block_centres[0][x];
                    position += 0.002 * r * r;
                    if (out.geometry) {
                        out.geometry->coordinates[p][x] += 0.004 * w * n * r;
                        out.geometry->block_centres[0][x] -= 0.004 * w * n * r;
                    }
                }
            }
            local += position * n;
            energy += w * local;
            out.vrho[j] = scale * w * (1.4 * n + 0.2 * t + position);
            out.vtau[j] = scale * (w * (0.6 * t + 0.2 * n) + d * 0.4 * moment * moment_scale);
            for (int x = 0; x < 3; x++) {
                int k = p + nr * (x + 3 * s);
                energy += w * 0.1 * in.gradient[k] * in.gradient[k];
                local += 0.1 * in.gradient[k] * in.gradient[k];
                out.vgradient[k] = scale * w * 0.2 * in.gradient[k];
            }
            if (out.geometry) {
                out.geometry->energy_weights[p] += local;
                out.geometry->descriptor_weights[p] += 0.4 * moment * moment_scale * t;
            }
        }
    }
    comm.allreduce(&energy, 1);
    out.energy = energy + 0.2 * moment * moment * moment_scale;
}

struct Geometry
{
    r3::vector<double> displacement;
    int mu{-1}, nu{-1};
    double strain{0}, energy{0};
    std::array<std::array<double, 5>, 2> fields{};
};

int
test_external_xc(bool gamma, int mag, bool points, bool nlcc, Geometry* deformation)
{
    auto conf                          = R"({"parameters": {
        "electronic_structure_method":"pseudopotential", "use_symmetry":false,
        "pw_cutoff":8.0, "gk_cutoff":2.6, "num_bands":4, "xc_functionals":[]
    }, "control":{"mpi_grid_dims":[1,1], "processing_unit":"CPU", "verbosity":0}})"_json;
    conf["parameters"]["gamma_point"]  = gamma;
    conf["parameters"]["num_mag_dims"] = mag;
    double charge                      = gamma ? 0 : 0.25;
    conf["parameters"]["extra_charge"] = charge;
    // Cover both FFT replication and a distributed FFT with the same test executable.
    if (!points || !gamma) {
        conf["control"]["mpi_grid_dims"] = {mpi::Communicator::world().size(), 1};
    }
    Simulation_context ctx(conf);
    r3::matrix<double> lattice{{8, 0, 0}, {0, 9, 0}, {0, 0, 10}};
    if (deformation && deformation->mu >= 0) {
        auto base = lattice;
        for (int j = 0; j < 3; j++) {
            lattice(deformation->mu, j) += deformation->strain * base(deformation->nu, j);
        }
    }
    ctx.unit_cell().set_lattice_vectors(lattice);
    auto atom = dot(lattice, r3::vector<double>{0.2, 0.3, 0.4});
    if (deformation) {
        atom += deformation->displacement;
    }
    auto& type = ctx.unit_cell().add_atom_type("He");
    type.zn(2);
    type.set_radial_grid(radial_grid_t::lin_exp, 400, 0.0, 20.0, 6);
    // A smooth Coulomb-tail PP also exercises the charged-cell G=0 stress contribution.
    std::vector<double> rho(400), orbital(400), vlocal(400), core(400), core_tau(400);
    for (int ir = 0; ir < 400; ir++) {
        double r    = type.radial_grid(ir);
        rho[ir]     = 4 * std::exp(-r * r) * r;
        orbital[ir] = std::exp(-r);
        vlocal[ir]  = r > 0 ? -2 * std::erf(r / 0.8) / r : -4 / (std::sqrt(pi) * 0.8);
        if (nlcc) {
            core[ir]     = 0.025 * std::exp(-2 * r * r);
            core_tau[ir] = 0.015 * std::exp(-3 * r * r);
        }
    }
    type.local_potential(vlocal);
    type.ps_core_charge_density(core);
    if (nlcc) {
        type.ps_core_kinetic_density(core_tau);
    }
    type.ps_total_charge_density(rho);
    type.add_ps_atomic_wf(1, angular_momentum(0), orbital);
    type.d_mtrx_ion(matrix<double>({0, 0}));
    ctx.unit_cell().add_atom("He", dot(r3::inverse(lattice), atom));
    std::array<std::array<double, 5>, 2> probe{};
    ctx.cb().external_xc_ = [&](auto const& c, auto const& in, auto& out) {
        if (in.points && in.num_points) {
            for (int s = 0; s < in.num_spins; s++) {
                auto n   = in.num_points;
                probe[s] = {in.rho[n * s], in.gradient[n * (3 * s)], in.gradient[n * (3 * s + 1)],
                            in.gradient[n * (3 * s + 2)], in.tau[n * s]};
            }
        }
        functional(c, in, out);
    };
    std::vector<r3::vector<double>> fractions;
    if (points) {
        auto q = std::make_shared<xc_point_set_t>(ctx);
        // A single owner deliberately leaves every other context rank empty.
        if (ctx.comm().rank() == 0) {
            for (int z = 0; z < 4; z++) {
                for (int y = 0; y < 4; y++) {
                    for (int x = 0; x < 4; x++) {
                        r3::vector<double> f{0.037 + x / 4.0, 0.029 + y / 4.0, 0.041 + z / 4.0};
                        if (x + y + z == 0) {
                            f = {1.237, 0.329, 0.441}; // Descriptor-only point near a periodic image of the core.
                        }
                        fractions.push_back(f);
                        auto r = dot(lattice, f) + atom - r3::vector<double>{1.6, 2.7, 4.0};
                        q->coordinates.push_back({r[0], r[1], r[2]});
                        q->energy_weights.push_back(ctx.unit_cell().omega() / 64);
                        q->descriptor_weights.push_back(1.1 * ctx.unit_cell().omega() / 64);
                    }
                }
            }
            q->energy_weights[0] = 0;
            q->block_sizes       = {64};
            q->block_centres     = {{atom[0], atom[1], atom[2]}};
        }
        ctx.cb().external_xc_points_   = q;
        ctx.cb().external_xc_geometry_ = functional;
        ctx.cb().external_xc_pullback_ = [&fractions](auto const& c, auto const& q, auto const& d, auto& out) {
            auto inverse = r3::inverse(c.unit_cell().lattice_vectors());
            for (size_t p = 0; p < q.coordinates.size(); p++) {
                double volume =
                        q.energy_weights[p] * d.energy_weights[p] + q.descriptor_weights[p] * d.descriptor_weights[p];
                for (int x = 0; x < 3; x++) {
                    out.atoms[0][x] += d.coordinates[p][x];
                    for (int y = 0; y < 3; y++) {
                        out.lattice[x + 3 * y] += d.coordinates[p][x] * fractions[p][y] + volume * inverse(y, x);
                    }
                }
            }
            for (auto const& centre : d.block_centres) {
                for (int x = 0; x < 3; x++) {
                    out.atoms[0][x] += centre[x];
                }
            }
            c.comm().allreduce(out.atoms[0].data(), 3);
            c.comm().allreduce(out.lattice.data(), 9);
        };
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
    if (deformation) {
        deformation->energy = center;
        deformation->fields = probe;
        return 0;
    }
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
    if (nlcc) {
        Geometry valence;
        test_external_xc(gamma, mag, points, false, &valence);
        double error{0};
        if (ctx.comm().rank() == 0) {
            r3::vector<double> d{0.037 * 8, 0.029 * 9, 0.041 * 10};
            double r2 = r3::dot(d, d);
            double n = 0.025 * std::exp(-2 * r2), t = 0.015 * std::exp(-3 * r2);
            std::array<double, 5> expected{n, -4 * n * d[0], -4 * n * d[1], -4 * n * d[2], t};
            for (int s = 0; s < ctx.num_spins(); s++) {
                for (size_t j = 0; j < expected.size(); j++) {
                    error = std::max(error,
                                     std::abs(probe[s][j] - valence.fields[s][j] - expected[j] / ctx.num_spins()));
                }
            }
        }
        ctx.comm().allreduce<double, mpi::op_t::max>(&error, 1);
        check("frozen core density, gradient and tau at an image point", error, 2e-7);
    }
    check("XC total-energy accounting", std::abs(ks_energy(ctx, ks, density, potential) - center), 2e-10);
    check("orbital kinetic-energy contraction",
          std::abs(energy_kin(ctx, ks, density, potential) - density.kinetic_density().scalar().integrate().total),
          2e-10);
    double h = 5e-5;
    check("nonlocal XC orbital derivative", std::abs((evaluate(h) - evaluate(-h)) / (2 * h) - derivative), 6e-9);
    if (points) {
        evaluate(0);
        Force force(ctx, density, potential, ks);
        auto const& forces = force.calc_forces_total(false);
        Stress stress(ctx, density, potential, ks);
        auto sigma  = stress.calc_stress_total();
        double step = 1e-5;
        r3::vector<double> direction{0.3, -0.2, 0.4};
        Geometry plus, minus;
        plus.displacement  = step * direction;
        minus.displacement = -step * direction;
        test_external_xc(gamma, mag, points, nlcc, &plus);
        test_external_xc(gamma, mag, points, nlcc, &minus);
        double projection{0};
        for (int x = 0; x < 3; x++) {
            projection += forces(x, 0) * direction[x];
        }
        check("external XC atomic force", std::abs(projection + (plus.energy - minus.energy) / (2 * step)), 2e-8);
        for (int nu : {0, 1}) {
            plus    = Geometry{};
            minus   = Geometry{};
            plus.mu = minus.mu = 0;
            plus.nu = minus.nu = nu;
            plus.strain        = step;
            minus.strain       = -step;
            test_external_xc(gamma, mag, points, nlcc, &plus);
            test_external_xc(gamma, mag, points, nlcc, &minus);
            check("external XC normal/shear stress",
                  std::abs(ctx.unit_cell().omega() * sigma(0, nu) - (plus.energy - minus.energy) / (2 * step)), 2e-8);
        }
        auto pullback                  = ctx.cb().external_xc_pullback_;
        ctx.cb().external_xc_pullback_ = [](auto const& c, auto const&, auto const&, auto&) {
            if (c.comm().rank() == 0) {
                throw std::runtime_error("test quadrature failure");
            }
        };
        bool rejected{false};
        try {
            potential.external_xc_geometry(density);
        } catch (std::exception const&) {
            rejected = true;
        }
        check("collective quadrature failure", rejected ? 0 : 1, 0);
        ctx.cb().external_xc_pullback_ = pullback;
    }
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
    int failures = call_test("external XC, FFT collinear", test_external_xc, false, 1, false, false, nullptr);
    failures += call_test("external XC, point Gamma", test_external_xc, true, 0, true, false, nullptr);
    failures += call_test("external XC, point collinear with NLCC", test_external_xc, false, 1, true, true, nullptr);
    sirius::finalize();
    return std::min(failures, 1);
}
