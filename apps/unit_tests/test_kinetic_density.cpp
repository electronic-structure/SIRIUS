/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include "core/wf/kinetic_density_operator.hpp"

using namespace sirius;

template <typename T>
int
test_kinetic_density(bool gamma__)
{
    auto const& comm = mpi::Communicator::world();
    r3::matrix<double> reciprocal{{1.1, 0.2, 0.1}, {0.0, 0.9, 0.15}, {0.0, 0.0, 1.2}};
    r3::vector<double> k = gamma__ ? r3::vector<double>(0, 0, 0) : r3::vector<double>(0.23, -0.17, 0.11);
    fft::Gvec gv(k, reciprocal, 2.5, comm, gamma__);
    fft::Gvec_fft gvp(gv, comm, mpi::Communicator::self());
    int const n = 12;
    auto spl_z  = fft::split_z_dimension(n, comm);
    fft::spfft_grid_type<T> grid(n, n, n, gvp.zcol_count(), spl_z.local_size(), SPFFT_PU_HOST, 1, comm.native(),
                                 SPFFT_EXCH_DEFAULT);
    fft::spfft_transform_type<T> transform(grid.create_transform(
            SPFFT_PU_HOST, gamma__ ? SPFFT_TRANS_R2C : SPFFT_TRANS_C2C, n, n, n, spl_z.local_size(), gvp.count(),
            SPFFT_INDEX_TRIPLETS, gvp.gvec_array().at(memory_t::host)));
    wf::Kinetic_density_operator<T> op(transform, gvp);

    auto coefficient = [&](r3::vector<int> const& g, int shift) -> std::complex<T> {
        if (dot(g, g) > 3) {
            return 0;
        }
        if (dot(g, g) == 0) {
            return T(0.13 + 0.02 * shift);
        }
        return {T(0.09 * std::cos(g[0] + 2 * g[1] + 3 * g[2] + shift)),
                T(0.07 * std::sin(2 * g[0] - g[1] + g[2] + shift))};
    };
    std::vector<std::complex<T>> phi(gvp.count()), direction(gvp.count()), hphi(gvp.count()), hd(gvp.count());
    for (int ig = 0; ig < gvp.count(); ig++) {
        r3::vector<int> g(&gvp.gvec_array()(0, ig));
        phi[ig]       = coefficient(g, 0);
        direction[ig] = coefficient(g, 1);
    }

    double const volume     = std::pow(twopi, 3) / std::abs(reciprocal.det());
    double const occupation = 0.73;
    T weight                = occupation / volume;
    double const dv         = volume / (n * n * n);
    int nr                  = transform.local_slice_size();
    std::vector<T> tau(nr, 0), potential(nr);
    op.add_density(weight, phi.data(), tau.data());

    double error{0};
    for (int iz = 0; iz < transform.local_z_length(); iz++) {
        for (int iy = 0; iy < n; iy++) {
            for (int ix = 0; ix < n; ix++) {
                int ir = ix + n * (iy + n * iz);
                r3::vector<double> r(double(ix) / n, double(iy) / n, double(iz + transform.local_z_offset()) / n);
                std::array<std::complex<double>, 3> grad{};
                for (int ig = 0; ig < gv.num_gvec(); ig++) {
                    auto g     = gv.gvec(gvec_index_t::global(ig));
                    auto q     = dot(reciprocal, g + k);
                    auto c     = static_cast<std::complex<double>>(coefficient(g, 0));
                    auto phase = std::exp(std::complex<double>(0, twopi * dot(g, r)));
                    for (int x = 0; x < 3; x++) {
                        auto z = std::complex<double>(0, q[x]) * c * phase;
                        grad[x] += gamma__ ? std::complex<double>(2 * z.real(), 0) : z;
                    }
                }
                double reference{0};
                for (auto z : grad) {
                    reference += 0.5 * weight * std::norm(z);
                }
                error         = std::max(error, std::abs(tau[ir] - reference));
                potential[ir] = 0.8 + 0.3 * std::cos(twopi * r[0]) - 0.2 * std::sin(twopi * r[2]);
            }
        }
    }
    comm.allreduce<double, mpi::op_t::max>(&error, 1);
    double const tolerance = std::is_same_v<T, double> ? 2e-11 : 2e-6;
    int failures{0};
    auto check = [&](char const* name, double difference, double threshold) {
        if (comm.rank() == 0) {
            std::cout << (gamma__ ? "Gamma " : "k-point ") << name << ": " << difference << '\n';
        }
        failures += !std::isfinite(difference) || difference > threshold;
    };
    check("analytic tau", error, tolerance);

    auto inner_pw = [&](std::vector<std::complex<T>> const& a, std::vector<std::complex<T>> const& b) {
        std::complex<double> value{0};
        for (int ig = 0; ig < gvp.count(); ig++) {
            auto z = std::conj(static_cast<std::complex<double>>(a[ig])) * static_cast<std::complex<double>>(b[ig]);
            auto q = gvp.gkvec_cart(ig);
            value += gamma__ ? std::complex<double>((q.length() == 0 ? 1 : 2) * z.real(), 0) : z;
        }
        comm.allreduce(&value, 1);
        return value;
    };

    op.add_potential(potential.data(), phi.data(), hphi.data());
    op.add_potential(potential.data(), direction.data(), hd.data());
    check("Hermiticity", std::abs(inner_pw(direction, hphi) - std::conj(inner_pw(phi, hd))), tolerance);
    double energy{0};
    for (int ir = 0; ir < nr; ir++) {
        energy += dv * potential[ir] * tau[ir];
    }
    comm.allreduce(&energy, 1);
    check("density/operator normalization", std::abs(energy - occupation * inner_pw(phi, hphi).real()), tolerance);

    std::fill(hphi.begin(), hphi.end(), std::complex<T>(0));
    std::fill(potential.begin(), potential.end(), T(1.7));
    op.add_potential(potential.data(), phi.data(), hphi.data());
    error = 0;
    for (int ig = 0; ig < gvp.count(); ig++) {
        auto q = gvp.gkvec_cart(ig);
        error  = std::max(error, double(std::abs(hphi[ig] - T(0.5 * 1.7 * dot(q, q)) * phi[ig])));
    }
    comm.allreduce<double, mpi::op_t::max>(&error, 1);
    check("constant v_tau", error, tolerance);

    // E[tau] = integral tau^2/2 gives a nonuniform v_tau = tau.
    std::fill(hphi.begin(), hphi.end(), std::complex<T>(0));
    op.add_potential(tau.data(), phi.data(), hphi.data());
    auto trial_energy = [&](T step) {
        std::vector<std::complex<T>> c(phi.size());
        for (size_t i = 0; i < c.size(); i++) {
            c[i] = phi[i] + step * direction[i];
        }
        std::vector<T> t(nr, 0);
        op.add_density(weight, c.data(), t.data());
        double e{0};
        for (auto value : t) {
            e += 0.5 * dv * double(value) * value;
        }
        comm.allreduce(&e, 1);
        return e;
    };
    T step            = std::is_same_v<T, double> ? 1e-4 : 1e-2;
    double fd         = (trial_energy(step) - trial_energy(-step)) / (2 * step);
    double derivative = 2 * occupation * inner_pw(direction, hphi).real();
    check("nonlinear energy derivative", std::abs(fd - derivative), tolerance);

    std::fill(tau.begin(), tau.end(), T(0));
    op.add_density(weight, phi.data(), tau.data());
    op.add_density(2 * weight, phi.data(), tau.data());
    std::vector<T> tau_sum(nr, 0);
    op.add_density(3 * weight, phi.data(), tau_sum.data());
    error = 0;
    for (int ir = 0; ir < nr; ir++) {
        error = std::max(error, double(std::abs(tau[ir] - tau_sum[ir])));
    }
    comm.allreduce<double, mpi::op_t::max>(&error, 1);
    check("occupation accumulation", error, tolerance);
    return failures;
}

int
test_kinetic_density_collection(bool gamma__, int num_mag_dims__)
{
    auto conf                          = R"({"parameters": {
        "electronic_structure_method": "pseudopotential",
        "use_symmetry": false, "pw_cutoff": 6.0, "gk_cutoff": 2.0,
        "num_bands": 4, "xc_functionals": ["XC_LDA_X"]
    }, "control": {"mpi_grid_dims": [1, 1], "processing_unit": "CPU", "verbosity": 0}})"_json;
    conf["parameters"]["gamma_point"]  = gamma__;
    conf["parameters"]["num_mag_dims"] = num_mag_dims__;
    auto ctx = create_simulation_context(conf, {{8, 0, 0}, {0, 9, 0}, {0, 0, 10}}, 0, {}, false, false);
    K_point_set ks(*ctx);
    std::vector<r3::vector<double>> kpoints =
            gamma__ ? std::vector<r3::vector<double>>{{0, 0, 0}}
                    : std::vector<r3::vector<double>>{{0.23, -0.1, 0.05}, {-0.15, 0.2, 0.3}};
    std::vector<double> weights = gamma__ ? std::vector<double>{1.0} : std::vector<double>{0.3, 0.7};
    for (size_t ik = 0; ik < kpoints.size(); ik++) {
        ks.add_kpoint(kpoints[ik], weights[ik]);
    }
    ks.initialize();
    auto occupancy = [](int band, int spin, int ik) { return (0.6 + 0.2 * band + 0.1 * ik) / (spin + 1); };
    for (auto it : ks.spl_num_kpoints()) {
        auto kp   = ks.get<double>(it.i);
        auto& psi = kp->spinor_wave_functions();
        for (int ispn = 0; ispn < ctx->num_spins(); ispn++) {
            for (int ib = 0; ib < ctx->num_bands(); ib++) {
                kp->band_occupancy(ib, ispn, ib < 2 ? occupancy(ib, ispn, it.i) : 0);
                for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                    auto g        = kp->gkvec().gvec(gvec_index_t::global(kp->gkvec().offset() + ig));
                    bool selected = ib < 2 && g[(ib + 1) % 2] == 0 && g[2] == 0 &&
                                    (gamma__ ? std::abs(g[ib]) == 1 : g[ib] == 1);
                    psi.pw_coeffs(ig, wf::spin_index(ispn), wf::band_index(ib)) =
                            selected ? (gamma__ ? std::sqrt(0.5) : 1.0) : 0.0;
                }
            }
        }
    }
    Density density(*ctx);
    auto tau        = density.generate_smooth_kinetic_density<double>(ks);
    auto const& fft = ctx->spfft<double>();
    double error{0};
    for (int ispn = 0; ispn < ctx->num_spins(); ispn++) {
        for (int iz = 0; iz < fft.local_z_length(); iz++) {
            for (int iy = 0; iy < fft.dim_y(); iy++) {
                for (int ix = 0; ix < fft.dim_x(); ix++) {
                    int ir = ix + fft.dim_x() * (iy + fft.dim_y() * iz);
                    r3::vector<double> r(double(ix) / fft.dim_x(), double(iy) / fft.dim_y(),
                                         double(iz + fft.local_z_offset()) / fft.dim_z());
                    double reference{0};
                    for (size_t ik = 0; ik < kpoints.size(); ik++) {
                        for (int ib = 0; ib < 2; ib++) {
                            r3::vector<double> g(0, 0, 0);
                            g[ib]            = 1;
                            auto q           = dot(ctx->unit_cell().reciprocal_lattice_vectors(), g + kpoints[ik]);
                            double amplitude = gamma__ ? 2 * std::pow(std::sin(twopi * r[ib]), 2) : 1;
                            reference += 0.5 * weights[ik] * occupancy(ib, ispn, ik) * dot(q, q) * amplitude /
                                         ctx->unit_cell().omega();
                        }
                    }
                    if (!std::isfinite(tau[ispn].value(ir))) {
                        error = 1;
                    }
                    error = std::max(error, std::abs(tau[ispn].value(ir) - reference));
                }
            }
        }
    }
    ctx->comm().allreduce<double, mpi::op_t::max>(&error, 1);
    if (ctx->comm().rank() == 0) {
        std::cout << "smooth density collection, gamma=" << gamma__ << ", magnetic dimensions=" << num_mag_dims__
                  << ": " << error << '\n';
    }
    return error > 1e-12;
}

int
test_density_copy()
{
    auto conf  = R"({"parameters": {
        "electronic_structure_method": "pseudopotential", "use_symmetry": false,
        "pw_cutoff": 6.0, "gk_cutoff": 2.0, "num_bands": 4,
        "num_mag_dims": 1, "xc_functionals": ["XC_LDA_X"]
    }, "control": {"mpi_grid_dims": [1, 1], "processing_unit": "CPU", "verbosity": 0}})"_json;
    auto ctx   = create_simulation_context(conf, {{8, 0, 0}, {0, 9, 0}, {0, 0, 10}}, 0, {}, false, false);
    auto other = create_simulation_context(conf, {{8, 0, 0}, {0, 9, 0}, {0, 0, 10}}, 0, {}, false, false);
    int failures{0};
    for (bool tau : {false, true}) {
        Density source(*ctx, tau), same(*ctx, tau), different(*other, tau), mismatch(*ctx, !tau);
        for (int j = 0; j < 2; j++) {
            source.component(j).rg().values()     = [j]() { return double(j + 1); };
            source.component(j).rg().f_pw_local() = [j]() { return std::complex<double>(j + 2, j + 3); };
            if (tau) {
                source.kinetic_density().component(j).rg().values()     = [j]() { return double(j + 4); };
                source.kinetic_density().component(j).rg().f_pw_local() = [j]() {
                    return std::complex<double>(j + 5, j + 6);
                };
            }
        }
        for (auto* dest : {&same, &different, &mismatch}) {
            bool rejected{false};
            try {
                copy(source, *dest);
            } catch (std::exception const&) {
                rejected = true;
            }
            bool expected = dest == &mismatch || (tau && dest == &different);
            failures += rejected != expected;
            if (!rejected && !expected) {
                for (int j = 0; j < 2; j++) {
                    for (size_t i = 0; i < source.component(j).rg().values().size(); i++) {
                        failures += dest->component(j).rg().value(i) != source.component(j).rg().value(i);
                        if (tau) {
                            failures += dest->kinetic_density().component(j).rg().value(i) !=
                                        source.kinetic_density().component(j).rg().value(i);
                        }
                    }
                    for (int ig = 0; ig < ctx->gvec().count(); ig++) {
                        failures += dest->component(j).rg().f_pw_local(ig) != source.component(j).rg().f_pw_local(ig);
                        if (tau) {
                            failures += dest->kinetic_density().component(j).rg().f_pw_local(ig) !=
                                        source.kinetic_density().component(j).rg().f_pw_local(ig);
                        }
                    }
                }
            }
        }
    }
    return failures;
}

int
main(int argc, char** argv)
{
    sirius::initialize(true);
    int result = call_test("density copy with optional kinetic fields", test_density_copy);
    result += call_test("kinetic density, complex k-point", test_kinetic_density<double>, false);
    result += call_test("kinetic density, real Gamma", test_kinetic_density<double>, true);
#ifdef SIRIUS_USE_FP32
    result += call_test("kinetic density FP32, complex k-point", test_kinetic_density<float>, false);
    result += call_test("kinetic density FP32, real Gamma", test_kinetic_density<float>, true);
#endif
    for (auto gamma : {false, true}) {
        for (int num_mag_dims : {0, 1}) {
            result += call_test("smooth kinetic density collection", test_kinetic_density_collection, gamma,
                                num_mag_dims);
        }
    }
    sirius::finalize();
    return std::min(result, 1);
}
