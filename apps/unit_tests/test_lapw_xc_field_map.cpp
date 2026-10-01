/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>
#include <testing.hpp>
#include <chrono>
#include <filesystem>
#include "density/lapw_xc_atom.hpp"
#include "density/lapw_xc_field_map.hpp"
#include "hamiltonian/local_operator.hpp"

using namespace sirius;

namespace {

std::complex<double>
orbital(int i, int ia, int s, bool direction)
{
    double x = 0.37 * (i + 1) + 0.23 * ia + 0.19 * s;
    return direction ? std::complex<double>(0.13 * std::sin(2 * x), -0.09 * std::cos(3 * x))
                     : std::complex<double>(0.27 * std::cos(x), 0.21 * std::sin(1.7 * x));
}

double
occupation(int s)
{
    return 0.8 / (s + 1);
}

std::complex<double>
fourier_coefficient(r3::vector<int> g, int s, bool tau, bool direction)
{
    if (r3::dot(g, g) == 0) {
        return (direction ? 0.1 : 0.7) / (s + 1) * (tau ? 0.6 : 1.0);
    }
    if (g[2] != 0 || std::abs(g[0]) + std::abs(g[1]) != 1) {
        return 0;
    }
    return std::complex<double>(direction ? -0.012 : 0.06, (direction ? 0.009 : -0.04) * (g[0] + g[1])) *
           (tau ? 0.4 : 1.0) / double(s + 1);
}

int
test_lapw_hamiltonian(Simulation_context& ctx, lapw_xc_field_adjoint_t full_data,
                      lapw_xc_field_adjoint_t interstitial_data)
{
    auto const& comm = ctx.comm();
    int failures{0};
    auto check = [&](char const* name, double error, double tolerance) {
        comm.allreduce<double, mpi::op_t::max>(&error, 1);
        if (comm.rank() == 0) {
            std::cout << "LAPW Hamiltonian " << name << ": " << error << '\n';
        }
        failures += !std::isfinite(error) || error > tolerance;
    };
    for (auto& atom : interstitial_data.local) {
        for (auto& part : atom) {
            for (auto& field : part) {
                field.zero();
            }
        }
    }
    for (auto& atom : interstitial_data.core) {
        for (auto& field : atom) {
            std::fill(field.begin(), field.end(), 0);
        }
    }
    Potential base(ctx), full(ctx), interstitial(ctx);
    for (auto* p : {&base, &full, &interstitial}) {
        p->zero();
        p->fft_transform(-1);
    }
    full.set_lapw_xc_derivatives(std::move(full_data));
    interstitial.set_lapw_xc_derivatives(std::move(interstitial_data));
    auto reject = [&](char const* name, auto&& operation) {
        bool rejected{false};
        try {
            operation();
        } catch (std::exception const&) {
            rejected = true;
        }
        check(name, rejected ? 0 : 1, 0);
    };
    Density density(ctx);
    reject("no incomplete XC total energy", [&] { full.energy_exc(density); });
    reject("no incomplete rho contraction", [&] { full.energy_vxc(density); });
    reject("no incomplete tau contraction", [&] { full.energy_vtau(density); });
    reject("no incomplete magnetic contraction", [&] { energy_bxc(density, full); });
    reject("no incomplete effective-potential contraction", [&] { energy_veff(density, full); });
    reject("no partial operator copy from staged source", [&] { copy(full, base); });
    reject("no stale staged operator after copy", [&] { copy(base, full); });
    auto const& d = *full.lapw_xc_derivatives();
    std::vector<std::vector<std::complex<double>>> vrho, vtau;
    for (int s = 0; s < ctx.num_spins(); s++) {
        vrho.push_back(d.rho[s].gather_f_pw());
        vtau.push_back(d.tau[s].gather_f_pw());
    }
    bool second    = ctx.cfg().control().use_second_variation();
    auto component = [&](auto const& fields, int g, int j, bool canonical) {
        if (canonical || ctx.num_spins() == 1) {
            return fields[j][g];
        }
        return 0.5 * (fields[0][g] + double(1 - 2 * j) * fields[1][g]);
    };
    Hamiltonian0<double> h0(base, true, false);
    Hamiltonian0<double> h1(full, true, false);
    Hamiltonian0<double> hi(interstitial, true, false);
    double error{0};
    for (int ig = 0; ig < ctx.gvec().num_gvec(); ig++) {
        for (int s = 0; s < (second ? 1 : ctx.num_spins()); s++) {
            error = std::max(error, std::abs(h1.veff_pw(ig, s) - h0.veff_pw(ig, s) - component(vrho, ig, s, !second)));
            error = std::max(error, std::abs(h1.vtau_pw(ig, s) - component(vtau, ig, s, !second)));
        }
    }
    check("unmasked exact Fourier covectors", error, 2e-13);
    auto mt_error = [&](Hamiltonian0<double> const& a, Hamiltonian0<double> const& b) {
        double error{0};
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            LAPW_xc_atom op(ctx.unit_cell().atom(ia), ctx.lmax_rho());
            std::vector<mdarray<std::complex<double>, 2>> matrices;
            for (int s = 0; s < ctx.num_spins(); s++) {
                matrices.push_back(op.matrix_elements(d.local[ia][0][s], d.local[ia][1][s]));
            }
            int n = ctx.unit_cell().atom(ia).type().mt_basis_size();
            for (int s = 0; s < ctx.num_spins(); s++) {
                for (int j = 0; j < n; j++) {
                    for (int i = 0; i < n; i++) {
                        auto expected = matrices[s](i, j);
                        if (second && ctx.num_spins() == 2) {
                            expected = 0.5 * (matrices[0](i, j) + double(1 - 2 * s) * matrices[1](i, j));
                        }
                        error = std::max(error, std::abs(b.hmt(ia)(i, j, s) - a.hmt(ia)(i, j, s) - expected));
                    }
                }
            }
        }
        return error;
    };
    check("AW/LO local matrices and spin conversion", mt_error(h0, h1), 2e-13);

    K_point_set ks(ctx);
    ks.add_kpoint({0.13, -0.17, 0.07}, 1);
    ks.initialize();
    double local_error{0}, matrix_error{0}, full_matrix_error{0}, full_apply_error{0}, boundary_error{0};
    for (auto it : ks.spl_num_kpoints()) {
        auto kp = ks.get<double>(it.i);
        auto k0 = h0(*kp);
        auto k1 = hi(*kp);
        auto k2 = h1(*kp);
        int n = kp->gklo_basis_size(), bs = ctx.cyclic_block_size();
        la::dmatrix<std::complex<double>> m0(n, n, ctx.blacs_grid(), bs, bs), m1(n, n, ctx.blacs_grid(), bs, bs);
        la::dmatrix<std::complex<double>> o0(n, n, ctx.blacs_grid(), bs, bs), o1(n, n, ctx.blacs_grid(), bs, bs);
        la::dmatrix<std::complex<double>> m2(n, n, ctx.blacs_grid(), bs, bs), o2(n, n, ctx.blacs_grid(), bs, bs);
        std::vector<mdarray<std::complex<double>, 2>> mt_basis;
        std::vector<int> num_lo, lo_offset;
        int next_lo = kp->gkvec().num_gvec();
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            auto const& atom = ctx.unit_cell().atom(ia);
            int naw = atom.type().mt_aw_basis_size(), nmt = atom.type().mt_basis_size();
            mdarray<std::complex<double>, 2> alm({kp->gkvec().count(), naw});
            kp->alm_coeffs_loc().generate<false>(atom, alm);
            auto centre   = r3::dot(ctx.unit_cell().lattice_vectors(), atom.position());
            double radius = atom.type().mt_radius();
            for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                auto q      = kp->gkvec().gkvec_cart(gvec_index_t::local(ig));
                auto angles = r3::spherical_coordinates(q);
                std::vector<std::complex<double>> ylm(sf::lmmax(atom.type().lmax_apw()));
                sf::spherical_harmonics(atom.type().lmax_apw(), angles[1], angles[2], ylm.data());
                for (int l = 0; l <= atom.type().lmax_apw(); l++) {
                    double bessel     = gsl_sf_bessel_jl(l, angles[0] * radius);
                    double derivative = l / radius * bessel - angles[0] * gsl_sf_bessel_jl(l + 1, angles[0] * radius);
                    for (int m = -l; m <= l; m++) {
                        auto factor = fourpi / std::sqrt(ctx.unit_cell().omega()) *
                                      std::polar(1.0, r3::dot(q, centre)) * std::pow(std::complex<double>(0, 1), l) *
                                      std::conj(ylm[sf::lm(l, m)]);
                        for (int deriv = 0; deriv <= 1; deriv++) {
                            std::complex<double> value{0};
                            for (int order = 0; order < static_cast<int>(atom.type().aw_descriptor(l).size());
                                 order++) {
                                value += atom.symmetry_class().aw_surface_deriv(l, order, deriv) *
                                         alm(ig, atom.type().indexb_by_l_m_order(l, m, order));
                            }
                            boundary_error =
                                    std::max(boundary_error, std::abs(value - factor * (deriv ? derivative : bessel)));
                        }
                    }
                }
            }
            mt_basis.push_back(mdarray<std::complex<double>, 2>({nmt, n}));
            auto& a = mt_basis.back();
            a.zero();
            for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                for (int xi = 0; xi < naw; xi++) {
                    a(xi, kp->gkvec().offset() + ig) = alm(ig, xi);
                }
            }
            kp->comm().allreduce(a.at(memory_t::host), a.size());
            num_lo.push_back(nmt - naw);
            lo_offset.push_back(next_lo);
            for (int ilo = 0; ilo < nmt - naw; ilo++) {
                a(naw + ilo, next_lo++) = 1;
            }
        }
        mdarray<std::complex<double>, 2> scalar_delta({n, n});
        scalar_delta.zero();
        for (int s = 0; s < (second ? 1 : ctx.num_spins()); s++) {
            k0.set_fv_h_o(s, m0, o0);
            k1.set_fv_h_o(s, m1, o1);
            k2.set_fv_h_o(s, m2, o2);
            std::vector<mdarray<std::complex<double>, 2>> local_matrices;
            for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
                LAPW_xc_atom op(ctx.unit_cell().atom(ia), ctx.lmax_rho());
                auto matrix = op.matrix_elements(d.local[ia][0][s], d.local[ia][1][s]);
                if (second && ctx.num_spins() == 2) {
                    auto down = op.matrix_elements(d.local[ia][0][1], d.local[ia][1][1]);
                    for (size_t i = 0; i < matrix.size(); i++) {
                        matrix[i] = 0.5 * (matrix[i] + down[i]);
                    }
                }
                local_matrices.push_back(std::move(matrix));
            }
            for (int j = 0; j < m0.num_cols_local(); j++) {
                for (int i = 0; i < m0.num_rows_local(); i++) {
                    std::complex<double> expected{0};
                    if (m0.irow(i) < kp->gkvec().num_gvec() && m0.icol(j) < kp->gkvec().num_gvec()) {
                        auto gi = kp->gkvec().gvec(gvec_index_t::global(m0.irow(i)));
                        auto gj = kp->gkvec().gvec(gvec_index_t::global(m0.icol(j)));
                        auto qi = kp->gkvec().gkvec_cart(gvec_index_t::global(m0.irow(i)));
                        auto qj = kp->gkvec().gkvec_cart(gvec_index_t::global(m0.icol(j)));
                        int g   = ctx.gvec().index_g12(gi, gj);
                        expected =
                                component(vrho, g, s, !second) + 0.5 * r3::dot(qi, qj) * component(vtau, g, s, !second);
                    }
                    matrix_error = std::max(
                            {matrix_error, std::abs(m1(i, j) - m0(i, j) - expected), std::abs(o1(i, j) - o0(i, j))});
                    // Independently transform each atomic matrix into the full PW/LO basis.
                    for (size_t ia = 0; ia < mt_basis.size(); ia++) {
                        auto const& a = mt_basis[ia];
                        auto const& h = local_matrices[ia];
                        for (int v = 0; v < static_cast<int>(h.size(1)); v++) {
                            for (int u = 0; u < static_cast<int>(h.size(0)); u++) {
                                expected += std::conj(a(u, m0.irow(i))) * h(u, v) * a(v, m0.icol(j));
                            }
                        }
                    }
                    full_matrix_error = std::max({full_matrix_error, std::abs(m2(i, j) - m0(i, j) - expected),
                                                  std::abs(o2(i, j) - o0(i, j))});
                    if (s == 0) {
                        scalar_delta(m0.irow(i), m0.icol(j)) = expected;
                    }
                }
            }
        }
        // LAPACK uses a replicated exact matrix, even with distributed orbitals.
        ctx.blacs_grid().comm().allreduce(scalar_delta.at(memory_t::host), scalar_delta.size());
        // The iterative first-variation operator uses the scalar channel.
        if (second || ctx.num_spins() == 1) {
            wf::Wave_functions<double> phi(kp->gkvec_sptr(), num_lo, wf::num_mag_dims(0), wf::num_bands(1),
                                           memory_t::host);
            wf::Wave_functions<double> a(kp->gkvec_sptr(), num_lo, wf::num_mag_dims(0), wf::num_bands(1),
                                         memory_t::host);
            wf::Wave_functions<double> b(kp->gkvec_sptr(), num_lo, wf::num_mag_dims(0), wf::num_bands(1),
                                         memory_t::host);
            for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                phi.pw_coeffs(ig, wf::spin_index(0), wf::band_index(0)) =
                        orbital(kp->gkvec().offset() + ig, 0, 0, false);
            }
            for (auto atom : phi.spl_num_atoms()) {
                for (int ilo = 0; ilo < num_lo[atom.i]; ilo++) {
                    phi.mt_coeffs(ilo, atom.li, wf::spin_index(0), wf::band_index(0)) =
                            orbital(lo_offset[atom.i] + ilo, 0, 0, false);
                }
            }
            k0.apply_fv_h_o(false, false, wf::band_range(0, 1), phi, &a, nullptr);
            k2.apply_fv_h_o(false, false, wf::band_range(0, 1), phi, &b, nullptr);
            auto residual = [&](int i, std::complex<double> value) {
                std::complex<double> expected{0};
                for (int j = 0; j < n; j++) {
                    expected += scalar_delta(i, j) * orbital(j, 0, 0, false);
                }
                full_apply_error = std::max(full_apply_error, std::abs(value - expected));
            };
            for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                residual(kp->gkvec().offset() + ig, b.pw_coeffs(ig, wf::spin_index(0), wf::band_index(0)) -
                                                            a.pw_coeffs(ig, wf::spin_index(0), wf::band_index(0)));
            }
            for (auto atom : phi.spl_num_atoms()) {
                for (int ilo = 0; ilo < num_lo[atom.i]; ilo++) {
                    residual(lo_offset[atom.i] + ilo,
                             b.mt_coeffs(ilo, atom.li, wf::spin_index(0), wf::band_index(0)) -
                                     a.mt_coeffs(ilo, atom.li, wf::spin_index(0), wf::band_index(0)));
                }
            }
        }
        wf::Wave_functions<double> phi(kp->gkvec_sptr(), wf::num_mag_dims(0), wf::num_bands(1), memory_t::host);
        wf::Wave_functions<double> a(kp->gkvec_sptr(), wf::num_mag_dims(0), wf::num_bands(1), memory_t::host);
        wf::Wave_functions<double> b(kp->gkvec_sptr(), wf::num_mag_dims(0), wf::num_bands(1), memory_t::host);
        wf::Wave_functions<double> za(kp->gkvec_sptr(), wf::num_mag_dims(0), wf::num_bands(1), memory_t::host);
        wf::Wave_functions<double> zb(kp->gkvec_sptr(), wf::num_mag_dims(0), wf::num_bands(1), memory_t::host);
        auto c = [&](int ig) { return orbital(ig, 0, 0, false); };
        for (int ig = 0; ig < kp->gkvec().count(); ig++) {
            phi.pw_coeffs(ig, wf::spin_index(0), wf::band_index(0)) = c(kp->gkvec().offset() + ig);
        }
        h0.local_op().apply_fplapw(kp->spfft_transform(), kp->gkvec_fft_sptr(), wf::band_range(0, 1), phi, &a, nullptr,
                                   ctx.num_spins() == 2 ? &za : nullptr, nullptr);
        hi.local_op().apply_fplapw(kp->spfft_transform(), kp->gkvec_fft_sptr(), wf::band_range(0, 1), phi, &b, nullptr,
                                   ctx.num_spins() == 2 ? &zb : nullptr, nullptr);
        for (int s = 0; s < ctx.num_spins(); s++) {
            auto const& wa = s == 0 ? a : za;
            auto const& wb = s == 0 ? b : zb;
            for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                auto gi = kp->gkvec().gvec(gvec_index_t::local(ig));
                auto qi = kp->gkvec().gkvec_cart(gvec_index_t::local(ig));
                std::complex<double> expected{0};
                for (int jg = 0; jg < kp->gkvec().num_gvec(); jg++) {
                    auto gj = kp->gkvec().gvec(gvec_index_t::global(jg));
                    auto qj = kp->gkvec().gkvec_cart(gvec_index_t::global(jg));
                    int g   = ctx.gvec().index_g12(gi, gj);
                    expected += (component(vrho, g, s, false) + 0.5 * r3::dot(qi, qj) * component(vtau, g, s, false)) *
                                c(jg);
                }
                local_error = std::max(local_error,
                                       std::abs(wb.pw_coeffs(ig, wf::spin_index(0), wf::band_index(0)) -
                                                wa.pw_coeffs(ig, wf::spin_index(0), wf::band_index(0)) - expected));
            }
        }
    }
    check("exact LAPW assembly, shifted k and unchanged overlap", matrix_error, 3e-12);
    check("matrix-free scalar/magnetic rho and tau", local_error, 3e-12);
    check("complete PW/AW/LO assembly", full_matrix_error, 3e-12);
    check("complete matrix-free PW/AW/LO application", full_apply_error, 3e-12);
    check("LAPW boundary value and radial slope", boundary_error, 3e-12);

    // Change a basis function after XC evaluation. Cached matrices would now be stale.
    auto& atom = ctx.unit_cell().atom(0);
    std::vector<double> saved(atom.type().num_mt_points()), changed(saved.size());
    for (int ir = 0; ir < atom.type().num_mt_points(); ir++) {
        saved[ir]   = atom.symmetry_class().radial_function(ir, 0);
        changed[ir] = 1.07 * saved[ir];
    }
    auto set_radial = [&](std::vector<double> const& values) {
        atom.symmetry_class().radial_function(0, values);
        Spline<double> spline(atom.type().radial_grid(), values);
        spline.interpolate();
        std::vector<double> derivative(values.size());
        for (int ir = 0; ir < atom.type().num_mt_points(); ir++) {
            derivative[ir] = atom.type().radial_grid(ir) * spline.deriv(1, ir);
        }
        atom.symmetry_class().radial_function_derivative(0, derivative);
        for (int order = 0; order <= 2; order++) {
            atom.symmetry_class().aw_surface_deriv(0, 0, order, spline.deriv(order, values.size() - 1));
        }
    };
    set_radial(changed);
    Hamiltonian0<double> updated_base(base, true, false), updated_full(full, true, false);
    check("reprojection after radial-basis update", mt_error(updated_base, updated_full), 2e-13);
    double change = std::abs(updated_full.hmt(0)(0, 0, 0) - updated_base.hmt(0)(0, 0, 0) - h1.hmt(0)(0, 0, 0) +
                             h0.hmt(0)(0, 0, 0));
    check("basis update changes the XC matrix", change > 1e-8 ? 0 : 1, 0);
    set_radial(saved);
    return failures;
}

int
test_lapw_map(int num_mag_dims, bool replicated_fft, bool second_variation)
{
    auto const& comm                          = mpi::Communicator::world();
    auto config                               = R"({"parameters": {
        "electronic_structure_method": "full_potential_lapwlo",
        "use_symmetry": false, "pw_cutoff": 4.0, "gk_cutoff": 1.0,
        "num_bands": 4, "xc_functionals": ["XC_LDA_X"], "core_relativity": "none", "valence_relativity": "none",
        "auto_rmt": 0, "lmax_apw": 2, "lmax_rho": 4, "lmax_pot": 4
    }, "control": {"processing_unit": "CPU", "verbosity": 0},
    "mixer": {"type": "linear", "beta": 0.4, "use_hartree": false}})"_json;
    config["parameters"]["num_mag_dims"]      = num_mag_dims;
    config["control"]["mpi_grid_dims"]        = {replicated_fft ? 1 : comm.size(), 1};
    config["control"]["use_second_variation"] = second_variation;
    config["iterative_solver"]["type"]        = "exact";
    Simulation_context ctx(config);
    ctx.unit_cell().set_lattice_vectors({{8, 0.7, 0.2}, {0, 9, 0.4}, {0, 0, 10}});
    for (auto label : {"Li", "H"}) {
        auto& type = ctx.unit_cell().add_atom_type(label);
        type.zn(std::string(label) == "Li" ? 3 : 1);
        type.set_radial_grid(radial_grid_t::power, 501, 1e-4, 2.0, 2.0);
        auto free_grid = Radial_grid_factory<double>(radial_grid_t::power, 3000, 1e-7, 20.0, 3.0);
        type.free_atom_density(std::vector<double>(free_grid.num_points(), 0));
        type.set_free_atom_radial_grid(std::move(free_grid));
        for (int l = 0; l <= 2; l++) {
            type.add_aw_descriptor(l + 1, l, -0.5, 0, 0);
            type.add_aw_descriptor(l + 1, l, -0.5, 1, 0);
        }
        for (int l = 0; l <= 1; l++) {
            type.add_lo_descriptor(l, l + 2, l, -0.3, 0, 0);
            type.add_lo_descriptor(l, l + 2, l, -0.3, 1, 0);
        }
        if (std::string(label) == "Li") {
            type.set_configuration(1, 0, 1, 2.0, true);
        }
    }
    ctx.unit_cell().add_atom("Li", {0.15, 0.2, 0.25});
    ctx.unit_cell().add_atom("H", {0.65, 0.7, 0.75});
    ctx.initialize();
    for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
        auto& atom = ctx.unit_cell().atom(ia);
        for (auto const& rf : atom.type().indexr()) {
            std::vector<double> values(atom.type().num_mt_points());
            for (int ir = 0; ir < atom.type().num_mt_points(); ir++) {
                double r   = atom.type().radial_grid(ir);
                values[ir] = std::pow(r, rf.am.l()) * std::exp(-(0.4 + 0.1 * rf.idxrf) * r * r);
                if (rf.idxlo >= 0) {
                    values[ir] *= std::pow(1 - r * r / 4, 2);
                }
            }
            atom.symmetry_class().radial_function(rf.idxrf, values);
            Spline<double> spline(atom.type().radial_grid(), values);
            spline.interpolate();
            for (int ir = 0; ir < atom.type().num_mt_points(); ir++) {
                values[ir] = atom.type().radial_grid(ir) * spline.deriv(1, ir);
            }
            atom.symmetry_class().radial_function_derivative(rf.idxrf, values);
            if (rf.idxlo < 0) {
                for (int d = 0; d <= 2; d++) {
                    atom.symmetry_class().aw_surface_deriv(rf.am.l(), rf.order, d,
                                                           spline.deriv(d, atom.type().num_mt_points() - 1));
                }
            }
        }
    }
    Density density(ctx, true);
    int failures{0};
    auto check = [&](char const* name, double error, double tolerance) {
        comm.allreduce<double, mpi::op_t::max>(&error, 1);
        if (comm.rank() == 0) {
            std::cout << "LAPW map spins=" << ctx.num_spins() << " replicated FFT=" << replicated_fft
                      << " second variation=" << second_variation << " " << name << ": " << error << '\n';
        }
        failures += !std::isfinite(error) || error > tolerance;
    };
    auto reject = [&](char const* name, auto&& operation) {
        bool rejected{false};
        try {
            operation();
        } catch (std::exception const&) {
            rejected = true;
        }
        check(name, rejected ? 0 : 1, 0);
    };
    reject("core required before tau construction", [&] { density.generate_mt_kinetic_density(true); });
    reject("stored tau required for field sampling", [&] {
        Density without_tau(ctx);
        LAPW_xc_field_map map(without_tau, {});
    });
    bool fields_initialized{false};
    auto update_mt_fields = [&] {
        for (auto it : ctx.unit_cell().spl_num_atoms()) {
            auto local = LAPW_xc_atom(ctx.unit_cell().atom(it.i), ctx.lmax_rho()).fields(density.density_matrix(it.i));
            for (size_t i = 0; i < local[0][0].size(); i++) {
                double up = local[0][0][i], dn = ctx.num_spins() == 2 ? local[0][1][i] : 0;
                density.rho().mt()[it.i][i] = up + dn;
                if (ctx.num_spins() == 2) {
                    density.mag(0).mt()[it.i][i] = up - dn;
                }
            }
            for (int ir = 0; ir < ctx.unit_cell().atom(it.i).num_mt_points(); ir++) {
                density.rho().mt()[it.i](0, ir) += density.core_charge_density(it.i)[ir] / y00;
            }
        }
        for (int j = 0; j < ctx.num_spins(); j++) {
            density.component(j).mt().sync(ctx.unit_cell().spl_num_atoms());
        }
        density.generate_mt_kinetic_density(true);
    };
    auto solve_core = [&](double scale) {
        std::vector<std::vector<double>> v(ctx.unit_cell().num_atom_symmetry_classes());
        for (int ic = 0; ic < ctx.unit_cell().num_atom_symmetry_classes(); ic++) {
            auto const& type = ctx.unit_cell().atom_symmetry_class(ic).atom_type();
            for (int ir = 0; ir < type.num_mt_points(); ir++) {
                v[ic].push_back(-scale * type.zn() / type.radial_grid(ir));
            }
        }
        density.generate_core_charge_density(v);
        if (fields_initialized) {
            update_mt_fields();
        }
    };
    solve_core(1);
    auto set_dm = [&](double step) {
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            auto& dm = density.density_matrix(ia);
            for (int s = 0; s < ctx.num_spins(); s++) {
                for (int j = 0; j < static_cast<int>(dm.size(1)); j++) {
                    for (int i = 0; i < static_cast<int>(dm.size(0)); i++) {
                        auto ci     = orbital(i, ia, s, false) + step * orbital(i, ia, s, true);
                        auto cj     = orbital(j, ia, s, false) + step * orbital(j, ia, s, true);
                        dm(i, j, s) = occupation(s) * std::conj(ci) * cj;
                    }
                }
            }
        }
        update_mt_fields();
        fields_initialized = true;
    };
    set_dm(0);
    std::vector<Smooth_periodic_function<double>> rho, tau;
    for (int s = 0; s < ctx.num_spins(); s++) {
        rho.emplace_back(ctx.spfft<double>(), ctx.gvec_fft_sptr());
        tau.emplace_back(ctx.spfft<double>(), ctx.gvec_fft_sptr());
    }
    auto set_pw = [&](double step) {
        for (int s = 0; s < ctx.num_spins(); s++) {
            for (int ig = 0; ig < ctx.gvec().count(); ig++) {
                auto g = ctx.gvec().gvec(gvec_index_t::local(ig));
                rho[s].f_pw_local(ig) =
                        fourier_coefficient(g, s, false, false) + step * fourier_coefficient(g, s, false, true);
                tau[s].f_pw_local(ig) =
                        fourier_coefficient(g, s, true, false) + step * fourier_coefficient(g, s, true, true);
            }
        }
    };
    set_pw(0);
    auto const& lattice = ctx.unit_cell().lattice_vectors();
    auto centre         = [&](int ia) { return r3::dot(lattice, ctx.unit_cell().atom(ia).position()); };
    // The exact surface belongs to MT, including when the point has a periodic image.
    std::vector<r3::vector<double>> all_points{centre(0) + r3::vector<double>(0.4, 0.2, -0.3),
                                               centre(1) + r3::vector<double>(-0.2, 0.33, 0.15),
                                               r3::dot(lattice, r3::vector<double>(0.4, 0.45, 0.5)),
                                               centre(0) + r3::vector<double>(0, 0, 2),
                                               centre(0) + r3::vector<double>(0, 0, 2 + 1e-6),
                                               centre(0) + r3::vector<double>(0, 0, 2 - 1e-6),
                                               centre(0) + r3::vector<double>(0.4, 0.2, -0.3) +
                                                       r3::dot(lattice, r3::vector<double>(2, -1, 1)),
                                               centre(1) + r3::vector<double>(1e-5, 2e-5, -1e-5)};
    std::vector<int> ids;
    std::vector<r3::vector<double>> points;
    for (int p = 0; p < static_cast<int>(all_points.size()); p++) {
        if ((replicated_fft ? 0 : p % comm.size()) == comm.rank()) {
            ids.push_back(p);
            points.push_back(all_points[p]);
        }
    }
    LAPW_xc_field_map map(density, points);
    auto fields = map.sample(rho, tau);
    double error{0};
    auto pw_values = Smooth_periodic_function_sampler(ctx.gvec(), points);
    for (int s = 0; s < ctx.num_spins(); s++) {
        auto rpw = pw_values.sample(rho[s]);
        auto tpw = pw_values.sample(tau[s]);
        for (int p = 0; p < static_cast<int>(points.size()); p++) {
            int id = ids[p];
            std::array<double, 4> r;
            double t;
            if (id == 2 || id == 4) {
                r = rpw[p];
                t = tpw[p][0];
            } else {
                int ia     = id == 1 || id == 7 ? 1 : 0;
                auto d     = all_points[id == 6 ? 0 : id] - centre(ia);
                auto local = LAPW_xc_atom(ctx.unit_cell().atom(ia), ctx.lmax_rho()).fields(density.density_matrix(ia));
                r = Spheric_function_interpolator(local[0][s], spheric_inner_extension_t::charge).value_gradient(d);
                t = Spheric_function_interpolator(local[1][s], spheric_inner_extension_t::kinetic).value_gradient(d)[0];
                Flm core(1, ctx.unit_cell().atom(ia).type().radial_grid());
                for (int ir = 0; ir < core.radial_grid().num_points(); ir++) {
                    core(0, ir) = density.core_charge_density(ia)[ir] / (ctx.num_spins() * y00);
                }
                auto cr = Spheric_function_interpolator(core, spheric_inner_extension_t::charge).value_gradient(d);
                for (int x = 0; x < 4; x++) {
                    r[x] += cr[x];
                }
                for (int ir = 0; ir < core.radial_grid().num_points(); ir++) {
                    core(0, ir) = density.core_kinetic_density(ia)[ir] / (ctx.num_spins() * y00);
                }
                t += Spheric_function_interpolator(core, spheric_inner_extension_t::charge).value_gradient(d)[0];
            }
            error = std::max({error, std::abs(fields.rho(p, s) - r[0]), std::abs(fields.tau(p, s) - t)});
            for (int x = 0; x < 3; x++) {
                error = std::max(error, std::abs(fields.gradient(p, x, s) - r[x + 1]));
            }
        }
    }
    check("piecewise values, gradients, core and periodic images", error, 3e-11);

    // Independent orbital values test the complex Gaunt contraction and DM convention.
    error = 0;
    for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
        auto const& atom = ctx.unit_cell().atom(ia);
        auto local       = LAPW_xc_atom(atom, ctx.lmax_rho()).fields(density.density_matrix(ia));
        auto native_tau  = density.generate_mt_valence_kinetic_density(ia);
        SHT sht(device_t::CPU, ctx.lmax_rho());
        for (int s = 0; s < ctx.num_spins(); s++) {
            auto values = transform(sht, local[0][s]);
            for (int ir : {0, 200, 500}) {
                for (int tp = 0; tp < sht.num_points(); tp++) {
                    auto angular = r3::spherical_coordinates(sht.coord(tp));
                    std::vector<std::complex<double>> ylm(sf::lmmax(2));
                    sf::spherical_harmonics(2, angular[1], angular[2], ylm.data());
                    std::complex<double> psi{0};
                    for (auto const& b : atom.type().indexb()) {
                        psi += orbital(b.xi, ia, s, false) * atom.symmetry_class().radial_function(ir, b.idxrf) *
                               ylm[b.lm];
                    }
                    error = std::max(error, std::abs(values(tp, ir) - occupation(s) * std::norm(psi)));
                }
            }
            for (int ir = 0; ir < atom.type().num_mt_points(); ir++) {
                for (int lm = 0; lm < sf::lmmax(ctx.lmax_rho()); lm++) {
                    error = std::max(error, std::abs(local[1][s](lm, ir) - native_tau[s](lm, ir)));
                }
            }
        }
    }
    check("charge from complex orbitals and native MT tau", error, 2e-12);

    // A single nonlinear nonlocal energy spans both domains and both spins.
    // Point 0 has no energy weight but participates in the descriptor moment.
    auto energy = [&](LAPW_xc_field_map const& m, xc_point_fields_t* derivatives) {
        auto f = m.sample(rho, tau);
        double moment{0}, e{0};
        for (int p = 0; p < static_cast<int>(ids.size()); p++) {
            double w = ids[p] == 0 ? 0 : 0.02 + 0.003 * ids[p];
            double d = 0.06 + 0.005 * ids[p];
            for (int s = 0; s < ctx.num_spins(); s++) {
                double r = f.rho(p, s), t = f.tau(p, s);
                moment += d * (r + 0.7 * t);
                e += w * (0.3 * r * r + 0.2 * t * t + 0.13 * r * t);
                for (int x = 0; x < 3; x++) {
                    e += 0.1 * w * std::pow(f.gradient(p, x, s), 2);
                }
            }
        }
        comm.allreduce(&moment, 1);
        comm.allreduce(&e, 1);
        if (derivatives) {
            for (int p = 0; p < static_cast<int>(ids.size()); p++) {
                double w = ids[p] == 0 ? 0 : 0.02 + 0.003 * ids[p];
                double d = 0.06 + 0.005 * ids[p];
                for (int s = 0; s < ctx.num_spins(); s++) {
                    derivatives->rho(p, s) = w * (0.6 * f.rho(p, s) + 0.13 * f.tau(p, s)) + 0.04 * moment * d;
                    derivatives->tau(p, s) = w * (0.4 * f.tau(p, s) + 0.13 * f.rho(p, s)) + 0.028 * moment * d;
                    for (int x = 0; x < 3; x++) {
                        derivatives->gradient(p, x, s) = 0.2 * w * f.gradient(p, x, s);
                    }
                }
            }
        }
        return e + 0.02 * moment * moment;
    };
    xc_point_fields_t derivatives(points.size(), ctx.num_spins());
    double xc_energy = energy(map, &derivatives);
    for (int p = 0; p < static_cast<int>(ids.size()); p++) {
        if (ids[p] == 0 && std::abs(derivatives.rho(p, 0)) < 1e-4) {
            failures++;
        }
    }
    auto reverse          = map.adjoint(derivatives);
    auto local_derivative = [&](lapw_xc_field_adjoint_t const& r, bool real_only) {
        double value{0};
        for (size_t ia = 0; ia < r.local.size(); ia++) {
            LAPW_xc_atom op(ctx.unit_cell().atom(ia), ctx.lmax_rho());
            for (int s = 0; s < ctx.num_spins(); s++) {
                auto matrix = op.matrix_elements(r.local[ia][0][s], r.local[ia][1][s]);
                for (int j = 0; j < static_cast<int>(matrix.size(1)); j++) {
                    for (int i = 0; i < static_cast<int>(matrix.size(0)); i++) {
                        auto h = real_only ? std::complex<double>(matrix(i, j).real()) : matrix(i, j);
                        value += 2 * occupation(s) *
                                 std::real(std::conj(orbital(i, ia, s, true)) * h * orbital(j, ia, s, false));
                    }
                }
            }
        }
        return value;
    };
    double interstitial{0};
    for (int s = 0; s < ctx.num_spins(); s++) {
        for (int ig = 0; ig < ctx.gvec().count(); ig++) {
            auto g              = ctx.gvec().gvec(gvec_index_t::local(ig));
            double multiplicity = ctx.gvec().reduced() && r3::dot(g, g) != 0 ? 2 : 1;
            interstitial +=
                    ctx.gvec().omega() * multiplicity *
                    std::real(std::conj(reverse.rho[s].f_pw_local(ig)) * fourier_coefficient(g, s, false, true) +
                              std::conj(reverse.tau[s].f_pw_local(ig)) * fourier_coefficient(g, s, true, true));
        }
    }
    comm.allreduce(&interstitial, 1);
    double local    = local_derivative(reverse, false);
    double analytic = local + interstitial;
    if (comm.rank() == 0) {
        std::cout << "analytic derivatives MT=" << local << " interstitial=" << interstitial << '\n';
    }
    for (double h : {1e-4, 5e-5}) {
        set_dm(h);
        set_pw(h);
        double plus = energy(LAPW_xc_field_map(density, points), nullptr);
        set_dm(-h);
        set_pw(-h);
        double minus = energy(LAPW_xc_field_map(density, points), nullptr);
        check("joint complex-orbital/PW derivative", std::abs((plus - minus) / (2 * h) - analytic), 2e-9);
    }
    set_dm(0);
    set_pw(0);
    for (bool mt_only : {false, true}) {
        double h = 1e-4;
        if (mt_only) {
            set_dm(h);
        } else {
            set_pw(h);
        }
        double plus = energy(LAPW_xc_field_map(density, points), nullptr);
        if (mt_only) {
            set_dm(-h);
        } else {
            set_pw(-h);
        }
        double minus = energy(LAPW_xc_field_map(density, points), nullptr);
        check(mt_only ? "MT-only orbital derivative" : "interstitial-only PW derivative",
              std::abs((plus - minus) / (2 * h) - (mt_only ? local : interstitial)), 2e-9);
        set_dm(0);
        set_pw(0);
    }
    check("interstitial derivative is exercised", std::abs(interstitial) > 1e-5 ? 0 : 1, 0);
    check("imaginary local matrix is exercised", std::abs(local - local_derivative(reverse, true)) > 1e-5 ? 0 : 1, 0);
    auto no_tau = xc_point_fields_t(points.size(), ctx.num_spins());
    std::copy(derivatives.rho.begin(), derivatives.rho.end(), no_tau.rho.begin());
    std::copy(derivatives.gradient.begin(), derivatives.gradient.end(), no_tau.gradient.begin());
    no_tau.tau.zero();
    check("local tau derivative is exercised",
          std::abs(local - local_derivative(map.adjoint(no_tau), false)) > 1e-5 ? 0 : 1, 0);

    // Differentiate the sampled core fields, without claiming a tau-dependent core solver.
    for (double h : {1e-4, 5e-5}) {
        solve_core(1 + h);
        double plus = energy(LAPW_xc_field_map(density, points), nullptr);
        std::vector<std::array<std::vector<double>, 2>> core_plus;
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            core_plus.push_back({density.core_charge_density(ia), density.core_kinetic_density(ia)});
        }
        solve_core(1 - h);
        double minus = energy(LAPW_xc_field_map(density, points), nullptr);
        double core_derivative{0};
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            std::array<std::vector<double> const*, 2> values{&density.core_charge_density(ia),
                                                             &density.core_kinetic_density(ia)};
            for (int part = 0; part < 2; part++) {
                for (size_t ir = 0; ir < values[part]->size(); ir++) {
                    core_derivative +=
                            reverse.core[ia][part][ir] * (core_plus[ia][part][ir] - (*values[part])[ir]) / (2 * h);
                }
            }
        }
        check("core-field adjoint", std::abs(core_derivative - (plus - minus) / (2 * h)), 2e-7);
        check("core response is not silently discarded", std::abs(core_derivative) > 1e-3 ? 0 : 1, 0);
    }
    solve_core(1);

    failures += test_lapw_hamiltonian(ctx, map.adjoint(derivatives), map.adjoint(derivatives));

    // The physical fields, not the current density matrix, define the energy
    // contraction. Compare the production contraction to the original point sum.
    for (int ig = 0; ig < ctx.gvec().count(); ig++) {
        density.rho().rg().f_pw_local(ig)                      = rho[0].f_pw_local(ig);
        density.kinetic_density().scalar().rg().f_pw_local(ig) = tau[0].f_pw_local(ig);
        if (ctx.num_spins() == 2) {
            density.rho().rg().f_pw_local(ig) += rho[1].f_pw_local(ig);
            density.kinetic_density().scalar().rg().f_pw_local(ig) += tau[1].f_pw_local(ig);
            density.mag(0).rg().f_pw_local(ig)                      = rho[0].f_pw_local(ig) - rho[1].f_pw_local(ig);
            density.kinetic_density().vector(0).rg().f_pw_local(ig) = tau[0].f_pw_local(ig) - tau[1].f_pw_local(ig);
        }
    }
    density.fft_transform(1);
    density.kinetic_density().fft_transform(1);
    std::array<double, 3> point_sum{0, 0, 0};
    for (int p = 0; p < static_cast<int>(points.size()); p++) {
        double r = fields.rho(p, 0);
        std::array<double, 3> g;
        for (int x = 0; x < 3; x++) {
            g[x] = fields.gradient(p, x, 0);
        }
        if (ctx.num_spins() == 2) {
            r = 0.5 * (r + fields.rho(p, 1));
            for (int x = 0; x < 3; x++) {
                g[x] = 0.5 * (g[x] + fields.gradient(p, x, 1));
            }
        }
        for (int s = 0; s < ctx.num_spins(); s++) {
            point_sum[0] += derivatives.rho(p, s) * r;
            point_sum[1] += derivatives.rho(p, s) * (fields.rho(p, s) - r);
            point_sum[2] += derivatives.tau(p, s) * fields.tau(p, s);
            for (int x = 0; x < 3; x++) {
                point_sum[0] += derivatives.gradient(p, x, s) * g[x];
                point_sum[1] += derivatives.gradient(p, x, s) * (fields.gradient(p, x, s) - g[x]);
            }
        }
    }
    comm.allreduce(point_sum.data(), point_sum.size());
    check("homogeneous functional Euler identity", std::abs(point_sum[0] + point_sum[1] + point_sum[2] - 2 * xc_energy),
          2e-12);

    Potential staged(ctx);
    staged.zero();
    staged.xc_potential().zero();
    for (int j = 0; j < ctx.num_spins(); j++) {
        for (int ir = 0; ir < ctx.spfft<double>().local_slice_size(); ir++) {
            staged.component(j).rg().value(ir) = 0.17 / (j + 1);
        }
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            for (int ir = 0; ir < ctx.unit_cell().atom(ia).num_mt_points(); ir++) {
                staged.component(j).mt()[ia](0, ir) = 0.23 / (j + 1);
            }
        }
    }
    staged.fft_transform(-1);
    double electrostatic = energy_veff(density, staged), ordinary_magnetic = energy_bxc(density, staged);
    staged.set_lapw_xc_derivatives(map.adjoint(derivatives), xc_energy);
    auto core_fields        = staged.get_core_xc_derivatives();
    auto const& atom_fields = staged.lapw_xc_derivatives()->core;
    double class_error{0};
    for (int ic = 0; ic < ctx.unit_cell().num_atom_symmetry_classes(); ic++) {
        auto const& cls = ctx.unit_cell().atom_symmetry_class(ic);
        for (int part = 0; part < 2; part++) {
            for (size_t ir = 0; ir < core_fields[ic][part].size(); ir++) {
                double average{0};
                for (int i = 0; i < cls.num_atoms(); i++) {
                    average += atom_fields[cls.atom_id(i)][part][ir] / cls.num_atoms();
                }
                class_error = std::max(class_error, std::abs(average - core_fields[ic][part][ir]));
            }
        }
    }
    check("class-resolved spherical core covectors", class_error, 0);
    auto contractions = staged.lapw_xc_contractions(density);
    check("discrete scalar-rho energy contraction", std::abs(contractions.rho - point_sum[0]), 2e-12);
    check("discrete magnetic energy contraction", std::abs(contractions.magnetic - point_sum[1]), 2e-12);
    check("discrete tau energy contraction", std::abs(contractions.tau - point_sum[2]), 2e-12);
    check("functional energy associated with its covectors", std::abs(staged.energy_exc(density) - xc_energy), 0);
    check("XC scalar contraction API", std::abs(staged.energy_vxc(density) - point_sum[0]), 2e-12);
    check("XC tau contraction API", std::abs(staged.energy_vtau(density) - point_sum[2]), 2e-12);
    check("XC addition to electrostatic contraction",
          std::abs(energy_veff(density, staged) - electrostatic - point_sum[0]), 2e-12);
    check("XC addition to magnetic contraction",
          std::abs(energy_bxc(density, staged) - ordinary_magnetic - point_sum[1]), 2e-12);
    reject("no pseudo-core contraction in LAPW", [&] { staged.energy_vxc_core(density); });
    reject("no LAPW energy density for nonlocal XC", [&] { staged.xc_energy_density(); });
    reject("energy contraction requires stored tau", [&] {
        Density without_tau(ctx);
        staged.energy_vtau(without_tau);
    });
    auto const* original = staged.lapw_xc_derivatives();
    for (int invalid : {0, 1, 2, 3, 4}) {
        auto bad = map.adjoint(derivatives);
        if (comm.rank() == 0) {
            switch (invalid) {
                case 0:
                    bad.rho.clear();
                    break;
                case 1:
                    bad.local[0][0][0] = Flm(1, ctx.unit_cell().atom(0).type().radial_grid());
                    break;
                case 2:
                    bad.core[0][1][0] = std::numeric_limits<double>::quiet_NaN();
                    break;
                case 3:
                    bad.tau[0].f_pw_local(0) = std::numeric_limits<double>::quiet_NaN();
                    break;
                case 4:
                    bad.core[0][1][0] += 0.1;
                    break;
            }
        }
        reject("collective invalid Hamiltonian covectors", [&] { staged.set_lapw_xc_derivatives(std::move(bad)); });
        check("rejected update preserves the staged operator", staged.lapw_xc_derivatives() == original ? 0 : 1, 0);
        check("rejected update preserves functional energy", std::abs(staged.energy_exc(density) - xc_energy), 0);
    }
    auto bad_energy = map.adjoint(derivatives);
    reject("collective nonfinite XC energy", [&] {
        staged.set_lapw_xc_derivatives(std::move(bad_energy),
                                       comm.rank() == 0 ? std::numeric_limits<double>::quiet_NaN() : xc_energy);
    });
    if (comm.size() > 1) {
        auto bad_presence = map.adjoint(derivatives);
        reject("collective mismatched XC energy presence", [&] {
            staged.set_lapw_xc_derivatives(std::move(bad_presence),
                                           comm.rank() == 0 ? std::optional<double>(xc_energy) : std::nullopt);
        });
        auto bad_value = map.adjoint(derivatives);
        reject("collective inconsistent XC energy",
               [&] { staged.set_lapw_xc_derivatives(std::move(bad_value), xc_energy + 0.1 * comm.rank()); });
    }
    check("rejected energy preserves the staged operator", staged.lapw_xc_derivatives() == original ? 0 : 1, 0);
    check("rejected energy preserves functional energy", std::abs(staged.energy_exc(density) - xc_energy), 0);
    staged.set_lapw_xc_derivatives(map.adjoint(derivatives));
    reject("operator-only restaging clears previous energy", [&] { staged.energy_exc(density); });

    for (int region : {0, 1, 2}) {
        std::vector<r3::vector<double>> selected;
        if (comm.rank() == 0 && region) {
            selected.push_back(all_points[region == 1 ? 1 : 2]);
        }
        LAPW_xc_field_map small(density, selected);
        auto f = small.sample(rho, tau);
        xc_point_fields_t zero(selected.size(), ctx.num_spins());
        zero.rho.zero();
        zero.gradient.zero();
        zero.tau.zero();
        auto r = small.adjoint(zero);
        double maximum{0};
        for (auto const& atom : r.local) {
            for (auto const& part : atom) {
                for (auto const& field : part) {
                    for (size_t i = 0; i < field.size(); i++) {
                        maximum = std::max(maximum, std::abs(field[i]));
                    }
                }
            }
        }
        for (auto const& atom : r.core) {
            for (auto const& field : atom) {
                for (double value : field) {
                    maximum = std::max(maximum, std::abs(value));
                }
            }
        }
        for (auto const* fields : {&r.rho, &r.tau}) {
            for (auto const& field : *fields) {
                for (int ig = 0; ig < ctx.gvec().count(); ig++) {
                    maximum = std::max(maximum, std::abs(field.f_pw_local(ig)));
                }
            }
        }
        check("empty, MT-only and interstitial-only target sets", maximum, 0);
    }
    auto bad_points = points;
    if (comm.rank() == 0) {
        bad_points.push_back(centre(0));
    }
    reject("collective exact-nucleus rejection", [&] { LAPW_xc_field_map bad(density, bad_points); });
    bad_points = points;
    if (comm.rank() == 0) {
        bad_points.push_back({std::numeric_limits<double>::quiet_NaN(), 0, 0});
    }
    reject("collective nonfinite-coordinate rejection", [&] { LAPW_xc_field_map bad(density, bad_points); });
    for (bool tau_field : {false, true}) {
        auto& f      = tau_field ? density.kinetic_density().scalar().mt()[0] : density.rho().mt()[0];
        double saved = f(0, 0);
        if (comm.rank() == 0) {
            f(0, 0) = std::numeric_limits<double>::quiet_NaN();
        }
        reject("collective nonfinite physical-field rejection", [&] { LAPW_xc_field_map bad(density, points); });
        f(0, 0) = saved;
    }
    auto& atom = ctx.unit_cell().atom(0);
    std::vector<double> radial(atom.type().num_mt_points());
    for (int ir = 0; ir < atom.type().num_mt_points(); ir++) {
        radial[ir] = atom.symmetry_class().radial_function(ir, 0);
    }
    auto changed = radial;
    for (auto& value : changed) {
        value *= 1.7;
    }
    atom.symmetry_class().radial_function(0, changed);
    density.density_matrix(0).zero();
    auto unchanged = LAPW_xc_field_map(density, points).sample(rho, tau);
    error          = 0;
    for (size_t i = 0; i < fields.rho.size(); i++) {
        error = std::max(
                {error, std::abs(unchanged.rho[i] - fields.rho[i]), std::abs(unchanged.tau[i] - fields.tau[i])});
    }
    for (size_t i = 0; i < fields.gradient.size(); i++) {
        error = std::max(error, std::abs(unchanged.gradient[i] - fields.gradient[i]));
    }
    check("physical fields survive radial-basis and density-matrix updates", error, 2e-12);
    atom.symmetry_class().radial_function(0, radial);
    set_dm(0);

    // Physical fields, including the core contribution, must be mixed and restarted as fields.
    Density saved(ctx, true), restored(ctx, true);
    copy(density, saved);
    auto state = [&](Density const& d) {
        std::vector<double> values;
        for (auto const* f : {static_cast<Field4D const*>(&d), &d.kinetic_density()}) {
            for (int j = 0; j < ctx.num_spins(); j++) {
                for (int ir = 0; ir < ctx.spfft<double>().local_slice_size(); ir++) {
                    values.push_back(f->component(j).rg().value(ir));
                }
                for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
                    auto const& mt = f->component(j).mt()[ia];
                    values.insert(values.end(), mt.begin(), mt.end());
                }
            }
        }
        return values;
    };
    auto compare = [&](std::vector<double> const& a, std::vector<double> const& b) {
        double delta{0};
        for (size_t i = 0; i < a.size(); i++) {
            delta = std::max(delta, std::abs(a[i] - b[i]));
        }
        return delta;
    };
    check("copy preserves physical MT tau and core metadata", compare(state(density), state(saved)), 0);
    check("copy preserves solved core tau",
          std::abs(saved.core_kinetic_density(0)[20] - density.core_kinetic_density(0)[20]), 0);
    Density without_tau(ctx);
    reject("copy rejects incompatible tau storage", [&] { copy(density, without_tau); });
    density.mixer_init(ctx.cfg().mixer());
    auto previous = state(density);
    for (auto* f : {static_cast<Field4D*>(&density), &density.kinetic_density()}) {
        for (int j = 0; j < ctx.num_spins(); j++) {
            for (int ir = 0; ir < ctx.spfft<double>().local_slice_size(); ir++) {
                f->component(j).rg().value(ir) = 0.04 * (j + 1) + 0.7 * f->component(j).rg().value(ir);
            }
            for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
                auto& mt = f->component(j).mt()[ia];
                for (size_t i = 0; i < mt.size(); i++) {
                    mt[i] = 0.7 * mt[i] + 0.001 * (j + 1) * std::cos(0.17 * i);
                }
            }
        }
        f->fft_transform(-1);
    }
    auto expected = state(density);
    for (size_t i = 0; i < expected.size(); i++) {
        expected[i] = 0.4 * expected[i] + 0.6 * previous[i];
    }
    density.mix();
    check("SCF mixes charge and tau on every owned and replicated sphere", compare(state(density), expected), 2e-12);
    auto mixed_points        = LAPW_xc_field_map(density, points).sample(rho, tau);
    unsigned long long stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    comm.bcast(&stamp, 1, 0);
    auto path = std::filesystem::temp_directory_path() / ("sirius-lapw-density-" + std::to_string(stamp) + ".h5");
    ctx.create_storage_file(path.string());
    density.save(path.string());
    comm.barrier();
    restored.load(path.string());
    check("restart preserves physical charge and tau", compare(state(density), state(restored)), 2e-12);
    auto restart_points = LAPW_xc_field_map(restored, points).sample(rho, tau);
    error               = 0;
    for (size_t i = 0; i < mixed_points.rho.size(); i++) {
        error = std::max({error, std::abs(mixed_points.rho[i] - restart_points.rho[i]),
                          std::abs(mixed_points.tau[i] - restart_points.tau[i])});
    }
    for (size_t i = 0; i < mixed_points.gradient.size(); i++) {
        error = std::max(error, std::abs(mixed_points.gradient[i] - restart_points.gradient[i]));
    }
    check("restarted point fields need neither a radial-basis rebuild nor a new core solve", error, 0);
    comm.barrier();
    if (comm.rank() == 0) {
        std::filesystem::remove(path);
    }

    density.zero();
    for (int j = 0; j < ctx.num_spins(); j++) {
        double value = j == 0 ? 0.8 : 0.2;
        for (int ir = 0; ir < ctx.spfft<double>().local_slice_size(); ir++) {
            density.component(j).rg().value(ir) = value;
        }
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            for (int ir = 0; ir < ctx.unit_cell().atom(ia).num_mt_points(); ir++) {
                density.component(j).mt()[ia](0, ir) = value / y00;
            }
        }
    }
    density.initial_kinetic_density();
    error = 0;
    double total{0}, difference{0};
    for (int s = 0; s < ctx.num_spins(); s++) {
        double n = ctx.num_spins() == 1 ? 0.8 : 0.5 * (0.8 + (1 - 2 * s) * 0.2);
        double t = 0.3 * std::pow(3 * pi * pi * ctx.num_spins(), 2.0 / 3.0) * std::pow(n, 5.0 / 3.0);
        total += t;
        difference += (1 - 2 * s) * t;
    }
    for (int j = 0; j < ctx.num_spins(); j++) {
        double value  = j == 0 ? total : difference;
        auto const& f = density.kinetic_density().component(j);
        for (int ir = 0; ir < ctx.spfft<double>().local_slice_size(); ir++) {
            error = std::max(error, std::abs(f.rg().value(ir) - value));
        }
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            for (int ir = 0; ir < ctx.unit_cell().atom(ia).num_mt_points(); ir++) {
                for (int lm = 0; lm < sf::lmmax(ctx.lmax_rho()); lm++) {
                    error = std::max(error, std::abs(f.mt()[ia](lm, ir) - (lm == 0 ? value / y00 : 0)));
                }
            }
        }
    }
    check("initial tau populates both real-space domains and spin channels", error, 2e-12);
    copy(saved, density);

    K_point_set density_ks(ctx);
    std::array<r3::vector<double>, 2> kpoints{{{0.13, -0.17, 0.07}, {-0.21, 0.11, 0.16}}};
    std::array<double, 2> weights{0.3, 0.7};
    for (int ik = 0; ik < 2; ik++) {
        density_ks.add_kpoint(kpoints[ik], weights[ik]);
    }
    density_ks.initialize();
    auto amplitude = [](int ik, int s) { return 1 + 0.1 * ik + 0.2 * s; };
    for (int ik = 0; ik < 2; ik++) {
        for (int s = 0; s < ctx.num_spins(); s++) {
            for (int ib = 0; ib < ctx.num_bands(); ib++) {
                density_ks.get<double>(ik)->band_occupancy(ib, s, ib == 0 ? 2.0 / ctx.num_spins() : 0);
            }
        }
    }
    for (auto it : density_ks.spl_num_kpoints()) {
        auto kp   = density_ks.get<double>(it.i);
        auto& psi = kp->spinor_wave_functions();
        psi.zero(memory_t::host);
        for (int s = 0; s < ctx.num_spins(); s++) {
            for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                auto g = kp->gkvec().gvec(gvec_index_t::local(ig));
                if (g[0] == 1 && g[1] == 0 && g[2] == 0) {
                    psi.pw_coeffs(ig, wf::spin_index(s), wf::band_index(0)) = amplitude(it.i, s);
                }
            }
            for (auto ia : psi.spl_num_atoms()) {
                auto local_atom = atom_index_t::local(ia.li);
                for (int xi = 0; xi < ctx.unit_cell().atom(ia.i).type().mt_basis_size(); xi++) {
                    psi.mt_coeffs(xi, local_atom, wf::spin_index(s), wf::band_index(0)) =
                            amplitude(it.i, s) * orbital(xi, ia.i, s, false);
                }
            }
        }
    }
    for (bool add_core : {false, true}) {
        density.generate<double>(density_ks, false, add_core, true);
        error = 0;
        std::vector<double> norms(ctx.num_spins()), smooth_tau(ctx.num_spins());
        for (int s = 0; s < ctx.num_spins(); s++) {
            for (int ik = 0; ik < 2; ik++) {
                double weight = weights[ik] * 2.0 / ctx.num_spins() * std::pow(amplitude(ik, s), 2);
                auto q        = r3::dot(ctx.unit_cell().reciprocal_lattice_vectors(),
                                        kpoints[ik] + r3::vector<double>(1, 0, 0));
                norms[s] += weight;
                smooth_tau[s] += weight * 0.5 * r3::dot(q, q) / ctx.unit_cell().omega();
            }
        }
        for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
            auto const& dm = density.density_matrix(ia);
            for (int s = 0; s < ctx.num_spins(); s++) {
                for (int j = 0; j < static_cast<int>(dm.size(1)); j++) {
                    for (int i = 0; i < static_cast<int>(dm.size(0)); i++) {
                        auto target = norms[s] * std::conj(orbital(i, ia, s, false)) * orbital(j, ia, s, false);
                        error       = std::max(error, std::abs(dm(i, j, s) - target));
                    }
                }
            }
            auto local = LAPW_xc_atom(ctx.unit_cell().atom(ia), ctx.lmax_rho()).fields(dm);
            for (int j = 0; j < ctx.num_spins(); j++) {
                for (int ir = 0; ir < ctx.unit_cell().atom(ia).num_mt_points(); ir++) {
                    for (int lm = 0; lm < sf::lmmax(ctx.lmax_rho()); lm++) {
                        double r = local[0][0](lm, ir), t = local[1][0](lm, ir);
                        if (ctx.num_spins() == 2) {
                            r += (1 - 2 * j) * local[0][1](lm, ir);
                            t += (1 - 2 * j) * local[1][1](lm, ir);
                        }
                        if (add_core && lm == 0 && j == 0) {
                            r += density.core_charge_density(ia)[ir] / y00;
                            t += density.core_kinetic_density(ia)[ir] / y00;
                        }
                        error = std::max({error, std::abs(density.component(j).mt()[ia](lm, ir) - r),
                                          std::abs(density.kinetic_density().component(j).mt()[ia](lm, ir) - t)});
                    }
                }
            }
        }
        for (int j = 0; j < ctx.num_spins(); j++) {
            double value = smooth_tau[0] + (ctx.num_spins() == 2 ? (1 - 2 * j) * smooth_tau[1] : 0);
            for (int ir = 0; ir < ctx.spfft<double>().local_slice_size(); ir++) {
                error = std::max(error, std::abs(density.kinetic_density().component(j).rg().value(ir) - value));
            }
        }
        check(add_core ? "orbital density update stores valence plus core rho/tau"
                       : "orbital density update stores valence rho/tau",
              error, 3e-12);
    }
    copy(saved, density);
    if (comm.rank() == 0) {
        rho[0].f_pw_local(0) = std::numeric_limits<double>::quiet_NaN();
    }
    reject("collective nonfinite-PW rejection", [&] { map.sample(rho, tau); });
    set_pw(0);
    std::vector<Smooth_periodic_function<double>> empty;
    reject("collective wrong-spin rejection", [&] { map.sample(comm.rank() == 0 ? empty : rho, tau); });
    xc_point_fields_t bad(points.size() + (comm.rank() == 0 ? 1 : 0), ctx.num_spins());
    bad.rho.zero();
    bad.tau.zero();
    bad.gradient.zero();
    reject("collective wrong-buffer rejection", [&] { map.adjoint(bad); });
    if (comm.rank() == 0) {
        derivatives.gradient(0, 0, 0) = std::numeric_limits<double>::quiet_NaN();
    }
    reject("collective nonfinite-covector rejection", [&] { map.adjoint(derivatives); });
    return failures;
}

} // namespace

int
main(int argc, char** argv)
{
    sirius::initialize(1);
    int result{0};
    for (int mag : {0, 1}) {
        for (bool replicated : {false, true}) {
            for (bool second : {false, true}) {
                result += call_test("LAPW XC field map", test_lapw_map, mag, replicated, second);
            }
        }
    }
    sirius::finalize();
    return result;
}
