/* This file is part of SIRIUS electronic structure library.
 * Copyright (c), ETH Zurich. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "dft_ground_state.hpp"
#include "hamiltonian/hamiltonian.hpp"

namespace sirius {

std::array<std::array<double, 3>, 2>
DFT_ground_state::check_lapw_xc_derivative(double step__)
{
    int invalid = !ctx_.full_potential() || !potential_.lapw_xc_derivatives() || ctx_.hubbard_correction() ||
                  ctx_.processing_unit() != device_t::CPU || ctx_.cfg().parameters().precision_wf() != "fp64" ||
                  !std::isfinite(step__) || step__ <= 0;
    ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    double hmin = step__, hmax = step__;
    ctx_.comm().allreduce<double, mpi::op_t::min>(&hmin, 1);
    ctx_.comm().allreduce<double, mpi::op_t::max>(&hmax, 1);
    if (invalid || hmin != hmax) {
        RTE_THROW("LAPW XC derivative check requires staged CPU fp64 operators and a common positive step");
    }
    using Wf   = wf::Wave_functions<double>;
    auto bands = wf::band_range(0, ctx_.num_bands());
    std::vector<int> nmt, nlo;
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        nmt.push_back(unit_cell_.atom(ia).type().mt_basis_size());
        nlo.push_back(unit_cell_.atom(ia).type().mt_lo_basis_size());
    }
    std::vector<std::unique_ptr<Wf>> saved, tangent;
    for (auto it : kset_.spl_num_kpoints()) {
        auto kp = kset_.get<double>(it.i);
        for (auto* storage : {&saved, &tangent}) {
            storage->push_back(std::make_unique<Wf>(kp->gkvec_sptr(), nmt, wf::num_mag_dims(ctx_.num_mag_dims()),
                                                    wf::num_bands(ctx_.num_bands()), memory_t::host));
        }
        for (int s = 0; s < ctx_.num_spins(); s++) {
            wf::copy(memory_t::host, kp->spinor_wave_functions(), wf::spin_index(s), bands, *saved.back(),
                     wf::spin_index(s), bands);
        }
    }
    auto restore = [&] {
        size_t ik{0};
        for (auto it : kset_.spl_num_kpoints()) {
            auto kp = kset_.get<double>(it.i);
            for (int s = 0; s < ctx_.num_spins(); s++) {
                wf::copy(memory_t::host, *saved[ik], wf::spin_index(s), bands, kp->spinor_wave_functions(),
                         wf::spin_index(s), bands);
            }
            ik++;
        }
    };
    // Perturb the independent PW/LO coefficients, then use the production
    // matching transformation for both the orbitals and their tangents.
    auto set_orbitals = [&](int direction, double t, bool derivative) {
        size_t ik{0};
        for (auto it : kset_.spl_num_kpoints()) {
            auto kp = kset_.get<double>(it.i);
            Wf evec(kp->gkvec_sptr(), nlo, wf::num_mag_dims(0), wf::num_bands(ctx_.num_bands()), memory_t::host);
            auto& target = derivative ? *tangent[ik] : kp->spinor_wave_functions();
            auto rotate  = [&](std::complex<double> c, double phase) {
                auto z = c * std::polar(1.0, t * phase);
                return derivative ? std::complex<double>(0, phase) * z : z;
            };
            for (int s = 0; s < ctx_.num_spins(); s++) {
                for (int b = 0; b < ctx_.num_bands(); b++) {
                    auto band = wf::band_index(b);
                    for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                        auto g       = kp->gkvec().gvec(gvec_index_t::local(ig));
                        double phase = direction == 0 ? std::sin(0.41 * g[0] + 0.73 * g[1] + 0.19 * g[2]) : 0;
                        evec.pw_coeffs(ig, wf::spin_index(0), band) =
                                rotate(saved[ik]->pw_coeffs(ig, wf::spin_index(s), band), phase);
                    }
                    for (auto atom : evec.spl_num_atoms()) {
                        int naw = unit_cell_.atom(atom.i).type().mt_aw_basis_size();
                        for (int j = 0; j < nlo[atom.i]; j++) {
                            double phase = direction == 1 ? std::sin(0.43 * (atom.i + 1) + 0.29 * (j + 1)) : 0;
                            evec.mt_coeffs(j, atom.li, wf::spin_index(0), band) =
                                    rotate(saved[ik]->mt_coeffs(naw + j, atom.li, wf::spin_index(s), band), phase);
                        }
                    }
                }
                kp->generate_lapw_wave_functions(evec, target, s);
            }
            ik++;
        }
    };

    std::array<std::array<double, 3>, 2> result{};
    try {
        Density density(ctx_);
        copy(density_, density);
        Potential potential(ctx_);
        auto expectation = [&] {
            // Do not regenerate radial functions or update atom-potential pointers.
            // Subtracting the zero-XC operator cancels the fixed non-XC terms.
            Hamiltonian0<double> h0(potential, false);
            double derivative{0};
            size_t ik{0};
            for (auto it : kset_.spl_num_kpoints()) {
                auto kp = kset_.get<double>(it.i);
                h0.local_op().prepare_k(*kp->gkvec_fft_sptr());
                Wf phi(kp->gkvec_sptr(), nmt, wf::num_mag_dims(0), wf::num_bands(ctx_.num_bands()), memory_t::host);
                Wf hphi(kp->gkvec_sptr(), wf::num_mag_dims(0), wf::num_bands(ctx_.num_bands()), memory_t::host);
                Wf bphi(kp->gkvec_sptr(), wf::num_mag_dims(0), wf::num_bands(ctx_.num_bands()), memory_t::host);
                for (int s = 0; s < ctx_.num_spins(); s++) {
                    int nb = kp->num_occupied_bands(s);
                    if (!nb) {
                        continue;
                    }
                    auto occupied = wf::band_range(0, nb);
                    wf::copy(memory_t::host, kp->spinor_wave_functions(), wf::spin_index(s), occupied, phi,
                             wf::spin_index(0), occupied);
                    h0.local_op().apply_fplapw(kp->spfft_transform(), kp->gkvec_fft_sptr(), occupied, phi, &hphi,
                                               nullptr, ctx_.num_spins() == 2 ? &bphi : nullptr, nullptr);
                    for (int b = 0; b < nb; b++) {
                        auto band     = wf::band_index(b);
                        double weight = 2 * kp->weight() * kp->band_occupancy(b, s);
                        for (int ig = 0; ig < kp->gkvec().count(); ig++) {
                            auto h = hphi.pw_coeffs(ig, wf::spin_index(0), band);
                            if (ctx_.num_spins() == 2) {
                                h += double(1 - 2 * s) * bphi.pw_coeffs(ig, wf::spin_index(0), band);
                            }
                            derivative += weight *
                                          std::real(std::conj(tangent[ik]->pw_coeffs(ig, wf::spin_index(s), band)) * h);
                        }
                        for (auto atom : phi.spl_num_atoms()) {
                            auto const& hmt = h0.hmt(atom.i);
                            for (int i = 0; i < nmt[atom.i]; i++) {
                                std::complex<double> h{0};
                                for (int j = 0; j < nmt[atom.i]; j++) {
                                    auto matrix = hmt(i, j, s);
                                    if (ctx_.num_spins() == 2 && ctx_.cfg().control().use_second_variation()) {
                                        matrix = hmt(i, j, 0) + double(1 - 2 * s) * hmt(i, j, 1);
                                    }
                                    h += matrix * phi.mt_coeffs(j, atom.li, wf::spin_index(0), band);
                                }
                                derivative += weight * std::real(std::conj(tangent[ik]->mt_coeffs(
                                                                         i, atom.li, wf::spin_index(s), band)) *
                                                                 h);
                            }
                        }
                    }
                }
                ik++;
            }
            ctx_.comm().allreduce(&derivative, 1);
            return derivative;
        };
        for (int direction = 0; direction < 2; direction++) {
            constexpr double offset{0.17};
            auto evaluate = [&](double t) {
                set_orbitals(direction, offset + t, false);
                density.generate<double>(kset_, false, true, true);
                potential.generate(density, false, true);
                return potential.energy_exc(density);
            };
            evaluate(0);
            set_orbitals(direction, offset, true);
            result[direction][0] = expectation();
            // Remove only XC, leaving the identical non-XC operator in place.
            lapw_xc_field_adjoint_t zero;
            for (int s = 0; s < ctx_.num_spins(); s++) {
                zero.rho.emplace_back(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
                zero.tau.emplace_back(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
                zero.rho.back().zero();
                zero.tau.back().zero();
            }
            zero.local.resize(unit_cell_.num_atoms());
            zero.core.resize(unit_cell_.num_atoms());
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                auto const& grid = unit_cell_.atom(ia).type().radial_grid();
                for (int part = 0; part < 2; part++) {
                    zero.core[ia][part].resize(grid.num_points());
                    for (int s = 0; s < ctx_.num_spins(); s++) {
                        zero.local[ia][part].emplace_back(sf::lmmax(ctx_.lmax_rho()), grid);
                        zero.local[ia][part].back().zero();
                    }
                }
            }
            potential.set_lapw_xc_derivatives(std::move(zero), 0);
            result[direction][0] -= expectation();
            for (int j = 1; j <= 2; j++) {
                double h    = step__ / j;
                double plus = evaluate(h), minus = evaluate(-h);
                result[direction][j] = (plus - minus) / (2 * h);
            }
        }
    } catch (...) {
        restore();
        throw;
    }
    restore();
    return result;
}

} // namespace sirius
