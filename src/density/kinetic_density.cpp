/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "density.hpp"
#include "core/wf/kinetic_density_operator.hpp"
#include "function3d/kinetic_density.hpp"
#include <gsl/gsl_spline.h>

namespace sirius {

void
Density::generate_pseudo_core_fields()
{
    int invalid{0};
    try {
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            auto const& type = unit_cell_.atom_type(iat);
            type.check_ps_core_kinetic_density();
            invalid |= type.has_ps_core_kinetic_density() &&
                       type.num_mt_points() < static_cast<int>(gsl_interp_type_min_size(gsl_interp_steffen));
        }
    } catch (std::exception const&) {
        invalid = 1;
    }
    ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("meta-GGA nonlinear core correction requires setup-supplied positive pseudo-core tau");
    }

    rho_pseudo_core_->zero();
    tau_pseudo_core_->zero();
    auto const& fft = ctx_.spfft<double>();
    std::array<int, 3> dims{fft.dim_x(), fft.dim_y(), fft.dim_z()};
    auto wrap = [](int i, int n) { return (i % n + n) % n; };
    try {
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            auto const& type = unit_cell_.atom_type(iat);
            if (!type.has_ps_core_kinetic_density()) {
                continue;
            }
            std::vector<double> grid(type.num_mt_points());
            for (int ir = 0; ir < type.num_mt_points(); ir++) {
                grid[ir] = type.radial_grid(ir);
            }
            auto core_rho = type.ps_core_charge_density();
            core_rho.resize(grid.size(), 0);
            std::unique_ptr<gsl_spline, decltype(&gsl_spline_free)> rho(
                    gsl_spline_alloc(gsl_interp_steffen, grid.size()), gsl_spline_free);
            std::unique_ptr<gsl_spline, decltype(&gsl_spline_free)> tau(
                    gsl_spline_alloc(gsl_interp_steffen, grid.size()), gsl_spline_free);
            std::unique_ptr<gsl_interp_accel, decltype(&gsl_interp_accel_free)> accel(gsl_interp_accel_alloc(),
                                                                                      gsl_interp_accel_free);
            if (!rho || !tau || !accel) {
                RTE_THROW("failed to allocate pseudo-core interpolation");
            }
            gsl_spline_init(rho.get(), grid.data(), core_rho.data(), grid.size());
            gsl_spline_init(tau.get(), grid.data(), type.ps_core_kinetic_density().data(), grid.size());
            // Sample the fixed radial fields directly. Fourier truncation of a positive
            // core can produce negative tails; monotone interpolation avoids this without clipping.
            for (int i = 0; i < type.num_atoms(); i++) {
                auto position = unit_cell_.atom(type.atom_id(i)).position();
                std::array<int, 3> first, last;
                for (int j = 0; j < 3; j++) {
                    position[j] -= std::floor(position[j]);
                    double length2{0};
                    for (int k = 0; k < 3; k++) {
                        length2 += std::pow(unit_cell_.inverse_lattice_vectors()(j, k), 2);
                    }
                    double extent = grid.back() * std::sqrt(length2);
                    first[j]      = static_cast<int>(std::floor(dims[j] * (position[j] - extent)));
                    last[j]       = static_cast<int>(std::ceil(dims[j] * (position[j] + extent)));
                }
                // Unwrapped grid indices enumerate every periodic image, including overlapping cores.
                for (int z = first[2]; z <= last[2]; z++) {
                    int iz = wrap(z, dims[2]) - fft.local_z_offset();
                    if (iz < 0 || iz >= fft.local_z_length()) {
                        continue;
                    }
                    for (int y = first[1]; y <= last[1]; y++) {
                        for (int x = first[0]; x <= last[0]; x++) {
                            r3::vector<double> dr(double(x) / dims[0] - position[0], double(y) / dims[1] - position[1],
                                                  double(z) / dims[2] - position[2]);
                            double r = unit_cell_.get_cartesian_coordinates(dr).length();
                            if (r > grid.back()) {
                                continue;
                            }
                            int ir = ctx_.fft_grid().index_by_coord(wrap(x, dims[0]), wrap(y, dims[1]), iz);
                            // The tabulated origin value also defines the unresolved interval below r_min.
                            r = std::max(r, grid.front());
                            // Preserve tabulated endpoints exactly, including a zero at the outer boundary.
                            auto sample = [&](gsl_spline const* spline) {
                                if (r == grid.front()) {
                                    return spline->y[0];
                                }
                                if (r == grid.back()) {
                                    return spline->y[grid.size() - 1];
                                }
                                return gsl_spline_eval(spline, r, accel.get());
                            };
                            rho_pseudo_core_->value(ir) += sample(rho.get());
                            tau_pseudo_core_->value(ir) += sample(tau.get());
                        }
                    }
                }
            }
        }
    } catch (std::exception const&) {
        invalid = 1;
    }
    ctx_.comm().allreduce<int, mpi::op_t::max>(&invalid, 1);
    if (invalid) {
        RTE_THROW("failed to sample pseudo-core density and kinetic density");
    }
    rho_pseudo_core_->fft_transform(-1);
    tau_pseudo_core_->fft_transform(-1);
}

namespace {

template <typename T>
std::vector<Spheric_function<function_domain_t::spectral, double>>
atom_kinetic_density(Atom_type const& type__, mdarray<double, 2> const& radial__, radial_wave_function_t storage__,
                     mdarray<std::complex<double>, 3> const& dm__, int num_spins__, int lmax__,
                     std::vector<int> const& radial_sizes__ = {}, mdarray<double, 2> const& derivatives__ = {})
{
    Spheric_kinetic_density_operator<T> op(type__.radial_grid(), type__.indexb(), radial__, lmax__, storage__,
                                           radial_sizes__, derivatives__);
    std::vector<Spheric_function<function_domain_t::spectral, double>> result;
    int n = type__.indexb().size();
    mdarray<std::complex<double>, 2> spin_dm({n, n});
    for (int ispn = 0; ispn < num_spins__; ispn++) {
        result.emplace_back(sf::lmmax(lmax__), type__.radial_grid());
        result.back().zero();
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                // LAPW stores <c_i^* c_j>; PAW uses the transposed projector convention.
                spin_dm(i, j) = std::is_same_v<T, double> ? dm__(j, i, ispn) : dm__(i, j, ispn);
            }
        }
        op.add_density(spin_dm, result.back());
    }
    return result;
}

} // namespace

void
Density::initial_kinetic_density()
{
    bool meta = ctx_.meta_gga();
    auto& tau = kinetic_density();
    tau.zero();
    auto guess = [&](double n) {
        return 0.3 * std::pow(3 * pi * pi * ctx_.num_spins(), 2.0 / 3.0) * std::pow(std::max(0.0, n), 5.0 / 3.0);
    };
    for (int ir = 0; ir < ctx_.spfft<double>().local_slice_size(); ir++) {
        for (int s = 0; s < ctx_.num_spins(); s++) {
            double n = rho().rg().value(ir);
            if (ctx_.num_spins() == 2) {
                n = 0.5 * (n + (1 - 2 * s) * mag(0).rg().value(ir));
            }
            double t = guess(n);
            tau.scalar().rg().value(ir) += t;
            if (ctx_.num_spins() == 2) {
                tau.vector(0).rg().value(ir) += (1 - 2 * s) * t;
            }
        }
    }
    if (meta) {
        // A Thomas-Fermi guess alone can violate tau >= |grad n|^2/(8 n).
        // Add the von Weizsaecker term only to the initial guess, not orbital tau.
        for (int s = 0; s < ctx_.num_spins(); s++) {
            Smooth_periodic_function<double> n(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
            for (int ir = 0; ir < ctx_.spfft<double>().local_slice_size(); ir++) {
                n.value(ir) = rho().rg().value(ir);
                if (ctx_.num_spins() == 2) {
                    n.value(ir) = 0.5 * (n.value(ir) + (1 - 2 * s) * mag(0).rg().value(ir));
                }
            }
            n.fft_transform(-1);
            auto grad = to_rg(gradient(n));
            for (int ir = 0; ir < ctx_.spfft<double>().local_slice_size(); ir++) {
                double sigma{0};
                for (int x = 0; x < 3; x++) {
                    sigma += std::pow(grad[x].value(ir), 2);
                }
                double tw = n.value(ir) > 0 ? sigma / (8 * n.value(ir)) : 0;
                tau.scalar().rg().value(ir) += tw;
                if (ctx_.num_spins() == 2) {
                    tau.vector(0).rg().value(ir) += (1 - 2 * s) * tw;
                }
            }
        }
    }
    if (ctx_.full_potential()) {
        SHT sht(device_t::CPU, ctx_.lmax_rho());
        for (auto it : unit_cell_.spl_num_atoms()) {
            auto n = transform(sht, rho().mt()[it.i]);
            Ftp m(sht.num_points(), n.radial_grid());
            m.zero();
            if (ctx_.num_spins() == 2) {
                transform(sht, mag(0).mt()[it.i], m);
            }
            std::vector<Ftp> grad_n, grad_m;
            if (meta) {
                auto g = gradient(rho().mt()[it.i]);
                for (int x = 0; x < 3; x++) {
                    grad_n.push_back(transform(sht, g[x]));
                }
                if (ctx_.num_spins() == 2) {
                    auto gm = gradient(mag(0).mt()[it.i]);
                    for (int x = 0; x < 3; x++) {
                        grad_m.push_back(transform(sht, gm[x]));
                    }
                }
            }
            Ftp t(sht.num_points(), n.radial_grid()), delta(sht.num_points(), n.radial_grid());
            t.zero();
            delta.zero();
            for (size_t i = 0; i < n.size(); i++) {
                for (int s = 0; s < ctx_.num_spins(); s++) {
                    double spin_n = ctx_.num_spins() == 1 ? n[i] : 0.5 * (n[i] + (1 - 2 * s) * m[i]);
                    double value  = guess(spin_n);
                    if (meta && spin_n > 0) {
                        double sigma{0};
                        for (int x = 0; x < 3; x++) {
                            double g = grad_n[x][i];
                            if (!grad_m.empty()) {
                                g = 0.5 * (g + (1 - 2 * s) * grad_m[x][i]);
                            }
                            sigma += g * g;
                        }
                        value += sigma / (8 * spin_n);
                    }
                    t[i] += value;
                    delta[i] += (1 - 2 * s) * value;
                }
            }
            transform(sht, t, tau.scalar().mt()[it.i]);
            if (ctx_.num_spins() == 2) {
                transform(sht, delta, tau.vector(0).mt()[it.i]);
            }
        }
        for (int j = 0; j < ctx_.num_spins(); j++) {
            tau.component(j).mt().sync(unit_cell_.spl_num_atoms());
        }
    }
    tau.fft_transform(-1);
}

void
Density::generate_mt_kinetic_density(bool add_core__)
{
    if (!ctx_.full_potential() || !has_kinetic_density() || (add_core__ && !core_kinetic_density_ready_)) {
        RTE_THROW("LAPW kinetic density requires enabled storage and, when requested, solved core states");
    }
    auto& tau = kinetic_density();
    for (auto it : unit_cell_.spl_num_atoms()) {
        auto spin_tau = generate_mt_valence_kinetic_density(it.i);
        for (size_t i = 0; i < spin_tau[0].size(); i++) {
            double up = spin_tau[0][i], dn = ctx_.num_spins() == 2 ? spin_tau[1][i] : 0;
            tau.scalar().mt()[it.i][i] = up + dn;
            if (ctx_.num_spins() == 2) {
                tau.vector(0).mt()[it.i][i] = up - dn;
            }
        }
        if (add_core__) {
            auto const& core = core_kinetic_density(it.i);
            for (int ir = 0; ir < static_cast<int>(core.size()); ir++) {
                tau.scalar().mt()[it.i](0, ir) += core[ir] / y00;
            }
        }
    }
    for (int j = 0; j < ctx_.num_spins(); j++) {
        tau.component(j).mt().sync(unit_cell_.spl_num_atoms());
    }
}

std::vector<Spheric_function<function_domain_t::spectral, double>>
Density::generate_mt_valence_kinetic_density(int ia__) const
{
    if (!ctx_.full_potential() || ctx_.num_mag_dims() == 3) {
        RTE_THROW("muffin-tin kinetic density requires nonmagnetic or collinear LAPW");
    }
    auto const& atom = unit_cell_.atom(ia__);
    auto const& type = atom.type();
    mdarray<double, 2> radial({type.num_mt_points(), type.indexr().size()});
    mdarray<double, 2> derivative({type.num_mt_points(), type.indexr().size()});
    for (auto const& rf : type.indexr()) {
        for (int ir = 0; ir < type.num_mt_points(); ir++) {
            radial(ir, rf.idxrf) = atom.symmetry_class().radial_function(ir, rf.idxrf);
            derivative(ir, rf.idxrf) =
                    atom.symmetry_class().radial_function_derivative(ir, rf.idxrf) / type.radial_grid(ir);
        }
    }
    return atom_kinetic_density<std::complex<double>>(type, radial, radial_wave_function_t::R, density_matrix(ia__),
                                                      ctx_.num_spins(), ctx_.lmax_rho(), {}, derivative);
}

std::array<std::vector<Spheric_function<function_domain_t::spectral, double>>, 2>
Density::generate_paw_valence_kinetic_density(int ia__) const
{
    auto const& type = unit_cell_.atom(ia__).type();
    if (ctx_.full_potential() || !type.is_paw() || ctx_.num_mag_dims() == 3 || type.spin_orbit_coupling()) {
        RTE_THROW("PAW kinetic density requires nonmagnetic or collinear scalar partial waves");
    }
    int lmax = 2 * type.indexr().lmax();
    std::vector<int> ae_sizes, ps_sizes;
    for (auto const& rf : type.indexr()) {
        ae_sizes.push_back(static_cast<int>(type.ae_paw_wf(rf.idxrf).size()));
        ps_sizes.push_back(static_cast<int>(type.ps_paw_wf(rf.idxrf).size()));
    }
    return {atom_kinetic_density<double>(type, type.ae_paw_wfs_array(), radial_wave_function_t::r_times_R,
                                         density_matrix(ia__), ctx_.num_spins(), lmax, ae_sizes),
            atom_kinetic_density<double>(type, type.ps_paw_wfs_array(), radial_wave_function_t::r_times_R,
                                         density_matrix(ia__), ctx_.num_spins(), lmax, ps_sizes)};
}

template <typename T>
std::vector<Smooth_periodic_function<double>>
Density::generate_smooth_kinetic_density(K_point_set const& ks__) const
{
    PROFILE("sirius::Density::generate_smooth_kinetic_density");
    if (ctx_.processing_unit() != device_t::CPU || ctx_.spfft_coarse<T>().processing_unit() != SPFFT_PU_HOST) {
        RTE_THROW("smooth kinetic-energy density currently requires CPU orbitals and FFTs");
    }
    if (ctx_.num_mag_dims() == 3) {
        RTE_THROW("noncollinear kinetic-energy density is not implemented");
    }

    std::vector<Smooth_periodic_function<double>> coarse, result;
    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        coarse.emplace_back(ctx_.spfft_coarse<double>(), ctx_.gvec_coarse_fft_sptr());
        result.emplace_back(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
    }
    int nr = ctx_.spfft_coarse<T>().local_slice_size();
    std::vector<T> tau(nr);
    for (auto it : ks__.spl_num_kpoints()) {
        auto kp = ks__.get<T>(it.i);
        wf::Kinetic_density_operator<T> op(kp->spfft_transform(), *kp->gkvec_fft_sptr());
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            int nbnd = kp->num_occupied_bands(ispn);
            if (nbnd == 0) {
                continue;
            }
            wf::Wave_functions_fft<T> psi(kp->gkvec_fft_sptr(), kp->spinor_wave_functions(), wf::spin_index(ispn),
                                          wf::band_range(0, nbnd), wf::shuffle_to::fft_layout);
            std::fill(tau.begin(), tau.end(), T(0));
            for (int ib = 0; ib < psi.num_wf_local(); ib++) {
                auto band = psi.spl_num_wf().global_index(ib);
                T weight  = kp->weight() * kp->band_occupancy(band, ispn) / unit_cell_.omega();
                op.add_density(weight, psi.at(memory_t::host, 0, wf::band_index(ib)), tau.data());
            }
            for (int ir = 0; ir < nr; ir++) {
                coarse[ispn].value(ir) += tau[ir];
            }
        }
    }

    // Bands and k-points are distributed orthogonally to the real-space FFT slabs.
    auto const& comm = ctx_.gvec_coarse_fft_sptr()->comm_ortho_fft();
    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        auto ptr = nr ? &coarse[ispn].value(0) : nullptr;
        comm.allreduce(ptr, nr);
        coarse[ispn].fft_transform(-1);
        for (int ig = 0; ig < ctx_.gvec_coarse().count(); ig++) {
            result[ispn].f_pw_local(ctx_.gvec().gvec_base_mapping(ig)) = coarse[ispn].f_pw_local(ig);
        }
        result[ispn].fft_transform(1);
    }
    return result;
}

template std::vector<Smooth_periodic_function<double>>
Density::generate_smooth_kinetic_density<double>(K_point_set const&) const;
#ifdef SIRIUS_USE_FP32
template std::vector<Smooth_periodic_function<double>>
Density::generate_smooth_kinetic_density<float>(K_point_set const&) const;
#endif

} // namespace sirius
