/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich. All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "density.hpp"
#include "core/wf/kinetic_density_operator.hpp"

namespace sirius {

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
    tau.fft_transform(-1);
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
