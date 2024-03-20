/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file sum_fg_fl_yg.hpp
 *
 *  \brief LAPW specific function to compute sum over plane-wave coefficients and spherical harmonics.
 */

#ifndef __SUM_FG_FL_YG_HPP__
#define __SUM_FG_FL_YG_HPP__

namespace sirius {

/// Compute sum over plane-wave coefficients and spherical harmonics that apperas in Poisson solver and finding of the
/// MT boundary values.
/** The following operation is performed:
 *  \f[
 *    q_{\ell m}^{\alpha} = \sum_{\bf G} 4\pi \rho({\bf G})
 *     e^{i{\bf G}{\bf r}_{\alpha}}i^{\ell}f_{\ell}^{\alpha}(G) Y_{\ell m}^{*}(\hat{\bf G})
 *  \f]
 */
inline auto
sum_fg_fl_yg(Simulation_context const& ctx__, int lmax__, std::complex<double> const* fpw__, mdarray<double, 3>& fl__,
             matrix<std::complex<double>>& gvec_ylm__)
{
    PROFILE("sirius::sum_fg_fl_yg");

    int ngv_loc = ctx__.gvec().count();

    int na_max{0};
    for (int iat = 0; iat < ctx__.unit_cell().num_atom_types(); iat++) {
        na_max = std::max(na_max, ctx__.unit_cell().atom_type(iat).num_atoms());
    }

    const int lmmax = sf::lmmax(lmax__);
    /* resuling matrix */
    mdarray<std::complex<double>, 2> flm({lmmax, ctx__.unit_cell().num_atoms()});

    matrix<std::complex<double>> phase_factors;
    matrix<std::complex<double>> zm;
    matrix<std::complex<double>> tmp;

    switch (ctx__.processing_unit()) {
        case device_t::CPU: {
            auto& mp      = get_memory_pool(memory_t::host);
            phase_factors = matrix<std::complex<double>>({ngv_loc, na_max}, mp);
            zm            = matrix<std::complex<double>>({lmmax, ngv_loc}, mp);
            tmp           = matrix<std::complex<double>>({lmmax, na_max}, mp);
            break;
        }
        case device_t::GPU: {
            auto& mp      = get_memory_pool(memory_t::host);
            auto& mpd     = get_memory_pool(memory_t::device);
            phase_factors = matrix<std::complex<double>>({ngv_loc, na_max}, mpd);
            zm            = matrix<std::complex<double>>({lmmax, ngv_loc}, mp);
            zm.allocate(mpd);
            tmp = matrix<std::complex<double>>({lmmax, na_max}, mp);
            tmp.allocate(mpd);
            break;
        }
    }

    std::vector<std::complex<double>> zil(lmax__ + 1);
    for (int l = 0; l <= lmax__; l++) {
        zil[l] = std::pow(std::complex<double>(0, 1), l);
    }

    for (int iat = 0; iat < ctx__.unit_cell().num_atom_types(); iat++) {
        const int na = ctx__.unit_cell().atom_type(iat).num_atoms();
        ctx__.generate_phase_factors(iat, phase_factors);
        PROFILE_START("sirius::sum_fg_fl_yg|zm");
        #pragma omp parallel for schedule(static)
        for (int igloc = 0; igloc < ngv_loc; igloc++) {
            for (int l = 0, lm = 0; l <= lmax__; l++) {
                std::complex<double> z = fourpi * fl__(l, igloc, iat) * zil[l] * fpw__[igloc];
                for (int m = -l; m <= l; m++, lm++) {
                    zm(lm, igloc) = z * std::conj(gvec_ylm__(lm, igloc));
                }
            }
        }
        PROFILE_STOP("sirius::sum_fg_fl_yg|zm");
        PROFILE_START("sirius::sum_fg_fl_yg|mul");
        switch (ctx__.processing_unit()) {
            case device_t::CPU: {
                la::wrap(la::lib_t::blas)
                        .gemm('N', 'N', lmmax, na, ngv_loc, &la::constant<std::complex<double>>::one(),
                              zm.at(memory_t::host), zm.ld(), phase_factors.at(memory_t::host), phase_factors.ld(),
                              &la::constant<std::complex<double>>::zero(), tmp.at(memory_t::host), tmp.ld());
                break;
            }
            case device_t::GPU: {
                zm.copy_to(memory_t::device);
                la::wrap(la::lib_t::gpublas)
                        .gemm('N', 'N', lmmax, na, ngv_loc, &la::constant<std::complex<double>>::one(),
                              zm.at(memory_t::device), zm.ld(), phase_factors.at(memory_t::device), phase_factors.ld(),
                              &la::constant<std::complex<double>>::zero(), tmp.at(memory_t::device), tmp.ld());
                tmp.copy_to(memory_t::host);
                break;
            }
        }
        PROFILE_STOP("sirius::sum_fg_fl_yg|mul");

        for (int i = 0; i < na; i++) {
            const int ia = ctx__.unit_cell().atom_type(iat).atom_id(i);
            std::copy(&tmp(0, i), &tmp(0, i) + lmmax, &flm(0, ia));
        }
    }

    ctx__.comm().allreduce(&flm(0, 0), (int)flm.size());

    return flm;
}

} // namespace sirius

#endif
