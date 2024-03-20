/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file symmetrize_pw_function.hpp
 *
 *  \brief Symmetrize plane-wave coefficients of regular-grid function.
 */

#ifndef __SYMMETRIZE_PW_FUNCTION_HPP__
#define __SYMMETRIZE_PW_FUNCTION_HPP__

#include "function3d/spheric_function.hpp"
#include "function3d/smooth_periodic_function.hpp"

namespace sirius {

/// Symmetrize scalar or vector function.
/** The following operation is performed:
    \f[
      f_{\mathrm{sym}}({\bf r}) = \frac{1}{N_{\mathrm{sym}}}
        \sum_{\hat{\bf S}\hat{\bf P}} \hat {\bf S} \hat {\bf P}f({\bf r})
    \f]
    where \f$ f({\bf r}) \f$ has to be understood as an unsymmetrized scalar or vector function.
    In the case of a scalar function \f$ \hat {\bf S} = 1 \f$. In the case of vector function
    \f$ \hat {\bf S} \f$ is rotation matrix acting on the Cartesian components of the function.
    \f$ \hat {\bf P} = \{{\bf R}|{\bf t}\} \f$ is a spacial part of the full magentic symmetry operatoin acting
    on the real-space coordinates.

    For the function expanded in plane-waves we have:
    \f[
      f_{\mathrm{sym}}({\bf r}) = \frac{1}{N_{\mathrm{sym}}}
        \sum_{\hat{\bf S}\hat{\bf P}} \hat {\bf S} \hat {\bf P}f({\bf r}) =
        \frac{1}{N_{\mathrm{sym}}} \sum_{\hat{\bf S}\hat{\bf P}} \hat {\bf S} \sum_{\bf G}
        f({\bf G}) e^{i{\bf G}\hat{\bf P}^{-1}{\bf r}} = \\
        \frac{1}{N_{\mathrm{sym}}} \sum_{\hat{\bf S}\hat{\bf P}} \sum_{\bf G} \hat {\bf S} f({\bf G})
        e^{i{\bf G}({\bf R}^{-1}{\bf r} - {\bf R}^{-1}{\bf t})} =
        \frac{1}{N_{\mathrm{sym}}} \sum_{\hat{\bf S}\hat{\bf P}} \sum_{\bf G} \hat {\bf S} f({\bf G})
        e^{i{\bf G}'{\bf r}} e^{-i{\bf G}'{\bf t}}
    \f]
    where \f$ {\bf G}' = {\bf G}{\bf R}^{-1} = {\bf R}^{-T}{\bf G} \f$. The last expression establishes the link
    between unsymmetrized plane-wave coefficient at <b>G</b>-vector and symmetrized coefficient at <b>G</b>'. We will
    rewrite the expression using inverse relation \f$ {\bf G} = {\bf R}^{T}{\bf G'} \f$ and summing over <b>G</b>'
    (which is just a permutaion of <b>G</b>):
    \f[
       f_{\mathrm{sym}}({\bf r}) =
        \sum_{\bf G'} e^{i{\bf G}'{\bf r}} \frac{1}{N_{\mathrm{sym}}} \sum_{\hat{\bf S}\hat{\bf P}}
        \hat {\bf S} f({\bf R}^{T}{\bf G'}) e^{-i{\bf G}'{\bf t}}
    \f]
    That gives an expression for the symmetrized plane-wave coefficient at <b>G</b>':
    \f[
      f_{\mathrm{sym}}({\bf G}') = \frac{1}{N_{\mathrm{sym}}} \sum_{\hat{\bf S}\hat{\bf P}}
         \hat {\bf S} f({\bf R}^{T}{\bf G'}) e^{-i{\bf G}'{\bf t}}
    \f]

    Once \f$ f_{\mathrm{sym}}({\bf G}) \f$ has been calculated for a single <b>G</b>, its values at a
    star of <b>G</b> can be calculated using the following relation:
    \f[
      f_{\mathrm{sym}}({\bf r}) = \hat{\bf S}\hat{\bf P} f_{\mathrm{sym}}({\bf r}) =
        \hat{\bf S} f_{\mathrm{sym}}(\hat {\bf P}^{-1}{\bf r})
    \f]
    which leads to the following relation for the plane-wave coefficient:
    \f[
      \sum_{\bf G} f_{\mathrm{sym}}({\bf G})e^{i{\bf G}{\bf r}} =
        \sum_{\bf G} \hat{\bf S}f_{\mathrm{sym}}({\bf G})e^{i{\bf G}\hat{\bf P}^{-1}{\bf r}} =
        \sum_{\bf G} \hat{\bf S}f_{\mathrm{sym}}({\bf G})e^{i{\bf G}{\bf R}^{-1}{\bf r}}
            e^{-i{\bf G}{\bf R}^{-1}{\bf t}} =
        \sum_{\bf G'} \hat{\bf S}f_{\mathrm{sym}}({\bf G})e^{i{\bf G}'{\bf r}}
             e^{-i{\bf G}'{\bf t}} =
        \sum_{\bf G'} \hat{\bf S}f_{\mathrm{sym}}({\bf G'})e^{i{\bf G}'{\bf r}}
    \f]
    and so
    \f[
       f_{\mathrm{sym}}({\bf G}') = \hat{\bf S}f_{\mathrm{sym}}({\bf G})e^{-i{\bf G'}{\bf t}}
    \f]

    \param [in] sym               Description of the crystal symmetry.
    \param [in] gvec_shells       Description of the G-vector shells.
    \param [in] sym_phase_factors Phase factors associated with fractional translations.
    \param [in] num_mag_dims      Number of magnetic dimensions.
    \param [inout] frg            Array of pointers to scalar and vector parts of the filed being symmetrized.
 */
inline void
symmetrize_pw_function(Crystal_symmetry const& sym__, fft::Gvec_shells const& gvec_shells__,
                       mdarray<std::complex<double>, 3> const& sym_phase_factors__, int num_mag_dims__,
                       std::vector<Smooth_periodic_function<double>*> frg__)
{
    PROFILE("sirius::symmetrize_pw_function");

    std::array<std::vector<std::complex<double>>, 4> fpw_remapped;
    std::array<std::vector<std::complex<double>>, 4> fpw_sym;

    /* local number of G-vectors in a distribution with complete G-vector shells */
    int ngv = gvec_shells__.gvec_count_remapped();

    for (int j = 0; j < num_mag_dims__ + 1; j++) {
        fpw_remapped[j] = gvec_shells__.remap_forward(&frg__[j]->f_pw_local(0));
        fpw_sym[j]      = std::vector<std::complex<double>>(ngv, 0);
    }

    std::vector<bool> is_done(ngv, false);

    double norm = 1 / double(sym__.size());

    auto phase_factor = [&](int isym, r3::vector<int> G) {
        return sym_phase_factors__(0, G[0], isym) * sym_phase_factors__(1, G[1], isym) *
               sym_phase_factors__(2, G[2], isym);
    };

    double const eps{1e-9};

    PROFILE_START("sirius::symmetrize|fpw|local");

    #pragma omp parallel
    {
        int nt  = omp_get_max_threads();
        int tid = omp_get_thread_num();

        for (int igloc = 0; igloc < gvec_shells__.gvec_count_remapped(); igloc++) {
            auto G = gvec_shells__.gvec_remapped(igloc);

            int igsh = gvec_shells__.gvec_shell_remapped(igloc);

#if !defined(NDEBUG)
            if (igsh != gvec_shells__.gvec().shell(G)) {
                std::stringstream s;
                s << "wrong index of G-shell" << std::endl
                  << "  G-vector : " << G << std::endl
                  << "  igsh in the remapped set : " << igsh << std::endl
                  << "  igsh in the original set : " << gvec_shells__.gvec().shell(G);
                RTE_THROW(s);
            }
#endif
            /* each thread is working on full shell of G-vectors */
            if (igsh % nt == tid && !is_done[igloc]) {

                std::complex<double> symf(0, 0);
                std::complex<double> symx(0, 0);
                std::complex<double> symy(0, 0);
                std::complex<double> symz(0, 0);

                /* find the symmetrized PW coefficient */

                for (int i = 0; i < sym__.size(); i++) {
                    auto G1 = r3::dot(G, sym__[i].spg_op.R);

                    auto S = sym__[i].spin_rotation;

                    auto phase = std::conj(phase_factor(i, G));

                    /* local index of a rotated G-vector */
                    int ig1 = gvec_shells__.index_by_gvec(G1);

                    bool conj_coeff{false};

                    /* check the reduced G-vector */
                    if (ig1 == -1) {
                        G1         = G1 * (-1);
                        conj_coeff = true;
                    }
                    auto do_conj = (conj_coeff) ? [](std::complex<double> z) { return std::conj(z); }
                                                : [](std::complex<double> z) { return z; };
#if !defined(NDEBUG)
                    if (igsh != gvec_shells__.gvec().shell(G1)) {
                        std::stringstream s;
                        s << "wrong index of G-shell" << std::endl
                          << "  index of G-shell : " << igsh << std::endl
                          << "  symmetry operation : " << sym__[i].spg_op.R << std::endl
                          << "  G-vector : " << G << std::endl
                          << "  rotated G-vector : " << G1 << std::endl
                          << "  G-vector index : " << gvec_shells__.index_by_gvec(G1) << std::endl
                          << "  index of rotated G-vector shell : " << gvec_shells__.gvec().shell(G1);
                        RTE_THROW(s);
                    }
#endif
                    ig1 = gvec_shells__.index_by_gvec(G1);
                    RTE_ASSERT(ig1 >= 0 && ig1 < ngv);

                    if (frg__[0]) {
                        symf += do_conj(fpw_remapped[0][ig1]) * phase;
                    }
                    if (num_mag_dims__ == 1 && frg__[1]) {
                        symz += do_conj(fpw_remapped[1][ig1]) * phase * S(2, 2);
                    }
                    if (num_mag_dims__ == 3) {
                        auto v =
                                r3::dot(S, r3::vector<std::complex<double>>(
                                                   {fpw_remapped[1][ig1], fpw_remapped[2][ig1], fpw_remapped[3][ig1]}));
                        symx += do_conj(v[0]) * phase;
                        symy += do_conj(v[1]) * phase;
                        symz += do_conj(v[2]) * phase;
                    }
                } /* loop over symmetries */

                symf *= norm;
                symx *= norm;
                symy *= norm;
                symz *= norm;

                /* apply symmetry operation and get all other plane-wave coefficients */

                for (int isym = 0; isym < sym__.size(); isym++) {
                    auto S = sym__[isym].spin_rotation;

                    /* get rotated G-vector */
                    auto G1 = r3::dot(sym__[isym].spg_op.invRT, G);
                    /* index of a rotated G-vector */
                    int ig1 = gvec_shells__.index_by_gvec(G1);

                    if (ig1 != -1) {
                        RTE_ASSERT(ig1 >= 0 && ig1 < ngv);
                        auto phase = std::conj(phase_factor(isym, G1));
                        std::complex<double> symf1, symx1, symy1, symz1;
                        if (frg__[0]) {
                            symf1 = symf * phase;
                        }
                        if (num_mag_dims__ == 1 && frg__[1]) {
                            symz1 = symz * phase * S(2, 2);
                        }
                        if (num_mag_dims__ == 3) {
                            auto v = r3::dot(S, r3::vector<std::complex<double>>({symx, symy, symz}));
                            symx1  = v[0] * phase;
                            symy1  = v[1] * phase;
                            symz1  = v[2] * phase;
                        }

                        if (is_done[ig1]) {
                            /* run checks */
                            if (frg__[0]) {
                                /* check that another symmetry operation leads to the same coefficient */
                                if (abs_diff(fpw_sym[0][ig1], symf1) > eps) {
                                    std::stringstream s;
                                    s << "inconsistent symmetry operation" << std::endl
                                      << "  existing value : " << fpw_sym[0][ig1] << std::endl
                                      << "  computed value : " << symf1 << std::endl
                                      << "  difference: " << std::abs(fpw_sym[0][ig1] - symf1) << std::endl;
                                    RTE_THROW(s);
                                }
                            }
                            if (num_mag_dims__ == 1 && frg__[1]) {
                                if (abs_diff(fpw_sym[1][ig1], symz1) > eps) {
                                    std::stringstream s;
                                    s << "inconsistent symmetry operation" << std::endl
                                      << "  existing value : " << fpw_sym[1][ig1] << std::endl
                                      << "  computed value : " << symz1 << std::endl
                                      << "  difference: " << std::abs(fpw_sym[1][ig1] - symz1) << std::endl;
                                    RTE_THROW(s);
                                }
                            }
                            if (num_mag_dims__ == 3) {
                                if (abs_diff(fpw_sym[1][ig1], symx1) > eps || abs_diff(fpw_sym[2][ig1], symy1) > eps ||
                                    abs_diff(fpw_sym[3][ig1], symz1) > eps) {

                                    RTE_THROW("inconsistent symmetry operation");
                                }
                            }
                        } else {
                            if (frg__[0]) {
                                fpw_sym[0][ig1] = symf1;
                            }
                            if (num_mag_dims__ == 1 && frg__[1]) {
                                fpw_sym[1][ig1] = symz1;
                            }
                            if (num_mag_dims__ == 3) {
                                fpw_sym[1][ig1] = symx1;
                                fpw_sym[2][ig1] = symy1;
                                fpw_sym[3][ig1] = symz1;
                            }
                            is_done[ig1] = true;
                        }
                    }
                } /* loop over symmetries */
            }
        } /* loop over igloc */
    }
    PROFILE_STOP("sirius::symmetrize|fpw|local");

#if !defined(NDEBUG)
    for (int igloc = 0; igloc < gvec_shells__.gvec_count_remapped(); igloc++) {
        auto G = gvec_shells__.gvec_remapped(igloc);
        for (int isym = 0; isym < sym__.size(); isym++) {
            auto S      = sym__[isym].spin_rotation;
            auto gv_rot = r3::dot(sym__[isym].spg_op.invRT, G);
            /* index of a rotated G-vector */
            int ig_rot                 = gvec_shells__.index_by_gvec(gv_rot);
            std::complex<double> phase = std::conj(phase_factor(isym, gv_rot));

            if (frg__[0] && ig_rot != -1) {
                auto diff = abs_diff(fpw_sym[0][ig_rot], fpw_sym[0][igloc] * phase);
                if (diff > eps) {
                    std::stringstream s;
                    s << "check of symmetrized PW coefficients failed" << std::endl << "difference : " << diff;
                    RTE_THROW(s);
                }
            }
            if (num_mag_dims__ == 1 && frg__[1] && ig_rot != -1) {
                auto diff = abs_diff(fpw_sym[1][ig_rot], fpw_sym[1][igloc] * phase * S(2, 2));
                if (diff > eps) {
                    std::stringstream s;
                    s << "check of symmetrized PW coefficients failed" << std::endl << "difference : " << diff;
                    RTE_THROW(s);
                }
            }
        }
    }
#endif

    for (int j = 0; j < num_mag_dims__ + 1; j++) {
        gvec_shells__.remap_backward(fpw_sym[j], &frg__[j]->f_pw_local(0));
    }
}

} // namespace sirius

#endif
