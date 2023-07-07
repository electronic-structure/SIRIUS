#ifndef __SYMMETRIZE_PW_FUNCTION_HPP__
#define __SYMMETRIZE_PW_FUNCTION_HPP__

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
    \param [inout] f_pw           Scalar function.
    \param [inout] x_pw           X-component of the vector function.
    \param [inout] y_pw           Y-component of the vector function.
    \param [inout] z_pw           Z-component of the vector function.
 */
inline void
symmetrize(Crystal_symmetry const& sym__, fft::Gvec_shells const& gvec_shells__,
           sddk::mdarray<std::complex<double>, 3> const& sym_phase_factors__,
           std::complex<double>* f_pw__,
           std::complex<double>* x_pw__, std::complex<double>* y_pw__, std::complex<double>* z_pw__)
{
    PROFILE("sirius::symmetrize|fpw");

    auto f_pw = f_pw__ ? gvec_shells__.remap_forward(f_pw__) : std::vector<std::complex<double>>();
    auto x_pw = x_pw__ ? gvec_shells__.remap_forward(x_pw__) : std::vector<std::complex<double>>();
    auto y_pw = y_pw__ ? gvec_shells__.remap_forward(y_pw__) : std::vector<std::complex<double>>();
    auto z_pw = z_pw__ ? gvec_shells__.remap_forward(z_pw__) : std::vector<std::complex<double>>();

    /* local number of G-vectors in a distribution with complete G-vector shells */
    int ngv = gvec_shells__.gvec_count_remapped();

    auto sym_f_pw = f_pw__ ? std::vector<std::complex<double>>(ngv, 0) : std::vector<std::complex<double>>();
    auto sym_x_pw = x_pw__ ? std::vector<std::complex<double>>(ngv, 0) : std::vector<std::complex<double>>();
    auto sym_y_pw = y_pw__ ? std::vector<std::complex<double>>(ngv, 0) : std::vector<std::complex<double>>();
    auto sym_z_pw = z_pw__ ? std::vector<std::complex<double>>(ngv, 0) : std::vector<std::complex<double>>();

    bool is_non_collin = ((x_pw__ != nullptr) && (y_pw__ != nullptr) && (z_pw__ != nullptr));

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

                    /* check the reduced G-vector */
                    if (ig1 == -1) {
                        G1 = G1 * (-1);
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
                        if (f_pw__) {
                            symf += std::conj(f_pw[ig1]) * phase;
                        }
                        if (!is_non_collin && z_pw__) {
                            symz += std::conj(z_pw[ig1]) * phase * S(2, 2);
                        }
                        if (is_non_collin) {
                            auto v = r3::dot(S, r3::vector<std::complex<double>>({x_pw[ig1], y_pw[ig1], z_pw[ig1]}));
                            symx += std::conj(v[0]) * phase;
                            symy += std::conj(v[1]) * phase;
                            symz += std::conj(v[2]) * phase;
                        }
                    } else {
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
                        RTE_ASSERT(ig1 >= 0 && ig1 < ngv);
                        if (f_pw__) {
                            symf += f_pw[ig1] * phase;
                        }
                        if (!is_non_collin && z_pw__) {
                            symz += z_pw[ig1] * phase * S(2, 2);
                        }
                        if (is_non_collin) {
                            auto v = r3::dot(S, r3::vector<std::complex<double>>({x_pw[ig1], y_pw[ig1], z_pw[ig1]}));
                            symx += v[0] * phase;
                            symy += v[1] * phase;
                            symz += v[2] * phase;
                        }
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
                        if (f_pw__) {
                            symf1 = symf * phase;
                        }
                        if (!is_non_collin && z_pw__) {
                            symz1 = symz * phase * S(2, 2);
                        }
                        if (is_non_collin) {
                            auto v = r3::dot(S, r3::vector<std::complex<double>>({symx, symy, symz}));
                            symx1  = v[0] * phase;
                            symy1  = v[1] * phase;
                            symz1  = v[2] * phase;
                        }

                        if (is_done[ig1]) {
                            if (f_pw__) {
                                /* check that another symmetry operation leads to the same coefficient */
                                if (utils::abs_diff(sym_f_pw[ig1], symf1) > eps) {
                                    std::stringstream s;
                                    s << "inconsistent symmetry operation" << std::endl
                                      << "  existing value : " << sym_f_pw[ig1] << std::endl
                                      << "  computed value : " << symf1 << std::endl
                                      << "  difference: " << std::abs(sym_f_pw[ig1] - symf1) << std::endl;
                                    RTE_THROW(s);
                                }
                            }
                            if (!is_non_collin && z_pw__) {
                                if (utils::abs_diff(sym_z_pw[ig1], symz1) > eps) {
                                    std::stringstream s;
                                    s << "inconsistent symmetry operation" << std::endl
                                      << "  existing value : " << sym_z_pw[ig1] << std::endl
                                      << "  computed value : " << symz1 << std::endl
                                      << "  difference: " << std::abs(sym_z_pw[ig1] - symz1) << std::endl;
                                    RTE_THROW(s);
                                }
                            }
                            if (is_non_collin) {
                                if (utils::abs_diff(sym_x_pw[ig1], symx1) > eps ||
                                    utils::abs_diff(sym_y_pw[ig1], symy1) > eps ||
                                    utils::abs_diff(sym_z_pw[ig1], symz1) > eps) {

                                    RTE_THROW("inconsistent symmetry operation");
                                }
                            }
                        } else {
                            if (f_pw__) {
                                sym_f_pw[ig1] = symf1;
                            }
                            if (!is_non_collin && z_pw__) {
                                sym_z_pw[ig1] = symz1;
                            }
                            if (is_non_collin) {
                                sym_x_pw[ig1] = symx1;
                                sym_y_pw[ig1] = symy1;
                                sym_z_pw[ig1] = symz1;
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
            int ig_rot           = gvec_shells__.index_by_gvec(gv_rot);
            std::complex<double> phase = std::conj(phase_factor(isym, gv_rot));

            if (f_pw__ && ig_rot != -1) {
                if (utils::abs_diff(sym_f_pw[ig_rot], sym_f_pw[igloc] * phase) > eps) {
                    std::stringstream s;
                    s << "check of symmetrized PW coefficients failed" << std::endl
                      << "difference : " << utils::abs_diff(sym_f_pw[ig_rot], sym_f_pw[igloc] * phase);
                    RTE_THROW(s);
                }
            }
            if (!is_non_collin && z_pw__ && ig_rot != -1) {
                if (utils::abs_diff(sym_z_pw[ig_rot], sym_z_pw[igloc] * phase * S(2, 2)) > eps) {
                    std::stringstream s;
                    s << "check of symmetrized PW coefficients failed" << std::endl
                      << "difference : " << utils::abs_diff(sym_z_pw[ig_rot], sym_z_pw[igloc] * phase * S(2, 2));
                    RTE_THROW(s);
                }
            }
        }
    }
#endif

    if (f_pw__) {
        gvec_shells__.remap_backward(sym_f_pw, f_pw__);
    }
    if (x_pw__) {
        gvec_shells__.remap_backward(sym_x_pw, x_pw__);
    }
    if (y_pw__) {
        gvec_shells__.remap_backward(sym_y_pw, y_pw__);
    }
    if (z_pw__) {
        gvec_shells__.remap_backward(sym_z_pw, z_pw__);
    }
}


}

#endif
