// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file symmetrize.hpp
 *
 *  \brief Symmetrize scalar and vector functions.
 */

#ifndef __SYMMETRIZE_HPP__
#define __SYMMETRIZE_HPP__

#include "crystal_symmetry.hpp"
#include "fft/gvec.hpp"
#include "SDDK/omp.hpp"
#include "typedefs.hpp"
#include "sht/sht.hpp"
#include "utils/profiler.hpp"
#include "utils/rte.hpp"
#include "function3d/spheric_function_set.hpp"

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
 */
inline void
symmetrize(Crystal_symmetry const& sym__, fft::Gvec_shells const& gvec_shells__,
           sddk::mdarray<std::complex<double>, 3> const& sym_phase_factors__, std::complex<double>* f_pw__,
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

inline void
symmetrize(Crystal_symmetry const& sym__, mpi::Communicator const& comm__, int num_mag_dims__,
        std::vector<Spheric_function_set<double>*> frlm__)
{
    PROFILE("sirius::symmetrize_function|flm");

    /* first (scalar) component is always available */
    auto& frlm = *frlm__[0];

    /* compute maximum lm size */
    int lmmax{0};
    for (auto ia : frlm.atoms()) {
        lmmax = std::max(lmmax, frlm[ia].angular_domain_size());
    }
    int lmax = utils::lmax(lmmax);

    /* split atoms between MPI ranks */
    sddk::splindex<sddk::splindex_t::block> spl_atoms(frlm.atoms().size(), comm__.size(), comm__.rank());

    /* space for real Rlm rotation matrix */
    sddk::mdarray<double, 2> rotm(lmmax, lmmax);

    /* symmetry-transformed functions */
    sddk::mdarray<double, 4> fsym_loc(lmmax, frlm.unit_cell().max_num_mt_points(), num_mag_dims__ + 1,
            spl_atoms.local_size());
    fsym_loc.zero();

    sddk::mdarray<double, 3> ftmp(lmmax, frlm.unit_cell().max_num_mt_points(), num_mag_dims__ + 1);

    double alpha = 1.0 / sym__.size();

    /* loop over crystal symmetries */
    for (int i = 0; i < sym__.size(); i++) {
        /* full space-group symmetry operation is S{R|t} */
        auto S = sym__[i].spin_rotation;
        /* compute Rlm rotation matrix */
        sht::rotation_matrix(lmax, sym__[i].spg_op.euler_angles, sym__[i].spg_op.proper, rotm);

        for (int ialoc = 0; ialoc < spl_atoms.local_size(); ialoc++) {
            /* get global index of the atom */
            int ia = frlm.atoms()[spl_atoms[ialoc]];
            int lmmax_ia = frlm[ia].angular_domain_size();
            int nrmax_ia = frlm.unit_cell().atom(ia).num_mt_points();
            int ja = sym__[i].spg_op.inv_sym_atom[ia];
            /* apply {R|t} part of symmetry operation to all components */
            for (int j = 0; j < num_mag_dims__ + 1; j++) {
                la::wrap(la::lib_t::blas).gemm('N', 'N', lmmax_ia, nrmax_ia, lmmax_ia, &alpha,
                    rotm.at(sddk::memory_t::host), rotm.ld(), (*frlm__[j])[ja].at(sddk::memory_t::host),
                    (*frlm__[j])[ja].ld(), &la::constant<double>::zero(),
                    ftmp.at(sddk::memory_t::host, 0, 0, j), ftmp.ld());
            }
            /* always symmetrize the scalar component */
            for (int ir = 0; ir < nrmax_ia; ir++) {
                for (int lm = 0; lm < lmmax_ia; lm++) {
                    fsym_loc(lm, ir, 0, ialoc) += ftmp(lm, ir, 0);
                }
            }
            /* apply S part to [0, 0, z] collinear vector */
            if (num_mag_dims__ == 1) {
                for (int ir = 0; ir < nrmax_ia; ir++) {
                    for (int lm = 0; lm < lmmax_ia; lm++) {
                        fsym_loc(lm, ir, 1, ialoc) += ftmp(lm, ir, 1) * S(2, 2);
                    }
                }
            }
            /* apply 3x3 S-matrix to [x, y, z] vector */
            if (num_mag_dims__ == 3) {
                for (int k : {0, 1, 2}) {
                    for (int j : {0, 1, 2}) {
                        for (int ir = 0; ir < nrmax_ia; ir++) {
                            for (int lm = 0; lm < lmmax_ia; lm++) {
                                fsym_loc(lm, ir, 1 + k, ialoc) += ftmp(lm, ir, 1 + j) * S(k, j);
                            }
                        }
                    }
                }
            }
        }
    }

    /* gather full function */
    double* sbuf = spl_atoms.local_size() ? fsym_loc.at(sddk::memory_t::host) : nullptr;
    auto ld = static_cast<int>(fsym_loc.size(0) * fsym_loc.size(1) * fsym_loc.size(2));

    sddk::mdarray<double, 4> fsym_glob(lmmax, frlm.unit_cell().max_num_mt_points(), num_mag_dims__ + 1,
            frlm.atoms().size());

    comm__.allgather(sbuf, fsym_glob.at(sddk::memory_t::host), ld * spl_atoms.local_size(),
            ld * spl_atoms.global_offset());

    /* copy back the result */
    for (int i = 0; i < static_cast<int>(frlm.atoms().size()); i++) {
        int ia = frlm.atoms()[i];
        for (int j = 0; j < num_mag_dims__ + 1; j++) {
            for (int ir = 0; ir < frlm.unit_cell().atom(ia).num_mt_points(); ir++) {
                for (int lm = 0; lm < frlm[ia].angular_domain_size(); lm++) {
                    (*frlm__[j])[ia](lm, ir) = fsym_glob(lm, ir, j, i);
                }
            }
        }
    }
}

/// Symmetrize density or occupancy matrix according to a given list of basis functions.
/** Density matrix arises in LAPW or PW methods. In PW it is computed in the basis of beta-projectors. Occupancy
 *  matrix is computed for the Hubbard-U correction. In both cases the matrix has the same structure and is
 *  symmetrized in the same way The symmetrization does depend explicitly on the beta or wfc. The last
 *  parameter is on when the atom has spin-orbit coupling and hubbard correction in
 *  that case, we must skip half of the indices because of the averaging of the
 *  radial integrals over the total angular momentum
 */
inline void
symmetrize(const sddk::mdarray<std::complex<double>, 4>& ns_, basis_functions_index const& indexb, const int ia, const int ja,
           const int ndm, sddk::mdarray<double, 2> const& rotm, sddk::mdarray<std::complex<double>, 2> const& spin_rot_su2,
           sddk::mdarray<std::complex<double>, 4>& dm_, const bool hubbard_)
{
    for (int xi1 = 0; xi1 < indexb.size(); xi1++) {
        int l1  = indexb[xi1].l;
        int lm1 = indexb[xi1].lm;
        int o1  = indexb[xi1].order;

        if ((hubbard_) && (xi1 >= (2 * l1 + 1))) {
            break;
        }

        for (int xi2 = 0; xi2 < indexb.size(); xi2++) {
            int l2                                       = indexb[xi2].l;
            int lm2                                      = indexb[xi2].lm;
            int o2                                       = indexb[xi2].order;
            std::array<std::complex<double>, 3> dm_rot_spatial = {0, 0, 0};

            //} the hubbard treatment when spin orbit coupling is present is
            // foundamentally wrong since we consider the full hubbard
            // correction with a averaged wave function (meaning we neglect the
            // L.S correction within hubbard). A better option (although still
            // wrong from physics pov) would be to consider a multi orbital case.

            if ((hubbard_) && (xi2 >= (2 * l2 + 1))) {
                break;
            }

            //      if (l1 == l2) {
            // the rotation matrix of the angular momentum is block
            // diagonal and does not couple different l.
            for (int j = 0; j < ndm; j++) {
                for (int m3 = -l1; m3 <= l1; m3++) {
                    int lm3 = utils::lm(l1, m3);
                    int xi3 = indexb.index_by_lm_order(lm3, o1);
                    for (int m4 = -l2; m4 <= l2; m4++) {
                        int lm4 = utils::lm(l2, m4);
                        int xi4 = indexb.index_by_lm_order(lm4, o2);
                        dm_rot_spatial[j] += ns_(xi3, xi4, j, ia) * rotm(lm1, lm3) * rotm(lm2, lm4);
                    }
                }
            }

            /* magnetic symmetrization */
            if (ndm == 1) {
                dm_(xi1, xi2, 0, ja) += dm_rot_spatial[0];
            } else {
                std::complex<double> spin_dm[2][2] = {{dm_rot_spatial[0], dm_rot_spatial[2]},
                                                {std::conj(dm_rot_spatial[2]), dm_rot_spatial[1]}};

                /* spin blocks of density matrix are: uu, dd, ud
                   the mapping from linear index (0, 1, 2) of density matrix components is:
                   for the first spin index: k & 1, i.e. (0, 1, 2) -> (0, 1, 0)
                   for the second spin index: min(k, 1), i.e. (0, 1, 2) -> (0, 1, 1)
                */
                for (int k = 0; k < ndm; k++) {
                    for (int is = 0; is < 2; is++) {
                        for (int js = 0; js < 2; js++) {
                            dm_(xi1, xi2, k, ja) +=
                                spin_rot_su2(k & 1, is) * spin_dm[is][js] * std::conj(spin_rot_su2(std::min(k, 1), js));
                        }
                    }
                }
            }
        }
    }
}

inline void
symmetrize(std::function<sddk::mdarray<std::complex<double>, 3>&(int ia__)> dm__, int num_mag_comp__,
           Crystal_symmetry const& sym__,
           std::function<sirius::experimental::basis_functions_index const*(int)> indexb__)
{
    /* quick exit */
    if (sym__.size() == 1) {
        return;
    }

    std::vector<sddk::mdarray<std::complex<double>, 3>> dmsym(sym__.num_atoms());
    for (int ia = 0; ia < sym__.num_atoms(); ia++) {
        int iat = sym__.atom_type(ia);
        if (indexb__(iat)) {
            dmsym[ia] = sddk::mdarray<std::complex<double>, 3>(indexb__(iat)->size(), indexb__(iat)->size(), 4);
            dmsym[ia].zero();
        }
    }

    int lmax{0};
    for (int iat = 0; iat < sym__.num_atom_types(); iat++) {
        if (indexb__(iat)) {
            lmax = std::max(lmax, indexb__(iat)->indexr().lmax());
        }
    }

    /* loop over symmetry operations */
    for (int isym = 0; isym < sym__.size(); isym++) {
        int pr            = sym__[isym].spg_op.proper;
        auto eang         = sym__[isym].spg_op.euler_angles;
        auto rotm         = sht::rotation_matrix<double>(lmax, eang, pr);
        auto spin_rot_su2 = rotation_matrix_su2(sym__[isym].spin_rotation);

        for (int ia = 0; ia < sym__.num_atoms(); ia++) {
            int iat = sym__.atom_type(ia);

            if (!indexb__(iat)) {
                continue;
            }

            int ja = sym__[isym].spg_op.inv_sym_atom[ia];

            auto& indexb = *indexb__(iat);
            auto& indexr = indexb.indexr();

            int mmax = 2 * indexb.indexr().lmax() + 1;
            sddk::mdarray<std::complex<double>, 3> dm_ia(mmax, mmax, num_mag_comp__);

            /* loop over radial functions */
            for (int idxrf1 = 0; idxrf1 < indexr.size(); idxrf1++) {
                /* angular momentum of radial function */
                auto am1     = indexr.am(idxrf1);
                auto ss1     = am1.subshell_size();
                auto offset1 = indexb.offset(idxrf1);
                for (int idxrf2 = 0; idxrf2 < indexr.size(); idxrf2++) {
                    /* angular momentum of radial function */
                    auto am2     = indexr.am(idxrf2);
                    auto ss2     = am2.subshell_size();
                    auto offset2 = indexb.offset(idxrf2);

                    dm_ia.zero();
                    for (int j = 0; j < num_mag_comp__; j++) {
                        /* apply spatial rotation */
                        for (int m1 = 0; m1 < ss1; m1++) {
                            for (int m2 = 0; m2 < ss2; m2++) {
                                for (int m1p = 0; m1p < ss1; m1p++) {
                                    for (int m2p = 0; m2p < ss2; m2p++) {
                                        dm_ia(m1, m2, j) += rotm[am1.l()](m1, m1p) *
                                                            dm__(ja)(offset1 + m1p, offset2 + m2p, j) *
                                                            rotm[am2.l()](m2, m2p);
                                    }
                                }
                            }
                        }
                    }
                    /* magnetic symmetry */
                    if (num_mag_comp__ == 1) { /* trivial non-magnetic case */
                        for (int m1 = 0; m1 < ss1; m1++) {
                            for (int m2 = 0; m2 < ss2; m2++) {
                                dmsym[ia](m1 + offset1, m2 + offset2, 0) += dm_ia(m1, m2, 0);
                            }
                        }
                    } else {
                        int const map_s[3][2] = {{0, 0}, {1, 1}, {0, 1}};
                        for (int j = 0; j < num_mag_comp__; j++) {
                            int s1 = map_s[j][0];
                            int s2 = map_s[j][1];

                            for (int m1 = 0; m1 < ss1; m1++) {
                                for (int m2 = 0; m2 < ss2; m2++) {
                                    std::complex<double> dm[2][2] = {{dm_ia(m1, m2, 0), 0}, {0, dm_ia(m1, m2, 1)}};
                                    if (num_mag_comp__ == 3) {
                                        dm[0][1] = dm_ia(m1, m2, 2);
                                        dm[1][0] = std::conj(dm[0][1]);
                                    }

                                    for (int s1p = 0; s1p < 2; s1p++) {
                                        for (int s2p = 0; s2p < 2; s2p++) {
                                            dmsym[ia](m1 + offset1, m2 + offset2, j) +=
                                                spin_rot_su2(s1, s1p) * dm[s1p][s2p] * std::conj(spin_rot_su2(s2, s2p));
                                        }
                                    }
                                }
                            }
                        }
                        if (num_mag_comp__ == 3) {
                            for (int m1 = 0; m1 < ss1; m1++) {
                                for (int m2 = 0; m2 < ss2; m2++) {
                                    dmsym[ia](m1 + offset1, m2 + offset2, 3) =
                                        std::conj(dmsym[ia](m1 + offset1, m2 + offset2, 2));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    double alpha = 1.0 / sym__.size();

    for (int ia = 0; ia < sym__.num_atoms(); ia++) {
        int iat = sym__.atom_type(ia);
        if (indexb__(iat)) {
            for (size_t i = 0; i < dm__(ia).size(); i++) {
                dm__(ia)[i] = dmsym[ia][i] * alpha;
            }
        }
    }
}

} // namespace sirius

#endif // __SYMMETRIZE_HPP__
