/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file hamiltonian.cpp
 *
 *  \brief Contains definition of sirius::Hamiltonian0 class.
 */

#include "potential/potential.hpp"
#include "local_operator.hpp"
#include "hamiltonian.hpp"

namespace sirius {

// TODO: radial integrals for the potential should be computed here; the problem is that they also can be set
//       externally by the host code

template <typename T>
Hamiltonian0<T>::Hamiltonian0(Potential& potential__, bool precompute_lapw__, bool update_lapw_rf__)
    : ctx_(potential__.ctx())
    , potential_(&potential__)
    , unit_cell_(potential__.ctx().unit_cell())
{
    PROFILE("sirius::Hamiltonian0");

    local_op_ = std::unique_ptr<Local_operator<T>>(
            new Local_operator<T>(ctx_, ctx_.spfft_coarse<T>(), ctx_.gvec_coarse_fft_sptr(), &potential__));

    if (!ctx_.full_potential()) {
        d_op_ = std::unique_ptr<D_operator<T>>(new D_operator<T>(ctx_));
        q_op_ = std::unique_ptr<Q_operator<T>>(new Q_operator<T>(ctx_));
    }
    if (ctx_.full_potential()) {
        if (precompute_lapw__) {
            potential_->generate_pw_coefs();
            potential_->update_atomic_potential();
            if (update_lapw_rf__) {
                ctx_.unit_cell().generate_radial_functions(ctx_.out());
            }
            ctx_.unit_cell().generate_radial_integrals();
        }
        hmt_    = std::vector<mdarray<std::complex<T>, 2>>(ctx_.unit_cell().num_atoms());
        auto pu = ctx_.processing_unit();
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for
            for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
                auto& atom = ctx_.unit_cell().atom(ia);
                auto& type = atom.type();

                int nmt = type.mt_basis_size();

                hmt_[ia] = mdarray<std::complex<T>, 2>({nmt, nmt}, mdarray_label("hmt"));

                /* compute muffin-tin Hamiltonian */
                for (int j2 = 0; j2 < nmt; j2++) {
                    int lm2    = type.indexb(j2).lm;
                    int idxrf2 = type.indexb(j2).idxrf;
                    for (int j1 = 0; j1 <= j2; j1++) {
                        int lm1          = type.indexb(j1).lm;
                        int idxrf1       = type.indexb(j1).idxrf;
                        hmt_[ia](j1, j2) = atom.radial_integrals_sum_L3(spin_block_t::nm, idxrf1, idxrf2,
                                                                        type.gaunt_coefs().gaunt_vector(lm1, lm2));
                        hmt_[ia](j2, j1) = std::conj(hmt_[ia](j1, j2));
                    }
                }
                if (pu == device_t::GPU) {
                    hmt_[ia].allocate(memory_t::device).copy_to(memory_t::device, acc::stream_id(tid));
                }
            }
            if (pu == device_t::GPU) {
                acc::sync_stream(acc::stream_id(tid));
            }
        }
    }
}

template <typename T>
Hamiltonian0<T>::~Hamiltonian0()
{
}

template <typename T>
void
Hamiltonian0<T>::apply_hmt_to_apw(Atom const& atom__, spin_block_t sblock__, int ngv__,
                                  mdarray<std::complex<T>, 2>& alm__, mdarray<std::complex<T>, 2>& halm__) const
{
    auto& type = atom__.type();

    // TODO: this is k-independent and can in principle be precomputed together with radial integrals if memory is
    // available
    // TODO: for spin-collinear case hmt is Hermitian; compute upper triangular part and use zhemm
    mdarray<std::complex<T>, 2> hmt({type.mt_aw_basis_size(), type.mt_aw_basis_size()});
    /* compute the muffin-tin Hamiltonian */
    for (int j2 = 0; j2 < type.mt_aw_basis_size(); j2++) {
        int lm2    = type.indexb(j2).lm;
        int idxrf2 = type.indexb(j2).idxrf;
        for (int j1 = 0; j1 < type.mt_aw_basis_size(); j1++) {
            int lm1    = type.indexb(j1).lm;
            int idxrf1 = type.indexb(j1).idxrf;
            hmt(j1, j2) =
                    atom__.radial_integrals_sum_L3(sblock__, idxrf1, idxrf2, type.gaunt_coefs().gaunt_vector(lm1, lm2));
        }
    }
    la::wrap(la::lib_t::blas)
            .gemm('N', 'T', ngv__, type.mt_aw_basis_size(), type.mt_aw_basis_size(),
                  &la::constant<std::complex<T>>::one(), alm__.at(memory_t::host), alm__.ld(), hmt.at(memory_t::host),
                  hmt.ld(), &la::constant<std::complex<T>>::zero(), halm__.at(memory_t::host), halm__.ld());
}

template <typename T>
void
Hamiltonian0<T>::add_o1mt_to_apw(Atom const& atom__, int num_gkvec__, mdarray<std::complex<T>, 2>& alm__) const
{
    // TODO: optimize for the loop layout using blocks of G-vectors
    auto& type = atom__.type();
    std::vector<std::complex<T>> alm(type.mt_aw_basis_size());
    std::vector<std::complex<T>> oalm(type.mt_aw_basis_size());
    for (int ig = 0; ig < num_gkvec__; ig++) {
        for (int j = 0; j < type.mt_aw_basis_size(); j++) {
            alm[j] = oalm[j] = alm__(ig, j);
        }
        for (int j = 0; j < type.mt_aw_basis_size(); j++) {
            int l     = type.indexb(j).am.l();
            int lm    = type.indexb(j).lm;
            int idxrf = type.indexb(j).idxrf;
            for (int order = 0; order < type.aw_order(l); order++) {
                int j1     = type.indexb().index_by_lm_order(lm, order);
                int idxrf1 = type.indexr().index_of(angular_momentum(l), order);
                oalm[j] += static_cast<const T>(atom__.symmetry_class().o1_radial_integral(idxrf, idxrf1)) * alm[j1];
            }
        }
        for (int j = 0; j < type.mt_aw_basis_size(); j++) {
            alm__(ig, j) = oalm[j];
        }
    }
}

template <typename T>
void
Hamiltonian0<T>::apply_bmt(wf::Wave_functions<T>& psi__, std::vector<wf::Wave_functions<T>>& bpsi__) const
{
    mdarray<std::complex<T>, 3> zm(
            {unit_cell_.max_mt_basis_size(), unit_cell_.max_mt_basis_size(), ctx_.num_mag_dims()});

    for (auto it : psi__.spl_num_atoms()) {
        auto ia           = it.i;
        auto& atom        = unit_cell_.atom(ia);
        int mt_basis_size = atom.type().mt_basis_size();

        zm.zero();

        /* only upper triangular part of zm is computed because it is a hermitian matrix */
        #pragma omp parallel for default(shared)
        for (int xi2 = 0; xi2 < mt_basis_size; xi2++) {
            int lm2    = atom.type().indexb(xi2).lm;
            int idxrf2 = atom.type().indexb(xi2).idxrf;

            for (int i = 0; i < ctx_.num_mag_dims(); i++) {
                for (int xi1 = 0; xi1 <= xi2; xi1++) {
                    int lm1    = atom.type().indexb(xi1).lm;
                    int idxrf1 = atom.type().indexb(xi1).idxrf;

                    zm(xi1, xi2, i) = atom.type().gaunt_coefs().sum_L3_gaunt(
                            lm1, lm2, atom.b_radial_integrals(idxrf1, idxrf2, i));
                }
            }
        }
        /* compute bwf = B_z*|wf_j> */
        la::wrap(la::lib_t::blas)
                .hemm('L', 'U', mt_basis_size, ctx_.num_fv_states(), &la::constant<std::complex<T>>::one(),
                      zm.at(memory_t::host), zm.ld(), &psi__.mt_coeffs(0, it.li, wf::spin_index(0), wf::band_index(0)),
                      psi__.ld(), &la::constant<std::complex<T>>::zero(),
                      &bpsi__[0].mt_coeffs(0, it.li, wf::spin_index(0), wf::band_index(0)), bpsi__[0].ld());

        /* compute bwf = (B_x - iB_y)|wf_j> */
        if (bpsi__.size() == 3) {
            /* reuse first (z) component of zm matrix to store (B_x - iB_y) */
            for (int xi2 = 0; xi2 < mt_basis_size; xi2++) {
                for (int xi1 = 0; xi1 <= xi2; xi1++) {
                    zm(xi1, xi2, 0) = zm(xi1, xi2, 1) - std::complex<T>(0, 1) * zm(xi1, xi2, 2);
                }

                /* remember: zm for x,y,z, components of magnetic field is hermitian and we computed
                 * only the upper triangular part */
                for (int xi1 = xi2 + 1; xi1 < mt_basis_size; xi1++) {
                    zm(xi1, xi2, 0) = std::conj(zm(xi2, xi1, 1)) - std::complex<T>(0, 1) * std::conj(zm(xi2, xi1, 2));
                }
            }

            la::wrap(la::lib_t::blas)
                    .gemm('N', 'N', mt_basis_size, ctx_.num_fv_states(), mt_basis_size,
                          &la::constant<std::complex<T>>::one(), zm.at(memory_t::host), zm.ld(),
                          &psi__.mt_coeffs(0, it.li, wf::spin_index(0), wf::band_index(0)), psi__.ld(),
                          &la::constant<std::complex<T>>::zero(),
                          &bpsi__[2].mt_coeffs(0, it.li, wf::spin_index(0), wf::band_index(0)), bpsi__[2].ld());
        }
    }
}

template <typename T>
void
Hamiltonian0<T>::apply_so_correction(wf::Wave_functions<T>& psi__, std::vector<wf::Wave_functions<T>>& hpsi__) const
{
    PROFILE("sirius::Hamiltonian0::apply_so_correction");

    wf::spin_index s(0);

    for (auto it : psi__.spl_num_atoms()) {
        auto ia    = it.i;
        auto& atom = unit_cell_.atom(ia);
        auto a     = it.li;

        for (int l = 0; l <= atom.type().lmax_apw(); l++) {
            /* number of radial functions for this l */
            int nrf = atom.type().indexr().max_order(l);

            for (int order1 = 0; order1 < nrf; order1++) {
                for (int order2 = 0; order2 < nrf; order2++) {
                    T sori = atom.symmetry_class().so_radial_integral(l, order1, order2);

                    for (int m = -l; m <= l; m++) {
                        int idx1 = atom.type().indexb_by_l_m_order(l, m, order1);
                        int idx2 = atom.type().indexb_by_l_m_order(l, m, order2);
                        int idx3 = (m + l != 0) ? atom.type().indexb_by_l_m_order(l, m - 1, order2) : 0;
                        // int idx4 = (m - l != 0) ? atom.type().indexb_by_l_m_order(l, m + 1, order2) : 0;

                        for (int ist = 0; ist < ctx_.num_fv_states(); ist++) {
                            wf::band_index b(ist);
                            auto z1 = psi__.mt_coeffs(idx2, a, s, b) * T(m) * sori;
                            /* u-u part */
                            hpsi__[0].mt_coeffs(idx1, a, s, b) += z1;
                            /* d-d part */
                            hpsi__[1].mt_coeffs(idx1, a, s, b) -= z1;
                            /* apply L_{-} operator; u-d part */
                            if (m + l) {
                                hpsi__[2].mt_coeffs(idx1, a, s, b) +=
                                        psi__.mt_coeffs(idx3, a, s, b) * sori * std::sqrt(T(l * (l + 1) - m * (m - 1)));
                            }
                            /* for the d-u part */
                            ///* apply L_{+} operator */
                            // if (m - l) {
                            //    hpsi[3].mt_coeffs(0).prime().at(memory_t::host, offset + idx1, ist) +=
                            //        fv_states__.mt_coeffs(0).prime().at(memory_t::host, offset + idx4, ist) *
                            //            sori * std::sqrt(double(l * (l + 1) - m * (m + 1)));
                            //}
                        }
                    }
                }
            }
        }
    }
}

template class Hamiltonian0<double>;
#ifdef SIRIUS_USE_FP32
template class Hamiltonian0<float>;
#endif

} // namespace sirius
