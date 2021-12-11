// Copyright (c) 2013-2018 Anton Kozhevnikov, Mathieu Taillefumier, Thomas Schulthess
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

/** \file k_point.cpp
 *
 *  \brief Contains partial implementation of sirius::K_point class.
 */

#include "k_point/k_point.hpp"
#include "hamiltonian/non_local_operator.hpp"
#include "SDDK/wf_inner.hpp"
#include "SDDK/wf_trans.hpp"

namespace sirius {

template <typename T>
void
K_point<T>::initialize()
{
    PROFILE("sirius::K_point::initialize");

    zil_.resize(ctx_.lmax_apw() + 1);
    for (int l = 0; l <= ctx_.lmax_apw(); l++) {
        zil_[l] = std::pow(std::complex<T>(0, 1), l);
    }

    l_by_lm_ = utils::l_by_lm(ctx_.lmax_apw());

    int bs = ctx_.cyclic_block_size();

    /* In case of collinear magnetism we store only non-zero spinor components.
     *
     * non magnetic case:
     * .---.
     * |   |
     * .---.
     *
     * collinear case:
     * .---.          .---.
     * |uu | 0        |uu |
     * .---.---.  ->  .---.
     *   0 |dd |      |dd |
     *     .---.      .---.
     *
     * non collinear case:
     * .-------.
     * |       |
     * .-------.
     * |       |
     * .-------.
     */
    int nst = ctx_.num_bands();

    auto mem_type_evp  = ctx_.std_evp_solver().host_memory_t();
    auto mem_type_gevp = ctx_.gen_evp_solver().host_memory_t();

    /* build a full list of G+k vectors for all MPI ranks */
    generate_gkvec(ctx_.gk_cutoff());
    /* build a list of basis functions */
    generate_gklo_basis();

    if (ctx_.full_potential()) {
        if (ctx_.cfg().control().use_second_variation()) {

            assert(ctx_.num_fv_states() > 0);
            fv_eigen_values_ = sddk::mdarray<double, 1>(ctx_.num_fv_states(), memory_t::host, "fv_eigen_values");

            if (ctx_.need_sv()) {
                /* in case of collinear magnetism store pure up and pure dn components, otherwise store the full matrix
                 */
                for (int is = 0; is < ctx_.num_spinors(); is++) {
                    sv_eigen_vectors_[is] = dmatrix<std::complex<T>>(nst, nst, ctx_.blacs_grid(), bs, bs, mem_type_evp);
                }
            }
            /* allocate fv eien vectors */
            fv_eigen_vectors_slab_ = std::unique_ptr<Wave_functions<T>>(new Wave_functions<T>(
                gkvec_partition(), unit_cell_.num_atoms(),
                [this](int ia) { return unit_cell_.atom(ia).mt_lo_basis_size(); }, ctx_.num_fv_states(),
                ctx_.preferred_memory_t()));

            fv_eigen_vectors_slab_->pw_coeffs(0).prime().zero();
            fv_eigen_vectors_slab_->mt_coeffs(0).prime().zero();
            /* starting guess for wave-functions */
            for (int i = 0; i < ctx_.num_fv_states(); i++) {
                for (int igloc = 0; igloc < gkvec().gvec_count(comm().rank()); igloc++) {
                    int ig = igloc + gkvec().gvec_offset(comm().rank());
                    if (ig == i) {
                        fv_eigen_vectors_slab_->pw_coeffs(0).prime(igloc, i) = 1.0;
                    }
                    if (ig == i + 1) {
                        fv_eigen_vectors_slab_->pw_coeffs(0).prime(igloc, i) = 0.5;
                    }
                    if (ig == i + 2) {
                        fv_eigen_vectors_slab_->pw_coeffs(0).prime(igloc, i) = 0.125;
                    }
                }
            }
            if (ctx_.cfg().iterative_solver().type() == "exact") {
                /* ELPA needs a full matrix of eigen-vectors as it uses it as a work space */
                if (ctx_.gen_evp_solver().type() == ev_solver_t::elpa) {
                    fv_eigen_vectors_ = dmatrix<std::complex<T>>(gklo_basis_size(), gklo_basis_size(),
                                                                 ctx_.blacs_grid(), bs, bs, mem_type_gevp);
                } else {
                    fv_eigen_vectors_ = dmatrix<std::complex<T>>(gklo_basis_size(), ctx_.num_fv_states(),
                                                                 ctx_.blacs_grid(), bs, bs, mem_type_gevp);
                }
            } else {
                int ncomp = ctx_.cfg().iterative_solver().num_singular();
                if (ncomp < 0) {
                    ncomp = ctx_.num_fv_states() / 2;
                }

                singular_components_ = std::unique_ptr<Wave_functions<T>>(
                    new Wave_functions<T>(gkvec_partition(), ncomp, ctx_.preferred_memory_t()));
                singular_components_->pw_coeffs(0).prime().zero();
                /* starting guess for wave-functions */
                for (int i = 0; i < ncomp; i++) {
                    for (int igloc = 0; igloc < gkvec().count(); igloc++) {
                        int ig = igloc + gkvec().offset();
                        if (ig == i) {
                            singular_components_->pw_coeffs(0).prime(igloc, i) = 1.0;
                        }
                        if (ig == i + 1) {
                            singular_components_->pw_coeffs(0).prime(igloc, i) = 0.5;
                        }
                        if (ig == i + 2) {
                            singular_components_->pw_coeffs(0).prime(igloc, i) = 0.125;
                        }
                    }
                }
                if (ctx_.cfg().control().print_checksum()) {
                    singular_components_->print_checksum(device_t::CPU, "singular_components", 0, ncomp);
                }
            }

            fv_states_ = std::unique_ptr<Wave_functions<T>>(new Wave_functions<T>(
                gkvec_partition(), unit_cell_.num_atoms(),
                [this](int ia) { return unit_cell_.atom(ia).mt_basis_size(); }, ctx_.num_fv_states(),
                ctx_.preferred_memory_t()));

            spinor_wave_functions_ = std::make_shared<Wave_functions<T>>(
                gkvec_partition(), unit_cell_.num_atoms(),
                [this](int ia) { return unit_cell_.atom(ia).mt_basis_size(); }, nst, ctx_.preferred_memory_t(),
                ctx_.num_spins());
        } else {
            throw std::runtime_error("not implemented");
        }
    } else {
        spinor_wave_functions_ =
            std::make_shared<Wave_functions<T>>(gkvec_partition(), nst, ctx_.preferred_memory_t(), ctx_.num_spins());
        if (ctx_.hubbard_correction()) {
            /* allocate Hubbard wave-functions */
            auto r                    = unit_cell_.num_hubbard_wf();
            hubbard_wave_functions_S_ = std::unique_ptr<Wave_functions<T>>(new Wave_functions<T>(
                gkvec_partition(), r.first * ctx_.num_spinor_comp(), ctx_.preferred_memory_t(), ctx_.num_spins()));
            hubbard_wave_functions_   = std::unique_ptr<Wave_functions<T>>(new Wave_functions<T>(
                gkvec_partition(), r.first * ctx_.num_spinor_comp(), ctx_.preferred_memory_t(), ctx_.num_spins()));
            atomic_wave_functions_    = std::unique_ptr<Wave_functions<T>>(new Wave_functions<T>(
                gkvec_partition(), r.first * ctx_.num_spinor_comp(), ctx_.preferred_memory_t(), ctx_.num_spins()));
            atomic_wave_functions_S_  = std::unique_ptr<Wave_functions<T>>(new Wave_functions<T>(
                gkvec_partition(), r.first * ctx_.num_spinor_comp(), ctx_.preferred_memory_t(), ctx_.num_spins()));
        }
    }

    update();
}

template <typename T>
void
K_point<T>::generate_hubbard_orbitals()
{
    PROFILE("sirius::K_point::generate_hubbard_orbitals");

    /* phi and s_phi are aliases for the atomic wave functions. They are *not*
     * the hubbard orbitals */

    auto& phi   = hubbard_atomic_wave_functions();
    auto& s_phi = hubbard_atomic_wave_functions_S();

    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        hubbard_wave_functions_S_->pw_coeffs(ispn).prime().zero();
        hubbard_wave_functions_->pw_coeffs(ispn).prime().zero();
        s_phi.pw_coeffs(ispn).prime().zero();
        phi.pw_coeffs(ispn).prime().zero();
    }
    /* total number of Hubbard orbitals */
    auto r = unit_cell_.num_hubbard_wf();

    std::vector<int> atoms;
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        auto& atom_type = unit_cell_.atom(ia).type();
        if (atom_type.hubbard_correction()) {
            atoms.push_back(ia);
        }
    }

    /* generate the initial atomic wavefunctions */
    this->generate_atomic_wave_functions(
        atoms, [&](int iat) { return &ctx_.unit_cell().atom_type(iat).indexb_hub(); }, ctx_.hubbard_wf_ri(), phi);

    // if (ctx_.cfg().hubbard().full_orthogonalization()) {
    //   /* generate the initial atomic wavefunctions */
    //   this->generate_atomic_wave_functions(atoms, [&](int iat){return
    //   &ctx_.unit_cell().atom_type(iat).indexb_hub();},
    //                                        ctx_.hubbard_wf_ri(), *hubbard_wave_functions_);
    // }
    if (ctx_.cfg().control().print_checksum()) {
        phi.print_checksum(device_t::CPU, "phi_hub_init", 0, phi.num_wf());
    }

    if (ctx_.num_spins() == 2) {
        /* copy up component to dn component in collinear case
         * +-------------------------------+
         * |  phi1_{lm}, phi2_{lm}, ...    |
         * +-------------------------------+
         * |  phi1_{lm}, phi2_{lm}, ...    |
         * +-------------------------------+
         *
         * or with offset in non-collinear case
         *
         * +-------------------------------+---------------------------------+
         * |  phi1_{lm}, phi2_{lm}, ...    |              0                  |
         * +-------------------------------+---------------------------------+
         * |           0                   |   phi1_{lm}, phi2_{lm}, ...     |
         * +-------------------------------+---------------------------------+
         */
        phi.copy_from(device_t::CPU, r.first, phi, 0, 0, 1, (ctx_.num_mag_dims() == 3) ? r.first : 0);
    }

    /* check if we have a norm conserving pseudo potential only */
    auto q_op = (unit_cell_.augment()) ? std::unique_ptr<Q_operator<T>>(new Q_operator<T>(ctx_)) : nullptr;

    auto sr = spin_range(ctx_.num_spins() == 2 ? 2 : 0);
    phi.prepare(sr, true);
    hubbard_wave_functions_S_->prepare(sr, false);
    s_phi.prepare(sr, false);
    hubbard_wave_functions_->prepare(sr, false);

    /* compute S|phi> */
    beta_projectors().prepare();
    for (int is = 0; is < ctx_.num_spinors(); is++) {
        /* spin range to apply S-operator.
         * if WFs are non-magnetic, sping range is [0] or [1] - apply to single component
         * if WFs have two components, spin range is [0,1] and S will be aplpied to both components */
        auto sr = ctx_.num_mag_dims() == 3 ? spin_range(2) : spin_range(is);

        sirius::apply_S_operator<std::complex<T>>(ctx_.processing_unit(), sr, 0, phi.num_wf(), beta_projectors(), phi,
                                                  q_op.get(), s_phi);
    }

    if (ctx_.cfg().control().print_checksum()) {
        s_phi.print_checksum(device_t::CPU, "hubbard_atomic_wfc_S", 0, s_phi.num_wf());
        phi.print_checksum(device_t::CPU, "hubbard_atomic_wfc", 0, s_phi.num_wf());
    }

    /* now compute the hubbard wfc from the atomic orbitals */
    orthogonalize_hubbard_orbitals(phi, s_phi, *hubbard_wave_functions_, *hubbard_wave_functions_S_);

    // for (int is = 0; is < ctx_.num_spinors(); is++) {
    //     /* spin range to apply S-operator.
    //      * if WFs are non-magnetic, sping range is [0] or [1] - apply to single component
    //      * if WFs have two components, spin range is [0,1] and S will be aplpied to both components */
    //     auto sr = ctx_.num_mag_dims() == 3 ? spin_range(2) : spin_range(is);

    //     sirius::apply_S_operator<std::complex<T>>(ctx_.processing_unit(), sr, 0, phi.num_wf(), beta_projectors(),
    //                                               *hubbard_wave_functions_, q_op.get(), *hubbard_wave_functions_S_);
    // }

    beta_projectors().dismiss();

    /* all calculations on GPU then we need to copy the final result back to the CPUs */
    hubbard_wave_functions_S_->dismiss(sr, true);
    phi.dismiss(sr, true);
    s_phi.dismiss(sr, true);
    hubbard_wave_functions_->dismiss(sr, true);

    if (ctx_.cfg().control().print_checksum()) {
        hubbard_wave_functions_S_->print_checksum(device_t::CPU, "hubbard_phi_S", 0,
                                                  hubbard_wave_functions_S_->num_wf());
        hubbard_wave_functions_->print_checksum(device_t::CPU, "hubbard_phi", 0, hubbard_wave_functions_->num_wf());
    }
}

template <typename T>
void
K_point<T>::compute_orthogonalization_operator(const int istep, Wave_functions<T>& phi__, Wave_functions<T>& sphi__,
                                               dmatrix<std::complex<T>>& S__, dmatrix<std::complex<T>>& Z__,
                                               std::vector<double>& eigenvalues__)
{
    auto sr        = spin_range(ctx_.num_mag_dims() == 3 ? 2 : istep);
    const int nwfu = phi__.num_wf();

    S__.zero();
    Z__.zero();
    /* compute inner product between full spinors or between indpendent components */
    inner<std::complex<T>>(ctx_.spla_context(), sr, phi__, 0, nwfu, sphi__, 0, nwfu, S__, 0, 0);

    // SPLA should return on CPU as well
    // if (ctx_.processing_unit() == device_t::GPU) {
    //    S.copy_to(memory_t::host);
    //}

    /* create transformation matrix */
    if (ctx_.cfg().hubbard().orthogonalize()) {

        auto ev_solver = Eigensolver_factory("lapack", nullptr);

        ev_solver->solve(nwfu, S__, eigenvalues__.data(), Z__);

        /* build the O^{-1/2} operator */
        for (int i = 0; i < static_cast<int>(eigenvalues__.size()); i++) {
            eigenvalues__[i] = 1.0 / std::sqrt(eigenvalues__[i]);
        }

        /* first compute S_{nm} = E_m Z_{nm} */
        S__.zero();
        for (int l = 0; l < nwfu; l++) {
            for (int m = 0; m < nwfu; m++) {
                for (int n = 0; n < nwfu; n++) {
                    S__(n, m) += eigenvalues__[l] * Z__(n, l) * std::conj(Z__(m, l));
                }
            }
        }
    } else {
        S__.zero();
        for (int l = 0; l < nwfu; l++) {
            S__(l, l) = 1.0 / std::sqrt(S__(l, l).real());
        }
    }
}

template <typename T>
void
K_point<T>::orthogonalize_hubbard_orbitals(Wave_functions<T>& phi__, Wave_functions<T>& sphi__,
                                           Wave_functions<T>& phi_hub__, Wave_functions<T>& phi_hub_S__)
{
    int nwfu = phi__.num_wf();
    auto la  = linalg_t::none;
    auto mt  = memory_t::none;
    switch (ctx_.processing_unit()) {
        case device_t::CPU: {
            la = linalg_t::blas;
            mt = memory_t::host;
            break;
        }
        case device_t::GPU: {
            la = linalg_t::gpublas;
            mt = memory_t::device;
            break;
        }
        default:
            break;
    }

    if (!(ctx_.cfg().hubbard().orthogonalize() || ctx_.cfg().hubbard().normalize())) {
        for (int s = 0; s < ctx_.num_spins(); s++) {
            phi_hub__.copy_from(ctx_.processing_unit(), nwfu, phi__, s, 0, s, 0);
        }
        return;
    }

    /* sphi_hub__ is used as scratch space */
    dmatrix<std::complex<T>> S(nwfu, nwfu);
    /* Z is not used here. it is just a scratch matrix. Howver it is important when calculating the derivative of O_  */
    dmatrix<std::complex<T>> Z(nwfu, nwfu);

    if (ctx_.processing_unit() == device_t::GPU) {
        S.allocate(memory_t::device);
        Z.allocate(memory_t::device);
    }

    std::vector<T> eigenvalues(nwfu, 0.0);

    for (int istep = 0; istep < ctx_.num_spinors(); istep++) {

        auto sr = spin_range(ctx_.num_mag_dims() == 3 ? 2 : istep);

        /* compute inner product between full spinors or between indpendent components */
        compute_orthogonalization_operator(istep, phi__, sphi__, S, Z, eigenvalues);

        if (ctx_.processing_unit() == device_t::GPU) {
            S.copy_to(memory_t::device);
        }

        // if(ctx_.cfg().hubbard().full_orthogonalization()) {
        // dmatrix<std::complex<T>> overlap_(phi_hub__.num_wf(), nwfu);
        // dmatrix<std::complex<T>> O_(phi_hub__.num_wf(), nwfu);
        // overlap_.zero();
        // /* compute inner product between full spinors or between independent components */
        // inner<std::complex<T>>(ctx_.spla_context(),
        //                        sr,
        //                        phi_hub__, 0, phi_hub__.num_wf(),
        //                        sphi__, 0, nwfu,
        //                        overlap_, 0, 0);

        // linalg(la).gemm('N', 'N', phi_hub__.num_wf(), nwfu,
        //                 nwfu, &linalg_const<double_complex>::one(),
        //                 overlap_.at(mt), overlap_.ld(), S.at(mt), S.ld(),
        //                 &linalg_const<double_complex>::zero(), O_.at(mt), O_.ld());

        /* sphi_hub__ is used as scratch memory. We orthogonalize the hubbard
         wave functions (which can be a subset of the orthomic orbitals) by doing this

         overlap = <phi^i_atomic | S | phi^j_hubbard>

         O' = S.overlap = O'_ij

         |phi'_h > = Tr(S.overlap) |phi_atomic>
        */

        // transform<std::complex<T>>(ctx_.spla_context(), sr(), phi__, 0, nwfu, O_, 0, 0, phi_hub__, 0, nwfu);
        //} else {
        transform<std::complex<T>>(ctx_.spla_context(), sr(), phi__, 0, nwfu, S, 0, 0, phi_hub__, 0, nwfu);
        transform<std::complex<T>>(ctx_.spla_context(), sr(), sphi__, 0, nwfu, S, 0, 0, phi_hub_S__, 0, nwfu);
        //}
    }
}

template <typename T>
void
K_point<T>::generate_gkvec(double gk_cutoff__)
{
    PROFILE("sirius::K_point::generate_gkvec");

    if (ctx_.full_potential() && (gk_cutoff__ * unit_cell_.max_mt_radius() > ctx_.lmax_apw()) &&
        ctx_.comm().rank() == 0 && ctx_.verbosity() >= 0) {
        std::stringstream s;
        s << "G+k cutoff (" << gk_cutoff__ << ") is too large for a given lmax (" << ctx_.lmax_apw()
          << ") and a maximum MT radius (" << unit_cell_.max_mt_radius() << ")" << std::endl
          << "suggested minimum value for lmax : " << int(gk_cutoff__ * unit_cell_.max_mt_radius()) + 1;
        WARNING(s);
    }

    if (gk_cutoff__ * 2 > ctx_.pw_cutoff()) {
        std::stringstream s;
        s << "G+k cutoff is too large for a given plane-wave cutoff" << std::endl
          << "  pw cutoff : " << ctx_.pw_cutoff() << std::endl
          << "  doubled G+k cutoff : " << gk_cutoff__ * 2;
        TERMINATE(s);
    }

    gkvec_partition_ = std::unique_ptr<Gvec_partition>(
        new Gvec_partition(this->gkvec(), ctx_.comm_fft_coarse(), ctx_.comm_band_ortho_fft_coarse()));

    gkvec_offset_ = gkvec().gvec_offset(comm().rank());

    const auto fft_type = gkvec_->reduced() ? SPFFT_TRANS_R2C : SPFFT_TRANS_C2C;
    const auto spfft_pu = ctx_.processing_unit() == device_t::CPU ? SPFFT_PU_HOST : SPFFT_PU_GPU;
    auto gv             = gkvec_partition_->get_gvec();
    /* create transformation */
    spfft_transform_.reset(new spfft_transform_type<T>(ctx_.spfft_grid_coarse<T>().create_transform(
        spfft_pu, fft_type, ctx_.fft_coarse_grid()[0], ctx_.fft_coarse_grid()[1], ctx_.fft_coarse_grid()[2],
        ctx_.spfft_coarse<double>().local_z_length(), gkvec_partition_->gvec_count_fft(), SPFFT_INDEX_TRIPLETS,
        gv.at(memory_t::host))));
}

template <typename T>
void
K_point<T>::update()
{
    PROFILE("sirius::K_point::update");

    gkvec_->lattice_vectors(ctx_.unit_cell().reciprocal_lattice_vectors());

    if (ctx_.full_potential()) {
        if (ctx_.cfg().iterative_solver().type() == "exact") {
            alm_coeffs_row_ = std::unique_ptr<Matching_coefficients>(
                new Matching_coefficients(unit_cell_, ctx_.lmax_apw(), num_gkvec_row(), igk_row_, gkvec()));
            alm_coeffs_col_ = std::unique_ptr<Matching_coefficients>(
                new Matching_coefficients(unit_cell_, ctx_.lmax_apw(), num_gkvec_col(), igk_col_, gkvec()));
        }
        alm_coeffs_loc_ = std::unique_ptr<Matching_coefficients>(
            new Matching_coefficients(unit_cell_, ctx_.lmax_apw(), num_gkvec_loc(), igk_loc_, gkvec()));
    }

    if (!ctx_.full_potential()) {
        /* compute |beta> projectors for atom types */
        beta_projectors_ = std::unique_ptr<Beta_projectors<T>>(new Beta_projectors<T>(ctx_, gkvec(), igk_loc_));

        if (ctx_.cfg().iterative_solver().type() == "exact") {
            beta_projectors_row_ = std::unique_ptr<Beta_projectors<T>>(new Beta_projectors<T>(ctx_, gkvec(), igk_row_));
            beta_projectors_col_ = std::unique_ptr<Beta_projectors<T>>(new Beta_projectors<T>(ctx_, gkvec(), igk_col_));
        }

        if (ctx_.hubbard_correction()) {
            generate_hubbard_orbitals();
        }

        // if (false) {
        //    p_mtrx_ = mdarray<double_complex, 3>(unit_cell_.max_mt_basis_size(), unit_cell_.max_mt_basis_size(),
        //    unit_cell_.num_atom_types()); p_mtrx_.zero();

        //    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        //        auto& atom_type = unit_cell_.atom_type(iat);

        //        if (!atom_type.pp_desc().augment) {
        //            continue;
        //        }
        //        int nbf = atom_type.mt_basis_size();
        //        int ofs = atom_type.offset_lo();

        //        matrix<double_complex> qinv(nbf, nbf);
        //        for (int xi1 = 0; xi1 < nbf; xi1++) {
        //            for (int xi2 = 0; xi2 < nbf; xi2++) {
        //                qinv(xi2, xi1) = ctx_.augmentation_op(iat).q_mtrx(xi2, xi1);
        //            }
        //        }
        //        linalg<device_t::CPU>::geinv(nbf, qinv);
        //
        //        /* compute P^{+}*P */
        //        linalg<device_t::CPU>::gemm(2, 0, nbf, nbf, num_gkvec_loc(),
        //                          beta_projectors_->beta_gk_t().at<CPU>(0, ofs), beta_projectors_->beta_gk_t().ld(),
        //                          beta_projectors_->beta_gk_t().at<CPU>(0, ofs), beta_projectors_->beta_gk_t().ld(),
        //                          &p_mtrx_(0, 0, iat), p_mtrx_.ld());
        //        comm().allreduce(&p_mtrx_(0, 0, iat), unit_cell_.max_mt_basis_size() *
        //        unit_cell_.max_mt_basis_size());

        //        for (int xi1 = 0; xi1 < nbf; xi1++) {
        //            for (int xi2 = 0; xi2 < nbf; xi2++) {
        //                qinv(xi2, xi1) += p_mtrx_(xi2, xi1, iat);
        //            }
        //        }
        //        /* compute (Q^{-1} + P^{+}*P)^{-1} */
        //        linalg<device_t::CPU>::geinv(nbf, qinv);
        //        for (int xi1 = 0; xi1 < nbf; xi1++) {
        //            for (int xi2 = 0; xi2 < nbf; xi2++) {
        //                p_mtrx_(xi2, xi1, iat) = qinv(xi2, xi1);
        //            }
        //        }
        //    }
        //}
    }
}

template <typename T>
void
K_point<T>::get_fv_eigen_vectors(mdarray<std::complex<T>, 2>& fv_evec__) const
{
    assert((int)fv_evec__.size(0) >= gklo_basis_size());
    assert((int)fv_evec__.size(1) == ctx_.num_fv_states());
    assert(gklo_basis_size_row() == fv_eigen_vectors_.num_rows_local());

    mdarray<std::complex<T>, 1> tmp(gklo_basis_size_row());

    fv_evec__.zero();

    for (int ist = 0; ist < ctx_.num_fv_states(); ist++) {
        auto loc = fv_eigen_vectors_.spl_col().location(ist);
        if (loc.rank == fv_eigen_vectors_.rank_col()) {
            std::copy(&fv_eigen_vectors_(0, loc.local_index),
                      &fv_eigen_vectors_(0, loc.local_index) + gklo_basis_size_row(), &tmp(0));
        }
        fv_eigen_vectors_.blacs_grid().comm_col().bcast(&tmp(0), gklo_basis_size_row(), loc.rank);
        for (int jloc = 0; jloc < gklo_basis_size_row(); jloc++) {
            int j             = fv_eigen_vectors_.irow(jloc);
            fv_evec__(j, ist) = tmp(jloc);
        }
        fv_eigen_vectors_.blacs_grid().comm_row().allreduce(&fv_evec__(0, ist), gklo_basis_size());
    }
}

//== void K_point::check_alm(int num_gkvec_loc, int ia, mdarray<double_complex, 2>& alm)
//== {
//==     static SHT* sht = NULL;
//==     if (!sht) sht = new SHT(ctx_.lmax_apw());
//==
//==     Atom* atom = unit_cell_.atom(ia);
//==     Atom_type* type = atom->type();
//==
//==     mdarray<double_complex, 2> z1(sht->num_points(), type->mt_aw_basis_size());
//==     for (int i = 0; i < type->mt_aw_basis_size(); i++)
//==     {
//==         int lm = type->indexb(i).lm;
//==         int idxrf = type->indexb(i).idxrf;
//==         double rf = atom->symmetry_class()->radial_function(atom->num_mt_points() - 1, idxrf);
//==         for (int itp = 0; itp < sht->num_points(); itp++)
//==         {
//==             z1(itp, i) = sht->ylm_backward(lm, itp) * rf;
//==         }
//==     }
//==
//==     mdarray<double_complex, 2> z2(sht->num_points(), num_gkvec_loc);
//==     blas<CPU>::gemm(0, 2, sht->num_points(), num_gkvec_loc, type->mt_aw_basis_size(), z1.ptr(), z1.ld(),
//==                     alm.ptr(), alm.ld(), z2.ptr(), z2.ld());
//==
//==     vector3d<double> vc = unit_cell_.get_cartesian_coordinates(unit_cell_.atom(ia)->position());
//==
//==     double tdiff = 0;
//==     for (int igloc = 0; igloc < num_gkvec_loc; igloc++)
//==     {
//==         vector3d<double> gkc = gkvec_cart(igkglob(igloc));
//==         for (int itp = 0; itp < sht->num_points(); itp++)
//==         {
//==             double_complex aw_value = z2(itp, igloc);
//==             vector3d<double> r;
//==             for (int x = 0; x < 3; x++) r[x] = vc[x] + sht->coord(x, itp) * type->mt_radius();
//==             double_complex pw_value = exp(double_complex(0, Utils::scalar_product(r, gkc))) /
//sqrt(unit_cell_.omega());
//==             tdiff += abs(pw_value - aw_value);
//==         }
//==     }
//==
//==     printf("atom : %i  absolute alm error : %e  average alm error : %e\n",
//==            ia, tdiff, tdiff / (num_gkvec_loc * sht->num_points()));
//== }

// Periodic_function<double_complex>* K_point::spinor_wave_function_component(Band* band, int lmax, int ispn, int jloc)
//{
//    Timer t("sirius::K_point::spinor_wave_function_component");
//
//    int lmmax = Utils::lmmax_by_lmax(lmax);
//
//    Periodic_function<double_complex, index_order>* func =
//        new Periodic_function<double_complex, index_order>(ctx_, lmax);
//    func->allocate(ylm_component | it_component);
//    func->zero();
//
//    if (basis_type == pwlo)
//    {
//        if (index_order != radial_angular) error(__FILE__, __LINE__, "wrong order of indices");
//
//        double fourpi_omega = fourpi / sqrt(ctx_.omega());
//
//        for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//        {
//            int igk = igkglob(igkloc);
//            double_complex z1 = spinor_wave_functions_(ctx_.mt_basis_size() + igk, ispn, jloc) * fourpi_omega;
//
//            // TODO: possilbe optimization with zgemm
//            for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//            {
//                int iat = ctx_.atom_type_index_by_id(ctx_.atom(ia)->type_id());
//                double_complex z2 = z1 * gkvec_phase_factors_(igkloc, ia);
//
//                #pragma omp parallel for default(shared)
//                for (int lm = 0; lm < lmmax; lm++)
//                {
//                    int l = l_by_lm_(lm);
//                    double_complex z3 = z2 * zil_[l] * conj(gkvec_ylm_(lm, igkloc));
//                    for (int ir = 0; ir < ctx_.atom(ia)->num_mt_points(); ir++)
//                        func->f_ylm(ir, lm, ia) += z3 * (*sbessel_[igkloc])(ir, l, iat);
//                }
//            }
//        }
//
//        for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//        {
//            Platform::allreduce(&func->f_ylm(0, 0, ia), lmmax * ctx_.max_num_mt_points(),
//                                ctx_.mpi_grid().communicator(1 << band->dim_row()));
//        }
//    }
//
//    for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//    {
//        for (int i = 0; i < ctx_.atom(ia)->type()->mt_basis_size(); i++)
//        {
//            int lm = ctx_.atom(ia)->type()->indexb(i).lm;
//            int idxrf = ctx_.atom(ia)->type()->indexb(i).idxrf;
//            switch (index_order)
//            {
//                case angular_radial:
//                {
//                    for (int ir = 0; ir < ctx_.atom(ia)->num_mt_points(); ir++)
//                    {
//                        func->f_ylm(lm, ir, ia) +=
//                            spinor_wave_functions_(ctx_.atom(ia)->offset_wf() + i, ispn, jloc) *
//                            ctx_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//                    }
//                    break;
//                }
//                case radial_angular:
//                {
//                    for (int ir = 0; ir < ctx_.atom(ia)->num_mt_points(); ir++)
//                    {
//                        func->f_ylm(ir, lm, ia) +=
//                            spinor_wave_functions_(ctx_.atom(ia)->offset_wf() + i, ispn, jloc) *
//                            ctx_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//                    }
//                    break;
//                }
//            }
//        }
//    }
//
//    // in principle, wave function must have an overall e^{ikr} phase factor
//    ctx_.fft().input(num_gkvec(), &fft_index_[0],
//                            &spinor_wave_functions_(ctx_.mt_basis_size(), ispn, jloc));
//    ctx_.fft().transform(1);
//    ctx_.fft().output(func->f_it());
//
//    for (int i = 0; i < ctx_.fft().size(); i++) func->f_it(i) /= sqrt(ctx_.omega());
//
//    return func;
//}

//== void K_point::spinor_wave_function_component_mt(int lmax, int ispn, int jloc, mt_functions<double_complex>& psilm)
//== {
//==     Timer t("sirius::K_point::spinor_wave_function_component_mt");
//==
//==     //int lmmax = Utils::lmmax_by_lmax(lmax);
//==
//==     psilm.zero();
//==
//==     //if (basis_type == pwlo)
//==     //{
//==     //    if (index_order != radial_angular) error(__FILE__, __LINE__, "wrong order of indices");
//==
//==     //    double fourpi_omega = fourpi / sqrt(ctx_.omega());
//==
//==     //    mdarray<double_complex, 2> zm(ctx_.max_num_mt_points(),  num_gkvec_row());
//==
//==     //    for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//==     //    {
//==     //        int iat = ctx_.atom_type_index_by_id(ctx_.atom(ia)->type_id());
//==     //        for (int l = 0; l <= lmax; l++)
//==     //        {
//==     //            #pragma omp parallel for default(shared)
//==     //            for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//==     //            {
//==     //                int igk = igkglob(igkloc);
//==     //                double_complex z1 = spinor_wave_functions_(ctx_.mt_basis_size() + igk, ispn, jloc) *
//fourpi_omega;
//==     //                double_complex z2 = z1 * gkvec_phase_factors_(igkloc, ia) * zil_[l];
//==     //                for (int ir = 0; ir < ctx_.atom(ia)->num_mt_points(); ir++)
//==     //                    zm(ir, igkloc) = z2 * (*sbessel_[igkloc])(ir, l, iat);
//==     //            }
//==     //            blas<CPU>::gemm(0, 2, ctx_.atom(ia)->num_mt_points(), (2 * l + 1), num_gkvec_row(),
//==     //                            &zm(0, 0), zm.ld(), &gkvec_ylm_(Utils::lm_by_l_m(l, -l), 0), gkvec_ylm_.ld(),
//==     //                            &fylm(0, Utils::lm_by_l_m(l, -l), ia), fylm.ld());
//==     //        }
//==     //    }
//==     //    //for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//==     //    //{
//==     //    //    int igk = igkglob(igkloc);
//==     //    //    double_complex z1 = spinor_wave_functions_(ctx_.mt_basis_size() + igk, ispn, jloc) * fourpi_omega;
//==
//==     //    //    // TODO: possilbe optimization with zgemm
//==     //    //    for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//==     //    //    {
//==     //    //        int iat = ctx_.atom_type_index_by_id(ctx_.atom(ia)->type_id());
//==     //    //        double_complex z2 = z1 * gkvec_phase_factors_(igkloc, ia);
//==     //    //
//==     //    //        #pragma omp parallel for default(shared)
//==     //    //        for (int lm = 0; lm < lmmax; lm++)
//==     //    //        {
//==     //    //            int l = l_by_lm_(lm);
//==     //    //            double_complex z3 = z2 * zil_[l] * conj(gkvec_ylm_(lm, igkloc));
//==     //    //            for (int ir = 0; ir < ctx_.atom(ia)->num_mt_points(); ir++)
//==     //    //                fylm(ir, lm, ia) += z3 * (*sbessel_[igkloc])(ir, l, iat);
//==     //    //        }
//==     //    //    }
//==     //    //}
//==
//==     //    for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//==     //    {
//==     //        Platform::allreduce(&fylm(0, 0, ia), lmmax * ctx_.max_num_mt_points(),
//==     //                            ctx_.mpi_grid().communicator(1 << band->dim_row()));
//==     //    }
//==     //}
//==
//==     for (int ia = 0; ia < ctx_.num_atoms(); ia++)
//==     {
//==         for (int i = 0; i < ctx_.atom(ia)->type()->mt_basis_size(); i++)
//==         {
//==             int lm = ctx_.atom(ia)->type()->indexb(i).lm;
//==             int idxrf = ctx_.atom(ia)->type()->indexb(i).idxrf;
//==             for (int ir = 0; ir < ctx_.atom(ia)->num_mt_points(); ir++)
//==             {
//==                 psilm(lm, ir, ia) +=
//==                     spinor_wave_functions_(ctx_.atom(ia)->offset_wf() + i, ispn, jloc) *
//==                     ctx_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//==             }
//==         }
//==     }
//== }

template <typename T>
void
K_point<T>::test_spinor_wave_functions(int use_fft)
{
    STOP();

    //==     if (num_ranks() > 1) error_local(__FILE__, __LINE__, "test of spinor wave functions on multiple ranks is
    //not implemented");
    //==
    //==     std::vector<double_complex> v1[2];
    //==     std::vector<double_complex> v2;
    //==
    //==     if (use_fft == 0 || use_fft == 1) v2.resize(fft_->size());
    //==
    //==     if (use_fft == 0)
    //==     {
    //==         for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) v1[ispn].resize(num_gkvec());
    //==     }
    //==
    //==     if (use_fft == 1)
    //==     {
    //==         for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) v1[ispn].resize(fft_->size());
    //==     }
    //==
    //==     double maxerr = 0;
    //==
    //==     for (int j1 = 0; j1 < ctx_.num_bands(); j1++)
    //==     {
    //==         if (use_fft == 0)
    //==         {
    //==             for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
    //==             {
    //==                 fft_->input(num_gkvec(), gkvec_.index_map(),
    //==                                        &spinor_wave_functions_(unit_cell_.mt_basis_size(), ispn, j1));
    //==                 fft_->transform(1);
    //==                 fft_->output(&v2[0]);
    //==
    //==                 for (int ir = 0; ir < fft_->size(); ir++) v2[ir] *= ctx_.step_function()->theta_r(ir);
    //==
    //==                 fft_->input(&v2[0]);
    //==                 fft_->transform(-1);
    //==                 fft_->output(num_gkvec(), gkvec_.index_map(), &v1[ispn][0]);
    //==             }
    //==         }
    //==
    //==         if (use_fft == 1)
    //==         {
    //==             for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
    //==             {
    //==                 fft_->input(num_gkvec(), gkvec_.index_map(),
    //==                                        &spinor_wave_functions_(unit_cell_.mt_basis_size(), ispn, j1));
    //==                 fft_->transform(1);
    //==                 fft_->output(&v1[ispn][0]);
    //==             }
    //==         }
    //==
    //==         for (int j2 = 0; j2 < ctx_.num_bands(); j2++)
    //==         {
    //==             double_complex zsum(0, 0);
    //==             for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
    //==             {
    //==                 for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
    //==                 {
    //==                     int offset_wf = unit_cell_.atom(ia)->offset_wf();
    //==                     Atom_type* type = unit_cell_.atom(ia)->type();
    //==                     Atom_symmetry_class* symmetry_class = unit_cell_.atom(ia)->symmetry_class();
    //==
    //==                     for (int l = 0; l <= ctx_.lmax_apw(); l++)
    //==                     {
    //==                         int ordmax = type->indexr().num_rf(l);
    //==                         for (int io1 = 0; io1 < ordmax; io1++)
    //==                         {
    //==                             for (int io2 = 0; io2 < ordmax; io2++)
    //==                             {
    //==                                 for (int m = -l; m <= l; m++)
    //==                                 {
    //==                                     zsum += conj(spinor_wave_functions_(offset_wf +
    //type->indexb_by_l_m_order(l, m, io1), ispn, j1)) *
    //==                                             spinor_wave_functions_(offset_wf + type->indexb_by_l_m_order(l, m,
    //io2), ispn, j2) *
    //==                                             symmetry_class->o_radial_integral(l, io1, io2);
    //==                                 }
    //==                             }
    //==                         }
    //==                     }
    //==                 }
    //==             }
    //==
    //==             if (use_fft == 0)
    //==             {
    //==                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
    //==                {
    //==                    for (int ig = 0; ig < num_gkvec(); ig++)
    //==                        zsum += conj(v1[ispn][ig]) * spinor_wave_functions_(unit_cell_.mt_basis_size() + ig,
    //ispn, j2);
    //==                }
    //==             }
    //==
    //==             if (use_fft == 1)
    //==             {
    //==                 for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
    //==                 {
    //==                     fft_->input(num_gkvec(), gkvec_.index_map(),
    //&spinor_wave_functions_(unit_cell_.mt_basis_size(), ispn, j2));
    //==                     fft_->transform(1);
    //==                     fft_->output(&v2[0]);
    //==
    //==                     for (int ir = 0; ir < fft_->size(); ir++)
    //==                         zsum += std::conj(v1[ispn][ir]) * v2[ir] * ctx_.step_function()->theta_r(ir) /
    //double(fft_->size());
    //==                 }
    //==             }
    //==
    //==             if (use_fft == 2)
    //==             {
    //==                 STOP();
    //==                 //for (int ig1 = 0; ig1 < num_gkvec(); ig1++)
    //==                 //{
    //==                 //    for (int ig2 = 0; ig2 < num_gkvec(); ig2++)
    //==                 //    {
    //==                 //        int ig3 = ctx_.gvec().index_g12(ig1, ig2);
    //==                 //        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
    //==                 //        {
    //==                 //            zsum += std::conj(spinor_wave_functions_(unit_cell_.mt_basis_size() + ig1, ispn,
    //j1)) *
    //==                 //                    spinor_wave_functions_(unit_cell_.mt_basis_size() + ig2, ispn, j2) *
    //==                 //                    ctx_.step_function()->theta_pw(ig3);
    //==                 //        }
    //==                 //    }
    //==                 //}
    //==             }
    //==
    //==             zsum = (j1 == j2) ? zsum - double_complex(1.0, 0.0) : zsum;
    //==             maxerr = std::max(maxerr, std::abs(zsum));
    //==         }
    //==     }
    //==     std :: cout << "maximum error = " << maxerr << std::endl;
}

/** The following HDF5 data structure is created:
  \verbatim
  /K_point_set/ik/vk
  /K_point_set/ik/band_energies
  /K_point_set/ik/band_occupancies
  /K_point_set/ik/gkvec
  /K_point_set/ik/gvec
  /K_point_set/ik/bands/ibnd/spinor_wave_function/ispn/pw
  /K_point_set/ik/bands/ibnd/spinor_wave_function/ispn/mt
  \endverbatim
*/
template <typename T>
void
K_point<T>::save(std::string const& name__, int id__) const
{
    /* rank 0 creates placeholders in the HDF5 file */
    if (comm().rank() == 0) {
        /* open file with write access */
        HDF5_tree fout(name__, hdf5_access_t::read_write);
        /* create /K_point_set/ik */
        fout["K_point_set"].create_node(id__);
        fout["K_point_set"][id__].write("vk", &vk_[0], 3);
        fout["K_point_set"][id__].write("band_energies", band_energies_);
        fout["K_point_set"][id__].write("band_occupancies", band_occupancies_);

        /* save the entire G+k object */
        //TODO: only the list of z-columns is probably needed to recreate the G+k vectors
        //serializer s;
        //gkvec().pack(s);
        //fout["K_point_set"][id__].write("gkvec", s.stream());

        /* save the order of G-vectors */
        mdarray<int, 2> gv(3, num_gkvec());
        for (int i = 0; i < num_gkvec(); i++) {
            auto v = gkvec().gvec(i);
            for (int x : {0, 1, 2}) {
                gv(x, i) = v[x];
            }
        }
        fout["K_point_set"][id__].write("gvec", gv);
        fout["K_point_set"][id__].create_node("bands");
        for (int i = 0; i < ctx_.num_bands(); i++) {
            fout["K_point_set"][id__]["bands"].create_node(i);
            fout["K_point_set"][id__]["bands"][i].create_node("spinor_wave_function");
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                fout["K_point_set"][id__]["bands"][i]["spinor_wave_function"].create_node(ispn);
            }
        }
    }
    /* wait for rank 0 */
    comm().barrier();
    int gkvec_count  = gkvec().count();
    int gkvec_offset = gkvec().offset();
    std::vector<std::complex<T>> wf_tmp(num_gkvec());

    std::unique_ptr<HDF5_tree> fout;
    /* rank 0 opens a file */
    if (comm().rank() == 0) {
        fout = std::unique_ptr<HDF5_tree>(new HDF5_tree(name__, hdf5_access_t::read_write));
    }

    /* store wave-functions */
    for (int i = 0; i < ctx_.num_bands(); i++) {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            /* gather full column of PW coefficients on rank 0 */
            comm().gather(&spinor_wave_functions_->pw_coeffs(ispn).prime(0, i), wf_tmp.data(), gkvec_offset,
                          gkvec_count, 0);
            if (comm().rank() == 0) {
                (*fout)["K_point_set"][id__]["bands"][i]["spinor_wave_function"][ispn].write("pw", wf_tmp);
            }
        }
        comm().barrier();
    }
}

template <typename T>
void
K_point<T>::load(HDF5_tree h5in, int id)
{
    STOP();
    //== band_energies_.resize(ctx_.num_bands());
    //== h5in[id].read("band_energies", band_energies_);

    //== band_occupancies_.resize(ctx_.num_bands());
    //== h5in[id].read("band_occupancies", band_occupancies_);
    //==
    //== h5in[id].read_mdarray("fv_eigen_vectors", fv_eigen_vectors_panel_);
    //== h5in[id].read_mdarray("sv_eigen_vectors", sv_eigen_vectors_);
}

//== void K_point::save_wave_functions(int id)
//== {
//==     if (ctx_.mpi_grid().root(1 << _dim_col_))
//==     {
//==         HDF5_tree fout(storage_file_name, false);
//==
//==         fout["K_points"].create_node(id);
//==         fout["K_points"][id].write("coordinates", &vk_[0], 3);
//==         fout["K_points"][id].write("mtgk_size", mtgk_size());
//==         fout["K_points"][id].create_node("spinor_wave_functions");
//==         fout["K_points"][id].write("band_energies", &band_energies_[0], ctx_.num_bands());
//==         fout["K_points"][id].write("band_occupancies", &band_occupancies_[0], ctx_.num_bands());
//==     }
//==
//==     Platform::barrier(ctx_.mpi_grid().communicator(1 << _dim_col_));
//==
//==     mdarray<double_complex, 2> wfj(NULL, mtgk_size(), ctx_.num_spins());
//==     for (int j = 0; j < ctx_.num_bands(); j++)
//==     {
//==         int rank = ctx_.spl_spinor_wf_col().location(_splindex_rank_, j);
//==         int offs = ctx_.spl_spinor_wf_col().location(_splindex_offs_, j);
//==         if (ctx_.mpi_grid().coordinate(_dim_col_) == rank)
//==         {
//==             HDF5_tree fout(storage_file_name, false);
//==             wfj.set_ptr(&spinor_wave_functions_(0, 0, offs));
//==             fout["K_points"][id]["spinor_wave_functions"].write_mdarray(j, wfj);
//==         }
//==         Platform::barrier(ctx_.mpi_grid().communicator(_dim_col_));
//==     }
//== }
//==
//== void K_point::load_wave_functions(int id)
//== {
//==     HDF5_tree fin(storage_file_name, false);
//==
//==     int mtgk_size_in;
//==     fin["K_points"][id].read("mtgk_size", &mtgk_size_in);
//==     if (mtgk_size_in != mtgk_size()) error_local(__FILE__, __LINE__, "wrong wave-function size");
//==
//==     band_energies_.resize(ctx_.num_bands());
//==     fin["K_points"][id].read("band_energies", &band_energies_[0], ctx_.num_bands());
//==
//==     band_occupancies_.resize(ctx_.num_bands());
//==     fin["K_points"][id].read("band_occupancies", &band_occupancies_[0], ctx_.num_bands());
//==
//==     spinor_wave_functions_.set_dimensions(mtgk_size(), ctx_.num_spins(),
//==                                           ctx_.spl_spinor_wf_col().local_size());
//==     spinor_wave_functions_.allocate();
//==
//==     mdarray<double_complex, 2> wfj(NULL, mtgk_size(), ctx_.num_spins());
//==     for (int jloc = 0; jloc < ctx_.spl_spinor_wf_col().local_size(); jloc++)
//==     {
//==         int j = ctx_.spl_spinor_wf_col(jloc);
//==         wfj.set_ptr(&spinor_wave_functions_(0, 0, jloc));
//==         fin["K_points"][id]["spinor_wave_functions"].read_mdarray(j, wfj);
//==     }
//== }

template <typename T>
void
K_point<T>::generate_atomic_wave_functions(
    std::vector<int> atoms__, std::function<sirius::experimental::basis_functions_index const*(int)> indexb__,
    Radial_integrals_atomic_wf<false> const& ri__, sddk::Wave_functions<T>& wf__)
{
    PROFILE("sirius::K_point::generate_atomic_wave_functions");

    int lmax{3};
    int lmmax = utils::lmmax(lmax);
    // for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
    //    auto& atom_type = unit_cell_.atom_type(iat);
    //    lmax            = std::max(lmax, atom_type.lmax_ps_atomic_wf());
    //}
    // lmax = std::max(lmax, unit_cell_.lmax());

    /* compute offset for each atom */
    std::vector<int> offset;
    int n{0};
    for (int ia : atoms__) {
        offset.push_back(n);
        int iat = unit_cell_.atom(ia).type_id();
        n += indexb__(iat)->size();
    }

    /* allocate memory to store wave-functions for atom types */
    std::vector<sddk::mdarray<std::complex<T>, 2>> wf_t(unit_cell_.num_atom_types());
    for (int ia : atoms__) {
        int iat = unit_cell_.atom(ia).type_id();
        if (wf_t[iat].size() == 0) {
            wf_t[iat] = sddk::mdarray<std::complex<T>, 2>(this->num_gkvec_loc(), indexb__(iat)->size(),
                                                          ctx_.mem_pool(memory_t::host));
        }
    }

    #pragma omp parallel for schedule(static)
    for (int igk_loc = 0; igk_loc < this->num_gkvec_loc(); igk_loc++) {
        /* vs = {r, theta, phi} */
        auto vs = geometry3d::spherical_coordinates(this->gkvec().template gkvec_cart<index_domain_t::local>(igk_loc));

        /* compute real spherical harmonics for G+k vector */
        std::vector<double> rlm(lmmax);
        sf::spherical_harmonics(lmax, vs[1], vs[2], &rlm[0]);

        /* get all values of the radial integrals for a given G+k vector */
        std::vector<mdarray<double, 1>> ri_values(unit_cell_.num_atom_types());
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            if (wf_t[iat].size() != 0) {
                ri_values[iat] = ri__.values(iat, vs[0]);
            }
        }
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            if (wf_t[iat].size() == 0) {
                continue;
            }
            auto const& indexb = *indexb__(iat);
            for (int xi = 0; xi < static_cast<int>(indexb.size()); xi++) {
                /*  orbital quantum  number of this atomic orbital */
                int l = indexb.l(xi);
                /*  composite l,m index */
                int lm = indexb.lm(xi);
                /* index of the radial function */
                int idxrf = indexb.idxrf(xi);

                auto z = std::pow(std::complex<double>(0, -1), l) * fourpi / std::sqrt(unit_cell_.omega());

                wf_t[iat](igk_loc, xi) = static_cast<std::complex<T>>(z * rlm[lm] * ri_values[iat](idxrf));
            }
        }
    }

    for (int ia : atoms__) {

        T phase                 = twopi * dot(gkvec().vk(), unit_cell_.atom(ia).position());
        std::complex<T> phase_k = std::exp(std::complex<T>(0.0, phase));

        /* quickly compute phase factors without calling exp() function */
        std::vector<std::complex<T>> phase_gk(num_gkvec_loc());
        #pragma omp parallel for schedule(static)
        for (int igk_loc = 0; igk_loc < num_gkvec_loc(); igk_loc++) {
            /* global index of G+k-vector */
            int igk = this->idxgk(igk_loc);
            auto G  = gkvec().gvec(igk);
            /* total phase e^{-i(G+k)r_{\alpha}} */
            phase_gk[igk_loc] = std::conj(static_cast<std::complex<T>>(ctx_.gvec_phase_factor(G, ia)) * phase_k);
        }

        int iat = unit_cell_.atom(ia).type_id();
        #pragma omp parallel
        for (int xi = 0; xi < static_cast<int>(indexb__(iat)->size()); xi++) {
            #pragma omp for schedule(static) nowait
            for (int igk_loc = 0; igk_loc < num_gkvec_loc(); igk_loc++) {
                wf__.pw_coeffs(0).prime(igk_loc, offset[ia] + xi) = wf_t[iat](igk_loc, xi) * phase_gk[igk_loc];
            }
        }
    }
}

template <typename T>
void
K_point<T>::compute_gradient_wave_functions(Wave_functions<T>& phi, const int starting_position_i, const int num_wf,
                                            Wave_functions<T>& dphi, const int starting_position_j, const int direction)
{
    std::vector<std::complex<T>> qalpha(this->num_gkvec_loc());
    auto k_cart = dot(ctx_.unit_cell().reciprocal_lattice_vectors(), this->vk());
    for (int igk_loc = 0; igk_loc < this->num_gkvec_loc(); igk_loc++) {
        auto G = this->gkvec().template gkvec_cart<index_domain_t::local>(igk_loc);

        qalpha[igk_loc] = std::complex<T>(0.0, -(G[direction] + k_cart[direction]));
    }

    #pragma omp parallel for schedule(static)
    for (int nphi = 0; nphi < num_wf; nphi++) {
        for (int ispn = 0; ispn < phi.num_sc(); ispn++) {
            for (int igk_loc = 0; igk_loc < this->num_gkvec_loc(); igk_loc++) {
                dphi.pw_coeffs(ispn).prime(igk_loc, nphi + starting_position_j) =
                    qalpha[igk_loc] * phi.pw_coeffs(ispn).prime(igk_loc, nphi + starting_position_i);
            }
        }
    }
}

template class K_point<double>;
#ifdef USE_FP32
template class K_point<float>;
#endif
} // namespace sirius
