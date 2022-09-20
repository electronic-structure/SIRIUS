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
#include "linalg/inverse_sqrt.hpp"
#include "SDDK/wf_inner.hpp"
#include "SDDK/wf_trans.hpp"

namespace sirius {

template <typename T>
void
K_point<T>::initialize()
{
    PROFILE("sirius::K_point::initialize");

    zil_.resize(ctx_.unit_cell().lmax_apw() + 1);
    for (int l = 0; l <= ctx_.unit_cell().lmax_apw(); l++) {
        zil_[l] = std::pow(std::complex<T>(0, 1), l);
    }

    l_by_lm_ = utils::l_by_lm(ctx_.unit_cell().lmax_apw());

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

            RTE_ASSERT(ctx_.num_fv_states() > 0);
            fv_eigen_values_ = sddk::mdarray<double, 1>(ctx_.num_fv_states(), sddk::memory_t::host, "fv_eigen_values");

            if (ctx_.need_sv()) {
                /* in case of collinear magnetism store pure up and pure dn components, otherwise store the full matrix
                 */
                for (int is = 0; is < ctx_.num_spinors(); is++) {
                    sv_eigen_vectors_[is] = sddk::dmatrix<std::complex<T>>(nst, nst, ctx_.blacs_grid(), bs, bs, mem_type_evp);
                }
            }
            /* allocate fv eien vectors */
            fv_eigen_vectors_slab_ = std::make_unique<sddk::Wave_functions<T>>(
                gkvec_partition(), unit_cell_.num_atoms(),
                [this](int ia) { return unit_cell_.atom(ia).mt_lo_basis_size(); }, ctx_.num_fv_states(),
                ctx_.preferred_memory_t());

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
                    fv_eigen_vectors_ = sddk::dmatrix<std::complex<T>>(gklo_basis_size(), gklo_basis_size(),
                                                                 ctx_.blacs_grid(), bs, bs, mem_type_gevp);
                } else {
                    fv_eigen_vectors_ = sddk::dmatrix<std::complex<T>>(gklo_basis_size(), ctx_.num_fv_states(),
                                                                 ctx_.blacs_grid(), bs, bs, mem_type_gevp);
                }
            } else {
                int ncomp = ctx_.cfg().iterative_solver().num_singular();
                if (ncomp < 0) {
                    ncomp = ctx_.num_fv_states() / 2;
                }

                singular_components_ = std::make_unique<sddk::Wave_functions<T>>(
                    gkvec_partition(), ncomp, ctx_.preferred_memory_t());

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
                    singular_components_->print_checksum(sddk::device_t::CPU, "singular_components", 0, ncomp, RTE_OUT(std::cout));
                }
            }

            fv_states_ = std::make_unique<sddk::Wave_functions<T>>(
                gkvec_partition(), unit_cell_.num_atoms(),
                [this](int ia) { return unit_cell_.atom(ia).mt_basis_size(); }, ctx_.num_fv_states(),
                ctx_.preferred_memory_t());

            spinor_wave_functions_ = std::make_shared<sddk::Wave_functions<T>>(
                gkvec_partition(), unit_cell_.num_atoms(),
                [this](int ia) { return unit_cell_.atom(ia).mt_basis_size(); }, nst, ctx_.preferred_memory_t(),
                ctx_.num_spins());
        } else {
            throw std::runtime_error("not implemented");
        }
    } else {
        spinor_wave_functions_ =
            std::make_shared<sddk::Wave_functions<T>>(gkvec_partition(), nst, ctx_.preferred_memory_t(), ctx_.num_spins());
        if (ctx_.hubbard_correction()) {
            /* allocate Hubbard wave-functions */
            int nwfh = unit_cell_.num_hubbard_wf().first;// * ctx_.num_spinor_comp();
            int nwf  = unit_cell_.num_ps_atomic_wf().first;// * ctx_.num_spinor_comp();
            auto mt  = ctx_.preferred_memory_t();
            int ns   = 1; //ctx_.num_spins();

            hubbard_wave_functions_   = std::make_unique<sddk::Wave_functions<T>>(gkvec_partition(), nwfh, mt, ns);
            hubbard_wave_functions_S_ = std::make_unique<sddk::Wave_functions<T>>(gkvec_partition(), nwfh, mt, ns);
            atomic_wave_functions_    = std::make_unique<sddk::Wave_functions<T>>(gkvec_partition(), nwf, mt, ns);
            atomic_wave_functions_S_  = std::make_unique<sddk::Wave_functions<T>>(gkvec_partition(), nwf, mt, ns);
        }
    }

    update();
}

template <typename T>
void
K_point<T>::generate_hubbard_orbitals()
{
    PROFILE("sirius::K_point::generate_hubbard_orbitals");

    auto& phi = atomic_wave_functions();
    auto& sphi = atomic_wave_functions_S();
    if (ctx_.so_correction()) {
        RTE_THROW("Hubbard+SO is not implemented");
    }
    if (ctx_.gamma_point()) {
        RTE_THROW("Hubbard+Gamma point is not implemented");
    }

    phi.zero(sddk::device_t::CPU);
    sphi.zero(sddk::device_t::CPU);

    auto num_ps_atomic_wf = unit_cell_.num_ps_atomic_wf();
    int nwf = num_ps_atomic_wf.first;

    /* generate the initial atomic wavefunctions (full set composed of all atoms wfs) */
    std::vector<int> atoms(ctx_.unit_cell().num_atoms());
    std::iota(atoms.begin(), atoms.end(), 0);
    this->generate_atomic_wave_functions(atoms, [&](int iat){ return &ctx_.unit_cell().atom_type(iat).indexb_wfs(); },
                ctx_.ps_atomic_wf_ri(), phi);

    if (ctx_.cfg().control().print_checksum()) {
        atomic_wave_functions_->print_checksum(sddk::device_t::CPU, "atomic_wave_functions", 0, nwf, RTE_OUT(std::cout));
    }

    /* check if we have a norm conserving pseudo potential only */
    auto q_op = (unit_cell_.augment()) ? std::make_unique<Q_operator<T>>(ctx_) : nullptr;

    phi.prepare(sddk::spin_range(0), true);
    sphi.prepare(sddk::spin_range(0), false);

    /* compute S|phi> */
    beta_projectors().prepare();

    sirius::apply_S_operator<std::complex<T>>(ctx_.processing_unit(), sddk::spin_range(0), 0, nwf, beta_projectors(),
            phi, q_op.get(), sphi);

    std::unique_ptr<sddk::Wave_functions<T>> wf_tmp;
    std::unique_ptr<sddk::Wave_functions<T>> swf_tmp;
    if (ctx_.cfg().hubbard().full_orthogonalization()) {
        wf_tmp = std::make_unique<sddk::Wave_functions<T>>(gkvec_partition(), nwf, sddk::memory_t::host, 1);
        swf_tmp = std::make_unique<sddk::Wave_functions<T>>(gkvec_partition(), nwf, sddk::memory_t::host, 1);

        wf_tmp->copy_from(sddk::device_t::CPU, nwf, phi, 0, 0, 0, 0);
        if (is_device_memory(sphi.preferred_memory_t())) {
            sphi.copy_to(sddk::spin_range(0), sddk::memory_t::host, 0, nwf);
        }
        swf_tmp->copy_from(sddk::device_t::CPU, nwf, sphi, 0, 0, 0, 0);

        int BS = ctx_.cyclic_block_size();
        sddk::dmatrix<complex_type<T>> ovlp(nwf, nwf, ctx_.blacs_grid(), BS, BS);
        sddk::inner(ctx_.spla_context(), sddk::spin_range(0), phi, 0, nwf, sphi, 0, nwf, ovlp, 0, 0);
        auto B = std::get<0>(inverse_sqrt(ovlp, nwf));

        sddk::transform<complex_type<T>>(ctx_.spla_context(), 0, {&phi}, 0, nwf, *B, 0, 0, {&sphi}, 0, nwf);
        phi.copy_from(sphi, nwf, 0, 0, 0, 0);

        sirius::apply_S_operator<std::complex<T>>(ctx_.processing_unit(), sddk::spin_range(0), 0, nwf, beta_projectors(),
                phi, q_op.get(), sphi);

        if (ctx_.cfg().control().verification() >= 1) {
            sddk::inner(ctx_.spla_context(), sddk::spin_range(0), phi, 0, nwf, sphi, 0, nwf, ovlp, 0, 0);

            auto diff = check_identity(ovlp, nwf);
            RTE_OUT(std::cout) << "orthogonalization error " << diff << std::endl;
        }
    }

    beta_projectors().dismiss();
    phi.dismiss(sddk::spin_range(0), true);
    sphi.dismiss(sddk::spin_range(0), true);

    if (ctx_.cfg().control().print_checksum()) {
        sphi.print_checksum(sddk::device_t::CPU, "atomic_wave_functions_S", 0, nwf, RTE_OUT(std::cout));
    }

    auto& phi_hub = hubbard_wave_functions();
    auto& sphi_hub = hubbard_wave_functions_S();

    auto num_hubbard_wf = unit_cell_.num_hubbard_wf();

    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        auto& atom = ctx_.unit_cell().atom(ia);
        auto& type = atom.type();
        if (type.hubbard_correction()) {
            /* loop over Hubbard orbitals of the atom */
            for (int idxrf = 0; idxrf < type.indexr_hub().size(); idxrf++) {
                /* Hubbard orbital descriptor */
                auto& hd = type.lo_descriptor_hub(idxrf);
                int l = type.indexr_hub().am(idxrf).l();
                int mmax = 2 * l + 1;

                int idxr_wf = hd.idx_wf();

                int offset_in_wf = num_ps_atomic_wf.second[ia] + type.indexb_wfs().offset(idxr_wf);
                int offset_in_hwf = num_hubbard_wf.second[ia] + type.indexb_hub().offset(idxrf);

                phi_hub.copy_from(sddk::device_t::CPU, mmax, phi, 0, offset_in_wf, 0, offset_in_hwf);
                sphi_hub.copy_from(sddk::device_t::CPU, mmax, sphi, 0, offset_in_wf, 0, offset_in_hwf);
            }
        }
    }
    /* restore phi and sphi */
    if (ctx_.cfg().hubbard().full_orthogonalization()) {
        phi.copy_from(sddk::device_t::CPU, nwf, *wf_tmp, 0, 0, 0, 0);
        sphi.copy_from(sddk::device_t::CPU, nwf, *swf_tmp, 0, 0, 0, 0);
    }

    //if (ctx_.num_spins() == 2) {
    //    /* copy up component to dn component in collinear case
    //     * +-------------------------------+
    //     * |  phi1_{lm}, phi2_{lm}, ...    |
    //     * +-------------------------------+
    //     * |  phi1_{lm}, phi2_{lm}, ...    |
    //     * +-------------------------------+
    //     *
    //     * or with offset in non-collinear case
    //     *
    //     * +-------------------------------+---------------------------------+
    //     * |  phi1_{lm}, phi2_{lm}, ...    |              0                  |
    //     * +-------------------------------+---------------------------------+
    //     * |           0                   |   phi1_{lm}, phi2_{lm}, ...     |
    //     * +-------------------------------+---------------------------------+
    //     */
    //    phi.copy_from(device_t::CPU, r.first, phi, 0, 0, 1, (ctx_.num_mag_dims() == 3) ? r.first : 0);
    //}

    if (ctx_.cfg().control().print_checksum()) {
        hubbard_wave_functions_->print_checksum(sddk::device_t::CPU, "hubbard_phi", 0,
                                                hubbard_wave_functions_->num_wf(), RTE_OUT(std::cout));
        hubbard_wave_functions_S_->print_checksum(sddk::device_t::CPU, "hubbard_phi_S", 0,
                                                  hubbard_wave_functions_S_->num_wf(), RTE_OUT(std::cout));
    }
}

template <typename T>
void
K_point<T>::generate_gkvec(double gk_cutoff__)
{
    PROFILE("sirius::K_point::generate_gkvec");

    if (ctx_.full_potential() && (gk_cutoff__ * unit_cell_.max_mt_radius() > ctx_.unit_cell().lmax_apw()) &&
        ctx_.comm().rank() == 0 && ctx_.verbosity() >= 0) {
        std::stringstream s;
        s << "G+k cutoff (" << gk_cutoff__ << ") is too large for a given lmax ("
          << ctx_.unit_cell().lmax_apw() << ") and a maximum MT radius (" << unit_cell_.max_mt_radius() << ")"
          << std::endl
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

    gkvec_partition_ = std::make_unique<sddk::Gvec_partition>(
        this->gkvec(), ctx_.comm_fft_coarse(), ctx_.comm_band_ortho_fft_coarse());

    const auto fft_type = gkvec_->reduced() ? SPFFT_TRANS_R2C : SPFFT_TRANS_C2C;
    const auto spfft_pu = ctx_.processing_unit() == sddk::device_t::CPU ? SPFFT_PU_HOST : SPFFT_PU_GPU;
    auto const& gv      = gkvec_partition_->gvec_array();
    /* create transformation */
    spfft_transform_.reset(new spfft_transform_type<T>(ctx_.spfft_grid_coarse<T>().create_transform(
        spfft_pu, fft_type, ctx_.fft_coarse_grid()[0], ctx_.fft_coarse_grid()[1], ctx_.fft_coarse_grid()[2],
        ctx_.spfft_coarse<double>().local_z_length(), gkvec_partition_->gvec_count_fft(), SPFFT_INDEX_TRIPLETS,
        gv.at(sddk::memory_t::host))));

    sddk::splindex<sddk::splindex_t::block_cyclic> spl_ngk_row(num_gkvec(), num_ranks_row_, rank_row_, ctx_.cyclic_block_size());
    num_gkvec_row_ = spl_ngk_row.local_size();
    sddk::mdarray<int, 2> gkvec_row(3, num_gkvec_row_);

    sddk::splindex<sddk::splindex_t::block_cyclic> spl_ngk_col(num_gkvec(), num_ranks_col_, rank_col_, ctx_.cyclic_block_size());
    num_gkvec_col_ = spl_ngk_col.local_size();
    sddk::mdarray<int, 2> gkvec_col(3, num_gkvec_col_);

    for (int rank = 0; rank < comm().size(); rank++) {
        auto gv = gkvec_->gvec_local(rank);
        for (int igloc = 0; igloc < gkvec_->gvec_count(rank); igloc++) {
            int ig = gkvec_->gvec_offset(rank) + igloc;
            auto loc_row = spl_ngk_row.location(ig);
            auto loc_col = spl_ngk_col.location(ig);
            if (loc_row.rank == comm_row().rank()) {
                for (int x : {0, 1, 2}) {
                    gkvec_row(x, loc_row.local_index) = gv(x, igloc);
                }
            }
            if (loc_col.rank == comm_col().rank()) {
                for (int x : {0, 1, 2}) {
                    gkvec_col(x, loc_col.local_index) = gv(x, igloc);
                }
            }
        }
    }
    gkvec_row_ = std::make_shared<sddk::Gvec>(vk_, unit_cell_.reciprocal_lattice_vectors(), num_gkvec_row_,
            &gkvec_row(0, 0), comm_row(), ctx_.gamma_point());

    gkvec_col_ = std::make_shared<sddk::Gvec>(vk_, unit_cell_.reciprocal_lattice_vectors(), num_gkvec_col_,
            &gkvec_col(0, 0), comm_col(), ctx_.gamma_point());
}

template <typename T>
void
K_point<T>::update()
{
    PROFILE("sirius::K_point::update");

    gkvec_->lattice_vectors(ctx_.unit_cell().reciprocal_lattice_vectors());
    gkvec_partition_->update_gkvec_cart();

    if (ctx_.full_potential()) {
        if (ctx_.cfg().iterative_solver().type() == "exact") {
            alm_coeffs_row_ = std::make_unique<Matching_coefficients>(unit_cell_, *gkvec_row_);
            alm_coeffs_col_ = std::make_unique<Matching_coefficients>(unit_cell_, *gkvec_col_);
        }
        alm_coeffs_loc_ = std::make_unique<Matching_coefficients>(unit_cell_, gkvec());
    }

    if (!ctx_.full_potential()) {
        /* compute |beta> projectors for atom types */
        beta_projectors_ = std::make_unique<Beta_projectors<T>>(ctx_, gkvec());

        if (ctx_.cfg().iterative_solver().type() == "exact") {
            beta_projectors_row_ = std::make_unique<Beta_projectors<T>>(ctx_, *gkvec_row_);
            beta_projectors_col_ = std::make_unique<Beta_projectors<T>>(ctx_, *gkvec_col_);
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
K_point<T>::get_fv_eigen_vectors(sddk::mdarray<std::complex<T>, 2>& fv_evec__) const
{
    assert((int)fv_evec__.size(0) >= gklo_basis_size());
    assert((int)fv_evec__.size(1) == ctx_.num_fv_states());
    assert(gklo_basis_size_row() == fv_eigen_vectors_.num_rows_local());

    sddk::mdarray<std::complex<T>, 1> tmp(gklo_basis_size_row());

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
        sddk::HDF5_tree fout(name__, sddk::hdf5_access_t::read_write);
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
        sddk::mdarray<int, 2> gv(3, num_gkvec());
        for (int i = 0; i < num_gkvec(); i++) {
            auto v = gkvec().template gvec<sddk::index_domain_t::global>(i);
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

    std::unique_ptr<sddk::HDF5_tree> fout;
    /* rank 0 opens a file */
    if (comm().rank() == 0) {
        fout = std::make_unique<sddk::HDF5_tree>(name__, sddk::hdf5_access_t::read_write);
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
K_point<T>::load(sddk::HDF5_tree h5in, int id)
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
K_point<T>::generate_atomic_wave_functions(std::vector<int> atoms__,
        std::function<sirius::experimental::basis_functions_index const*(int)> indexb__,
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
                                                          ctx_.mem_pool(sddk::memory_t::host));
        }
    }

    #pragma omp parallel for schedule(static)
    for (int igk_loc = 0; igk_loc < this->num_gkvec_loc(); igk_loc++) {
        /* vs = {r, theta, phi} */
        auto vs = geometry3d::spherical_coordinates(this->gkvec().template gkvec_cart<sddk::index_domain_t::local>(igk_loc));

        /* compute real spherical harmonics for G+k vector */
        std::vector<double> rlm(lmmax);
        sf::spherical_harmonics(lmax, vs[1], vs[2], &rlm[0]);

        /* get all values of the radial integrals for a given G+k vector */
        std::vector<sddk::mdarray<double, 1>> ri_values(unit_cell_.num_atom_types());
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

        T phase      = twopi * dot(gkvec().vk(), unit_cell_.atom(ia).position());
        auto phase_k = std::exp(std::complex<T>(0.0, phase));

        /* quickly compute phase factors without calling exp() function */
        std::vector<std::complex<T>> phase_gk(num_gkvec_loc());
        #pragma omp parallel for schedule(static)
        for (int igk_loc = 0; igk_loc < num_gkvec_loc(); igk_loc++) {
            auto G  = gkvec().template gvec<sddk::index_domain_t::local>(igk_loc);
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
K_point<T>::generate_gklo_basis()
{
    /* find local number of row G+k vectors */
    sddk::splindex<sddk::splindex_t::block_cyclic> spl_ngk_row(num_gkvec(), num_ranks_row_, rank_row_, ctx_.cyclic_block_size());
    num_gkvec_row_ = spl_ngk_row.local_size();

    igk_row_.resize(num_gkvec_row_);
    for (int i = 0; i < num_gkvec_row_; i++) {
        igk_row_[i] = spl_ngk_row[i];
    }

    /* find local number of column G+k vectors */
    sddk::splindex<sddk::splindex_t::block_cyclic> spl_ngk_col(num_gkvec(), num_ranks_col_, rank_col_, ctx_.cyclic_block_size());
    num_gkvec_col_ = spl_ngk_col.local_size();

    igk_col_.resize(num_gkvec_col_);
    for (int i = 0; i < num_gkvec_col_; i++) {
        igk_col_[i] = spl_ngk_col[i];
    }

    /* mapping between local and global G+k vecotor indices */
    igk_loc_.resize(num_gkvec_loc());
    for (int i = 0; i < num_gkvec_loc(); i++) {
        igk_loc_[i] = gkvec().offset() + i;
    }

    if (ctx_.full_potential()) {
        sddk::splindex<sddk::splindex_t::block_cyclic> spl_nlo_row(num_gkvec() + unit_cell_.mt_lo_basis_size(),
                num_ranks_row_, rank_row_, ctx_.cyclic_block_size());
        sddk::splindex<sddk::splindex_t::block_cyclic> spl_nlo_col(num_gkvec() + unit_cell_.mt_lo_basis_size(),
                num_ranks_col_, rank_col_, ctx_.cyclic_block_size());

        lo_basis_descriptor lo_desc;

        int idx{0};
        /* local orbital basis functions */
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto& atom = unit_cell_.atom(ia);
            auto& type = atom.type();

            int lo_index_offset = type.mt_aw_basis_size();

            for (int j = 0; j < type.mt_lo_basis_size(); j++) {
                int l         = type.indexb(lo_index_offset + j).l;
                int lm        = type.indexb(lo_index_offset + j).lm;
                int order     = type.indexb(lo_index_offset + j).order;
                int idxrf     = type.indexb(lo_index_offset + j).idxrf;
                lo_desc.ia    = static_cast<uint16_t>(ia);
                lo_desc.l     = static_cast<uint8_t>(l);
                lo_desc.lm    = static_cast<uint16_t>(lm);
                lo_desc.order = static_cast<uint8_t>(order);
                lo_desc.idxrf = static_cast<uint8_t>(idxrf);

                if (spl_nlo_row.local_rank(num_gkvec() + idx) == rank_row_) {
                    lo_basis_descriptors_row_.push_back(lo_desc);
                }
                if (spl_nlo_col.local_rank(num_gkvec() + idx) == rank_col_) {
                    lo_basis_descriptors_col_.push_back(lo_desc);
                }

                idx++;
            }
        }
        RTE_ASSERT(idx == unit_cell_.mt_lo_basis_size());

        atom_lo_cols_.clear();
        atom_lo_cols_.resize(unit_cell_.num_atoms());
        for (int i = 0; i < num_lo_col(); i++) {
            int ia = lo_basis_descriptor_col(i).ia;
            atom_lo_cols_[ia].push_back(i);
        }

        atom_lo_rows_.clear();
        atom_lo_rows_.resize(unit_cell_.num_atoms());
        for (int i = 0; i < num_lo_row(); i++) {
            int ia = lo_basis_descriptor_row(i).ia;
            atom_lo_rows_[ia].push_back(i);
        }
    }
}

template class K_point<double>;
#ifdef USE_FP32
template class K_point<float>;
#endif
} // namespace sirius
