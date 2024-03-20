/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file k_point.cpp
 *
 *  \brief Contains partial implementation of sirius::K_point class.
 */

#include "k_point/k_point.hpp"
#include "hamiltonian/non_local_operator.hpp"
#include "core/la/inverse_sqrt.hpp"

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

    l_by_lm_ = sf::l_by_lm(ctx_.unit_cell().lmax_apw());

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
            fv_eigen_values_ = mdarray<double, 1>({ctx_.num_fv_states()}, mdarray_label("fv_eigen_values"));

            if (ctx_.need_sv()) {
                /* in case of collinear magnetism store pure up and pure dn components, otherwise store the full matrix
                 */
                for (int is = 0; is < ctx_.num_spinors(); is++) {
                    sv_eigen_vectors_[is] =
                            la::dmatrix<std::complex<T>>(nst, nst, ctx_.blacs_grid(), bs, bs, mem_type_evp);
                }
            }

            std::vector<int> num_mt_coeffs(unit_cell_.num_atoms());
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                num_mt_coeffs[ia] = unit_cell_.atom(ia).mt_lo_basis_size();
            }

            /* allocate fv eien vectors */
            fv_eigen_vectors_slab_ =
                    std::make_unique<wf::Wave_functions<T>>(gkvec_, num_mt_coeffs, wf::num_mag_dims(0),
                                                            wf::num_bands(ctx_.num_fv_states()), ctx_.host_memory_t());

            fv_eigen_vectors_slab_->zero(memory_t::host, wf::spin_index(0), wf::band_range(0, ctx_.num_fv_states()));
            for (int i = 0; i < ctx_.num_fv_states(); i++) {
                for (int igloc = 0; igloc < gkvec().count(comm().rank()); igloc++) {
                    int ig = igloc + gkvec().offset(comm().rank());
                    if (ig == i) {
                        fv_eigen_vectors_slab_->pw_coeffs(igloc, wf::spin_index(0), wf::band_index(i)) = 1.0;
                    }
                    if (ig == i + 1) {
                        fv_eigen_vectors_slab_->pw_coeffs(igloc, wf::spin_index(0), wf::band_index(i)) = 0.5;
                    }
                    if (ig == i + 2) {
                        fv_eigen_vectors_slab_->pw_coeffs(igloc, wf::spin_index(0), wf::band_index(i)) = 0.125;
                    }
                }
            }
            if (ctx_.cfg().iterative_solver().type() == "exact") {
                /* ELPA needs a full matrix of eigen-vectors as it uses it as a work space */
                if (ctx_.gen_evp_solver().type() == la::ev_solver_t::elpa ||
                    ctx_.gen_evp_solver().type() == la::ev_solver_t::dlaf) {
                    fv_eigen_vectors_ = la::dmatrix<std::complex<T>>(gklo_basis_size(), gklo_basis_size(),
                                                                     ctx_.blacs_grid(), bs, bs, mem_type_gevp);
                } else {
                    fv_eigen_vectors_ = la::dmatrix<std::complex<T>>(gklo_basis_size(), ctx_.num_fv_states(),
                                                                     ctx_.blacs_grid(), bs, bs, mem_type_gevp);
                }
            } else {
                int ncomp = ctx_.cfg().iterative_solver().num_singular();
                if (ncomp < 0) {
                    ncomp = ctx_.num_fv_states() / 2;
                }

                singular_components_ = std::make_unique<wf::Wave_functions<T>>(
                        gkvec_, num_mt_coeffs, wf::num_mag_dims(0), wf::num_bands(ncomp), ctx_.host_memory_t());

                singular_components_->zero(memory_t::host, wf::spin_index(0), wf::band_range(0, ncomp));
                /* starting guess for wave-functions */
                for (int i = 0; i < ncomp; i++) {
                    for (int igloc = 0; igloc < gkvec().count(); igloc++) {
                        int ig = igloc + gkvec().offset();
                        if (ig == i) {
                            singular_components_->pw_coeffs(igloc, wf::spin_index(0), wf::band_index(i)) = 1.0;
                        }
                        if (ig == i + 1) {
                            singular_components_->pw_coeffs(igloc, wf::spin_index(0), wf::band_index(i)) = 0.5;
                        }
                        if (ig == i + 2) {
                            singular_components_->pw_coeffs(igloc, wf::spin_index(0), wf::band_index(i)) = 0.125;
                        }
                    }
                }
            }

            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                num_mt_coeffs[ia] = unit_cell_.atom(ia).mt_basis_size();
            }
            fv_states_ =
                    std::make_unique<wf::Wave_functions<T>>(gkvec_, num_mt_coeffs, wf::num_mag_dims(0),
                                                            wf::num_bands(ctx_.num_fv_states()), ctx_.host_memory_t());

            spinor_wave_functions_ = std::make_unique<wf::Wave_functions<T>>(gkvec_, num_mt_coeffs,
                                                                             wf::num_mag_dims(ctx_.num_mag_dims()),
                                                                             wf::num_bands(nst), ctx_.host_memory_t());
        } else {
            RTE_THROW("not implemented");
        }
    } else {
        spinor_wave_functions_ = std::make_unique<wf::Wave_functions<T>>(gkvec_, wf::num_mag_dims(ctx_.num_mag_dims()),
                                                                         wf::num_bands(nst), ctx_.host_memory_t());

        if (ctx_.hubbard_correction()) {
            /* allocate Hubbard wave-functions */
            int nwfh = unit_cell_.num_hubbard_wf().first;
            int nwf  = unit_cell_.num_ps_atomic_wf().first;

            hubbard_wave_functions_ = std::make_unique<wf::Wave_functions<T>>(
                    gkvec_, wf::num_mag_dims(0), wf::num_bands(nwfh), ctx_.host_memory_t());
            hubbard_wave_functions_S_ = std::make_unique<wf::Wave_functions<T>>(
                    gkvec_, wf::num_mag_dims(0), wf::num_bands(nwfh), ctx_.host_memory_t());
            atomic_wave_functions_   = std::make_unique<wf::Wave_functions<T>>(gkvec_, wf::num_mag_dims(0),
                                                                             wf::num_bands(nwf), ctx_.host_memory_t());
            atomic_wave_functions_S_ = std::make_unique<wf::Wave_functions<T>>(
                    gkvec_, wf::num_mag_dims(0), wf::num_bands(nwf), ctx_.host_memory_t());
        }
    }

    update();
}

template <typename T>
void
K_point<T>::generate_hubbard_orbitals()
{
    PROFILE("sirius::K_point::generate_hubbard_orbitals");

    if (ctx_.so_correction()) {
        RTE_THROW("Hubbard+SO is not implemented");
    }
    if (ctx_.gamma_point()) {
        RTE_THROW("Hubbard+Gamma point is not implemented");
    }

    auto num_ps_atomic_wf = unit_cell_.num_ps_atomic_wf();
    int nwf               = num_ps_atomic_wf.first;

    /* generate the initial atomic wavefunctions (full set composed of all atoms wfs) */
    std::vector<int> atoms(ctx_.unit_cell().num_atoms());
    std::iota(atoms.begin(), atoms.end(), 0);

    this->generate_atomic_wave_functions(
            atoms, [&](int iat) { return &ctx_.unit_cell().atom_type(iat).indexb_wfs(); }, *ctx_.ri().ps_atomic_wf_,
            *atomic_wave_functions_);

    auto pcs = env::print_checksum();
    if (pcs) {
        auto cs = atomic_wave_functions_->checksum(memory_t::host, wf::spin_index(0), wf::band_range(0, nwf));
        if (this->comm().rank() == 0) {
            print_checksum("atomic_wave_functions", cs, RTE_OUT(std::cout));
        }
    }

    /* check if we have a norm conserving pseudo potential only */
    auto q_op = (unit_cell_.augment()) ? std::make_unique<Q_operator<T>>(ctx_) : nullptr;

    auto mem = ctx_.processing_unit_memory_t();

    std::unique_ptr<wf::Wave_functions<T>> wf_tmp;
    std::unique_ptr<wf::Wave_functions<T>> swf_tmp;

    {
        auto mg1 = atomic_wave_functions_->memory_guard(mem, wf::copy_to::device | wf::copy_to::host);
        auto mg2 = atomic_wave_functions_S_->memory_guard(mem, wf::copy_to::host);

        /* compute S|phi> */
        auto bp_gen    = beta_projectors().make_generator();
        auto bp_coeffs = bp_gen.prepare();

        sirius::apply_S_operator<T, std::complex<T>>(mem, wf::spin_range(0), wf::band_range(0, nwf), bp_gen, bp_coeffs,
                                                     *atomic_wave_functions_, q_op.get(), *atomic_wave_functions_S_);

        if (ctx_.cfg().hubbard().hubbard_subspace_method() == "full_orthogonalization") {
            /* save phi and sphi */

            wf_tmp  = std::make_unique<wf::Wave_functions<T>>(gkvec_, wf::num_mag_dims(0), wf::num_bands(nwf),
                                                             ctx_.host_memory_t());
            swf_tmp = std::make_unique<wf::Wave_functions<T>>(gkvec_, wf::num_mag_dims(0), wf::num_bands(nwf),
                                                              ctx_.host_memory_t());

            auto mg3 = wf_tmp->memory_guard(mem, wf::copy_to::host);
            auto mg4 = swf_tmp->memory_guard(mem, wf::copy_to::host);

            wf::copy(mem, *atomic_wave_functions_, wf::spin_index(0), wf::band_range(0, nwf), *wf_tmp,
                     wf::spin_index(0), wf::band_range(0, nwf));
            wf::copy(mem, *atomic_wave_functions_S_, wf::spin_index(0), wf::band_range(0, nwf), *swf_tmp,
                     wf::spin_index(0), wf::band_range(0, nwf));

            int BS = ctx_.cyclic_block_size();
            la::dmatrix<std::complex<T>> ovlp(nwf, nwf, ctx_.blacs_grid(), BS, BS);

            wf::inner(ctx_.spla_context(), mem, wf::spin_range(0), *atomic_wave_functions_, wf::band_range(0, nwf),
                      *atomic_wave_functions_S_, wf::band_range(0, nwf), ovlp, 0, 0);
            auto B = std::get<0>(inverse_sqrt(ovlp, nwf));

            /* use sphi as temporary */
            wf::transform(ctx_.spla_context(), mem, *B, 0, 0, 1.0, *atomic_wave_functions_, wf::spin_index(0),
                          wf::band_range(0, nwf), 0.0, *atomic_wave_functions_S_, wf::spin_index(0),
                          wf::band_range(0, nwf));

            wf::copy(mem, *atomic_wave_functions_S_, wf::spin_index(0), wf::band_range(0, nwf), *atomic_wave_functions_,
                     wf::spin_index(0), wf::band_range(0, nwf));

            apply_S_operator<T, std::complex<T>>(mem, wf::spin_range(0), wf::band_range(0, nwf), bp_gen, bp_coeffs,
                                                 *atomic_wave_functions_, q_op.get(), *atomic_wave_functions_S_);

            // if (ctx_.cfg().control().verification() >= 1) {
            //     sddk::inner(ctx_.spla_context(), sddk::spin_range(0), phi, 0, nwf, sphi, 0, nwf, ovlp, 0, 0);

            //    auto diff = check_identity(ovlp, nwf);
            //    RTE_OUT(std::cout) << "orthogonalization error " << diff << std::endl;
            //}
        }

        // beta_projectors().dismiss();
    }

    if (pcs) {
        auto cs = atomic_wave_functions_S_->checksum(memory_t::host, wf::spin_index(0), wf::band_range(0, nwf));
        if (this->comm().rank() == 0) {
            print_checksum("atomic_wave_functions_S", cs, RTE_OUT(std::cout));
        }
    }

    auto num_hubbard_wf = unit_cell_.num_hubbard_wf();

    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        auto& atom = ctx_.unit_cell().atom(ia);
        auto& type = atom.type();
        if (type.hubbard_correction()) {
            /* loop over Hubbard orbitals of the atom */
            for (auto e : type.indexr_hub()) {
                /* Hubbard orbital descriptor */
                auto& hd = type.lo_descriptor_hub(e.idxrf);
                int l    = e.am.l();
                int mmax = 2 * l + 1;

                int idxr_wf = hd.idx_wf();

                int offset_in_wf  = num_ps_atomic_wf.second[ia] + type.indexb_wfs().index_of(rf_index(idxr_wf));
                int offset_in_hwf = num_hubbard_wf.second[ia] + type.indexb_hub().index_of(e.idxrf);

                wf::copy(memory_t::host, *atomic_wave_functions_, wf::spin_index(0),
                         wf::band_range(offset_in_wf, offset_in_wf + mmax), *hubbard_wave_functions_, wf::spin_index(0),
                         wf::band_range(offset_in_hwf, offset_in_hwf + mmax));

                wf::copy(memory_t::host, *atomic_wave_functions_S_, wf::spin_index(0),
                         wf::band_range(offset_in_wf, offset_in_wf + mmax), *hubbard_wave_functions_S_,
                         wf::spin_index(0), wf::band_range(offset_in_hwf, offset_in_hwf + mmax));
            }
        }
    }
    /* restore phi and sphi */
    if (ctx_.cfg().hubbard().hubbard_subspace_method() == "full_orthogonalization") {

        wf::copy(memory_t::host, *wf_tmp, wf::spin_index(0), wf::band_range(0, nwf), *atomic_wave_functions_,
                 wf::spin_index(0), wf::band_range(0, nwf));
        wf::copy(memory_t::host, *swf_tmp, wf::spin_index(0), wf::band_range(0, nwf), *atomic_wave_functions_S_,
                 wf::spin_index(0), wf::band_range(0, nwf));
    }

    if (pcs) {
        auto cs1 = hubbard_wave_functions_->checksum(memory_t::host, wf::spin_index(0),
                                                     wf::band_range(0, num_hubbard_wf.first));
        auto cs2 = hubbard_wave_functions_S_->checksum(memory_t::host, wf::spin_index(0),
                                                       wf::band_range(0, num_hubbard_wf.first));
        if (comm().rank() == 0) {
            print_checksum("hubbard_wave_functions", cs1, RTE_OUT(std::cout));
            print_checksum("hubbard_wave_functions_S", cs2, RTE_OUT(std::cout));
        }
    }
}

template <typename T>
void
K_point<T>::generate_gkvec(double gk_cutoff__)
{
    PROFILE("sirius::K_point::generate_gkvec");

    gkvec_partition_ =
            std::make_shared<fft::Gvec_fft>(this->gkvec(), ctx_.comm_fft_coarse(), ctx_.comm_band_ortho_fft_coarse());

    const auto fft_type = gkvec_->reduced() ? SPFFT_TRANS_R2C : SPFFT_TRANS_C2C;
    const auto spfft_pu = ctx_.processing_unit() == device_t::CPU ? SPFFT_PU_HOST : SPFFT_PU_GPU;
    auto const& gv      = gkvec_partition_->gvec_array();
    /* create transformation */
    spfft_transform_.reset(new fft::spfft_transform_type<T>(ctx_.spfft_grid_coarse<T>().create_transform(
            spfft_pu, fft_type, ctx_.fft_coarse_grid()[0], ctx_.fft_coarse_grid()[1], ctx_.fft_coarse_grid()[2],
            ctx_.spfft_coarse<double>().local_z_length(), gkvec_partition_->count(), SPFFT_INDEX_TRIPLETS,
            gv.at(memory_t::host))));

    splindex_block_cyclic<> spl_ngk_row(num_gkvec(), n_blocks(num_ranks_row_), block_id(rank_row_),
                                        ctx_.cyclic_block_size());
    num_gkvec_row_ = spl_ngk_row.local_size();
    mdarray<int, 2> gkvec_row({3, num_gkvec_row_});

    splindex_block_cyclic<> spl_ngk_col(num_gkvec(), n_blocks(num_ranks_col_), block_id(rank_col_),
                                        ctx_.cyclic_block_size());
    num_gkvec_col_ = spl_ngk_col.local_size();
    mdarray<int, 2> gkvec_col({3, num_gkvec_col_});

    for (int rank = 0; rank < comm().size(); rank++) {
        auto gv = gkvec_->gvec_local(rank);
        for (int igloc = 0; igloc < gkvec_->count(rank); igloc++) {
            int ig       = gkvec_->offset(rank) + igloc;
            auto loc_row = spl_ngk_row.location(ig);
            auto loc_col = spl_ngk_col.location(ig);
            if (loc_row.ib == comm_row().rank()) {
                for (int x : {0, 1, 2}) {
                    gkvec_row(x, loc_row.index_local) = gv(x, igloc);
                }
            }
            if (loc_col.ib == comm_col().rank()) {
                for (int x : {0, 1, 2}) {
                    gkvec_col(x, loc_col.index_local) = gv(x, igloc);
                }
            }
        }
    }
    gkvec_row_ = std::make_shared<fft::Gvec>(vk_, unit_cell_.reciprocal_lattice_vectors(), num_gkvec_row_,
                                             &gkvec_row(0, 0), comm_row(), ctx_.gamma_point());

    gkvec_col_ = std::make_shared<fft::Gvec>(vk_, unit_cell_.reciprocal_lattice_vectors(), num_gkvec_col_,
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
    }
}

template <typename T>
void
K_point<T>::get_fv_eigen_vectors(mdarray<std::complex<T>, 2>& fv_evec__) const
{
    RTE_ASSERT((int)fv_evec__.size(0) >= gklo_basis_size());
    RTE_ASSERT((int)fv_evec__.size(1) == ctx_.num_fv_states());
    RTE_ASSERT(gklo_basis_size_row() == fv_eigen_vectors_.num_rows_local());

    mdarray<std::complex<T>, 1> tmp({gklo_basis_size_row()});

    /* zero global array */
    fv_evec__.zero();

    try {
        for (int ist = 0; ist < ctx_.num_fv_states(); ist++) {
            for (int igloc = 0; igloc < this->num_gkvec_loc(); igloc++) {
                int ig             = this->gkvec().offset() + igloc;
                fv_evec__(ig, ist) = fv_eigen_vectors_slab_->pw_coeffs(igloc, wf::spin_index(0), wf::band_index(ist));
            }
            this->comm().allgather(fv_evec__.at(memory_t::host, 0, ist), this->gkvec().count(), this->gkvec().offset());
        }
    } catch (std::exception const& e) {
        std::stringstream s;
        s << e.what() << std::endl;
        s << "Error in getting plane-wave coefficients";
        RTE_THROW(s);
    }

    try {
        for (int ist = 0; ist < ctx_.num_fv_states(); ist++) {
            /* offset in the global index of local orbitals */
            int offs{0};
            for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
                /* number of atom local orbitals */
                int nlo  = ctx_.unit_cell().atom(ia).mt_lo_basis_size();
                auto loc = fv_eigen_vectors_slab_->spl_num_atoms().location(typename atom_index_t::global(ia));
                if (loc.ib == this->comm().rank()) {
                    for (int xi = 0; xi < nlo; xi++) {
                        fv_evec__(this->num_gkvec() + offs + xi, ist) = fv_eigen_vectors_slab_->mt_coeffs(
                                xi, atom_index_t::local(loc.index_local), wf::spin_index(0), wf::band_index(ist));
                    }
                }
                offs += nlo;
            }
            auto& mtd = fv_eigen_vectors_slab_->mt_coeffs_distr();
            this->comm().allgather(fv_evec__.at(memory_t::host, this->num_gkvec(), ist), mtd.counts.data(),
                                   mtd.offsets.data());
        }
    } catch (std::exception const& e) {
        std::stringstream s;
        s << e.what() << std::endl;
        s << "Error in getting muffin-tin coefficients";
        RTE_THROW(s);
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
// sqrt(unit_cell_.omega());
//==             tdiff += abs(pw_value - aw_value);
//==         }
//==     }
//==
//==     printf("atom : %i  absolute alm error : %e  average alm error : %e\n",
//==            ia, tdiff, tdiff / (num_gkvec_loc * sht->num_points()));
//== }

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
        // TODO: only the list of z-columns is probably needed to recreate the G+k vectors
        // serializer s;
        // gkvec().pack(s);
        // fout["K_point_set"][id__].write("gkvec", s.stream());

        /* save the order of G-vectors */
        mdarray<int, 2> gv({3, num_gkvec()});
        for (int i = 0; i < num_gkvec(); i++) {
            auto v = gkvec().gvec(gvec_index_t::global(i));
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
    // int gkvec_count  = gkvec().count();
    // int gkvec_offset = gkvec().offset();
    // std::vector<std::complex<T>> wf_tmp(num_gkvec());

    // std::unique_ptr<sddk::HDF5_tree> fout;
    /* rank 0 opens a file */
    // if (comm().rank() == 0) {
    //     fout = std::make_unique<sddk::HDF5_tree>(name__, sddk::hdf5_access_t::read_write);
    // }

    RTE_THROW("re-implement");

    ///* store wave-functions */
    // for (int i = 0; i < ctx_.num_bands(); i++) {
    //     for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
    //         /* gather full column of PW coefficients on rank 0 */
    //         comm().gather(&spinor_wave_functions_->pw_coeffs(ispn).prime(0, i), wf_tmp.data(), gkvec_offset,
    //                       gkvec_count, 0);
    //         if (comm().rank() == 0) {
    //             (*fout)["K_point_set"][id__]["bands"][i]["spinor_wave_function"][ispn].write("pw", wf_tmp);
    //         }
    //     }
    //     comm().barrier();
    // }
}

template <typename T>
void
K_point<T>::load(HDF5_tree h5in, int id)
{
    RTE_THROW("not implemented");
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
                                           std::function<basis_functions_index const*(int)> indexb__,
                                           Radial_integrals_atomic_wf<false> const& ri__, wf::Wave_functions<T>& wf__)
{
    PROFILE("sirius::K_point::generate_atomic_wave_functions");

    int lmax{3};
    int lmmax = sf::lmmax(lmax);

    /* compute offset for each atom */
    std::vector<int> offset;
    int n{0};
    for (int ia : atoms__) {
        offset.push_back(n);
        int iat = unit_cell_.atom(ia).type_id();
        n += indexb__(iat)->size();
    }

    /* allocate memory to store wave-functions for atom types */
    std::vector<mdarray<std::complex<T>, 2>> wf_t(unit_cell_.num_atom_types());
    for (int ia : atoms__) {
        int iat = unit_cell_.atom(ia).type_id();
        if (wf_t[iat].size() == 0) {
            wf_t[iat] = mdarray<std::complex<T>, 2>({this->num_gkvec_loc(), indexb__(iat)->size()},
                                                    get_memory_pool(memory_t::host));
        }
    }

    #pragma omp parallel for schedule(static)
    for (int igk_loc = 0; igk_loc < this->num_gkvec_loc(); igk_loc++) {
        /* vs = {r, theta, phi} */
        auto vs = r3::spherical_coordinates(this->gkvec().gkvec_cart(gvec_index_t::local(igk_loc)));

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
            for (auto const& e : indexb) {
                auto z = std::pow(std::complex<double>(0, -1), e.am.l()) * fourpi / std::sqrt(unit_cell_.omega());

                wf_t[iat](igk_loc, e.xi) = static_cast<std::complex<T>>(z * rlm[e.lm] * ri_values[iat](e.idxrf));
            }
        }
    }

    for (int ia : atoms__) {

        T phase      = twopi * dot(gkvec().vk(), unit_cell_.atom(ia).position());
        auto phase_k = std::exp(std::complex<T>(0.0, phase));

        /* quickly compute phase factors without calling exp() function */
        std::vector<std::complex<T>> phase_gk(num_gkvec_loc());
        #pragma omp parallel for
        for (int igk_loc = 0; igk_loc < num_gkvec_loc(); igk_loc++) {
            auto G = gkvec().gvec(gvec_index_t::local(igk_loc));
            /* total phase e^{-i(G+k)r_{\alpha}} */
            phase_gk[igk_loc] = std::conj(static_cast<std::complex<T>>(ctx_.gvec_phase_factor(G, ia)) * phase_k);
        }

        int iat = unit_cell_.atom(ia).type_id();
        #pragma omp parallel
        for (int xi = 0; xi < static_cast<int>(indexb__(iat)->size()); xi++) {
            #pragma omp for nowait
            for (int igk_loc = 0; igk_loc < num_gkvec_loc(); igk_loc++) {
                wf__.pw_coeffs(igk_loc, wf::spin_index(0), wf::band_index(offset[ia] + xi)) =
                        wf_t[iat](igk_loc, xi) * phase_gk[igk_loc];
            }
        }
    }
}

template <typename T>
void
K_point<T>::generate_gklo_basis()
{
    /* find local number of row G+k vectors */
    splindex_block_cyclic<> spl_ngk_row(num_gkvec(), n_blocks(num_ranks_row_), block_id(rank_row_),
                                        ctx_.cyclic_block_size());
    num_gkvec_row_ = spl_ngk_row.local_size();

    /* find local number of column G+k vectors */
    splindex_block_cyclic<> spl_ngk_col(num_gkvec(), n_blocks(num_ranks_col_), block_id(rank_col_),
                                        ctx_.cyclic_block_size());
    num_gkvec_col_ = spl_ngk_col.local_size();

    if (ctx_.full_potential()) {
        splindex_block_cyclic<> spl_nlo_row(num_gkvec() + unit_cell_.mt_lo_basis_size(), n_blocks(num_ranks_row_),
                                            block_id(rank_row_), ctx_.cyclic_block_size());
        splindex_block_cyclic<> spl_nlo_col(num_gkvec() + unit_cell_.mt_lo_basis_size(), n_blocks(num_ranks_col_),
                                            block_id(rank_col_), ctx_.cyclic_block_size());

        lo_basis_descriptor lo_desc;

        int idx{0};
        /* local orbital basis functions */
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto& atom = unit_cell_.atom(ia);
            auto& type = atom.type();

            int lo_index_offset = type.mt_aw_basis_size();

            for (int j = 0; j < type.mt_lo_basis_size(); j++) {
                int l         = type.indexb(lo_index_offset + j).am.l();
                int lm        = type.indexb(lo_index_offset + j).lm;
                int order     = type.indexb(lo_index_offset + j).order;
                int idxrf     = type.indexb(lo_index_offset + j).idxrf;
                lo_desc.ia    = static_cast<uint16_t>(ia);
                lo_desc.l     = static_cast<uint8_t>(l);
                lo_desc.lm    = static_cast<uint16_t>(lm);
                lo_desc.order = static_cast<uint8_t>(order);
                lo_desc.idxrf = static_cast<uint8_t>(idxrf);

                if (spl_nlo_row.location(num_gkvec() + idx).ib == rank_row_) {
                    lo_basis_descriptors_row_.push_back(lo_desc);
                }
                if (spl_nlo_col.location(num_gkvec() + idx).ib == rank_col_) {
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
#ifdef SIRIUS_USE_FP32
template class K_point<float>;
#endif
} // namespace sirius
