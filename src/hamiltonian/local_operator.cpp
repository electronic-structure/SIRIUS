/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file local_operator.cpp
 *
 *  \brief Implementation of sirius::Local_operator class.
 */

#include "local_operator.hpp"
#include "potential/potential.hpp"
#include "function3d/smooth_periodic_function.hpp"
#include "core/profiler.hpp"
#include "core/wf/wave_functions.hpp"

namespace sirius {

template <typename T>
Local_operator<T>::Local_operator(Simulation_context const& ctx__, fft::spfft_transform_type<T>& fft_coarse__,
                                  std::shared_ptr<fft::Gvec_fft> gvec_coarse_p__, Potential* potential__)
    : ctx_(ctx__)
    , fft_coarse_(fft_coarse__)
    , gvec_coarse_p_(gvec_coarse_p__)

{
    PROFILE("sirius::Local_operator");

    /* allocate functions */
    for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
        veff_vec_[j] = std::make_unique<Smooth_periodic_function<T>>(fft_coarse__, gvec_coarse_p__);
        #pragma omp parallel for schedule(static)
        for (int ir = 0; ir < fft_coarse__.local_slice_size(); ir++) {
            veff_vec_[j]->value(ir) = 2.71828;
        }
    }
    /* map Theta(r) to the coarse mesh */
    if (ctx_.full_potential()) {
        auto& gvec_dense_p = ctx_.gvec_fft();
        veff_vec_[v_local_index_t::theta] =
                std::make_unique<Smooth_periodic_function<T>>(fft_coarse__, gvec_coarse_p__);
        /* map unit-step function */
        #pragma omp parallel for schedule(static)
        for (int igloc = 0; igloc < gvec_coarse_p_->gvec().count(); igloc++) {
            /* map from fine to coarse set of G-vectors */
            veff_vec_[v_local_index_t::theta]->f_pw_local(igloc) =
                    ctx_.theta_pw(gvec_dense_p.gvec().gvec_base_mapping(igloc) + gvec_dense_p.gvec().offset());
        }
        veff_vec_[v_local_index_t::theta]->fft_transform(1);
        if (fft_coarse_.processing_unit() == SPFFT_PU_GPU) {
            veff_vec_[v_local_index_t::theta]
                    ->values()
                    .allocate(get_memory_pool(memory_t::device))
                    .copy_to(memory_t::device);
        }
        if (env::print_checksum()) {
            auto cs1 = veff_vec_[v_local_index_t::theta]->checksum_pw();
            auto cs2 = veff_vec_[v_local_index_t::theta]->checksum_rg();
            print_checksum("theta_pw", cs1, ctx_.out());
            print_checksum("theta_rg", cs2, ctx_.out());
        }
    }

    /* map potential */
    if (potential__) {

        if (ctx_.full_potential()) {

            auto& fft_dense    = ctx_.spfft<T>();
            auto& gvec_dense_p = ctx_.gvec_fft();

            Smooth_periodic_function<T> ftmp(ctx_.spfft<T>(), ctx_.gvec_fft_sptr());

            for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                /* multiply potential by step function theta(r) */
                for (int ir = 0; ir < fft_dense.local_slice_size(); ir++) {
                    ftmp.value(ir) = potential__->component(j).rg().value(ir) * ctx_.theta(ir);
                }
                /* transform to plane-wave domain */
                ftmp.fft_transform(-1);
                if (j == 0) {
                    v0_[0] = ftmp.f_0().real();
                }
                /* loop over local set of coarse G-vectors */
                #pragma omp parallel for schedule(static)
                for (int igloc = 0; igloc < gvec_coarse_p_->gvec().count(); igloc++) {
                    /* map from fine to coarse set of G-vectors */
                    veff_vec_[j]->f_pw_local(igloc) = ftmp.f_pw_local(gvec_dense_p.gvec().gvec_base_mapping(igloc));
                }
                /* transform to real space */
                veff_vec_[j]->fft_transform(1);
            }
            if (ctx_.valence_relativity() == relativity_t::zora) {
                veff_vec_[v_local_index_t::rm_inv] =
                        std::make_unique<Smooth_periodic_function<T>>(fft_coarse__, gvec_coarse_p__);
                /* loop over local set of coarse G-vectors */
                #pragma omp parallel for schedule(static)
                for (int igloc = 0; igloc < gvec_coarse_p_->gvec().count(); igloc++) {
                    /* map from fine to coarse set of G-vectors */
                    veff_vec_[v_local_index_t::rm_inv]->f_pw_local(igloc) = potential__->rm_inv_pw(
                            gvec_dense_p.gvec().offset() + gvec_dense_p.gvec().gvec_base_mapping(igloc));
                }
                /* transform to real space */
                veff_vec_[v_local_index_t::rm_inv]->fft_transform(1);
            }

        } else {

            for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                /* loop over local set of coarse G-vectors */
                #pragma omp parallel for schedule(static)
                for (int igloc = 0; igloc < gvec_coarse_p_->gvec().count(); igloc++) {
                    /* map from fine to coarse set of G-vectors */
                    veff_vec_[j]->f_pw_local(igloc) = potential__->component(j).rg().f_pw_local(
                            potential__->component(j).rg().gvec().gvec_base_mapping(igloc));
                }
                /* transform to real space */
                veff_vec_[j]->fft_transform(1);
            }

            /* change to canonical form */
            if (ctx_.num_mag_dims()) {
                #pragma omp parallel for schedule(static)
                for (int ir = 0; ir < fft_coarse_.local_slice_size(); ir++) {
                    T v0                                      = veff_vec_[v_local_index_t::v0]->value(ir);
                    T v1                                      = veff_vec_[v_local_index_t::v1]->value(ir);
                    veff_vec_[v_local_index_t::v0]->value(ir) = v0 + v1; // v + Bz
                    veff_vec_[v_local_index_t::v1]->value(ir) = v0 - v1; // v - Bz
                }
            }

            if (ctx_.num_mag_dims() == 0) {
                v0_[0] = potential__->component(0).rg().f_0().real();
            } else {
                v0_[0] = potential__->component(0).rg().f_0().real() + potential__->component(1).rg().f_0().real();
                v0_[1] = potential__->component(0).rg().f_0().real() - potential__->component(1).rg().f_0().real();
            }
        }

        if (env::print_checksum()) {
            for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                auto cs1 = veff_vec_[j]->checksum_pw();
                auto cs2 = veff_vec_[j]->checksum_rg();
                print_checksum("veff_pw", cs1, ctx_.out());
                print_checksum("veff_rg", cs2, ctx_.out());
            }
        }
    }

    buf_rg_ = mdarray<std::complex<T>, 1>({fft_coarse_.local_slice_size()}, get_memory_pool(memory_t::host),
                                          mdarray_label("Local_operator::buf_rg_"));
    /* move functions to GPU */
    if (fft_coarse_.processing_unit() == SPFFT_PU_GPU) {
        for (int j = 0; j < 6; j++) {
            if (veff_vec_[j]) {
                veff_vec_[j]->values().allocate(get_memory_pool(memory_t::device)).copy_to(memory_t::device);
            }
        }
        buf_rg_.allocate(get_memory_pool(memory_t::device));
    }
}

template <typename T>
void
Local_operator<T>::prepare_k(fft::Gvec_fft const& gkvec_p__)
{
    PROFILE("sirius::Local_operator::prepare_k");

    int ngv_fft = gkvec_p__.count();

    /* cache kinteic energy of plane-waves */
    pw_ekin_ = mdarray<T, 1>({ngv_fft}, get_memory_pool(memory_t::host), mdarray_label("Local_operator::pw_ekin"));
    gkvec_cart_ =
            mdarray<T, 2>({ngv_fft, 3}, get_memory_pool(memory_t::host), mdarray_label("Local_operator::gkvec_cart"));
    vphi_ = mdarray<std::complex<T>, 1>({ngv_fft}, get_memory_pool(memory_t::host),
                                        mdarray_label("Local_operator::vphi"));

    #pragma omp parallel for schedule(static)
    for (int ig_loc = 0; ig_loc < ngv_fft; ig_loc++) {
        /* get G+k in Cartesian coordinates */
        auto gv          = gkvec_p__.gkvec_cart(ig_loc);
        pw_ekin_[ig_loc] = 0.5 * dot(gv, gv);
        for (int x : {0, 1, 2}) {
            gkvec_cart_(ig_loc, x) = gv[x];
        }
    }

    if (fft_coarse_.processing_unit() == SPFFT_PU_GPU) {
        pw_ekin_.allocate(get_memory_pool(memory_t::device)).copy_to(memory_t::device);
        vphi_.allocate(get_memory_pool(memory_t::device));
        gkvec_cart_.allocate(get_memory_pool(memory_t::device)).copy_to(memory_t::device);
    }
}

/// Multiply FFT buffer by the effective potential.
template <typename T>
static inline void
mul_by_veff(fft::spfft_transform_type<T>& spfftk__, T const* in__,
            std::array<std::unique_ptr<Smooth_periodic_function<T>>, 6> const& veff_vec__, int idx_veff__, T* out__)
{
    int nr = spfftk__.local_slice_size();

    switch (spfftk__.processing_unit()) {
        case SPFFT_PU_HOST: {
            if (idx_veff__ <= 1 || idx_veff__ >= 4) { /* up-up or dn-dn block or Theta(r) */
                switch (spfftk__.type()) {
                    case SPFFT_TRANS_R2C: {
                        #pragma omp parallel for
                        for (int ir = 0; ir < nr; ir++) {
                            /* multiply by V+Bz or V-Bz (in PP-PW case) or by V(r), B_z(r) or Theta(r) (in LAPW case) */
                            out__[ir] = in__[ir] * veff_vec__[idx_veff__]->value(ir);
                        }
                        break;
                    }
                    case SPFFT_TRANS_C2C: {
                        auto in  = reinterpret_cast<std::complex<T> const*>(in__);
                        auto out = reinterpret_cast<std::complex<T>*>(out__);
                        #pragma omp parallel for
                        for (int ir = 0; ir < nr; ir++) {
                            /* multiply by V+Bz or V-Bz (in PP-PW case) or by V(r), B_z(r) or Theta(r) (in LAPW case) */
                            out[ir] = in[ir] * veff_vec__[idx_veff__]->value(ir);
                        }
                        break;
                    }
                }
            } else { /* special case for idx_veff = 2 or idx_veff__ = 3 */
                T pref   = (idx_veff__ == 2) ? -1 : 1;
                auto in  = reinterpret_cast<std::complex<T> const*>(in__);
                auto out = reinterpret_cast<std::complex<T>*>(out__);
                #pragma omp parallel for schedule(static)
                for (int ir = 0; ir < nr; ir++) {
                    /* multiply by Bx +/- i*By */
                    out[ir] = in[ir] * std::complex<T>(veff_vec__[2]->value(ir), pref * veff_vec__[3]->value(ir));
                }
            }
            break;
        }
        case SPFFT_PU_GPU: {
            if (idx_veff__ <= 1 || idx_veff__ >= 4) { /* up-up or dn-dn block or Theta(r) */
                switch (spfftk__.type()) {
                    case SPFFT_TRANS_R2C: {
                        /* multiply by V+Bz or V-Bz (in PP-PW case) or by V(r), B_z(r) or Theta(r) (in LAPW case) */
                        mul_by_veff_real_real_gpu(nr, in__, veff_vec__[idx_veff__]->values().at(memory_t::device),
                                                  out__);
                        break;
                    }
                    case SPFFT_TRANS_C2C: {
                        auto in  = reinterpret_cast<std::complex<T> const*>(in__);
                        auto out = reinterpret_cast<std::complex<T>*>(out__);
                        /* multiply by V+Bz or V-Bz (in PP-PW case) or by V(r), B_z(r) or Theta(r) (in LAPW case) */
                        mul_by_veff_complex_real_gpu(nr, in, veff_vec__[idx_veff__]->values().at(memory_t::device),
                                                     out);
                        break;
                    }
                }
            } else {
                /* multiply by Bx +/- i*By */
                T pref   = (idx_veff__ == 2) ? -1 : 1;
                auto in  = reinterpret_cast<std::complex<T> const*>(in__);
                auto out = reinterpret_cast<std::complex<T>*>(out__);

                mul_by_veff_complex_complex_gpu(nr, in, pref, veff_vec__[2]->values().at(memory_t::device),
                                                veff_vec__[3]->values().at(memory_t::device), out);
            }
            break;
        } break;
    }
}

template <typename T>
void
Local_operator<T>::apply_h(fft::spfft_transform_type<T>& spfftk__, std::shared_ptr<fft::Gvec_fft> gkvec_fft__,
                           wf::spin_range spins__, wf::Wave_functions<T> const& phi__, wf::Wave_functions<T>& hphi__,
                           wf::band_range br__)
{
    PROFILE("sirius::Local_operator::apply_h");

    if ((spfftk__.dim_x() != fft_coarse_.dim_x()) || (spfftk__.dim_y() != fft_coarse_.dim_y()) ||
        (spfftk__.dim_z() != fft_coarse_.dim_z())) {
        RTE_THROW("wrong FFT dimensions");
    }

    /* increment the counter by the number of wave-functions */
    ctx_.num_loc_op_applied(br__.size());

    /* local number of G-vectors for the FFT transformation */
    int ngv_fft = gkvec_fft__->count();

    if (ngv_fft != spfftk__.num_local_elements()) {
        RTE_THROW("wrong number of G-vectors");
    }

    std::array<wf::Wave_functions_fft<T>, 2> phi_fft;
    std::array<wf::Wave_functions_fft<T>, 2> hphi_fft;
    for (auto s = spins__.begin(); s != spins__.end(); s++) {
        phi_fft[s.get()] = wf::Wave_functions_fft<T>(gkvec_fft__, const_cast<wf::Wave_functions<T>&>(phi__), s, br__,
                                                     wf::shuffle_to::fft_layout);

        hphi_fft[s.get()] = wf::Wave_functions_fft<T>(gkvec_fft__, hphi__, s, br__, wf::shuffle_to::wf_layout);
        auto hphi_mem     = hphi_fft[s.get()].on_device() ? memory_t::device : memory_t::host;
        hphi_fft[s.get()].zero(hphi_mem, wf::spin_index(0), wf::band_range(0, hphi_fft[s.get()].num_wf_local()));
    }

    auto spl_num_wf = phi_fft[spins__.begin().get()].spl_num_wf();

    /* assume the location of data on the current processing unit */
    auto spfft_pu  = spfftk__.processing_unit();
    auto spfft_mem = fft::spfft_memory_t.at(spfft_pu);

    /* number of real-space points in the local part of FFT buffer */
    int nr = spfftk__.local_slice_size();

    /* pointer to FFT buffer */
    auto spfft_buf = spfftk__.space_domain_data(spfft_pu);

    /* transform wave-function to real space; the result of the transformation is stored in the FFT buffer */
    auto phi_to_r = [&](wf::spin_index ispn, wf::band_index i) {
        auto phi_mem = phi_fft[ispn.get()].on_device() ? memory_t::device : memory_t::host;
        spfftk__.backward(phi_fft[ispn.get()].pw_coeffs_spfft(phi_mem, i), spfft_pu);
    };

    /* transform function to PW domain */
    auto vphi_to_G = [&]() {
        spfftk__.forward(spfft_pu, reinterpret_cast<T*>(vphi_.at(spfft_mem)), SPFFT_FULL_SCALING);
    };

    /* store the resulting hphi
       spin block (ispn_block) is used as a bit mask:
        - first bit: spin component which is updated
        - second bit: add or not kinetic energy term */
    auto add_to_hphi = [&](int ispn_block, wf::band_index i) {
        /* index of spin component */
        int ispn = ispn_block & 1;
        /* add kinetic energy if this is a diagonal block */
        int ekin = (ispn_block & 2) ? 0 : 1;

        auto hphi_mem = hphi_fft[ispn].on_device() ? memory_t::device : memory_t::host;

        switch (hphi_mem) {
            case memory_t::host: {
                if (spfft_pu == SPFFT_PU_GPU) {
                    vphi_.copy_to(memory_t::host);
                }
                /* CPU case */
                if (ekin) {
                    #pragma omp parallel for
                    for (int ig = 0; ig < ngv_fft; ig++) {
                        hphi_fft[ispn].pw_coeffs(ig, i) += phi_fft[ispn].pw_coeffs(ig, i) * pw_ekin_[ig] + vphi_[ig];
                    }
                } else {
                    #pragma omp parallel for
                    for (int ig = 0; ig < ngv_fft; ig++) {
                        hphi_fft[ispn].pw_coeffs(ig, wf::band_index(i)) += vphi_[ig];
                    }
                }
                break;
            }
            case memory_t::device: {
                add_to_hphi_pw_gpu(ngv_fft, ekin, pw_ekin_.at(memory_t::device),
                                   phi_fft[ispn].at(memory_t::device, 0, wf::band_index(i)), vphi_.at(memory_t::device),
                                   hphi_fft[ispn].at(memory_t::device, 0, wf::band_index(i)));
                break;
            }
            default: {
                break;
            }
        }
    };

    auto copy_phi = [&]() {
        switch (spfft_pu) {
            /* this is a non-collinear case, so the wave-functions and FFT buffer are complex and
               we can copy memory */
            case SPFFT_PU_HOST: {
                auto inp = reinterpret_cast<std::complex<T>*>(spfft_buf);
                std::copy(inp, inp + nr, buf_rg_.at(memory_t::host));
                break;
            }
            case SPFFT_PU_GPU: {
                acc::copy(buf_rg_.at(memory_t::device), reinterpret_cast<std::complex<T>*>(spfft_buf), nr);
                break;
            }
        }
    };

    PROFILE_START("sirius::Local_operator::apply_h|bands");
    for (int i = 0; i < spl_num_wf.local_size(); i++) {

        /* non-collinear case */
        /* 2x2 Hamiltonian in applied to spinor wave-functions
           .--------.--------.   .-----.   .------.
           |        |        |   |     |   |      |
           | H_{uu} | H_{ud} |   |phi_u|   |hphi_u|
           |        |        |   |     |   |      |
           .--------.--------. x .-----. = .------.
           |        |        |   |     |   |      |
           | H_{du} | H_{dd} |   |phi_d|   |hphi_d|
           |        |        |   |     |   |      |
           .--------.--------.   .-----.   .------.

           hphi_u = H_{uu} phi_u + H_{ud} phi_d
           hphi_d = H_{du} phi_u + H_{dd} phi_d

           The following indexing scheme will be used for spin-blocks
           .---.---.
           | 0 | 2 |
           .---.---.
           | 3 | 1 |
           .---.---.
        */
        if (spins__.size() == 2) {
            /* phi_u(G) -> phi_u(r) */
            phi_to_r(wf::spin_index(0), wf::band_index(i));
            /* save phi_u(r) in temporary buf_rg array */
            copy_phi();
            /* multiply phi_u(r) by effective potential */
            mul_by_veff<T>(spfftk__, spfft_buf, veff_vec_, v_local_index_t::v0, spfft_buf);

            /* V_{uu}(r)phi_{u}(r) -> [V*phi]_{u}(G) */
            vphi_to_G();
            /* add kinetic energy */
            add_to_hphi(0, wf::band_index(i));
            /* multiply phi_{u} by V_{du} and copy to FFT buffer */
            mul_by_veff<T>(spfftk__, reinterpret_cast<T*>(buf_rg_.at(spfft_mem)), veff_vec_, 3, spfft_buf);
            /* V_{du}(r)phi_{u}(r) -> [V*phi]_{d}(G) */
            vphi_to_G();
            /* add to hphi_{d} */
            add_to_hphi(3, wf::band_index(i));

            /* for the second spin component */

            /* phi_d(G) -> phi_d(r) */
            phi_to_r(wf::spin_index(1), wf::band_index(i));
            /* save phi_d(r) */
            copy_phi();
            /* multiply phi_d(r) by effective potential */
            mul_by_veff<T>(spfftk__, spfft_buf, veff_vec_, v_local_index_t::v1, spfft_buf);
            /* V_{dd}(r)phi_{d}(r) -> [V*phi]_{d}(G) */
            vphi_to_G();
            /* add kinetic energy */
            add_to_hphi(1, wf::band_index(i));
            /* multiply phi_{d} by V_{ud} and copy to FFT buffer */
            mul_by_veff<T>(spfftk__, reinterpret_cast<T*>(buf_rg_.at(spfft_mem)), veff_vec_, 2, spfft_buf);
            /* V_{ud}(r)phi_{d}(r) -> [V*phi]_{u}(G) */
            vphi_to_G();
            /* add to hphi_{u} */
            add_to_hphi(2, wf::band_index(i));
        } else { /* spin-collinear or non-magnetic case */
            /* phi(G) -> phi(r) */
            phi_to_r(spins__.begin(), wf::band_index(i));
            /* multiply by effective potential */
            mul_by_veff<T>(spfftk__, spfft_buf, veff_vec_, spins__.begin().get(), spfft_buf);
            /* V(r)phi(r) -> [V*phi](G) */
            vphi_to_G();
            /* add kinetic energy */
            add_to_hphi(spins__.begin().get(), wf::band_index(i));
        }
    }
    PROFILE_STOP("sirius::Local_operator::apply_h|bands");
}

// This is full-potential case. Only C2C FFT transformation is considered here.
// TODO: document the data location on input/output
template <typename T>
void
Local_operator<T>::apply_fplapw(fft::spfft_transform_type<T>& spfftk__, std::shared_ptr<fft::Gvec_fft> gkvec_fft__,
                                wf::band_range b__, wf::Wave_functions<T>& phi__, wf::Wave_functions<T>* hphi__,
                                wf::Wave_functions<T>* ophi__, wf::Wave_functions<T>* bzphi__,
                                wf::Wave_functions<T>* bxyphi__)
{
    PROFILE("sirius::Local_operator::apply_h_o");

    ctx_.num_loc_op_applied(b__.size());

    /* assume the location of data on the current processing unit */
    auto spfft_pu = spfftk__.processing_unit();

    auto spfft_mem = fft::spfft_memory_t.at(spfft_pu);

    auto s0 = wf::spin_index(0);

    wf::Wave_functions_fft<T> phi_fft(gkvec_fft__, phi__, s0, b__, wf::shuffle_to::fft_layout);

    std::array<wf::Wave_functions_fft<T>, 4> wf_fft;
    if (hphi__) {
        wf_fft[0] = wf::Wave_functions_fft<T>(gkvec_fft__, *hphi__, s0, b__, wf::shuffle_to::wf_layout);
    }
    if (ophi__) {
        wf_fft[1] = wf::Wave_functions_fft<T>(gkvec_fft__, *ophi__, s0, b__, wf::shuffle_to::wf_layout);
    }
    if (bzphi__) {
        wf_fft[2] = wf::Wave_functions_fft<T>(gkvec_fft__, *bzphi__, s0, b__, wf::shuffle_to::wf_layout);
    }
    if (bxyphi__) {
        wf_fft[3] = wf::Wave_functions_fft<T>(gkvec_fft__, *bxyphi__, s0, b__, wf::shuffle_to::wf_layout);
    }

    auto pcs = env::print_checksum();

    if (pcs) {
        auto cs = phi__.checksum_pw(spfft_mem, wf::spin_index(0), b__);
        if (phi__.gkvec().comm().rank() == 0) {
            print_checksum("theta_pw", cs, RTE_OUT(std::cout));
        }
    }

    // auto& mp = const_cast<Simulation_context&>(ctx_).mem_pool(memory_t::host);

    auto spl_num_wf = phi_fft.spl_num_wf();

    /* number of real-space points in the local part of FFT buffer */
    int nr = spfftk__.local_slice_size();

    /* pointer to memory where SpFFT stores real-space data */
    auto spfft_buf = spfftk__.space_domain_data(spfft_pu);

    mdarray<std::complex<T>, 1> buf_pw({gkvec_fft__->count()}, get_memory_pool(ctx_.host_memory_t()));
    if (ctx_.processing_unit() == device_t::GPU) {
        buf_pw.allocate(get_memory_pool(memory_t::device));
    }

    auto phi_mem = phi_fft.on_device() ? memory_t::device : memory_t::host;

    auto phi_r = buf_rg_.at(spfft_mem);

    for (int j = 0; j < spl_num_wf.local_size(); j++) {
        /* phi(G) -> phi(r) */
        spfftk__.backward(phi_fft.pw_coeffs_spfft(phi_mem, wf::band_index(j)), spfft_pu);

        /* we are going to apply various terms of the Hamiltonian to the wave-function; save wave-function on the
         * real space grid first */

        /* save phi(r); real-space data is complex */
        auto inp = reinterpret_cast<std::complex<T>*>(spfft_buf);
        switch (spfft_pu) {
            case SPFFT_PU_HOST: {
                std::copy(inp, inp + nr, phi_r);
                break;
            }
            case SPFFT_PU_GPU: {
                acc::copy(phi_r, inp, nr);
                break;
            }
        }

        if (ophi__) {
            /* multiply phi(r) by step function */
            mul_by_veff(spfftk__, reinterpret_cast<T*>(phi_r), veff_vec_, v_local_index_t::theta, spfft_buf);

            auto mem = wf_fft[1].on_device() ? memory_t::device : memory_t::host;

            /* phi(r) * Theta(r) -> ophi(G) */
            spfftk__.forward(spfft_pu, wf_fft[1].pw_coeffs_spfft(mem, wf::band_index(j)), SPFFT_FULL_SCALING);
        }

        if (bzphi__) {
            mul_by_veff(spfftk__, reinterpret_cast<T*>(phi_r), veff_vec_, v_local_index_t::v1, spfft_buf);

            auto mem = wf_fft[2].on_device() ? memory_t::device : memory_t::host;

            /* phi(r) * Bz(r) -> bzphi(G) */
            spfftk__.forward(spfft_pu, wf_fft[2].pw_coeffs_spfft(mem, wf::band_index(j)), SPFFT_FULL_SCALING);
        }

        if (bxyphi__) {
            mul_by_veff(spfftk__, reinterpret_cast<T*>(phi_r), veff_vec_, 2, spfft_buf);

            auto mem = wf_fft[3].on_device() ? memory_t::device : memory_t::host;

            /* phi(r) * (Bx(r) - iBy(r)) -> bxyphi(G) */
            spfftk__.forward(spfft_pu, wf_fft[3].pw_coeffs_spfft(mem, wf::band_index(j)), SPFFT_FULL_SCALING);
        }

        if (hphi__) {
            mul_by_veff(spfftk__, reinterpret_cast<T*>(phi_r), veff_vec_, v_local_index_t::v0, spfft_buf);

            auto mem = wf_fft[0].on_device() ? memory_t::device : memory_t::host;

            /* phi(r) * Theta(r) * V(r) -> hphi(G) */
            spfftk__.forward(spfft_pu, wf_fft[0].pw_coeffs_spfft(mem, wf::band_index(j)), SPFFT_FULL_SCALING);

            /* add kinetic energy */
            for (int x : {0, 1, 2}) {
                if (is_host_memory(mem)) {
                    #pragma omp parallel for
                    for (int igloc = 0; igloc < gkvec_fft__->count(); igloc++) {
                        auto gvc = gkvec_fft__->gkvec_cart(igloc);
                        /* \hat P phi = phi(G+k) * (G+k), \hat P is momentum operator */
                        buf_pw[igloc] = phi_fft.pw_coeffs(igloc, wf::band_index(j)) * static_cast<T>(gvc[x]);
                    }
                } else {
                    grad_phi_lapw_gpu(gkvec_fft__->count(), phi_fft.at(mem, 0, wf::band_index(j)),
                                      gkvec_cart_.at(mem, 0, x), buf_pw.at(mem));
                }

                /* transform Cartesian component of wave-function gradient to real space */
                spfftk__.backward(reinterpret_cast<T const*>(buf_pw.at(mem)), spfft_pu);
                /* multiply by real-space function */
                switch (ctx_.valence_relativity()) {
                    case relativity_t::iora:
                    case relativity_t::zora: {
                        /* multiply be inverse relative mass */
                        mul_by_veff(spfftk__, spfft_buf, veff_vec_, v_local_index_t::rm_inv, spfft_buf);
                        break;
                    }
                    case relativity_t::none: {
                        /* multiply be step function */
                        mul_by_veff(spfftk__, spfft_buf, veff_vec_, v_local_index_t::theta, spfft_buf);
                        break;
                    }
                    default: {
                        break;
                    }
                }
                /* transform back to PW domain */
                spfftk__.forward(spfft_pu, reinterpret_cast<T*>(buf_pw.at(mem)), SPFFT_FULL_SCALING);
                if (is_host_memory(mem)) {
                    #pragma omp parallel for
                    for (int igloc = 0; igloc < gkvec_fft__->count(); igloc++) {
                        auto gvc = gkvec_fft__->gkvec_cart(igloc);
                        wf_fft[0].pw_coeffs(igloc, wf::band_index(j)) += buf_pw[igloc] * static_cast<T>(0.5 * gvc[x]);
                    }
                } else {
                    add_to_hphi_lapw_gpu(gkvec_fft__->count(), buf_pw.at(memory_t::device),
                                         gkvec_cart_.at(memory_t::device, 0, x),
                                         wf_fft[0].at(memory_t::device, 0, wf::band_index(j)));
                }
            } // x
        }
    }
}

// instantiate for supported precision
template class Local_operator<double>;
#ifdef SIRIUS_USE_FP32
template class Local_operator<float>;
#endif
} // namespace sirius
