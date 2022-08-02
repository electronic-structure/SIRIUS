// Copyright (c) 2013-2019 Anton Kozhevnikov, Mathieu Taillefumier, Thomas Schulthess
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

/** \file local_operator.cpp
 *
 *  \brief Implementation of sirius::Local_operator class.
 */

#include "local_operator.hpp"
#include "potential/potential.hpp"
#include "function3d/smooth_periodic_function.hpp"
#include "utils/profiler.hpp"

namespace sirius {

template <typename T>
Local_operator<T>::Local_operator(Simulation_context const& ctx__, spfft_transform_type<T>& fft_coarse__,
                                  sddk::Gvec_partition const& gvec_coarse_p__, Potential* potential__)
    : ctx_(ctx__)
    , fft_coarse_(fft_coarse__)
    , gvec_coarse_p_(gvec_coarse_p__)

{
    PROFILE("sirius::Local_operator");

    /* allocate functions */
    for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
        veff_vec_[j] = std::unique_ptr<Smooth_periodic_function<T>>(
            new Smooth_periodic_function<T>(fft_coarse__, gvec_coarse_p__, &ctx_.mem_pool(sddk::memory_t::host)));
        #pragma omp parallel for schedule(static)
        for (int ir = 0; ir < fft_coarse__.local_slice_size(); ir++) {
            veff_vec_[j]->f_rg(ir) = 2.71828;
        }
    }
    /* map Theta(r) to the coarse mesh */
    if (ctx_.full_potential()) {
        auto& gvec_dense_p = ctx_.gvec_partition();
        veff_vec_[v_local_index_t::theta] = std::unique_ptr<Smooth_periodic_function<T>>(
            new Smooth_periodic_function<T>(fft_coarse__, gvec_coarse_p__, &ctx_.mem_pool(sddk::memory_t::host)));
        /* map unit-step function */
        #pragma omp parallel for schedule(static)
        for (int igloc = 0; igloc < gvec_coarse_p_.gvec().count(); igloc++) {
            /* map from fine to coarse set of G-vectors */
            veff_vec_[v_local_index_t::theta]->f_pw_local(igloc) =
                ctx_.theta_pw(gvec_dense_p.gvec().gvec_base_mapping(igloc) + gvec_dense_p.gvec().offset());
        }
        veff_vec_[v_local_index_t::theta]->fft_transform(1);
        if (fft_coarse_.processing_unit() == SPFFT_PU_GPU) {
            veff_vec_[v_local_index_t::theta]->f_rg().allocate(ctx_.mem_pool(sddk::memory_t::device)).copy_to(sddk::memory_t::device);
        }
        if (ctx_.print_checksum()) {
            auto cs1 = veff_vec_[v_local_index_t::theta]->checksum_pw();
            auto cs2 = veff_vec_[v_local_index_t::theta]->checksum_rg();
            if (ctx_.comm().rank() == 0) {
                utils::print_checksum("theta_pw", cs1);
                utils::print_checksum("theta_rg", cs2);
            }
        }
    }

    /* map potential */
    if (potential__) {

        if (ctx_.full_potential()) {

            auto& fft_dense    = ctx_.spfft<T>();
            auto& gvec_dense_p = ctx_.gvec_partition();

            Smooth_periodic_function<T> ftmp(const_cast<Simulation_context&>(ctx_).spfft<T>(), gvec_dense_p,
                                             &ctx_.mem_pool(sddk::memory_t::host));

            for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                /* multiply potential by step function theta(r) */
                for (int ir = 0; ir < fft_dense.local_slice_size(); ir++) {
                    ftmp.f_rg(ir) = potential__->component(j).f_rg(ir) * ctx_.theta(ir);
                }
                /* transform to plane-wave domain */
                ftmp.fft_transform(-1);
                if (j == 0) {
                    v0_[0] = ftmp.f_0().real();
                }
                /* loop over local set of coarse G-vectors */
                #pragma omp parallel for schedule(static)
                for (int igloc = 0; igloc < gvec_coarse_p_.gvec().count(); igloc++) {
                    /* map from fine to coarse set of G-vectors */
                    veff_vec_[j]->f_pw_local(igloc) = ftmp.f_pw_local(gvec_dense_p.gvec().gvec_base_mapping(igloc));
                }
                /* transform to real space */
                veff_vec_[j]->fft_transform(1);
            }
            if (ctx_.valence_relativity() == relativity_t::zora) {
                veff_vec_[v_local_index_t::rm_inv] = std::unique_ptr<Smooth_periodic_function<T>>(
                    new Smooth_periodic_function<T>(fft_coarse__, gvec_coarse_p__, &ctx_.mem_pool(sddk::memory_t::host)));
                /* loop over local set of coarse G-vectors */
                #pragma omp parallel for schedule(static)
                for (int igloc = 0; igloc < gvec_coarse_p_.gvec().count(); igloc++) {
                    /* map from fine to coarse set of G-vectors */
                    veff_vec_[v_local_index_t::rm_inv]->f_pw_local(igloc) =
                        potential__->rm_inv_pw(gvec_dense_p.gvec().offset() + gvec_dense_p.gvec().gvec_base_mapping(igloc));
                }
                /* transform to real space */
                veff_vec_[v_local_index_t::rm_inv]->fft_transform(1);
            }

        } else {

            for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                /* loop over local set of coarse G-vectors */
                #pragma omp parallel for schedule(static)
                for (int igloc = 0; igloc < gvec_coarse_p_.gvec().count(); igloc++) {
                    /* map from fine to coarse set of G-vectors */
                    veff_vec_[j]->f_pw_local(igloc) =
                        potential__->component(j).f_pw_local(potential__->component(j).gvec().gvec_base_mapping(igloc));
                }
                /* transform to real space */
                veff_vec_[j]->fft_transform(1);
            }

            /* change to canonical form */
            if (ctx_.num_mag_dims()) {
                #pragma omp parallel for schedule(static)
                for (int ir = 0; ir < fft_coarse_.local_slice_size(); ir++) {
                    T v0             = veff_vec_[v_local_index_t::v0]->f_rg(ir);
                    T v1             = veff_vec_[v_local_index_t::v1]->f_rg(ir);
                    veff_vec_[v_local_index_t::v0]->f_rg(ir) = v0 + v1; // v + Bz
                    veff_vec_[v_local_index_t::v1]->f_rg(ir) = v0 - v1; // v - Bz
                }
            }

            if (ctx_.num_mag_dims() == 0) {
                v0_[0] = potential__->component(0).f_0().real();
            } else {
                v0_[0] = potential__->component(0).f_0().real() + potential__->component(1).f_0().real();
                v0_[1] = potential__->component(0).f_0().real() - potential__->component(1).f_0().real();
            }
        }

        if (ctx_.print_checksum()) {
            for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                auto cs1 = veff_vec_[j]->checksum_pw();
                auto cs2 = veff_vec_[j]->checksum_rg();
                if (ctx_.comm().rank() == 0) {
                    utils::print_checksum("veff_pw", cs1);
                    utils::print_checksum("veff_rg", cs2);
                }
            }
        }
    }

    buf_rg_ = sddk::mdarray<std::complex<T>, 1>(fft_coarse_.local_slice_size(), ctx_.mem_pool(sddk::memory_t::host),
                                         "Local_operator::buf_rg_");
    /* move functions to GPU */
    if (fft_coarse_.processing_unit() == SPFFT_PU_GPU) {
        for (int j = 0; j < 6; j++) {
            if (veff_vec_[j]) {
                veff_vec_[j]->f_rg().allocate(ctx_.mem_pool(sddk::memory_t::device)).copy_to(sddk::memory_t::device);
            }
        }
        buf_rg_.allocate(ctx_.mem_pool(sddk::memory_t::device));
    }
}

template <typename T>
void Local_operator<T>::prepare_k(sddk::Gvec_partition const& gkvec_p__)
{
    PROFILE("sirius::Local_operator::prepare_k");

    int ngv_fft = gkvec_p__.gvec_count_fft();

    /* cache kinteic energy of plane-waves */
    if (static_cast<int>(pw_ekin_.size()) < ngv_fft) {
        pw_ekin_ = sddk::mdarray<T, 1>(ngv_fft, ctx_.mem_pool(sddk::memory_t::host), "Local_operator::pw_ekin");
    }
    #pragma omp parallel for schedule(static)
    for (int ig_loc = 0; ig_loc < ngv_fft; ig_loc++) {
        /* get G+k in Cartesian coordinates */
        auto gv          = gkvec_p__.gkvec_cart(ig_loc);
        pw_ekin_[ig_loc] = 0.5 * dot(gv, gv);
    }

    if (static_cast<int>(vphi_.size(0)) < ngv_fft) {
        vphi_ = sddk::mdarray<std::complex<T>, 1>(ngv_fft, ctx_.mem_pool(sddk::memory_t::host), "Local_operator::vphi");
    }

    if (fft_coarse_.processing_unit() == SPFFT_PU_GPU) {
        pw_ekin_.allocate(ctx_.mem_pool(sddk::memory_t::device)).copy_to(sddk::memory_t::device);
        vphi_.allocate(ctx_.mem_pool(sddk::memory_t::device));
    }
}

#ifdef SIRIUS_GPU
void mul_by_veff_real_real_gpu(int nr__, float* buf__, float* veff__)
{
    mul_by_veff_real_real_gpu_float(nr__, buf__, veff__);
}

void mul_by_veff_real_real_gpu(int nr__, double* buf__, double* veff__)
{
    mul_by_veff_real_real_gpu_double(nr__, buf__, veff__);
}

void mul_by_veff_complex_real_gpu(int nr__, std::complex<float>* buf__, float* veff__)
{
    mul_by_veff_complex_real_gpu_float(nr__, buf__, veff__);
}

void mul_by_veff_complex_real_gpu(int nr__, double_complex* buf__, double* veff__)
{
    mul_by_veff_complex_real_gpu_double(nr__, buf__, veff__);
}

void mul_by_veff_complex_complex_gpu(int nr__, std::complex<float>* buf__, float pref__, float* vx__, float* vy__)
{
    mul_by_veff_complex_complex_gpu_float(nr__, buf__, pref__, vx__, vy__);
}

void mul_by_veff_complex_complex_gpu(int nr__, double_complex* buf__, double pref__, double* vx__, double* vy__)
{
    mul_by_veff_complex_complex_gpu_double(nr__, buf__, pref__, vx__, vy__);
}
#endif

template <typename T, typename = std::enable_if_t<std::is_scalar<T>::value>>
static inline void mul_by_veff(spfft_transform_type<T>& spfftk__, T* buff__,
                               std::array<std::unique_ptr<Smooth_periodic_function<T>>, 6>& veff_vec__,
                               int idx_veff__)
{
    int nr = spfftk__.local_slice_size();

    switch (spfftk__.processing_unit()) {
        case SPFFT_PU_HOST: {
            if (idx_veff__ <= 1 || idx_veff__ >= 4) { /* up-up or dn-dn block or Theta(r) */
                switch (spfftk__.type()) {
                    case SPFFT_TRANS_R2C: {
                        #pragma omp parallel for schedule(static)
                        for (int ir = 0; ir < nr; ir++) {
                            /* multiply by V+Bz or V-Bz (in PP-PW case) or by V(r), B_z(r) or Theta(r) (in LAPW case) */
                            buff__[ir] *= veff_vec__[idx_veff__]->f_rg(ir);
                        }
                        break;
                    }
                    case SPFFT_TRANS_C2C: {
                        auto wf = reinterpret_cast<std::complex<T>*>(buff__);
                        #pragma omp parallel for schedule(static)
                        for (int ir = 0; ir < nr; ir++) {
                            /* multiply by V+Bz or V-Bz (in PP-PW case) or by V(r), B_z(r) or Theta(r) (in LAPW case) */
                            wf[ir] *= veff_vec__[idx_veff__]->f_rg(ir);
                        }
                        break;
                    }
                }
            } else {
                T pref = (idx_veff__ == 2) ? -1 : 1;
                auto wf     = reinterpret_cast<std::complex<T>*>(buff__);
                #pragma omp parallel for schedule(static)
                for (int ir = 0; ir < nr; ir++) {
                    /* multiply by Bx +/- i*By */
                    wf[ir] *= std::complex<T>(veff_vec__[2]->f_rg(ir), pref * veff_vec__[3]->f_rg(ir));
                }
            }
            break;
        }
        case SPFFT_PU_GPU: {
#if defined(SIRIUS_GPU)
            if (idx_veff__ <= 1 || idx_veff__ >= 4) { /* up-up or dn-dn block or Theta(r) */
                switch (spfftk__.type()) {
                    case SPFFT_TRANS_R2C: {
                        /* multiply by V+Bz or V-Bz (in PP-PW case) or by V(r), B_z(r) or Theta(r) (in LAPW case) */
                        mul_by_veff_real_real_gpu(nr, buff__, veff_vec__[idx_veff__]->f_rg().at(sddk::memory_t::device));
                        break;
                    }
                    case SPFFT_TRANS_C2C: {
                        auto wf = reinterpret_cast<std::complex<T>*>(buff__);
                        /* multiply by V+Bz or V-Bz (in PP-PW case) or by V(r), B_z(r) or Theta(r) (in LAPW case) */
                        mul_by_veff_complex_real_gpu(nr, wf, veff_vec__[idx_veff__]->f_rg().at(sddk::memory_t::device));
                        break;
                    }
                }
            } else {
                /* multiply by Bx +/- i*By */
                T pref = (idx_veff__ == 2) ? -1 : 1;
                auto wf     = reinterpret_cast<std::complex<T>*>(buff__);
                mul_by_veff_complex_complex_gpu(nr, wf, pref, veff_vec__[2]->f_rg().at(sddk::memory_t::device),
                    veff_vec__[3]->f_rg().at(sddk::memory_t::device));
            }
            break;
#endif
        }
        break;
    }
}

#ifdef SIRIUS_GPU
void add_pw_ekin_gpu(int                        num_gvec__,
                     float                      alpha__,
                     float const*               pw_ekin__,
                     std::complex<float> const* phi__,
                     std::complex<float> const* vphi__,
                     std::complex<float>*       hphi__)
{
    add_pw_ekin_gpu_float(num_gvec__, alpha__, pw_ekin__, phi__, vphi__, hphi__);
}

void add_pw_ekin_gpu(int                   num_gvec__,
                     double                alpha__,
                     double const*         pw_ekin__,
                     double_complex const* phi__,
                     double_complex const* vphi__,
                     double_complex*       hphi__)
{
    add_pw_ekin_gpu_double(num_gvec__, alpha__, pw_ekin__, phi__, vphi__, hphi__);
}
#endif

template <typename T>
void Local_operator<T>::apply_h(spfft_transform_type<T>& spfftk__, sddk::Gvec_partition const& gkvec_p__,
                                sddk::spin_range spins__, sddk::Wave_functions<T>& phi__,
                                sddk::Wave_functions<T>& hphi__, int idx0__, int n__)
{
    PROFILE("sirius::Local_operator::apply_h");

    if ((spfftk__.dim_x() != fft_coarse_.dim_x()) || (spfftk__.dim_y() != fft_coarse_.dim_y()) ||
        (spfftk__.dim_z() != fft_coarse_.dim_z())) {
        TERMINATE("wrong FFT dimensions");
    }

    /* increment the counter by the number of wave-functions */
    ctx_.num_loc_op_applied(n__);

    /* this memory pool will be used to allocate extra storage in the host memory */
    auto& mp = const_cast<Simulation_context&>(ctx_).mem_pool(ctx_.host_memory_t());
    /* this memory pool will be used to allocate extra storage in the device memory */
#if defined(SIRIUS_GPU)
    sddk::memory_pool* mpd = &const_cast<Simulation_context&>(ctx_).mem_pool(sddk::memory_t::device);
#else
    sddk::memory_pool* mpd{nullptr};
#endif
    /* alias array for all wave functions in FFT-friendly storage */
    std::array<sddk::mdarray<std::complex<T>, 2>, 2> phi;
    /* alias array for all hphi in FFT-friendly storage */
    std::array<sddk::mdarray<std::complex<T>, 2>, 2> hphi;

    /* alias for wave-functions that are currently computed */
    std::array<sddk::mdarray<std::complex<T>, 1>, 2> phi1;
    std::array<sddk::mdarray<std::complex<T>, 1>, 2> hphi1;

    /* local number of G-vectors for the FFT transformation */
    int ngv_fft = gkvec_p__.gvec_count_fft();

    if (ngv_fft != spfftk__.num_local_elements()) {
        TERMINATE("wrong number of G-vectors");
    }

    sddk::memory_t mem_phi{sddk::memory_t::none};
    sddk::memory_t mem_hphi{sddk::memory_t::none};

    /* remap wave-functions to FFT friendly distribution */
    for (int ispn : spins__) {
        /* if we store wave-functions in the device memory and if the wave functions are remapped
           we need to copy the wave functions to host memory */
        if (is_device_memory(phi__.preferred_memory_t()) && phi__.pw_coeffs(ispn).is_remapped()) {
            phi__.pw_coeffs(ispn).copy_to(sddk::memory_t::host, idx0__, n__);
        }
        /* set FFT friendly distribution */
        phi__.pw_coeffs(ispn).remap_forward(n__, idx0__, &mp);
        /* memory location of phi in extra storage */
        mem_phi = (phi__.pw_coeffs(ispn).is_remapped()) ? sddk::memory_t::host : phi__.preferred_memory_t();
        /* set FFT friednly distribution */
        hphi__.pw_coeffs(ispn).set_num_extra(n__, idx0__, &mp);
        /* memory location of hphi in extra storage */
        mem_hphi = (hphi__.pw_coeffs(ispn).is_remapped()) ? sddk::memory_t::host : hphi__.preferred_memory_t();

        /* local number of wave-functions in extra-storage distribution */
        int num_wf_loc = phi__.pw_coeffs(ispn).spl_num_col().local_size();

        /* set alias for phi extra storage */
        std::complex<T>* ptr_d{nullptr};
        if (phi__.pw_coeffs(ispn).extra().on_device()) {
            ptr_d = phi__.pw_coeffs(ispn).extra().at(sddk::memory_t::device);
        }
        phi[ispn] =
            sddk::mdarray<std::complex<T>, 2>(phi__.pw_coeffs(ispn).extra().at(sddk::memory_t::host), ptr_d, ngv_fft, num_wf_loc);

        /* set alias for hphi extra storage */
        ptr_d = nullptr;
        if (hphi__.pw_coeffs(ispn).extra().on_device()) {
            ptr_d = hphi__.pw_coeffs(ispn).extra().at(sddk::memory_t::device);
        }
        hphi[ispn] =
            sddk::mdarray<std::complex<T>, 2>(hphi__.pw_coeffs(ispn).extra().at(sddk::memory_t::host), ptr_d, ngv_fft, num_wf_loc);
    }

    /* assume the location of data on the current processing unit */
    auto spfft_mem = spfftk__.processing_unit();

    /* number of real-space points in the local part of FFT buffer */
    int nr = spfftk__.local_slice_size();

    /* pointer to FFT buffer */
    auto spfft_buf = spfftk__.space_domain_data(spfft_mem);

    auto prepare_phi_hphi = [&](int i) {
        for (int ispn : spins__) {
            switch (spfft_mem) {
                case SPFFT_PU_HOST: {              /* FFT is done on CPU */
                    if (is_host_memory(mem_phi)) { /* wave-functions are also on host memory */
                        phi1[ispn] = sddk::mdarray<std::complex<T>, 1>(phi[ispn].at(sddk::memory_t::host, 0, i), ngv_fft);
                    } else { /* wave-functions are on the device memory */
                        phi1[ispn] = sddk::mdarray<std::complex<T>, 1>(ngv_fft, mp);
                        /* copy wave-functions to host memory */
                        acc::copyout(phi1[ispn].at(sddk::memory_t::host), phi[ispn].at(sddk::memory_t::device, 0, i), ngv_fft);
                    }
                    if (is_host_memory(mem_hphi)) {
                        hphi1[ispn] = sddk::mdarray<std::complex<T>, 1>(hphi[ispn].at(sddk::memory_t::host, 0, i), ngv_fft);
                    } else {
                        hphi1[ispn] = sddk::mdarray<std::complex<T>, 1>(ngv_fft, mp);
                    }
                    hphi1[ispn].zero(sddk::memory_t::host);
                    break;
                }
                case SPFFT_PU_GPU: { /* FFT is done on GPU */
                    if (is_host_memory(mem_phi)) {
                        phi1[ispn] = sddk::mdarray<std::complex<T>, 1>(ngv_fft, *mpd);
                        /* copy wave-functions to device */
                        acc::copyin(phi1[ispn].at(sddk::memory_t::device), phi[ispn].at(sddk::memory_t::host, 0, i), ngv_fft);
                    } else {
                        phi1[ispn] =
                            sddk::mdarray<std::complex<T>, 1>(nullptr, phi[ispn].at(sddk::memory_t::device, 0, i), ngv_fft);
                    }
                    if (is_host_memory(mem_hphi)) {
                        /* small buffers in the device memory */
                        hphi1[ispn] = sddk::mdarray<std::complex<T>, 1>(ngv_fft, *mpd);
                    } else {
                        hphi1[ispn] =
                            sddk::mdarray<std::complex<T>, 1>(nullptr, hphi[ispn].at(sddk::memory_t::device, 0, i), ngv_fft);
                    }
                    hphi1[ispn].zero(sddk::memory_t::device);
                    break;
                }
            }
        }
    };

    auto store_hphi = [&](int i) {
        for (int ispn : spins__) {
            switch (spfft_mem) {
                case SPFFT_PU_HOST: { /* FFT is done on CPU */
                    if (is_device_memory(mem_hphi)) {
                        /* copy to device */
                        acc::copyin(hphi[ispn].at(sddk::memory_t::device, 0, i), hphi1[ispn].at(sddk::memory_t::host), ngv_fft);
                    }
                    break;
                }
                case SPFFT_PU_GPU: { /* FFT is done on GPU */
                    if (is_host_memory(mem_hphi)) {
                        /* copy back to host */
                        acc::copyout(hphi[ispn].at(sddk::memory_t::host, 0, i), hphi1[ispn].at(sddk::memory_t::device), ngv_fft);
                    }
                    break;
                }
            }
        }
    };

    /* transform wave-function to real space; the result of the transformation is stored in the FFT buffer */
    auto phi_to_r = [&](int ispn) {
        spfftk__.backward(reinterpret_cast<T const*>(phi1[ispn].at(spfft_memory_t.at(spfft_mem))), spfft_mem);
    };

    /* transform function to PW domain */
    auto vphi_to_G = [&]() {
        spfftk__.forward(spfft_mem, reinterpret_cast<T*>(vphi_.at(spfft_memory_t.at(spfft_mem))), SPFFT_FULL_SCALING);
    };

    /* store the resulting hphi
       spin block (ispn_block) is used as a bit mask:
        - first bit: spin component which is updated
        - second bit: add or not kinetic energy term */
    auto add_to_hphi = [&](int ispn_block) {
        /* index of spin component */
        int ispn = ispn_block & 1;
        /* add kinetic energy if this is a diagonal block */
        int ekin = (ispn_block & 2) ? 0 : 1;

        switch (spfft_mem) {
            case SPFFT_PU_HOST: {
                /* CPU case */
                if (ekin) {
                    #pragma omp parallel for schedule(static)
                    for (int ig = 0; ig < gkvec_p__.gvec_count_fft(); ig++) {
                        hphi1[ispn][ig] += (phi1[ispn][ig] * pw_ekin_[ig] + vphi_[ig]);
                    }
                } else {
                    #pragma omp parallel for schedule(static)
                    for (int ig = 0; ig < gkvec_p__.gvec_count_fft(); ig++) {
                        hphi1[ispn][ig] += vphi_[ig];
                    }
                }
                break;
            }
            case SPFFT_PU_GPU: {
#if defined(SIRIUS_GPU)
                T alpha = static_cast<T>(ekin);
                add_pw_ekin_gpu(gkvec_p__.gvec_count_fft(), alpha, pw_ekin_.at(sddk::memory_t::device),
                                phi1[ispn].at(sddk::memory_t::device), vphi_.at(sddk::memory_t::device),
                                hphi1[ispn].at(sddk::memory_t::device));
#endif
                break;
            }
        }
    };

    /* local number of wave-functions in extra-storage distribution */
    int num_wf_loc = phi__.pw_coeffs(0).spl_num_col().local_size();

    /* if we don't have G-vector reductions, first = 0 and we start a normal loop */
    for (int i = 0; i < num_wf_loc; i++) {

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
        if (spins__() == 2) {
            prepare_phi_hphi(i);
            /* phi_u(G) -> phi_u(r) */
            phi_to_r(0);
            /* save phi_u(r) in temporary buf_rg array */
            switch (spfft_mem) {
                /* this is a non-collinear case, so the wave-functions and FFT buffer are complex and
                   we can copy memory */
                case SPFFT_PU_HOST: {
                    auto inp = reinterpret_cast<std::complex<T>*>(spfft_buf);
                    std::copy(inp, inp + nr, buf_rg_.at(sddk::memory_t::host));
                    break;
                }
                case SPFFT_PU_GPU: {
                    acc::copy(buf_rg_.at(sddk::memory_t::device), reinterpret_cast<std::complex<T>*>(spfft_buf), nr);
                    break;
                }
            }
            /* multiply phi_u(r) by effective potential */
            mul_by_veff<T>(spfftk__, spfft_buf, veff_vec_, 0);

            /* V_{uu}(r)phi_{u}(r) -> [V*phi]_{u}(G) */
            vphi_to_G();
            /* add kinetic energy */
            add_to_hphi(0);
            /* multiply phi_{u} by V_{du} and copy to FFT buffer */
            switch (spfft_mem) {
                case SPFFT_PU_HOST: {
                    mul_by_veff<T>(spfftk__, reinterpret_cast<T*>(buf_rg_.at(sddk::memory_t::host)), veff_vec_, 3);
                    std::copy(buf_rg_.at(sddk::memory_t::host), buf_rg_.at(sddk::memory_t::host) + nr,
                              reinterpret_cast<std::complex<T>*>(spfft_buf));
                    break;
                }
                case SPFFT_PU_GPU: {
                    mul_by_veff<T>(spfftk__, reinterpret_cast<T*>(buf_rg_.at(sddk::memory_t::device)), veff_vec_, 3);
                    acc::copy(reinterpret_cast<std::complex<T>*>(spfft_buf), buf_rg_.at(sddk::memory_t::device), nr);
                    break;
                }
            }
            /* V_{du}(r)phi_{u}(r) -> [V*phi]_{d}(G) */
            vphi_to_G();
            /* add to hphi_{d} */
            add_to_hphi(3);

            /* for the second spin component */

            /* phi_d(G) -> phi_d(r) */
            phi_to_r(1);
            /* save phi_d(r) */
            switch (spfft_mem) {
                case SPFFT_PU_HOST: {
                    auto inp = reinterpret_cast<std::complex<T>*>(spfft_buf);
                    std::copy(inp, inp + nr, buf_rg_.at(sddk::memory_t::host));
                    break;
                }
                case SPFFT_PU_GPU: {
                    acc::copy(buf_rg_.at(sddk::memory_t::device), reinterpret_cast<std::complex<T>*>(spfft_buf), nr);
                    break;
                }
            }
            /* multiply phi_d(r) by effective potential */
            mul_by_veff<T>(spfftk__, spfft_buf, veff_vec_, 1);
            /* V_{dd}(r)phi_{d}(r) -> [V*phi]_{d}(G) */
            vphi_to_G();
            /* add kinetic energy */
            add_to_hphi(1);
            /* multiply phi_{d} by V_{ud} and copy to FFT buffer */
            switch (spfft_mem) {
                case SPFFT_PU_HOST: {
                    mul_by_veff<T>(spfftk__, reinterpret_cast<T*>(buf_rg_.at(sddk::memory_t::host)), veff_vec_, 2);
                    std::copy(buf_rg_.at(sddk::memory_t::host), buf_rg_.at(sddk::memory_t::host) + nr,
                              reinterpret_cast<std::complex<T>*>(spfft_buf));
                    break;
                }
                case SPFFT_PU_GPU: {
                    mul_by_veff<T>(spfftk__, reinterpret_cast<T*>(buf_rg_.at(sddk::memory_t::device)), veff_vec_, 2);
                    acc::copy(reinterpret_cast<std::complex<T>*>(spfft_buf), buf_rg_.at(sddk::memory_t::device), nr);
                    break;
                }
            }
            /* V_{ud}(r)phi_{d}(r) -> [V*phi]_{u}(G) */
            vphi_to_G();
            /* add to hphi_{u} */
            add_to_hphi(2);
            /* copy to main hphi array */
            store_hphi(i);
        } else { /* spin-collinear or non-magnetic case */
            prepare_phi_hphi(i);
            /* phi(G) -> phi(r) */
            phi_to_r(spins__());
            /* multiply by effective potential */
            mul_by_veff<T>(spfftk__, spfft_buf, veff_vec_, spins__());
            /* V(r)phi(r) -> [V*phi](G) */
            vphi_to_G();
            /* add kinetic energy */
            add_to_hphi(spins__());
            store_hphi(i);
        }
    }

    /* remap hphi backward */
    for (int ispn : spins__) {
        hphi__.pw_coeffs(ispn).remap_backward(n__, idx0__);
        if (is_device_memory(hphi__.preferred_memory_t()) && hphi__.pw_coeffs(ispn).is_remapped()) {
            hphi__.pw_coeffs(ispn).copy_to(sddk::memory_t::device, idx0__, n__);
        }
    }
    /* at this point hphi in prime storage is both on CPU and GPU memory; however if the memory pool
       was used for the device memory allocation, device storage is destroyed */
}

template <typename T>
void Local_operator<T>::apply_h_o(spfft_transform_type<T>& spfftk__, sddk::Gvec_partition const& gkvec_p__,
        int N__, int n__, sddk::Wave_functions<T>& phi__, sddk::Wave_functions<T>* hphi__, sddk::Wave_functions<T>* ophi__)
{
    PROFILE("sirius::Local_operator::apply_h_o");

    ctx_.num_loc_op_applied(n__);

    sddk::mdarray<std::complex<T>, 1> buf_pw(gkvec_p__.gvec_count_fft(), ctx_.mem_pool(sddk::memory_t::host));

    if (ctx_.processing_unit() == sddk::device_t::GPU) {
        phi__.pw_coeffs(0).copy_to(sddk::memory_t::host, N__, n__);
    }
    // if (ctx_->control().print_checksum_) {
    //    auto cs = phi__.checksum_pw(N__, n__, ctx_->processing_unit());
    //    if (phi__.comm().rank() == 0) {
    //        DUMP("checksum(phi_pw): %18.10f %18.10f", cs.real(), cs.imag());
    //    }
    //}

    auto& mp = const_cast<Simulation_context&>(ctx_).mem_pool(sddk::memory_t::host);

    phi__.pw_coeffs(0).remap_forward(n__, N__, &mp);

    if (hphi__ != nullptr) {
        hphi__->pw_coeffs(0).set_num_extra(n__, N__, &mp);
    }

    if (ophi__ != nullptr) {
        ophi__->pw_coeffs(0).set_num_extra(n__, N__, &mp);
    }

    /* assume the location of data on the current processing unit */
    auto spfft_mem = spfftk__.processing_unit();

    /* number of real-space points in the local part of FFT buffer */
    int nr = spfftk__.local_slice_size();

    /* pointer to FFT buffer */
    auto spfft_buf = spfftk__.space_domain_data(spfft_mem);

    for (int j = 0; j < phi__.pw_coeffs(0).spl_num_col().local_size(); j++) {
        /* phi(G) -> phi(r) */
        spfftk__.backward(reinterpret_cast<T const*>(phi__.pw_coeffs(0).extra().at(sddk::memory_t::host, 0, j)), spfft_mem);
        if (ophi__ != nullptr) {
            /* save phi(r) */
            if (hphi__ != nullptr) {
                switch (spfft_mem) {
                    case SPFFT_PU_HOST: {
                        auto inp = reinterpret_cast<std::complex<T>*>(spfft_buf);
                        std::copy(inp, inp + nr, buf_rg_.at(sddk::memory_t::host));
                        break;
                    }
                    case SPFFT_PU_GPU: {
                        acc::copy(buf_rg_.at(sddk::memory_t::device), reinterpret_cast<std::complex<T>*>(spfft_buf), nr);
                        break;
                    }
                }
            }

            /* multiply phi(r) by step function */
            mul_by_veff<T>(spfftk__, spfft_buf, veff_vec_, 4);

            /* phi(r) * Theta(r) -> ophi(G) */
            spfftk__.forward(spfft_mem, reinterpret_cast<T*>(ophi__->pw_coeffs(0).extra().at(sddk::memory_t::host, 0, j)),
                             SPFFT_FULL_SCALING);
            /* load phi(r) back */
            if (hphi__ != nullptr) {
                switch (spfft_mem) {
                    case SPFFT_PU_HOST: {
                        std::copy(buf_rg_.at(sddk::memory_t::host), buf_rg_.at(sddk::memory_t::host) + nr,
                                  reinterpret_cast<std::complex<T>*>(spfft_buf));
                        break;
                    }
                    case SPFFT_PU_GPU: {
                        acc::copy(reinterpret_cast<std::complex<T>*>(spfft_buf), buf_rg_.at(sddk::memory_t::device), nr);
                        break;
                    }
                }
            }
        }
        if (hphi__ != nullptr) {
            /* multiply by effective potential, which itself was multiplied by the step function */
            mul_by_veff<T>(spfftk__, spfft_buf, veff_vec_, 0);
            /* phi(r) * Theta(r) * V(r) -> hphi(G) */
            spfftk__.forward(spfft_mem, reinterpret_cast<T*>(hphi__->pw_coeffs(0).extra().at(sddk::memory_t::host, 0, j)),
                             SPFFT_FULL_SCALING);
        }

        if (hphi__ != nullptr) {
            /* add kinetic energy */
            for (int x : {0, 1, 2}) {
                #pragma omp parallel for schedule(static)
                for (int igloc = 0; igloc < gkvec_p__.gvec_count_fft(); igloc++) {
                    auto gvc = gkvec_p__.gkvec_cart(igloc);
                    /* \hat P phi = phi(G+k) * (G+k), \hat P is momentum operator */
                    buf_pw[igloc] = phi__.pw_coeffs(0).extra()(igloc, j) * static_cast<T>(gvc[x]);
                }
                /* transform Cartesian component of wave-function gradient to real space */
                spfftk__.backward(reinterpret_cast<T const*>(&buf_pw[0]), spfft_mem);
                /* multiply by real-space function */
                switch (ctx_.valence_relativity()) {
                    case relativity_t::iora:
                    case relativity_t::zora: {
                        /* multiply be inverse relative mass */
                        mul_by_veff<T>(spfftk__, spfft_buf, veff_vec_, 5);
                        break;
                    }
                    case relativity_t::none: {
                        /* multiply be step function */
                        mul_by_veff<T>(spfftk__, spfft_buf, veff_vec_, 4);
                        break;
                    }
                    default: {
                        break;
                    }
                }
                /* transform back to PW domain */
                spfftk__.forward(spfft_mem, reinterpret_cast<T*>(&buf_pw[0]), SPFFT_FULL_SCALING);
                #pragma omp parallel for schedule(static)
                for (int igloc = 0; igloc < gkvec_p__.gvec_count_fft(); igloc++) {
                    auto gvc = gkvec_p__.gkvec_cart(igloc);
                    hphi__->pw_coeffs(0).extra()(igloc, j) += static_cast<T>(0.5) * buf_pw[igloc] * static_cast<T>(gvc[x]);
                }
            }
        }
    }

    if (hphi__ != nullptr) {
        hphi__->pw_coeffs(0).remap_backward(n__, N__);
    }
    if (ophi__ != nullptr) {
        ophi__->pw_coeffs(0).remap_backward(n__, N__);
    }

    if (ctx_.processing_unit() == sddk::device_t::GPU) {
        if (hphi__ != nullptr) {
            hphi__->pw_coeffs(0).copy_to(sddk::memory_t::device, N__, n__);
        }
        if (ophi__ != nullptr) {
            ophi__->pw_coeffs(0).copy_to(sddk::memory_t::device, N__, n__);
        }
    }
}

template <typename T>
void Local_operator<T>::apply_b(spfft_transform_type<T>& spfftk__, int N__, int n__, sddk::Wave_functions<T>& phi__,
        std::vector<sddk::Wave_functions<T>>& bphi__)
{
    PROFILE("sirius::Local_operator::apply_b");

    // TODO: bphi[1] is not used here (it will compied from bphi[0] with a negative sign later;
    //       so it's remapping here is useless

    /* components of H|psi> to which H is applied */
    std::vector<int> iv(1, 0);
    if (bphi__.size() == 3) {
        iv.push_back(2);
    }

    auto& mp = const_cast<Simulation_context&>(ctx_).mem_pool(sddk::memory_t::host);

    /* assume the location of data on the current processing unit */
    auto spfft_mem = spfftk__.processing_unit();

    /* number of real-space points in the local part of FFT buffer */
    int nr = spfftk__.local_slice_size();

    /* pointer to FFT buffer */
    auto spfft_buf = spfftk__.space_domain_data(spfft_mem);

    phi__.pw_coeffs(0).remap_forward(n__, N__, &mp);
    for (int i : iv) {
        bphi__[i].pw_coeffs(0).set_num_extra(n__, N__, &mp);
    }

    for (int j = 0; j < phi__.pw_coeffs(0).spl_num_col().local_size(); j++) {
        /* phi(G) -> phi(r) */
        spfftk__.backward(reinterpret_cast<T const*>(phi__.pw_coeffs(0).extra().at(sddk::memory_t::host, 0, j)), spfft_mem);

        /* save phi(r) */
        if (bphi__.size() == 3) {
            switch (spfft_mem) {
                case SPFFT_PU_HOST: {
                    auto inp = reinterpret_cast<std::complex<T>*>(spfft_buf);
                    std::copy(inp, inp + nr, buf_rg_.at(sddk::memory_t::host));
                    break;
                }
                case SPFFT_PU_GPU: {
                    acc::copy(buf_rg_.at(sddk::memory_t::device), reinterpret_cast<std::complex<T>*>(spfft_buf), nr);
                    break;
                }
            }
        }
        /* multiply by Bz */
        mul_by_veff<T>(spfftk__, spfft_buf, veff_vec_, 1);

        /* phi(r) * Bz(r) -> bphi[0](G) */
        spfftk__.forward(spfft_mem,
                         reinterpret_cast<T*>(bphi__[0].pw_coeffs(0).extra().at(sddk::memory_t::host, 0, j)),
                         SPFFT_FULL_SCALING);

        /* non-collinear case */
        if (bphi__.size() == 3) {
            /* multiply by Bx-iBy and copy to FFT buffer */
            switch (spfft_mem) {
                case SPFFT_PU_HOST: {
                    mul_by_veff<T>(spfftk__, reinterpret_cast<T*>(buf_rg_.at(sddk::memory_t::host)), veff_vec_, 2);
                    std::copy(buf_rg_.at(sddk::memory_t::host), buf_rg_.at(sddk::memory_t::host) + nr,
                              reinterpret_cast<std::complex<T>*>(spfft_buf));
                    break;
                }
                case SPFFT_PU_GPU: {
                    mul_by_veff<T>(spfftk__, reinterpret_cast<T*>(buf_rg_.at(sddk::memory_t::device)), veff_vec_, 2);
                    acc::copy(reinterpret_cast<std::complex<T>*>(spfft_buf), buf_rg_.at(sddk::memory_t::device), nr);
                    break;
                }
            }
            /* phi(r) * (Bx(r)-iBy(r)) -> bphi[2](G) */
            spfftk__.forward(spfft_mem,
                             reinterpret_cast<T*>(bphi__[2].pw_coeffs(0).extra().at(sddk::memory_t::host, 0, j)),
                             SPFFT_FULL_SCALING);
        }
    }

    for (int i : iv) {
        bphi__[i].pw_coeffs(0).remap_backward(n__, N__);
    }
}

// This is full-potential case. Only C2C FFT transformation is considered here.
// TODO: document the data location on input/output
template <typename T>
void Local_operator<T>::apply_h_o(spfft_transform_type<T>& spfftk__, std::shared_ptr<sddk::Gvec_partition> gkvec_fft__,
        wf::band_range b__, wf::Wave_functions<T>& phi__, wf::Wave_functions<T>* hphi__, wf::Wave_functions<T>* ophi__)
{
    PROFILE("sirius::Local_operator::apply_h_o");

    ctx_.num_loc_op_applied(b__.size());

    sddk::mdarray<std::complex<T>, 1> buf_pw(gkvec_fft__->gvec_count_fft(), ctx_.mem_pool(sddk::memory_t::host));

    // TODO: need to pass temporaty functions or allocate from the pool
    wf::Wave_functions_fft<T> phi_fft(gkvec_fft__, wf::num_bands(b__.size()), sddk::memory_t::host);
    wf::Wave_functions_fft<T> hphi_fft(gkvec_fft__, wf::num_bands(b__.size()), sddk::memory_t::host);
    wf::Wave_functions_fft<T> ophi_fft(gkvec_fft__, wf::num_bands(b__.size()), sddk::memory_t::host);

    //if (ctx_.processing_unit() == device_t::GPU) {
    //    phi__.pw_coeffs(0).copy_to(sddk::memory_t::host, N__, n__);
    //}
    // if (ctx_->control().print_checksum_) {
    //    auto cs = phi__.checksum_pw(N__, n__, ctx_->processing_unit());
    //    if (phi__.comm().rank() == 0) {
    //        DUMP("checksum(phi_pw): %18.10f %18.10f", cs.real(), cs.imag());
    //    }
    //}

    auto& mp = const_cast<Simulation_context&>(ctx_).mem_pool(sddk::memory_t::host);

    wf::transform_to_fft_layout(phi__, phi_fft, wf::spin_index(0), b__);

    auto spl_num_wf = phi_fft.spl_num_wf();

    //phi__.pw_coeffs(0).remap_forward(n__, N__, &mp);

    //if (hphi__ != nullptr) {
    //    hphi__->pw_coeffs(0).set_num_extra(n__, N__, &mp);
    //}

    //if (ophi__ != nullptr) {
    //    ophi__->pw_coeffs(0).set_num_extra(n__, N__, &mp);
    //}

    /* assume the location of data on the current processing unit */
    auto spfft_mem = spfftk__.processing_unit();

    /* number of real-space points in the local part of FFT buffer */
    int nr = spfftk__.local_slice_size();

    /* pointer to memory where SpFFT stores real-space data */
    auto spfft_buf = spfftk__.space_domain_data(spfft_mem);

    for (int j = 0; j < spl_num_wf.local_size(); j++) {
        /* phi(G) -> phi(r) */
        spfftk__.backward(phi_fft.pw_coeffs(sddk::memory_t::host, j), spfft_mem);

        if (ophi__) {
            /* save phi(r) */
            if (hphi__) {
                /* reeal-space data is complex */
                auto inp = reinterpret_cast<std::complex<T>*>(spfft_buf);
                switch (spfft_mem) {
                    case SPFFT_PU_HOST: {
                        std::copy(inp, inp + nr, buf_rg_.at(sddk::memory_t::host));
                        break;
                    }
                    case SPFFT_PU_GPU: {
                        acc::copy(buf_rg_.at(sddk::memory_t::device), inp, nr);
                        break;
                    }
                }
            }

            /* multiply phi(r) by step function */
            mul_by_veff<T>(spfftk__, spfft_buf, veff_vec_, v_local_index_t::theta);

            /* phi(r) * Theta(r) -> ophi(G) */
            spfftk__.forward(spfft_mem, ophi_fft.pw_coeffs(sddk::memory_t::host, j), SPFFT_FULL_SCALING);
            /* load phi(r) back */
            if (hphi__) {
                auto outp = reinterpret_cast<std::complex<T>*>(spfft_buf);
                switch (spfft_mem) {
                    case SPFFT_PU_HOST: {
                        std::copy(buf_rg_.at(sddk::memory_t::host), buf_rg_.at(sddk::memory_t::host) + nr, outp);
                        break;
                    }
                    case SPFFT_PU_GPU: {
                        acc::copy(outp, buf_rg_.at(sddk::memory_t::device), nr);
                        break;
                    }
                }
            }
        }
        if (hphi__) {
            /* multiply by effective potential, which itself was multiplied by the step function */
            mul_by_veff<T>(spfftk__, spfft_buf, veff_vec_, v_local_index_t::v0);
            /* phi(r) * Theta(r) * V(r) -> hphi(G) */
            spfftk__.forward(spfft_mem, hphi_fft.pw_coeffs(sddk::memory_t::host, j), SPFFT_FULL_SCALING);
        }

        if (hphi__) {
            /* add kinetic energy */
            for (int x : {0, 1, 2}) {
                #pragma omp parallel for schedule(static)
                for (int igloc = 0; igloc < gkvec_fft__->gvec_count_fft(); igloc++) {
                    auto gvc = gkvec_fft__->gkvec_cart(igloc);
                    /* \hat P phi = phi(G+k) * (G+k), \hat P is momentum operator */
                    buf_pw[igloc] = phi_fft.pw_coeffs(igloc, j) * static_cast<T>(gvc[x]);
                }
                /* transform Cartesian component of wave-function gradient to real space */
                spfftk__.backward(reinterpret_cast<T const*>(&buf_pw[0]), spfft_mem);
                /* multiply by real-space function */
                switch (ctx_.valence_relativity()) {
                    case relativity_t::iora:
                    case relativity_t::zora: {
                        /* multiply be inverse relative mass */
                        mul_by_veff<T>(spfftk__, spfft_buf, veff_vec_, v_local_index_t::rm_inv);
                        break;
                    }
                    case relativity_t::none: {
                        /* multiply be step function */
                        mul_by_veff<T>(spfftk__, spfft_buf, veff_vec_, v_local_index_t::theta);
                        break;
                    }
                    default: {
                        break;
                    }
                }
                /* transform back to PW domain */
                spfftk__.forward(spfft_mem, reinterpret_cast<T*>(&buf_pw[0]), SPFFT_FULL_SCALING);
                #pragma omp parallel for schedule(static)
                for (int igloc = 0; igloc < gkvec_fft__->gvec_count_fft(); igloc++) {
                    auto gvc = gkvec_fft__->gkvec_cart(igloc);
                    hphi_fft.pw_coeffs(igloc, j) += buf_pw[igloc] * static_cast<T>(0.5 * gvc[x]);
                }
            }
        }
    }
    if (hphi__) {
        wf::transform_from_fft_layout(hphi_fft, *hphi__, wf::spin_index(0), b__);
    }
    if (ophi__) {
        wf::transform_from_fft_layout(ophi_fft, *ophi__, wf::spin_index(0), b__);
    }

    //if (hphi__ != nullptr) {
    //    hphi__->pw_coeffs(0).remap_backward(n__, N__);
    //}
    //if (ophi__ != nullptr) {
    //    ophi__->pw_coeffs(0).remap_backward(n__, N__);
    //}

    //if (ctx_.processing_unit() == device_t::GPU) {
    //    if (hphi__ != nullptr) {
    //        hphi__->pw_coeffs(0).copy_to(sddk::memory_t::device, N__, n__);
    //    }
    //    if (ophi__ != nullptr) {
    //        ophi__->pw_coeffs(0).copy_to(sddk::memory_t::device, N__, n__);
    //    }
    //}
}

// instantiate for supported precision
template class Local_operator<double>;
#ifdef USE_FP32
template class Local_operator<float>;
#endif
} // namespace sirius
