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

#include "local_operator.hpp"
#include "Potential/potential.hpp"

using namespace sddk;

namespace sirius {

Local_operator::Local_operator(Simulation_context const& ctx__, spfft::Transform& fft_coarse__,
                               Gvec_partition const& gvec_coarse_p__)
    : ctx_(ctx__)
    , fft_coarse_(fft_coarse__)
    , gvec_coarse_p_(gvec_coarse_p__)

{
    PROFILE("sirius::Local_operator");

    for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
        veff_vec_[j] = Smooth_periodic_function<double>(fft_coarse__, gvec_coarse_p__);
        for (int ir = 0; ir < fft_coarse__.local_slice_size(); ir++) {
            veff_vec_[j].f_rg(ir) = 2.71828;
        }
    }
    if (ctx_.full_potential()) {
        theta_ = Smooth_periodic_function<double>(fft_coarse__, gvec_coarse_p__);
    }

    if (fft_coarse_.processing_unit() == SPFFT_PU_GPU) {
        for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
            veff_vec_[j].f_rg().allocate(memory_t::device).copy_to(memory_t::device);
        }
        buf_rg_.allocate(memory_t::device);
    }
}

void Local_operator::prepare(Potential& potential__)
{
    PROFILE("sirius::Local_operator::prepare");

    if (!buf_rg_.size()) {
        buf_rg_ = mdarray<double_complex, 1>(fft_coarse_.local_slice_size(), memory_t::host, "Local_operator::buf_rg_");
    }

    if (ctx_.full_potential()) {

        auto& fft_dense    = ctx_.spfft();
        auto& gvec_dense_p = ctx_.gvec_partition();

        Smooth_periodic_function<double> ftmp(const_cast<Simulation_context&>(ctx_).spfft(), gvec_dense_p);
        for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
            for (int ir = 0; ir < fft_dense.local_slice_size(); ir++) {
                ftmp.f_rg(ir) = potential__.component(j).f_rg(ir) * ctx_.theta(ir);
            }
            ftmp.fft_transform(-1);
            if (j == 0) {
                v0_[0] = ftmp.f_0().real();
            }
            /* loop over local set of coarse G-vectors */
            #pragma omp parallel for schedule(static)
            for (int igloc = 0; igloc < gvec_coarse_p_.gvec().count(); igloc++) {
                /* map from fine to coarse set of G-vectors */
                veff_vec_[j].f_pw_local(igloc) = ftmp.f_pw_local(gvec_dense_p.gvec().gvec_base_mapping(igloc));
            }
            /* transform to real space */
            veff_vec_[j].fft_transform(1);
        }

        /* map unit-step function */
        #pragma omp parallel for schedule(static)
        for (int igloc = 0; igloc < gvec_coarse_p_.gvec().count(); igloc++) {
            /* map from fine to coarse set of G-vectors */
            theta_.f_pw_local(igloc) =
                ctx_.theta_pw(gvec_dense_p.gvec().gvec_base_mapping(igloc) + gvec_dense_p.gvec().offset());
        }
        theta_.fft_transform(1);

        if (fft_coarse_.processing_unit() == SPFFT_PU_GPU) {
            for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                veff_vec_[j].f_rg().allocate(memory_t::device).copy_to(memory_t::device);
            }
            theta_.f_rg().allocate(memory_t::device).copy_to(memory_t::device);
            buf_rg_.allocate(memory_t::device);
        }

        if (ctx_.control().print_checksum_) {
            auto cs1 = theta_.checksum_pw();
            auto cs2 = theta_.checksum_rg();
            if (ctx_.comm().rank() == 0) {
                utils::print_checksum("theta_pw", cs1);
                utils::print_checksum("theta_rg", cs2);
            }
            for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                cs1 = veff_vec_[j].checksum_pw();
                cs2 = veff_vec_[j].checksum_rg();
                if (ctx_.comm().rank() == 0) {
                    utils::print_checksum("veff_pw", cs1);
                    utils::print_checksum("veff_rg", cs2);
                }
            }
        }

    } else {

        for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
            /* loop over local set of coarse G-vectors */
            #pragma omp parallel for schedule(static)
            for (int igloc = 0; igloc < gvec_coarse_p_.gvec().count(); igloc++) {
                /* map from fine to coarse set of G-vectors */
                veff_vec_[j].f_pw_local(igloc) =
                    potential__.component(j).f_pw_local(potential__.component(j).gvec().gvec_base_mapping(igloc));
            }
            /* transform to real space */
            veff_vec_[j].fft_transform(1);
        }

        if (ctx_.num_mag_dims()) {
            for (int ir = 0; ir < fft_coarse_.local_slice_size(); ir++) {
                double v0             = veff_vec_[0].f_rg(ir);
                double v1             = veff_vec_[1].f_rg(ir);
                veff_vec_[0].f_rg(ir) = v0 + v1; // v + Bz
                veff_vec_[1].f_rg(ir) = v0 - v1; // v - Bz
            }
        }

        if (ctx_.num_mag_dims() == 0) {
            v0_[0] = potential__.component(0).f_0().real();
        } else {
            v0_[0] = potential__.component(0).f_0().real() + potential__.component(1).f_0().real();
            v0_[1] = potential__.component(0).f_0().real() - potential__.component(1).f_0().real();
        }

        /* copy veff to device */
        if (fft_coarse_.processing_unit() == SPFFT_PU_GPU) {
            for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                veff_vec_[j].f_rg().allocate(memory_t::device).copy_to(memory_t::device);
            }
            if (ctx_.num_mag_dims() == 3) {
                buf_rg_.allocate(memory_t::device);
            }
        }

        if (ctx_.control().print_checksum_) {
            for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                auto cs = veff_vec_[j].checksum_pw();
                if (gvec_coarse_p_.gvec().comm().rank() == 0) {
                    utils::print_checksum("veff_vec", cs);
                }
            }
        }
    }
}

void Local_operator::prepare(Gvec_partition const& gkvec_p__)
{
    PROFILE("sirius::Local_operator::prepare");

    gkvec_p_ = &gkvec_p__;

    int ngv_fft = gkvec_p__.gvec_count_fft();

    /* cache kinteic energy of plane-waves */
    if (static_cast<int>(pw_ekin_.size()) < ngv_fft) {
        pw_ekin_ = mdarray<double, 1>(ngv_fft, memory_t::host, "Local_operator::pw_ekin");
    }
    for (int ig_loc = 0; ig_loc < ngv_fft; ig_loc++) {
        /* global index of G-vector */
        int ig = gkvec_p__.idx_gvec(ig_loc);
        /* get G+k in Cartesian coordinates */
        auto gv          = gkvec_p__.gvec().gkvec_cart<index_domain_t::global>(ig);
        pw_ekin_[ig_loc] = 0.5 * dot(gv, gv);
    }

    if (static_cast<int>(vphi_.size(0)) < ngv_fft) {
        vphi_ = mdarray<double_complex, 1>(ngv_fft, memory_t::host, "Local_operator::vphi");
    }

    if (fft_coarse_.processing_unit() == SPFFT_PU_GPU) {
        pw_ekin_.allocate(memory_t::device).copy_to(memory_t::device);
        vphi_.allocate(memory_t::device);
    }
}

void Local_operator::dismiss()
{
    if (fft_coarse_.processing_unit() == SPFFT_PU_GPU) {
        for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
            veff_vec_[j].f_rg().deallocate(memory_t::device);
        }
        pw_ekin_.deallocate(memory_t::device);
        vphi_.deallocate(memory_t::device);
        theta_.f_rg().deallocate(memory_t::device);
        buf_rg_.deallocate(memory_t::device);
    }
    gkvec_p_ = nullptr;
}

void Local_operator::apply_h(spfft::Transform& spfft__, int ispn__, Wave_functions& phi__, Wave_functions& hphi__,
                             int idx0__, int n__)
{
    PROFILE("sirius::Local_operator::apply_h");

    if (!gkvec_p_) {
        TERMINATE("Local operator is not prepared");
    }

    /* increment the counter by the number of wave-functions */
    num_applied(n__);

    /* this memory pool will be used to allocate extra storage in the host memory*/
    auto& mp = const_cast<Simulation_context&>(ctx_).mem_pool(ctx_.host_memory_t());
    /* this memory pool will be used to allocate extra storage in the device memory */
#if defined(__GPU)
    auto& mpd = const_cast<Simulation_context&>(ctx_).mem_pool(memory_t::device);
#endif
    /* alias array for all wave functions in FFT-friendly storage */
    std::array<mdarray<double_complex, 2>, 2> phi;
    /* alias array for all hphi in FFT-friendly storage */
    std::array<mdarray<double_complex, 2>, 2> hphi;

    /* alias for wave-functions that are currently computed */
    std::array<mdarray<double_complex, 1>, 2> phi1;
    std::array<mdarray<double_complex, 1>, 2> hphi1;

    /* spin component to which H is applied */
    auto spins = get_spins(ispn__);

    /* local number of G-vectors for the FFT transformation */
    int ngv_fft = gkvec_p_->gvec_count_fft();

    memory_t mem_phi{memory_t::none};
    memory_t mem_hphi{memory_t::none};

    /* remap wave-functions to FFT friendly distribution */
    for (int ispn : spins) {
        /* if we store wave-functions in the device memory and if the wave functions are remapped
           we need to copy the wave functions to host memory */
        if (is_device_memory(phi__.preferred_memory_t()) && phi__.pw_coeffs(ispn).is_remapped()) {
            phi__.pw_coeffs(ispn).copy_to(memory_t::host, idx0__, n__);
        }
        /* set FFT friendly distribution */
        phi__.pw_coeffs(ispn).remap_forward(n__, idx0__, &mp);
        /* memory location of phi in extra storage */
        mem_phi = (phi__.pw_coeffs(ispn).is_remapped()) ? memory_t::host : phi__.preferred_memory_t();
        /* set FFT friednly distribution */
        hphi__.pw_coeffs(ispn).set_num_extra(n__, idx0__, &mp);
        /* memory location of hphi in extra storage */
        mem_hphi = (hphi__.pw_coeffs(ispn).is_remapped()) ? memory_t::host : hphi__.preferred_memory_t();

        /* local number of wave-functions in extra-storage distribution */
        int num_wf_loc = phi__.pw_coeffs(ispn).spl_num_col().local_size();

        /* set alias for phi extra storage */
        double_complex* ptr_d{nullptr};
        if (phi__.pw_coeffs(ispn).extra().on_device()) {
            ptr_d = phi__.pw_coeffs(ispn).extra().at(memory_t::device);
        }
        phi[ispn] =
            mdarray<double_complex, 2>(phi__.pw_coeffs(ispn).extra().at(memory_t::host), ptr_d, ngv_fft, num_wf_loc);

        /* set alias for hphi extra storage */
        ptr_d = nullptr;
        if (hphi__.pw_coeffs(ispn).extra().on_device()) {
            ptr_d = hphi__.pw_coeffs(ispn).extra().at(memory_t::device);
        }
        hphi[ispn] =
            mdarray<double_complex, 2>(hphi__.pw_coeffs(ispn).extra().at(memory_t::host), ptr_d, ngv_fft, num_wf_loc);
    }

#if defined(__GPU)
    /* pointers to effective potential on GPU */
    mdarray<double*, 1> vptr(4);
    vptr.zero();
    switch (spfft__.processing_unit()) {
        case SPFFT_PU_HOST: {
            vptr.allocate(memory_t::device);
            for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                vptr[j] = veff_vec_[j].f_rg().at(memory_t::device);
            }
            vptr.copy_to(memory_t::device);
            break;
        }
        case SPFFT_PU_GPU:
            break;
    }
#endif

    auto prepare_phi_hphi = [&](int i) {
        for (int ispn : spins) {
            switch (spfft__.processing_unit()) {
                case SPFFT_PU_HOST: {              /* FFT is done on CPU */
                    if (is_host_memory(mem_phi)) { /* wave-functions are also on host memory */
                        phi1[ispn] = mdarray<double_complex, 1>(phi[ispn].at(memory_t::host, 0, i), ngv_fft);
                    } else { /* wave-functions are on the device memory */
                        phi1[ispn] = mdarray<double_complex, 1>(mp, ngv_fft);
#if defined(__GPU)
                        /* copy wave-functions to host memory */
                        acc::copyout(phi1[ispn].at(memory_t::host), phi[ispn].at(memory_t::device, 0, i), ngv_fft);
#endif
                    }
                    if (is_host_memory(mem_hphi)) {
                        hphi1[ispn] = mdarray<double_complex, 1>(hphi[ispn].at(memory_t::host, 0, i), ngv_fft);
                    } else {
                        hphi1[ispn] = mdarray<double_complex, 1>(mp, ngv_fft);
                    }
                    hphi1[ispn].zero(memory_t::host);
                    break;
                }
                case SPFFT_PU_GPU: { /* FFT is done on GPU */
                    if (is_host_memory(mem_phi)) {
#ifdef __GPU
                        phi1[ispn] = mdarray<double_complex, 1>(mpd, ngv_fft);
                        /* copy wave-functions to device */
                        acc::copyin(phi1[ispn].at(memory_t::device), phi[ispn].at(memory_t::host, 0, i), ngv_fft);
#endif
                    } else {
                        phi1[ispn] = mdarray<double_complex, 1>(nullptr, phi[ispn].at(memory_t::device, 0, i), ngv_fft);
                    }
                    if (is_host_memory(mem_hphi)) {
#ifdef __GPU
                        /* small buffers in the device memory */
                        hphi1[ispn] = mdarray<double_complex, 1>(mpd, ngv_fft);
#endif
                    } else {
                        hphi1[ispn] =
                            mdarray<double_complex, 1>(nullptr, hphi[ispn].at(memory_t::device, 0, i), ngv_fft);
                    }
                    hphi1[ispn].zero(memory_t::device);
                    break;
                }
            }
        }
    };

    auto store_hphi = [&](int i) {
#ifdef __GPU
        for (int ispn : spins) {
            switch (spfft__.pu()) {
                case device_t::CPU: { /* FFT is done on CPU */
                    if (is_device_memory(mem_hphi)) {
                        /* copy to device */
                        acc::copyin(hphi[ispn].at(memory_t::device, 0, i), hphi1[ispn].at(memory_t::host), ngv_fft);
                    }
                    break;
                }
                case device_t::GPU: { /* FFT is done on GPU */
                    if (is_host_memory(mem_hphi)) {
                        /* copy back to host */
                        acc::copyout(hphi[ispn].at(memory_t::host, 0, i), hphi1[ispn].at(memory_t::device), ngv_fft);
                    }
                    break;
                }
            }
        }
#endif
    };

    /* transform wave-function to real space; the result of the transformation is stored in the FFT buffer */
    auto phi_to_r = [&](int ispn) {
        switch (spfft__.processing_unit()) {
            case SPFFT_PU_HOST: {
                spfft__.backward(reinterpret_cast<double const*>(phi1[ispn].at(memory_t::host)),
                                 SPFFT_PU_HOST);
                break;
            }
            case SPFFT_PU_GPU: {
                ///* parallel FFT starting from device pointer is not implemented */
                // assert(fft_coarse_.comm().size() == 1);
                // if (gamma) { /* warning: GPU pointer works only in case of serial FFT */
                //    fft_coarse_.transform<1, memory_t::device>(phi1[ispn].at(memory_t::device, 0, 0),
                //                                               phi1[ispn].at(memory_t::device, 0, 1));
                //} else {
                //    fft_coarse_.transform<1, memory_t::device>(phi1[ispn].at(memory_t::device, 0, 0));
                //}
                spfft__.backward(reinterpret_cast<double const*>(phi1[ispn].at(memory_t::device)),
                                 SPFFT_PU_GPU);
                break;
            }
        }
    };

    auto mul_by_veff = [&](double* buf, int ispn_block) {
        switch (spfft__.processing_unit()) {
            case SPFFT_PU_HOST: {
                if (ispn_block < 2) { /* up-up or dn-dn block */
                    if (ctx_.gamma_point()) {
                        #pragma omp parallel for schedule(static)
                        for (int ir = 0; ir < spfft__.local_slice_size(); ir++) {
                            /* multiply by V+Bz or V-Bz */
                            buf[ir] *= veff_vec_[ispn_block].f_rg(ir);
                        }
                    } else {
                        auto wf = reinterpret_cast<double_complex*>(buf);
                        #pragma omp parallel for schedule(static)
                        for (int ir = 0; ir < spfft__.local_slice_size(); ir++) {
                            /* multiply by V+Bz or V-Bz */
                            wf[ir] *= veff_vec_[ispn_block].f_rg(ir);
                        }
                    }
                } else {
                    double pref = (ispn_block == 2) ? -1 : 1;
                    auto wf     = reinterpret_cast<double_complex*>(buf);
                    #pragma omp parallel for schedule(static)
                    for (int ir = 0; ir < spfft__.local_slice_size(); ir++) {
                        /* multiply by Bx +/- i*By */
                        wf[ir] *= double_complex(veff_vec_[2].f_rg(ir), pref * veff_vec_[3].f_rg(ir));
                    }
                }
                break;
            }
            case SPFFT_PU_GPU: {
                STOP();
                //#ifdef __GPU
                //                mul_by_veff_gpu(ispn_block, fft_coarse_.local_size(), vptr.at(memory_t::device),
                //                                buf.at(memory_t::device));
                //#endif
                break;
            }
        }
    };

    /* transform one or two functions to PW domain */
    auto vphi_to_G = [&]() {
        switch (spfft__.processing_unit()) {
            case SPFFT_PU_HOST: {
                spfft__.forward(spfft__.processing_unit(), reinterpret_cast<double*>(vphi_.at(memory_t::host)),
                                SPFFT_FULL_SCALING);
                break;
            }
            case SPFFT_PU_GPU: {
                STOP();
                // if (gamma) {
                //    fft_coarse_.transform<-1, memory_t::device>(vphi_.at(memory_t::device, 0, 0),
                //                                                vphi_.at(memory_t::device, 0, 1));
                //} else {
                //    fft_coarse_.transform<-1, memory_t::device>(vphi_.at(memory_t::device, 0, 0));
                //}
                break;
            }
        }
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

        switch (spfft__.processing_unit()) {
            case SPFFT_PU_HOST: {
                /* CPU case */
                if (ekin) {
                    #pragma omp parallel for schedule(static)
                    for (int ig = 0; ig < gkvec_p_->gvec_count_fft(); ig++) {
                        hphi1[ispn][ig] += (phi1[ispn][ig] * pw_ekin_[ig] + vphi_[ig]);
                    }
                } else {
                    #pragma omp parallel for schedule(static)
                    for (int ig = 0; ig < gkvec_p_->gvec_count_fft(); ig++) {
                        hphi1[ispn][ig] += vphi_[ig];
                    }
                }
                break;
            }
            case SPFFT_PU_GPU: {
#ifdef __GPU
                double alpha = static_cast<double>(ekin);
                add_pw_ekin_gpu(gkvec_p_->gvec_count_fft(), alpha, pw_ekin_.at(memory_t::device),
                                phi1[ispn].at(memory_t::device), vphi_.at(memory_t::device),
                                hphi1[ispn].at(memory_t::device));
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
        if (ispn__ == 2) {
            prepare_phi_hphi(i);
            /* phi_u(G) -> phi_u(r) */
            phi_to_r(0);
            /* save phi_u(r) */
            switch (spfft__.processing_unit()) {
                case SPFFT_PU_HOST: {
                    // fft_coarse_.output(buf_rg_.at(memory_t::host));
                    spfft_output(spfft__, buf_rg_.at(memory_t::host));
                    //auto ptr = reinterpret_cast<double_complex*>(spfft__.space_domain_data(SPFFT_PU_HOST));
                    //std::copy(ptr, ptr + fft_coarse_.local_slice_size(), buf_rg_.at(memory_t::host));
                    break;
                }
                case SPFFT_PU_GPU: {
#ifdef __GPU
                    STOP();
                    // acc::copy(buf_rg_.at(memory_t::device), fft_coarse_.buffer().at(memory_t::device),
                    //          fft_coarse_.local_size());
#endif
                    break;
                }
            }
            /* multiply phi_u(r) by effective potential */
            mul_by_veff(spfft__.space_domain_data(SPFFT_PU_HOST), 0);
            /* V_{uu}(r)phi_{u}(r) -> [V*phi]_{u}(G) */
            vphi_to_G();
            /* add kinetic energy */
            add_to_hphi(0);
            /* multiply phi_{u} by V_{du} */
            mul_by_veff(reinterpret_cast<double*>(buf_rg_.at(memory_t::host)), 3);
            /* copy to FFT buffer */
            switch (spfft__.processing_unit()) {
                case SPFFT_PU_HOST: {
                    // fft_coarse_.input(buf_rg_.at(memory_t::host));
                    //auto ptr = reinterpret_cast<double_complex*>(spfft__.space_domain_data(SPFFT_PU_HOST));
                    //std::copy(buf_rg_.at(memory_t::host), buf_rg_.at(memory_t::host) + fft_coarse_.local_size(), ptr);
                    spfft_input(spfft__, buf_rg_.at(memory_t::host));
                    break;
                }
                case SPFFT_PU_GPU: {
#ifdef __GPU
                    STOP();
//                    acc::copy(fft_coarse_.buffer().at(memory_t::device), buf_rg_.at(memory_t::device),
//                              fft_coarse_.local_size());
#endif
                    break;
                }
            }
            /* V_{du}(r)phi_{u}(r) -> [V*phi]_{d}(G) */
            vphi_to_G();
            /* add to hphi_{d} */
            add_to_hphi(3);

            /* for the second spin */

            /* phi_d(G) -> phi_d(r) */
            phi_to_r(1);
            /* save phi_d(r) */
            switch (spfft__.processing_unit()) {
                case SPFFT_PU_HOST: {
                    // fft_coarse_.output(buf_rg_.at(memory_t::host));
                    //auto ptr = reinterpret_cast<double_complex*>(spfft__.space_domain_data(SPFFT_PU_HOST));
                    //std::copy(ptr, ptr + fft_coarse_.local_size(), buf_rg_.at(memory_t::host));
                    spfft_output(spfft__, buf_rg_.at(memory_t::host));
                    break;
                }
                case SPFFT_PU_GPU: {
#ifdef __GPU
                    STOP();
                    // acc::copy(buf_rg_.at(memory_t::device), fft_coarse_.buffer().at(memory_t::device),
                    //          fft_coarse_.local_size());
#endif
                    break;
                }
            }
            /* multiply phi_d(r) by effective potential */
            mul_by_veff(spfft__.space_domain_data(SPFFT_PU_HOST), 1);
            /* V_{dd}(r)phi_{d}(r) -> [V*phi]_{d}(G) */
            vphi_to_G();
            /* add kinetic energy */
            add_to_hphi(1);
            /* multiply phi_{d} by V_{ud} */
            mul_by_veff(reinterpret_cast<double*>(buf_rg_.at(memory_t::host)), 2);
            /* copy to FFT buffer */
            switch (spfft__.processing_unit()) {
                case SPFFT_PU_HOST: {
                    spfft_input(spfft__, buf_rg_.at(memory_t::host));
                    //auto ptr = reinterpret_cast<double_complex*>(spfft__.space_domain_data(SPFFT_PU_HOST));
                    //std::copy(buf_rg_.at(memory_t::host), buf_rg_.at(memory_t::host) + fft_coarse_.local_size(), ptr);
                    break;
                }
                case SPFFT_PU_GPU: {
#ifdef __GPU
                    STOP();
//                    acc::copy(fft_coarse_.buffer().at(memory_t::device), buf_rg_.at(memory_t::device),
//                              fft_coarse_.local_size());
#endif
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
            phi_to_r(ispn__);
            /* multiply by effective potential */
            mul_by_veff(spfft__.space_domain_data(SPFFT_PU_HOST), ispn__);
            /* V(r)phi(r) -> [V*phi](G) */
            vphi_to_G();
            /* add kinetic energy */
            add_to_hphi(ispn__);
            store_hphi(i);
        }
    }

    /* remap hphi backward */
    for (int ispn : spins) {
        hphi__.pw_coeffs(ispn).remap_backward(n__, idx0__);
        if (is_device_memory(hphi__.preferred_memory_t()) && hphi__.pw_coeffs(ispn).is_remapped()) {
            hphi__.pw_coeffs(ispn).copy_to(memory_t::device, idx0__, n__);
        }
    }
    /* at this point hphi in prime storage is both on CPU and GPU memory; however if the memory pool
       was used for the device memory allocation, device storage is destroyed */
}

void Local_operator::apply_h_o(spfft::Transform& spfftk__,int N__, int n__, Wave_functions& phi__, Wave_functions* hphi__, Wave_functions* ophi__)
{
    PROFILE("sirius::Local_operator::apply_h_o");

    if (!gkvec_p_) {
        TERMINATE("Local operator is not prepared");
    }

    num_applied(n__);

    mdarray<double_complex, 1> buf_pw(gkvec_p_->gvec_count_fft());

    if (ctx_.processing_unit() == device_t::GPU) {
        phi__.pw_coeffs(0).copy_to(memory_t::host, N__, n__);
    }
    // if (ctx_->control().print_checksum_) {
    //    auto cs = phi__.checksum_pw(N__, n__, ctx_->processing_unit());
    //    if (phi__.comm().rank() == 0) {
    //        DUMP("checksum(phi_pw): %18.10f %18.10f", cs.real(), cs.imag());
    //    }
    //}

    auto& mp = const_cast<Simulation_context&>(ctx_).mem_pool(memory_t::host);

    phi__.pw_coeffs(0).remap_forward(n__, N__, &mp);

    if (hphi__ != nullptr) {
        hphi__->pw_coeffs(0).set_num_extra(n__, N__, &mp);
    }

    if (ophi__ != nullptr) {
        ophi__->pw_coeffs(0).set_num_extra(n__, N__, &mp);
    }

    for (int j = 0; j < phi__.pw_coeffs(0).spl_num_col().local_size(); j++) {
        utils::timer t1("sirius::Local_operator::apply_h_o|pot");
        switch (spfftk__.processing_unit()) {
            case SPFFT_PU_HOST: {
                /* phi(G) -> phi(r) */
                spfftk__.backward(reinterpret_cast<double const*>(phi__.pw_coeffs(0).extra().at(memory_t::host, 0, j)),
                                 spfftk__.processing_unit());

                if (ophi__ != nullptr) {
                    /* save phi(r) */
                    if (hphi__ != nullptr) {
                        spfft_output(spfftk__, &buf_rg_[0]);
                    }
                    /* multiply phi(r) by step function */
                    spfft_multiply(spfftk__, [&](int ir)
                                             {
                                                 return theta_.f_rg(ir);
                                             });
                    /* phi(r) * Theta(r) -> ophi(G) */
                    spfftk__.forward(spfftk__.processing_unit(),
                                     reinterpret_cast<double*>(ophi__->pw_coeffs(0).extra().at(memory_t::host, 0, j)),
                                     SPFFT_FULL_SCALING);
                    /* load phi(r) back */
                    if (hphi__ != nullptr) {
                        spfft_input(spfftk__, buf_rg_.at(memory_t::host));
                    }
                }
                if (hphi__ != nullptr) {
                    /* multiply be effective potential, which itself was multiplied by the step function
                           in the prepare() method */
                    spfft_multiply(spfftk__, [&](int ir)
                                             {
                                                 return veff_vec_[0].f_rg(ir);
                                             });
                    /* phi(r) * Theta(r) * V(r) -> hphi(G) */
                    spfftk__.forward(spfftk__.processing_unit(),
                                     reinterpret_cast<double*>(hphi__->pw_coeffs(0).extra().at(memory_t::host, 0, j)),
                                     SPFFT_FULL_SCALING);
                }
                break;
            }
            case SPFFT_PU_GPU: {
#if defined(__GPU)
                STOP();
                ///* phi(G) -> phi(r) */
                //fft_coarse_.transform<1>(phi__.pw_coeffs(0).extra().at(memory_t::host, 0, j));

                //if (ophi__ != nullptr) {
                //    /* save phi(r) */
                //    if (hphi__ != nullptr) {
                //        acc::copy(buf_rg_.at(memory_t::device), fft_coarse_.buffer().at(memory_t::device),
                //                  fft_coarse_.local_size());
                //    }
                //    /* multiply by step function */
                //    scale_matrix_rows_gpu(fft_coarse_.local_size(), 1,
                //                          (acc_complex_double_t*)fft_coarse_.buffer().at(memory_t::device),
                //                          theta_.f_rg().at(memory_t::device));
                //    /* phi(r) * Theta(r) -> ophi(G) */
                //    fft_coarse_.transform<-1>(ophi__->pw_coeffs(0).extra().at(memory_t::host, 0, j));
                //    /* load phi(r) back */
                //    if (hphi__ != nullptr) {
                //        acc::copy(fft_coarse_.buffer().at(memory_t::device), buf_rg_.at(memory_t::device),
                //                  fft_coarse_.local_size());
                //    }
                //}
                //if (hphi__ != nullptr) {
                //    /* multiply by effective potential */
                //    scale_matrix_rows_gpu(fft_coarse_.local_size(), 1,
                //                          (acc_complex_double_t*)fft_coarse_.buffer().at(memory_t::device),
                //                          veff_vec_[0].f_rg().at(memory_t::device));
                //    /* phi(r) * Theta(r) * V(r) -> hphi(G) */
                //    fft_coarse_.transform<-1>(hphi__->pw_coeffs(0).extra().at(memory_t::host, 0, j));
                //}
#endif
                break;
            }
        }
        t1.stop();

        if (hphi__ != nullptr) {
            utils::timer t2("sirius::Local_operator::apply_h_o|kin");
            /* add kinetic energy */
            for (int x : {0, 1, 2}) {
                #pragma omp parallel for schedule(static)
                for (int igloc = 0; igloc < gkvec_p_->gvec_count_fft(); igloc++) {
                    /* global index of G-vector */
                    int ig = gkvec_p_->idx_gvec(igloc);
                    /* \hat P phi = phi(G+k) * (G+k), \hat P is momentum operator */
                    buf_pw[igloc] = phi__.pw_coeffs(0).extra()(igloc, j) *
                                    gkvec_p_->gvec().gkvec_cart<index_domain_t::global>(ig)[x];
                }
                /* transform Cartesian component of wave-function gradient to real space */
                spfftk__.backward(reinterpret_cast<double const*>(&buf_pw[0]),
                                  spfftk__.processing_unit());
                switch (spfftk__.processing_unit()) {
                    case SPFFT_PU_HOST: {
                        /* multiply be step function */
                        spfft_multiply(spfftk__, [&](int ir)
                                                 {
                                                     return theta_.f_rg(ir);
                                                 });

                        break;
                    }
                    case SPFFT_PU_GPU: {
#if defined(__GPU)
                        STOP();
                        /* multiply by step function */
                        //scale_matrix_rows_gpu(fft_coarse_.local_size(), 1,
                        //                      (acc_complex_double_t*)fft_coarse_.buffer().at(memory_t::device),
                        //                      theta_.f_rg().at(memory_t::device));
#endif
                        break;
                    }
                }
                /* transform back to PW domain */
                spfftk__.forward(spfftk__.processing_unit(), reinterpret_cast<double*>(&buf_pw[0]),
                                 SPFFT_FULL_SCALING);
                #pragma omp parallel for schedule(static)
                for (int igloc = 0; igloc < gkvec_p_->gvec_count_fft(); igloc++) {
                    int ig = gkvec_p_->idx_gvec(igloc);
                    hphi__->pw_coeffs(0).extra()(igloc, j) +=
                        0.5 * buf_pw[igloc] * gkvec_p_->gvec().gkvec_cart<index_domain_t::global>(ig)[x];
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

    if (ctx_.processing_unit() == device_t::GPU) {
        if (hphi__ != nullptr) {
            hphi__->pw_coeffs(0).copy_to(memory_t::device, N__, n__);
        }
        if (ophi__ != nullptr) {
            ophi__->pw_coeffs(0).copy_to(memory_t::device, N__, n__);
        }
    }
    // if (ctx_->control().print_checksum_) {
    //    auto cs1 = hphi__.checksum_pw(N__, n__, ctx_->processing_unit());
    //    auto cs2 = ophi__.checksum_pw(N__, n__, ctx_->processing_unit());
    //    if (phi__.comm().rank() == 0) {
    //        DUMP("checksum(hphi_pw): %18.10f %18.10f", cs1.real(), cs1.imag());
    //        DUMP("checksum(ophi_pw): %18.10f %18.10f", cs2.real(), cs2.imag());
    //    }
    //}
}

void Local_operator::apply_b(spfft::Transform& spfftk__, int N__, int n__, Wave_functions& phi__, std::vector<Wave_functions>& bphi__)
{
    PROFILE("sirius::Local_operator::apply_b");

    if (!gkvec_p_) {
        TERMINATE("Local operator is not prepared");
    }

    /* components of H|psi> to which H is applied */
    std::vector<int> iv(1, 0);
    if (bphi__.size() == 3) {
        iv.push_back(2);
    }

    auto& mp = const_cast<Simulation_context&>(ctx_).mem_pool(memory_t::host);

    phi__.pw_coeffs(0).remap_forward(n__, N__, &mp);
    for (int i : iv) {
        bphi__[i].pw_coeffs(0).set_num_extra(n__, N__, &mp);
    }

    for (int j = 0; j < phi__.pw_coeffs(0).spl_num_col().local_size(); j++) {
        switch (spfftk__.processing_unit()) {
            case SPFFT_PU_HOST: {
                /* phi(G) -> phi(r) */
                spfftk__.backward(reinterpret_cast<double const*>(phi__.pw_coeffs(0).extra().at(memory_t::host, 0, j)),
                                  spfftk__.processing_unit());

                /* save phi(r) */
                if (bphi__.size() == 3) {
                    spfft_output(spfftk__, buf_rg_.at(memory_t::host));
                }
                /* multiply by Bz */
                spfft_multiply(spfftk__, [&](int ir)
                                         {
                                             return veff_vec_[1].f_rg(ir);
                                         });

                /* phi(r) * Bz(r) -> bphi[0](G) */
                spfftk__.forward(spfftk__.processing_unit(),
                                 reinterpret_cast<double*>(bphi__[0].pw_coeffs(0).extra().at(memory_t::host, 0, j)),
                                 SPFFT_FULL_SCALING);

                /* non-collinear case */
                if (bphi__.size() == 3) {
                    /* multiply by Bx-iBy */
                    spfft_input(spfftk__, [&](int ir)
                                          {
                                              return buf_rg_[ir] * double_complex(veff_vec_[2].f_rg(ir), -veff_vec_[3].f_rg(ir));
                                          });
                    /* phi(r) * (Bx(r)-iBy(r)) -> bphi[2](G) */
                    spfftk__.forward(spfftk__.processing_unit(),
                                     reinterpret_cast<double*>(bphi__[2].pw_coeffs(0).extra().at(memory_t::host, 0, j)),
                                     SPFFT_FULL_SCALING);
                }
                break;
            }
            case SPFFT_PU_GPU: {
#if defined(__GPU)
                STOP();
                ///* phi(G) -> phi(r) */
                //fft_coarse_.transform<1>(phi__.pw_coeffs(0).extra().at(memory_t::host, 0, j));
                ///* multiply by Bz */
                //scale_matrix_rows_gpu(fft_coarse_.local_size(), 1,
                //                      (acc_complex_double_t*)fft_coarse_.buffer().at(memory_t::device),
                //                      veff_vec_[1].f_rg().at(memory_t::device));
                ///* phi(r) * Bz(r) -> bphi[0](G) */
                //fft_coarse_.transform<-1>(bphi__[0].pw_coeffs(0).extra().at(memory_t::host, 0, j));
#endif
                break;
            }
        }
    }

    for (int i : iv) {
        bphi__[i].pw_coeffs(0).remap_backward(n__, N__);
    }
}

} // namespace sirius
