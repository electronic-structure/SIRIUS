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

/** \file local_operator.hpp
 *
 *  \brief Contains declaration and implementation of sirius::Local_operator class.
 */

#ifndef __LOCAL_OPERATOR_HPP__
#define __LOCAL_OPERATOR_HPP__

#include "Potential/potential.hpp"

#ifdef __GPU
extern "C" void mul_by_veff_gpu(int ispn__, int size__, double* const* veff__, double_complex* buf__);

extern "C" void add_pw_ekin_gpu(int                   num_gvec__,
                                double                alpha__,
                                double const*         pw_ekin__,
                                double_complex const* phi__,
                                double_complex const* vphi__,
                                double_complex*       hphi__);
#endif

namespace sirius {

/// Representation of the local operator.
/** The following functionality is implementated:
 *    - application of the local part of Hamiltonian (kinetic + potential) to the wave-fucntions in the PP-PW case
 *    - application of the interstitial part of H and O in the case of FP-LAPW
 *    - application of the interstitial part of effective magnetic field to the first-variational functios
 *    - remapping of potential and unit-step functions from fine to coarse mesh of G-vectors
 */
class Local_operator
{
  private:
    /// Common parameters.
    Simulation_context const& ctx_;

    /// Coarse-grid FFT driver for this operator.
    FFT3D& fft_coarse_;

    /// Distribution of the G-vectors for the FFT transformation.
    Gvec_partition const& gvec_coarse_p_;

    Gvec_partition const* gkvec_p_{nullptr};

    /// Kinetic energy of G+k plane-waves.
    mdarray<double, 1> pw_ekin_;

    /// Effective potential components on a coarse FFT grid.
    std::array<Smooth_periodic_function<double>, 4> veff_vec_;

    /// Temporary array to store [V*phi](G)
    mdarray<double_complex, 2> vphi_;

    /// LAPW unit step function on a coarse FFT grid.
    Smooth_periodic_function<double> theta_;

    /// Temporary array to store psi_{up}(r).
    /** The size of the array is equal to the size of FFT buffer. */
    mdarray<double_complex, 1> buf_rg_;

    /// V(G=0) matrix elements.
    double v0_[2];

  public:
    /// Constructor.
    Local_operator(Simulation_context const& ctx__,
                   FFT3D&                    fft_coarse__,
                   Gvec_partition     const& gvec_coarse_p__)
        : ctx_(ctx__)
        , fft_coarse_(fft_coarse__)
        , gvec_coarse_p_(gvec_coarse_p__)

    {
        for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
            veff_vec_[j] = Smooth_periodic_function<double>(fft_coarse__, gvec_coarse_p__);
            for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
                veff_vec_[j].f_rg(ir) = 2.71828;
            }
        }
        if (ctx_.full_potential()) {
            theta_ = Smooth_periodic_function<double>(fft_coarse__, gvec_coarse_p__);
        }

        if (fft_coarse_.pu() == GPU) {
            for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                veff_vec_[j].f_rg().allocate(memory_t::device);
                veff_vec_[j].f_rg().copy<memory_t::host, memory_t::device>();
            }
            buf_rg_.allocate(memory_t::device);
        }
    }

    /// Keep track of the total number of wave-functions to which the local operator was applied.
    static int num_applied(int n = 0)
    {
        static int num_applied_{0};
        num_applied_ += n;
        return num_applied_;
    }

    /// Map effective potential and magnetic field to a coarse FFT mesh.
    /** \param [in] potential      \f$ V_{eff}({\bf r}) \f$ and \f$ {\bf B}_{eff}({\bf r}) \f$ on the fine grid FFT grid.
     *
     *  This function should be called prior to the band diagonalziation. In case of GPU execution all
     *  effective fields on the coarse grid will be copied to the device and will remain there until the
     *  dismiss() method is called after band diagonalization.
     */
    inline void prepare(Potential& potential__)
    {
        PROFILE("sirius::Local_operator::prepare");

        if (!buf_rg_.size()) {
            buf_rg_ = mdarray<double_complex, 1>(fft_coarse_.local_size(), memory_t::host, "Local_operator::buf_rg_");
        }

        fft_coarse_.prepare(gvec_coarse_p_);

        if (ctx_.full_potential()) {

            auto& fft_dense    = ctx_.fft();
            auto& gvec_dense_p = ctx_.gvec_partition();

            Smooth_periodic_function<double> ftmp(fft_dense, gvec_dense_p);
            for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                for (int ir = 0; ir < fft_dense.local_size(); ir++) {
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
                theta_.f_pw_local(igloc) = ctx_.theta_pw(gvec_dense_p.gvec().gvec_base_mapping(igloc) +
                                                         gvec_dense_p.gvec().offset());
            }
            theta_.fft_transform(1);
            /* release FFT driver */
            fft_coarse_.dismiss();

            if (fft_coarse_.pu() == GPU) {
                for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                    veff_vec_[j].f_rg().allocate(memory_t::device);
                    veff_vec_[j].f_rg().copy<memory_t::host, memory_t::device>();
                }
                theta_.f_rg().allocate(memory_t::device);
                theta_.f_rg().copy<memory_t::host, memory_t::device>();
                buf_rg_.allocate(memory_t::device);
            }

            //if (ctx_.control().print_checksum_) {
            //    double cs[] = {veff_vec_.checksum(), theta_.checksum()};
            //    fft_coarse_.comm().allreduce(&cs[0], 2);
            //    if (mpi_comm_world().rank() == 0) {
            //        print_checksum("veff_vec", cs[0]);
            //        print_checksum("theta", cs[1]);
            //    }
            //}

        } else {

            for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                /* loop over local set of coarse G-vectors */
                #pragma omp parallel for schedule(static)
                for (int igloc = 0; igloc < gvec_coarse_p_.gvec().count(); igloc++) {
                    /* map from fine to coarse set of G-vectors */
                    veff_vec_[j].f_pw_local(igloc) = potential__.component(j).f_pw_local(potential__.component(j).gvec().gvec_base_mapping(igloc));
                }
                /* transform to real space */
                veff_vec_[j].fft_transform(1);
            }

            if (ctx_.num_mag_dims()) {
                for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
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
            if (fft_coarse_.pu() == GPU) {
                for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                    veff_vec_[j].f_rg().allocate(memory_t::device);
                    veff_vec_[j].f_rg().copy<memory_t::host, memory_t::device>();
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

        fft_coarse_.dismiss();
    }

    /// Prepare the k-point dependent arrays.
    inline void prepare(Gvec_partition const& gkvec_p__)
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
            int p = (gkvec_p__.gvec().reduced()) ? 2 : 1;
            vphi_ = mdarray<double_complex, 2>(ngv_fft, p, memory_t::host, "Local_operator::vphi1");
        }

        if (fft_coarse_.pu() == device_t::GPU) {
            pw_ekin_.allocate(memory_t::device);
            pw_ekin_.copy<memory_t::host, memory_t::device>();
            vphi_.allocate(memory_t::device);
        }
    }

    /// Cleanup the local operator.
    inline void dismiss()
    {
        if (fft_coarse_.pu() == GPU) {
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

    /// Apply local part of Hamiltonian to wave-functions.
    /** \param [in]  ispn Index of spin.
     *  \param [in]  phi  Input wave-functions.
     *  \param [out] hphi Hamiltonian applied to wave-function.
     *  \param [in]  idx0 Starting index of wave-functions.
     *  \param [in]  n    Number of wave-functions to which H is applied.
     *
     *  Index of spin can take the following values:
     *    - 0: apply H_{uu} to the up- component of wave-functions
     *    - 1: apply H_{dd} to the dn- component of wave-functions
     *    - 2: apply full Hamiltonian to the spinor wave-functions
     *
     *  In the current implementation for the GPUs sequential FFT is assumed.
     */
    void apply_h(int ispn__, Wave_functions& phi__, Wave_functions& hphi__, int idx0__, int n__)
    {
        PROFILE("sirius::Local_operator::apply_h");

        if (!gkvec_p_) {
            TERMINATE("Local operator is not prepared");
        }

        /* increment the counter by the number of wave-functions */
        num_applied(n__);

        /* this memory pool will be used to allocate extra storage in the host memory*/
        auto& mp = const_cast<Simulation_context&>(ctx_).mem_pool(memory_t::host);
        /* this memory pool will be used to allocate extra storage in the device memory */
#ifdef __GPU
        auto& mpd = const_cast<Simulation_context&>(ctx_).mem_pool(memory_t::device);
#endif
        /* alias array for wave functions */
        std::array<mdarray<double_complex, 2>, 2> phi;
        /* alias array for hphi */
        std::array<mdarray<double_complex, 2>, 2> hphi;

        /* alias for one or two wave-functions that are currently computed */
        std::array<mdarray<double_complex, 2>, 2> phi1;
        std::array<mdarray<double_complex, 2>, 2> hphi1;

        /* spin component to which H is applied */
        auto spins = get_spins(ispn__);

        /* local number of G-vectors for the FFT transformation */
        int ngv_fft = gkvec_p_->gvec_count_fft();

        /* small buffers in the host memory */
        auto buff = mp.get_unique_ptr<double_complex>(4 * ngv_fft);
        /* small buffers in the device memory */
#ifdef __GPU
        auto buff_d = mpd.get_unique_ptr<double_complex>(4 * ngv_fft);
#endif
        memory_t mem_phi{memory_t::none};
        memory_t mem_hphi{memory_t::none};

        /* remap wave-functions to FFT friendly distribution */
        for (int ispn: spins) {
            /* if we store wave-functions in the device memory and if the wave functions are remapped
               we need to copy the wave functions to host memory */
            if (is_device_memory(ctx_.preferred_memory_t()) && phi__.pw_coeffs(ispn).is_remapped()) {
#ifdef __GPU
                phi__.pw_coeffs(ispn).copy_to_host(idx0__, n__);
#endif
            }
            /* set FFT friendly distribution */
            phi__.pw_coeffs(ispn).remap_forward(n__, idx0__, &mp);
            /* memory location of phi in extra storage */
            mem_phi = (phi__.pw_coeffs(ispn).is_remapped()) ? memory_t::host : ctx_.preferred_memory_t();
            /* set FFT friednly distribution */
            hphi__.pw_coeffs(ispn).set_num_extra(n__, idx0__, &mp);
            /* memory location of hphi in extra storage */
            mem_hphi = (hphi__.pw_coeffs(ispn).is_remapped()) ? memory_t::host : ctx_.preferred_memory_t();

            /* local number of wave-functions in extra-storage distribution */
            int num_wf_loc = phi__.pw_coeffs(ispn).spl_num_col().local_size();

            /* set alias for phi extra storage */
            double_complex* ptr{nullptr};
            if (phi__.pw_coeffs(ispn).extra().on_device()) {
                ptr = phi__.pw_coeffs(ispn).extra().at(memory_t::device);
            }
            phi[ispn] = mdarray<double_complex, 2>(phi__.pw_coeffs(ispn).extra().at(memory_t::host), ptr, ngv_fft, num_wf_loc);

            /* set alias for hphi extra storage */
            ptr = nullptr;
            if (hphi__.pw_coeffs(ispn).extra().on_device()) {
                ptr = hphi__.pw_coeffs(ispn).extra().at(memory_t::device);
            }
            hphi[ispn] = mdarray<double_complex, 2>(hphi__.pw_coeffs(ispn).extra().at(memory_t::host), ptr, ngv_fft, num_wf_loc);
        }

#ifdef __GPU
        memory_t vptr_mem = acc::num_devices() > 0 ? memory_t::host | memory_t::device : memory_t::host;
        mdarray<double*, 1> vptr(4, vptr_mem);
        vptr.zero();
        switch (fft_coarse_.pu()) {
            case device_t::GPU: {
                for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                    vptr[j] = veff_vec_[j].f_rg().at<GPU>();
                }
                vptr.copy<memory_t::host, memory_t::device>();
                break;
            }
            case device_t::CPU: break;
        }
#endif
        auto prepare_phi_hphi = [&](int i, bool gamma = false)
        {
            /* number of simultaneously transformed wave-functions */
            int p = (gamma) ? 2 : 1;
            /* offset in the auxiliary buffer */
            int o{0};

            for (int ispn: spins) {
                switch (fft_coarse_.pu()) {
                    case device_t::CPU: { /* FFT is done on CPU */
                        if (is_host_memory(mem_phi)) { /* wave-functions are also on host memory */
                            phi1[ispn] = mdarray<double_complex, 2>(phi[ispn].at(memory_t::host, 0, i * p),
                                                                    ngv_fft, p);
                        } else { /* wave-functions are on the device memory */
                            phi1[ispn] = mdarray<double_complex, 2>(buff.get() + o, ngv_fft, p);
                            o += ngv_fft * p;
#ifdef __GPU
                            /* copy wave-functions to host memory */
                            acc::copyout(phi1[ispn].at(memory_t::host), phi[ispn].at(memory_t::device, 0, i * p),
                                         ngv_fft * p);
#endif
                        }
                        if (is_host_memory(mem_hphi)) {
                            hphi1[ispn] = mdarray<double_complex, 2>(hphi[ispn].at(memory_t::host, 0, i * p),
                                                                     ngv_fft, p);
                        } else {
                            hphi1[ispn] = mdarray<double_complex, 2>(buff.get() + o, ngv_fft, p);
                            o += ngv_fft * p;
                        }
                        break;
                    }
                    case device_t::GPU: { /* FFT is done on GPU */
                        if (is_host_memory(mem_phi)) {
#ifdef __GPU
                            phi1[ispn] = mdarray<double_complex, 2>(nullptr, buff_d.get() + o, ngv_fft, p);
                            o += ngv_fft * p;
                            /* copy wave-functions to device */
                            acc::copyin(phi1[ispn].at(memory_t::device), phi[ispn].at(memory_t::host, 0, i * p),
                                        p * ngv_fft);
#endif
                        } else {
                            phi1[ispn] = mdarray<double_complex, 2>(nullptr, phi[ispn].at(memory_t::device, 0, i * p),
                                                                    ngv_fft, p);
                        }
                        if (is_host_memory(mem_hphi)) {
#ifdef __GPU
                            hphi1[ispn] = mdarray<double_complex, 2>(nullptr, buff_d.get() + o, ngv_fft, p);
                            o += ngv_fft * p;
#endif
                        } else {
                            hphi1[ispn] = mdarray<double_complex, 2>(nullptr, hphi[ispn].at(memory_t::device, 0, i * p),
                                                                     ngv_fft, p);
                        }
                        break;
                    }
                }
                hphi1[ispn].zero(memory_t::host);
                hphi1[ispn].zero(memory_t::device);
            }
        };

        auto store_hphi = [&](int i, bool gamma = false)
        {
#ifdef __GPU
            /* number of simultaneously transformed wave-functions */
            int p = (gamma) ? 2 : 1;

            for (int ispn: spins) {
                switch (fft_coarse_.pu()) {
                    case device_t::CPU: { /* FFT is done on CPU */
                        if (is_device_memory(mem_hphi)) {
                            /* copy to device */
                            acc::copyin(hphi[ispn].at(memory_t::device, 0, i * p), hphi1[ispn].at(memory_t::host),
                                        p * ngv_fft);
                        }
                        break;
                    }
                    case device_t::GPU: { /* FFT is done on GPU */
                        if (is_host_memory(mem_hphi)) {
                            /* copy back to host */
                             acc::copyout(hphi[ispn].at(memory_t::host, 0, i * p), hphi1[ispn].at(memory_t::device),
                                          p * ngv_fft);
                        }
                        break;
                    }
                }
            }
#endif
        };

        /* transform one or two wave-functions to real space; the result of
         * transformation is stored in the FFT buffer */
        auto phi_to_r = [&](int ispn, bool gamma = false) {
            switch (fft_coarse_.pu()) {
                case device_t::CPU: {
                    if (gamma) {
                        fft_coarse_.transform<1, memory_t::host>(phi1[ispn].at(memory_t::host, 0, 0),
                                                                 phi1[ispn].at(memory_t::host, 0, 1));

                    } else {
                        fft_coarse_.transform<1, memory_t::host>(phi1[ispn].at(memory_t::host, 0, 0));
                    }
                    break;
                }
                case device_t::GPU: {
                    /* parallel FFT starting from device pointer is not implemented */
                    assert(fft_coarse_.comm().size() == 1);
                    if (gamma) { /* warning: GPU pointer works only in case of serial FFT */
                        fft_coarse_.transform<1, memory_t::device>(phi1[ispn].at(memory_t::device, 0, 0),
                                                                   phi1[ispn].at(memory_t::device, 0, 1));
                    } else {
                        fft_coarse_.transform<1, memory_t::device>(phi1[ispn].at(memory_t::device, 0, 0));
                    }
                    break;
                }
            }
        };

        /* multiply by effective potential */
        auto mul_by_veff = [&](mdarray<double_complex, 1>& buf, int ispn_block) {
            switch (fft_coarse_.pu()) {
                case device_t::CPU: {
                    if (ispn_block < 2) {
                        #pragma omp parallel for schedule(static)
                        for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
                            /* multiply by V+Bz or V-Bz */
                            buf[ir] *= veff_vec_[ispn_block].f_rg(ir);
                        }
                    } else {
                        double pref = (ispn_block == 2) ? -1 : 1;
                        #pragma omp parallel for schedule(static)
                        for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
                            /* multiply by Bx +/- i*By */
                            buf[ir] *= double_complex(veff_vec_[2].f_rg(ir), pref * veff_vec_[3].f_rg(ir));
                        }
                    }
                    break;
                }
                case device_t::GPU: {
#ifdef __GPU
                    mul_by_veff_gpu(ispn_block, fft_coarse_.local_size(), vptr.at<GPU>(), buf.at<GPU>());
#endif
                    break;
                }
            }
        };

        /* transform one or two functions to PW domain */
        auto vphi_to_G = [&](bool gamma = false) {
            switch (fft_coarse_.pu()) {
                case device_t::CPU: {
                    if (gamma) {
                        fft_coarse_.transform<-1, memory_t::host>(vphi_.at(memory_t::host, 0, 0), vphi_.at(memory_t::host, 0, 1));
                    } else {
                        fft_coarse_.transform<-1, memory_t::host>(vphi_.at(memory_t::host, 0, 0));
                    }
                    break;
                }
                case device_t::GPU: {
                    if (gamma) {
                        fft_coarse_.transform<-1, memory_t::device>(vphi_.at(memory_t::device, 0, 0), vphi_.at(memory_t::device, 0, 1));
                    } else {
                        fft_coarse_.transform<-1, memory_t::device>(vphi_.at(memory_t::device, 0, 0));
                    }
                    break;
                }
            }
        };

        /* store the resulting hphi
           spin block (ispn_block) is used as a bit mask: 
            - first bit: spin component which is updated
            - second bit: add or not kinetic energy term */
        auto add_to_hphi = [&](int ispn_block, bool gamma = false) {
            /* index of spin component */
            int ispn = ispn_block & 1;
            /* add kinetic energy if this is a diagonal block */
            int ekin = (ispn_block & 2) ? 0 : 1;

            switch (fft_coarse_.pu()) {
                case device_t::CPU: {
                    /* CPU case */
                    if (gamma) { /* update two wave functions */
                        if (ekin) {
                            #pragma omp parallel for schedule(static)
                            for (int ig = 0; ig < gkvec_p_->gvec_count_fft(); ig++) {
                                hphi1[ispn](ig, 0) += (phi1[ispn](ig, 0) * pw_ekin_[ig] + vphi_(ig, 0));
                                hphi1[ispn](ig, 1) += (phi1[ispn](ig, 1) * pw_ekin_[ig] + vphi_(ig, 1));
                            }
                        } else {
                            #pragma omp parallel for schedule(static)
                            for (int ig = 0; ig < gkvec_p_->gvec_count_fft(); ig++) {
                                hphi1[ispn](ig, 0) += vphi_(ig, 0);
                                hphi1[ispn](ig, 1) += vphi_(ig, 1);
                            }
                        }
                    } else { /* update single wave function */
                        if (ekin) {
                            #pragma omp parallel for schedule(static)
                            for (int ig = 0; ig < gkvec_p_->gvec_count_fft(); ig++) {
                                hphi1[ispn](ig, 0) += (phi1[ispn](ig, 0) * pw_ekin_[ig] + vphi_(ig, 0));
                            }
                        } else {
                            #pragma omp parallel for schedule(static)
                            for (int ig = 0; ig < gkvec_p_->gvec_count_fft(); ig++) {
                                hphi1[ispn](ig, 0) += vphi_(ig, 0);
                            }
                        }
                    }
                    break;
                }
                case device_t::GPU: {
#ifdef __GPU
                    double alpha = static_cast<double>(ekin);
                    if (gamma) {
                        add_pw_ekin_gpu(gkvec_p_->gvec_count_fft(),
                                        alpha,
                                        pw_ekin_.at<GPU>(),
                                        phi1[ispn].at<GPU>(0, 0),
                                        vphi_.at<GPU>(0, 0),
                                        hphi1[ispn].at<GPU>(0, 0));
                        add_pw_ekin_gpu(gkvec_p_->gvec_count_fft(),
                                        alpha,
                                        pw_ekin_.at<GPU>(),
                                        phi1[ispn].at<GPU>(0, 1),
                                        vphi_.at<GPU>(0, 1),
                                        hphi1[ispn].at<GPU>(0, 1));
                    } else {
                        add_pw_ekin_gpu(gkvec_p_->gvec_count_fft(),
                                        alpha,
                                        pw_ekin_.at<GPU>(),
                                        phi1[ispn].at<GPU>(0, 0),
                                        vphi_.at<GPU>(0, 0),
                                        hphi1[ispn].at<GPU>(0, 0));
                    }
#endif
                    break;
                }
            }
        };

        /* local number of wave-functions in extra-storage distribution */
        int num_wf_loc = phi__.pw_coeffs(0).spl_num_col().local_size();

        int first{0};
        /* If G-vectors are reduced, wave-functions are real and we can transform two of them at once.
           Non-collinear case is not treated here because nc wave-functions are complex and G+k vectors 
           can't be reduced. In this case input spin index can only be 0 or 1. */
        if (gkvec_p_->gvec().reduced()) {
            int npairs = num_wf_loc / 2;
            /* Gamma-point case can only be non-magnetic or spin-collinear */
            for (int i = 0; i < npairs; i++) {
                /* setup pointers */
                prepare_phi_hphi(i, true);
                /* phi(G) -> phi(r) */
                phi_to_r(ispn__, true);
                /* multiply by effective potential */
                mul_by_veff(fft_coarse_.buffer(), ispn__);
                /* V(r)phi(r) -> [V*phi](G) */
                vphi_to_G(true);
                /* add kinetic energy */
                add_to_hphi(ispn__, true);
                /* copy to main hphi array */
                store_hphi(i, true);
            }
            /* check if we have to do last wave-function which had no pair */
            first = num_wf_loc - num_wf_loc % 2;
        }

        /* if we don't have G-vector reductions, first = 0 and we start a normal loop */
        for (int i = first; i < num_wf_loc; i++) {

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
                switch (fft_coarse_.pu()) {
                    case device_t::CPU: {
                        fft_coarse_.output(buf_rg_.at<CPU>());
                        break;
                    }
                    case device_t::GPU: {
#ifdef __GPU
                        acc::copy(buf_rg_.at<GPU>(), fft_coarse_.buffer().at<GPU>(), fft_coarse_.local_size());
#endif
                        break;
                    }
                }
                /* multiply phi_u(r) by effective potential */
                mul_by_veff(fft_coarse_.buffer(), 0);
                /* V_{uu}(r)phi_{u}(r) -> [V*phi]_{u}(G) */
                vphi_to_G();
                /* add kinetic energy */
                add_to_hphi(0);
                /* multiply phi_{u} by V_{du} */
                mul_by_veff(buf_rg_, 3);
                /* copy to FFT buffer */
                switch (fft_coarse_.pu()) {
                    case device_t::CPU: {
                        fft_coarse_.input(buf_rg_.at<CPU>());
                        break;
                    }
                    case device_t::GPU: {
#ifdef __GPU
                        acc::copy(fft_coarse_.buffer().at<GPU>(), buf_rg_.at<GPU>(), fft_coarse_.local_size());
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
                switch (fft_coarse_.pu()) {
                    case device_t::CPU: {
                        fft_coarse_.output(buf_rg_.at<CPU>());
                        break;
                    }
                    case device_t::GPU: {
#ifdef __GPU
                        acc::copy(buf_rg_.at<GPU>(), fft_coarse_.buffer().at<GPU>(), fft_coarse_.local_size());
#endif
                        break;
                    }
                }
                /* multiply phi_d(r) by effective potential */
                mul_by_veff(fft_coarse_.buffer(), 1);
                /* V_{dd}(r)phi_{d}(r) -> [V*phi]_{d}(G) */
                vphi_to_G();
                /* add kinetic energy */
                add_to_hphi(1);
                /* multiply phi_{d} by V_{ud} */
                mul_by_veff(buf_rg_, 2);
                /* copy to FFT buffer */
                switch (fft_coarse_.pu()) {
                    case CPU: {
                        fft_coarse_.input(buf_rg_.at<CPU>());
                        break;
                    }
                    case GPU: {
#ifdef __GPU
                        acc::copy(fft_coarse_.buffer().at<GPU>(), buf_rg_.at<GPU>(), fft_coarse_.local_size());
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
                mul_by_veff(fft_coarse_.buffer(), ispn__);
                /* V(r)phi(r) -> [V*phi](G) */
                vphi_to_G();
                /* add kinetic energy */
                add_to_hphi(ispn__);
                store_hphi(i);
            }
        }

        /* remap hphi backward */
        for (int ispn: spins) {
            hphi__.pw_coeffs(ispn).remap_backward(n__, idx0__);
            if (is_device_memory(ctx_.preferred_memory_t()) && hphi__.pw_coeffs(ispn).is_remapped()) {
#ifdef __GPU
                hphi__.pw_coeffs(ispn).copy_to_device(idx0__, n__);
#endif
            }
        }
        /* at this point hphi in prime storage is both on CPU and GPU memory; however if the memory pool
           was used for the device memory allocation, device storage is destroyed */
    }

    void apply_h_o(int             N__,
                   int             n__,
                   Wave_functions& phi__,
                   Wave_functions* hphi__,
                   Wave_functions* ophi__)
    {
        PROFILE("sirius::Local_operator::apply_h_o");

        if (!gkvec_p_) {
            TERMINATE("Local operator is not prepared");
        }

        num_applied(n__);

        fft_coarse_.prepare(*gkvec_p_);

        mdarray<double_complex, 1> buf_pw(gkvec_p_->gvec_count_fft());

#if defined(__GPU)
        if (ctx_.processing_unit() == GPU) {
            phi__.pw_coeffs(0).copy_to_host(N__, n__);
        }
#endif
        //if (ctx_->control().print_checksum_) {
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
            switch (fft_coarse_.pu()) {
                case CPU: {
                    /* phi(G) -> phi(r) */
                    fft_coarse_.transform<1>(phi__.pw_coeffs(0).extra().at<CPU>(0, j));

                    if (ophi__ != nullptr) {
                        /* save phi(r) */
                        if (hphi__ != nullptr) {
                            fft_coarse_.output(buf_rg_.at<CPU>());
                        }
                        #pragma omp parallel for schedule(static)
                        for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
                            /* multiply phi(r) by step function */
                            fft_coarse_.buffer(ir) *= theta_.f_rg(ir);
                        }
                        /* phi(r) * Theta(r) -> ophi(G) */
                        fft_coarse_.transform<-1>(ophi__->pw_coeffs(0).extra().at<CPU>(0, j));
                        /* load phi(r) back */
                        if (hphi__ != nullptr) {
                            fft_coarse_.input(buf_rg_.at<CPU>());
                        }
                    }
                    if (hphi__ != nullptr) {
                        #pragma omp parallel for schedule(static)
                        for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
                            /* multiply be effective potential, which itself was multiplied by the step function 
                                   in the prepare() method */
                            fft_coarse_.buffer(ir) *= veff_vec_[0].f_rg(ir);
                        }
                        /* phi(r) * Theta(r) * V(r) -> hphi(G) */
                        fft_coarse_.transform<-1>(hphi__->pw_coeffs(0).extra().at<CPU>(0, j));
                    }
                    break;
                }
                case GPU: {
#if defined(__GPU)
                    /* phi(G) -> phi(r) */
                    fft_coarse_.transform<1>(phi__.pw_coeffs(0).extra().at<CPU>(0, j));

                    if (ophi__ != nullptr) {
                        /* save phi(r) */
                        if (hphi__ != nullptr) {
                            acc::copy(buf_rg_.at<GPU>(), fft_coarse_.buffer().at<GPU>(), fft_coarse_.local_size());
                        }
                        /* multiply by step function */
                        scale_matrix_rows_gpu(fft_coarse_.local_size(), 1, fft_coarse_.buffer().at<GPU>(),
                                              theta_.f_rg().at<GPU>());
                        /* phi(r) * Theta(r) -> ophi(G) */
                        fft_coarse_.transform<-1>(ophi__->pw_coeffs(0).extra().at<CPU>(0, j));
                        /* load phi(r) back */
                        if (hphi__ != nullptr) {
                            acc::copy(fft_coarse_.buffer().at<GPU>(), buf_rg_.at<GPU>(), fft_coarse_.local_size());
                        }
                    }
                    if (hphi__ != nullptr) {
                        /* multiply by effective potential */
                        scale_matrix_rows_gpu(fft_coarse_.local_size(), 1, fft_coarse_.buffer().at<GPU>(),
                                              veff_vec_[0].f_rg().at<GPU>());
                        /* phi(r) * Theta(r) * V(r) -> hphi(G) */
                        fft_coarse_.transform<-1>(hphi__->pw_coeffs(0).extra().at<CPU>(0, j));
                    }
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
                        buf_pw[igloc] = phi__.pw_coeffs(0).extra()(igloc, j) * gkvec_p_->gvec().gkvec_cart<index_domain_t::global>(ig)[x];
                    }
                    /* transform Cartesian component of wave-function gradient to real space */
                    fft_coarse_.transform<1>(&buf_pw[0]);
                    switch (fft_coarse_.pu()) {
                        case CPU: {
                            #pragma omp parallel for schedule(static)
                            for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
                                /* multiply be step function */
                                fft_coarse_.buffer(ir) *= theta_.f_rg(ir);
                            }
                            break;
                        }
                        case GPU: {
#if defined(__GPU)
                            /* multiply by step function */
                            scale_matrix_rows_gpu(fft_coarse_.local_size(), 1, fft_coarse_.buffer().at<GPU>(), theta_.f_rg().at<GPU>());
#endif
                            break;
                        }
                    }
                    /* transform back to PW domain */
                    fft_coarse_.transform<-1>(&buf_pw[0]);
                    #pragma omp parallel for schedule(static)
                    for (int igloc = 0; igloc < gkvec_p_->gvec_count_fft(); igloc++) {
                        int ig = gkvec_p_->idx_gvec(igloc);
                        hphi__->pw_coeffs(0).extra()(igloc, j) += 0.5 * buf_pw[igloc] * gkvec_p_->gvec().gkvec_cart<index_domain_t::global>(ig)[x];
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

        fft_coarse_.dismiss();

#ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            if (hphi__ != nullptr) {
                hphi__->pw_coeffs(0).copy_to_device(N__, n__);
            }
            if (ophi__ != nullptr) {
                ophi__->pw_coeffs(0).copy_to_device(N__, n__);
            }
        }
#endif
        //if (ctx_->control().print_checksum_) {
        //    auto cs1 = hphi__.checksum_pw(N__, n__, ctx_->processing_unit());
        //    auto cs2 = ophi__.checksum_pw(N__, n__, ctx_->processing_unit());
        //    if (phi__.comm().rank() == 0) {
        //        DUMP("checksum(hphi_pw): %18.10f %18.10f", cs1.real(), cs1.imag());
        //        DUMP("checksum(ophi_pw): %18.10f %18.10f", cs2.real(), cs2.imag());
        //    }
        //}
    }

    /// Apply magnetic field to the wave-functions.
    /** In case of collinear magnetism only Bz is applied to <tt>phi</tt> and stored in the first component of
     *  <tt>bphi</tt>. In case of non-collinear magnetims Bx-iBy is also applied and stored in the third
     *  component of <tt>bphi</tt>. The second component of <tt>bphi</tt> is used to store -Bz|phi>. */
    void apply_b(int                          N__,
                 int                          n__,
                 Wave_functions&              phi__,
                 std::vector<Wave_functions>& bphi__)
    {
        PROFILE("sirius::Local_operator::apply_b");

        if (!gkvec_p_) {
            TERMINATE("Local operator is not prepared");
        }

        fft_coarse_.prepare(*gkvec_p_);

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
            switch (fft_coarse_.pu()) {
                case CPU: {
                    /* phi(G) -> phi(r) */
                    fft_coarse_.transform<1>(phi__.pw_coeffs(0).extra().at<CPU>(0, j));
                    /* save phi(r) */
                    if (bphi__.size() == 3) {
                        fft_coarse_.output(buf_rg_.at<CPU>());
                    }
                    #pragma omp parallel for schedule(static)
                    for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
                        /* multiply by Bz */
                        fft_coarse_.buffer(ir) *= veff_vec_[1].f_rg(ir);
                    }
                    /* phi(r) * Bz(r) -> bphi[0](G) */
                    fft_coarse_.transform<-1>(bphi__[0].pw_coeffs(0).extra().at<CPU>(0, j));
                    /* non-collinear case */
                    if (bphi__.size() == 3) {
                        #pragma omp parallel for schedule(static)
                        for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
                            /* multiply by Bx-iBy */
                            fft_coarse_.buffer(ir) = buf_rg_[ir] * double_complex(veff_vec_[2].f_rg(ir), -veff_vec_[3].f_rg(ir));
                        }
                        /* phi(r) * (Bx(r)-iBy(r)) -> bphi[2](G) */
                        fft_coarse_.transform<-1>(bphi__[2].pw_coeffs(0).extra().at<CPU>(0, j));
                    }
                    break;
                }
                case GPU: {
#if defined(__GPU)
                    /* phi(G) -> phi(r) */
                    fft_coarse_.transform<1>(phi__.pw_coeffs(0).extra().at<CPU>(0, j));
                    /* multiply by Bz */
                    scale_matrix_rows_gpu(fft_coarse_.local_size(), 1, fft_coarse_.buffer().at<GPU>(), veff_vec_[1].f_rg().at<GPU>());
                    /* phi(r) * Bz(r) -> bphi[0](G) */
                    fft_coarse_.transform<-1>(bphi__[0].pw_coeffs(0).extra().at<CPU>(0, j));
#else
                    TERMINATE_NO_GPU
#endif
                    break;
                }
            }
        }

        for (int i : iv) {
            bphi__[i].pw_coeffs(0).remap_backward(n__, N__);
        }

        fft_coarse_.dismiss();
    }

    inline double v0(int ispn__) const
    {
        return v0_[ispn__];
    }
};

} // namespace sirius

#endif // __LOCAL_OPERATOR_H__
