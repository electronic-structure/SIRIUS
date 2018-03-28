// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

#include "periodic_function.h"
#include "potential.h"

#ifdef __GPU
extern "C" void mul_by_veff_gpu(int ispn__, int size__, double* const* veff__, double_complex* buf__);

extern "C" void add_pw_ekin_gpu(int num_gvec__,
                                double alpha__,
                                double const* pw_ekin__,
                                double_complex const* phi__,
                                double_complex const* vphi__,
                                double_complex* hphi__);
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
        Simulation_parameters const& param_;

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
        mdarray<double_complex, 1> vphi1_;

        /// Second temporary array to store [V*phi](G)
        mdarray<double_complex, 1> vphi2_;
        
        /// LAPW unit step function on a coarse FFT grid.
        Smooth_periodic_function<double> theta_;
       
        /// Temporary array to store psi_{up}(r).
        mdarray<double_complex, 1> buf_rg_;
        
        /// V(G=0) matrix elements.
        double v0_[2];

    public:
        /// Constructor.
        Local_operator(Simulation_parameters const& param__,
                       FFT3D&                       fft_coarse__,
                       Gvec_partition        const& gvec_coarse_p__)
            : param_(param__)
            , fft_coarse_(fft_coarse__)
            , gvec_coarse_p_(gvec_coarse_p__)

        {
            for (int j = 0; j < param__.num_mag_dims() + 1; j++) {
                veff_vec_[j] = Smooth_periodic_function<double>(fft_coarse__, gvec_coarse_p__);
                for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
                    veff_vec_[j].f_rg(ir) = 2.71828;
                }
            }
            if (param__.full_potential()) {
                theta_ = Smooth_periodic_function<double>(fft_coarse__, gvec_coarse_p__);
            }

            if (fft_coarse_.pu() == GPU) {
                for (int j = 0; j < param_.num_mag_dims() + 1; j++) {
                    veff_vec_[j].f_rg().allocate(memory_t::device);
                    veff_vec_[j].f_rg().copy<memory_t::host, memory_t::device>();
                }
                buf_rg_.allocate(memory_t::device);
            }
        }

        static int& num_applied()
        {
            static int num_applied_{0};
            return num_applied_;
        }

        ///// This constructor is used internally in the debug and performance tests only.
        //Local_operator(Simulation_parameters const& param__,
        //               FFT3D&                       fft_coarse__,
        //               Gvec_partition const&        gvecp__)
        //    : param_(&param__)
        //    , fft_coarse_(fft_coarse__)
        //{
        //    veff_vec_ = mdarray<double, 2>(fft_coarse_.local_size(), 1, memory_t::host, "Local_operator::veff_vec_");
        //    for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
        //        veff_vec_(ir, 0) = 2.71828;
        //    }

        //    int ngv_fft = gvecp__.gvec_count_fft();
        //    
        //    pw_ekin_ = mdarray<double, 1>(ngv_fft, memory_t::host, "Local_operator::pw_ekin");
        //    pw_ekin_.zero();

        //    vphi1_ = mdarray<double_complex, 1>(ngv_fft, memory_t::host, "Local_operator::vphi1");
        //    vphi2_ = mdarray<double_complex, 1>(ngv_fft, memory_t::host, "Local_operator::vphi2");

        //    if (fft_coarse_.pu() == GPU) {
        //        veff_vec_.allocate(memory_t::device);
        //        veff_vec_.copy<memory_t::host, memory_t::device>();
        //        pw_ekin_.allocate(memory_t::device);
        //        pw_ekin_.copy<memory_t::host, memory_t::device>();
        //        vphi1_.allocate(memory_t::device);
        //        vphi2_.allocate(memory_t::device);
        //    }
        //}
        
        /// Map effective potential and magnetic field to a coarse FFT mesh in case of PP-PW.
        /** \param [in] potential      \f$ V_{eff}({\bf r}) \f$ and \f$ {\bf B}_{eff}({\bf r}) \f$ on the fine grid FFT grid.
         *
         *  This function should be called prior to the band diagonalziation. In case of GPU execution all
         *  effective fields on the coarse grid will be copied to the device and will remain there until the
         *  dismiss() method is called after band diagonalization.
         */
        inline void prepare(Potential& potential__)
        {
            PROFILE("sirius::Local_operator::prepare");

            /* group effective fields into single vector */
            std::vector<Periodic_function<double>*> veff_vec(param_.num_mag_dims() + 1);
            veff_vec[0] = potential__.effective_potential();
            for (int j = 0; j < param_.num_mag_dims(); j++) {
                veff_vec[1 + j] = potential__.effective_magnetic_field(j);
            }
            
            if (!buf_rg_.size() && param_.num_mag_dims() == 3) {
                buf_rg_ = mdarray<double_complex, 1>(fft_coarse_.local_size(), memory_t::host, "Local_operator::buf_rg_");
            }

            fft_coarse_.prepare(gvec_coarse_p_);
            for (int j = 0; j < param_.num_mag_dims() + 1; j++) {
                /* loop over local set of coarse G-vectors */
                #pragma omp parallel for schedule(static)
                for (int igloc = 0; igloc < gvec_coarse_p_.gvec().count(); igloc++) {
                    /* map from fine to coarse set of G-vectors */
                    veff_vec_[j].f_pw_local(igloc) = veff_vec[j]->f_pw_local(veff_vec[j]->gvec().gvec_base_mapping(igloc));
                }
                /* transform to real space */
                veff_vec_[j].fft_transform(1);
            }
            fft_coarse_.dismiss();

            if (param_.num_mag_dims()) {
                for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
                    double v0 = veff_vec_[0].f_rg(ir);
                    double v1 = veff_vec_[1].f_rg(ir);
                    veff_vec_[0].f_rg(ir) = v0 + v1; // v + Bz
                    veff_vec_[1].f_rg(ir) = v0 - v1; // v - Bz
                }
            }
            
            if (param_.num_mag_dims() == 0) {
                v0_[0] = veff_vec[0]->f_0().real();
            } else {
                v0_[0] = veff_vec[0]->f_0().real() + veff_vec[1]->f_0().real();
                v0_[1] = veff_vec[0]->f_0().real() - veff_vec[1]->f_0().real();
            }
            
            /* copy veff to device */
            if (fft_coarse_.pu() == GPU) {
                for (int j = 0; j < param_.num_mag_dims() + 1; j++) {
                    veff_vec_[j].f_rg().allocate(memory_t::device);
                    veff_vec_[j].f_rg().copy<memory_t::host, memory_t::device>();
                }
                if (param_.num_mag_dims() == 3) {
                    buf_rg_.allocate(memory_t::device);
                }
            }

            //if (param_.control().print_checksum_) {
            //    auto cs = veff_vec_.checksum();
            //    fft_coarse_.comm().allreduce(&cs, 1);
            //    if (gvec_coarse_p_.gvec().comm().rank() == 0) {
            //        print_checksum("Local_operator::prepare::veff_vec", cs);
            //    }
            //}
        }

        /// Map effective potential and magnetic field to a coarse FFT mesh in case of FP-LAPW.
        /** \param [in] potential      \f$ V_{eff}({\bf r}) \f$ and \f$ {\bf B}_{eff}({\bf r}) \f$ on the fine grid FFT grid.
         *  \param [in] step_function  Unit step function of the LAPW method.
         */
        inline void prepare(Potential&           potential__,
                            Step_function const& step_function__)
        {
            PROFILE("sirius::Local_operator::prepare");

            /* group effective fields into single vector */
            std::vector<Periodic_function<double>*> veff_vec(param_.num_mag_dims() + 1);
            veff_vec[0] = potential__.effective_potential();
            for (int j = 0; j < param_.num_mag_dims(); j++) {
                veff_vec[1 + j] = potential__.effective_magnetic_field(j);
            }

            if (!buf_rg_.size()) {
                buf_rg_ = mdarray<double_complex, 1>(fft_coarse_.local_size(), memory_t::host, "Local_operator::buf_rg_");
            }

            auto& fft_dense    = potential__.effective_potential()->fft();
            auto& gvec_dense_p = potential__.effective_potential()->gvec_partition();

            fft_coarse_.prepare(gvec_coarse_p_);

            Smooth_periodic_function<double> ftmp(fft_dense, gvec_dense_p);
            for (int j = 0; j < param_.num_mag_dims() + 1; j++) {
                for (int ir = 0; ir < fft_dense.local_size(); ir++) {
                    ftmp.f_rg(ir) = veff_vec[j]->f_rg(ir) * step_function__.theta_r(ir);
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
                theta_.f_pw_local(igloc) = step_function__.theta_pw(gvec_dense_p.gvec().gvec_base_mapping(igloc) +
                                                                    gvec_dense_p.gvec().offset());
            }
            theta_.fft_transform(1);
            /* release FFT driver */ 
            fft_coarse_.dismiss();

            if (fft_coarse_.pu() == GPU) {
                for (int j = 0; j < param_.num_mag_dims() + 1; j++) {
                    veff_vec_[j].f_rg().allocate(memory_t::device);
                    veff_vec_[j].f_rg().copy<memory_t::host, memory_t::device>();
                }
                theta_.f_rg().allocate(memory_t::device);
                theta_.f_rg().copy<memory_t::host, memory_t::device>();
                buf_rg_.allocate(memory_t::device);
            }

            //if (param_.control().print_checksum_) {
            //    double cs[] = {veff_vec_.checksum(), theta_.checksum()};
            //    fft_coarse_.comm().allreduce(&cs[0], 2);
            //    if (mpi_comm_world().rank() == 0) {
            //        print_checksum("veff_vec", cs[0]);
            //        print_checksum("theta", cs[1]);
            //    }
            //}
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
                auto gv = gkvec_p__.gvec().gkvec_cart(ig);
                pw_ekin_[ig_loc] = 0.5 * dot(gv, gv);
            }

            if (static_cast<int>(vphi1_.size()) < ngv_fft) {
                vphi1_ = mdarray<double_complex, 1>(ngv_fft, memory_t::host, "Local_operator::vphi1");
            }
            if (gkvec_p__.gvec().reduced() && static_cast<int>(vphi2_.size()) < ngv_fft) {
                vphi2_ = mdarray<double_complex, 1>(ngv_fft, memory_t::host, "Local_operator::vphi2");
            }

            if (fft_coarse_.pu() == GPU) {
                pw_ekin_.allocate(memory_t::device);
                pw_ekin_.copy<memory_t::host, memory_t::device>();
                vphi1_.allocate(memory_t::device);
                if (gkvec_p__.gvec().reduced()) {
                    vphi2_.allocate(memory_t::device);
                }
            }
        }
        
        /// Cleanup the local operator.
        inline void dismiss()
        {
            if (fft_coarse_.pu() == GPU) { 
                for (int j = 0; j < param_.num_mag_dims() + 1; j++) {
                    veff_vec_[j].f_rg().deallocate(memory_t::device);
                }
                pw_ekin_.deallocate(memory_t::device);
                vphi1_.deallocate(memory_t::device);
                vphi2_.deallocate(memory_t::device);
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
         */
        void apply_h(int ispn__, Wave_functions& phi__, Wave_functions& hphi__, int idx0__, int n__)
        {
            PROFILE("sirius::Local_operator::apply_h");

            if (!gkvec_p_) {
                TERMINATE("Local operator is not prepared");
            }

            num_applied() += n__;

            /* remap wave-functions */
            for (int ispn = 0; ispn < phi__.num_sc(); ispn++) {

                phi__.pw_coeffs(ispn).remap_forward(fft_coarse_.pu(), n__, idx0__);

                hphi__.pw_coeffs(ispn).set_num_extra(CPU, n__, idx0__);
                hphi__.pw_coeffs(ispn).extra().zero<memory_t::host | memory_t::device>();
            }
            
            #ifdef __GPU
            mdarray<double*, 1> vptr(4, memory_t::host | memory_t::device);
            vptr.zero();
            if (fft_coarse_.pu() == GPU) {
                for (int j = 0; j < param_.num_mag_dims() + 1; j++) {
                    vptr[j] = veff_vec_[j].f_rg().at<GPU>();
                }
                vptr.copy<memory_t::host, memory_t::device>();
            }
            #endif

            /* transform one or two wave-functions to real space; the result of
             * transformation is stored in the FFT buffer */
            auto phi_to_r = [&](int i, int ispn, bool gamma = false)
            {
                switch (fft_coarse_.pu()) {
                    case CPU: {
                        if (gamma) {
                            fft_coarse_.transform<1, CPU>(phi__.pw_coeffs(ispn).extra().at<CPU>(0, 2 * i),
                                                          phi__.pw_coeffs(ispn).extra().at<CPU>(0, 2 * i + 1));

                        } else {
                            fft_coarse_.transform<1, CPU>(phi__.pw_coeffs(ispn).extra().at<CPU>(0, i));
                        }
                        break;
                    }
                    case GPU: {
                        if (gamma) { /* warning: GPU pointer works only in case of serial FFT */
                            fft_coarse_.transform<1, GPU>(phi__.pw_coeffs(ispn).extra().at<GPU>(0, 2 * i),
                                                          phi__.pw_coeffs(ispn).extra().at<GPU>(0, 2 * i + 1));
                        } else {
                            fft_coarse_.transform<1, GPU>(phi__.pw_coeffs(ispn).extra().at<GPU>(0, i));
                        }
                        break;
                    }
                }
            };

            auto mul_by_veff = [&](mdarray<double_complex, 1>& buf, int ispn_block)
            {
                /* multiply by effective potential */
                switch (fft_coarse_.pu()) {
                    case CPU: {
                        if (ispn_block < 2) {
                            #pragma omp parallel for schedule(static)
                            for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
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
                    case GPU: {
                        #ifdef __GPU
                        mul_by_veff_gpu(ispn_block, fft_coarse_.local_size(), vptr.at<GPU>(), buf.at<GPU>());
                        #endif
                        break;
                    }
                }
            };

            /* transform one or two functions to PW domain */
            auto vphi_to_G = [&](bool gamma = false)
            {
                switch (fft_coarse_.pu()) {
                    case CPU: {
                        if (gamma) {
                            fft_coarse_.transform<-1, CPU>(vphi1_.at<CPU>(), vphi2_.at<CPU>());
                        } else {
                            fft_coarse_.transform<-1, CPU>(vphi1_.at<CPU>());
                        }
                        break;
                    }
                    case GPU: {
                        if (gamma) {
                            fft_coarse_.transform<-1, GPU>(vphi1_.at<GPU>(), vphi2_.at<GPU>());
                        } else {
                            fft_coarse_.transform<-1, GPU>(vphi1_.at<GPU>());
                        }
                        break;
                    }
                }
            };
            
            /* store the resulting hphi
               spin block (ispn_block) is used as a bit mask: 
                - first bit: spin component which is updated
                - second bit: add or not kinetic energy term */
            auto add_to_hphi = [&](int i, int ispn_block, bool gamma = false)
            {
                int ispn = ispn_block & 1;
                int ekin = (ispn_block & 2) ? 0 : 1;

                if (!phi__.pw_coeffs(ispn).is_remapped() && fft_coarse_.pu() == GPU) {
                    #ifdef __GPU
                    double alpha = static_cast<double>(ekin);
                    if (gamma) {
                        add_pw_ekin_gpu(gkvec_p_->gvec_count_fft(),
                                        alpha,
                                        pw_ekin_.at<GPU>(),
                                        phi__.pw_coeffs(ispn).extra().at<GPU>(0, 2 * i),
                                        vphi1_.at<GPU>(),
                                        hphi__.pw_coeffs(ispn).extra().at<GPU>(0, 2 * i));
                        add_pw_ekin_gpu(gkvec_p_->gvec_count_fft(),
                                        alpha,
                                        pw_ekin_.at<GPU>(),
                                        phi__.pw_coeffs(ispn).extra().at<GPU>(0, 2 * i + 1),
                                        vphi2_.at<GPU>(),
                                        hphi__.pw_coeffs(ispn).extra().at<GPU>(0, 2 * i + 1));
                    } else {
                        add_pw_ekin_gpu(gkvec_p_->gvec_count_fft(),
                                        alpha,
                                        pw_ekin_.at<GPU>(),
                                        phi__.pw_coeffs(ispn).extra().at<GPU>(0, i),
                                        vphi1_.at<GPU>(),
                                        hphi__.pw_coeffs(ispn).extra().at<GPU>(0, i));
                    }
                    #else
                    TERMINATE_NO_GPU
                    #endif
                    return;
                }
                /* data was remapped and hphi extra storage is allocated only on CPU */
                if (phi__.pw_coeffs(ispn).is_remapped() && fft_coarse_.pu() == GPU) {
                    if (gamma) {
                        vphi2_.copy<memory_t::device, memory_t::host>();
                    }
                    vphi1_.copy<memory_t::device, memory_t::host>();
                }
                /* CPU case */
                if (gamma) { /* update two wave functions */
                    if (ekin) {
                        #pragma omp parallel for schedule(static)
                        for (int ig = 0; ig < gkvec_p_->gvec_count_fft(); ig++) {
                            hphi__.pw_coeffs(ispn).extra()(ig, 2 * i)     += (phi__.pw_coeffs(ispn).extra()(ig, 2 * i)     * pw_ekin_[ig] + vphi1_[ig]);
                            hphi__.pw_coeffs(ispn).extra()(ig, 2 * i + 1) += (phi__.pw_coeffs(ispn).extra()(ig, 2 * i + 1) * pw_ekin_[ig] + vphi2_[ig]);
                        }
                    } else {
                        #pragma omp parallel for schedule(static)
                        for (int ig = 0; ig < gkvec_p_->gvec_count_fft(); ig++) {
                            hphi__.pw_coeffs(ispn).extra()(ig, 2 * i)     += vphi1_[ig];
                            hphi__.pw_coeffs(ispn).extra()(ig, 2 * i + 1) += vphi2_[ig];
                        }
                    }
                } else { /* update single wave function */
                    if (ekin) {
                        #pragma omp parallel for schedule(static)
                        for (int ig = 0; ig < gkvec_p_->gvec_count_fft(); ig++) {
                            hphi__.pw_coeffs(ispn).extra()(ig, i) += (phi__.pw_coeffs(ispn).extra()(ig, i) * pw_ekin_[ig] + vphi1_[ig]);
                        }
                    } else {
                        #pragma omp parallel for schedule(static)
                        for (int ig = 0; ig < gkvec_p_->gvec_count_fft(); ig++) {
                            hphi__.pw_coeffs(ispn).extra()(ig, i) += vphi1_[ig];
                        }
                    }
                }
            };
            /* local number of wave-functions in extra-storage distribution */
            int num_wf_loc = phi__.pw_coeffs(0).spl_num_col().local_size();

            int first{0};
            /* If G-vectors are reduced, wave-functions are real and we can transform two of them at once.
             * Non-collinear case is not treated here because nc wave-functions are complex and G+k vectors 
             * can't be reduced */
            if (gkvec_p_->gvec().reduced()) {
                int npairs = num_wf_loc / 2;
                /* Gamma-point case can only be non-magnetic or spin-collinear */
                for (int i = 0; i < npairs; i++) {
                    /* phi(G) -> phi(r) */
                    phi_to_r(i, ispn__, true);
                    /* multiply by effective potential */
                    mul_by_veff(fft_coarse_.buffer(), ispn__);
                    /* V(r)phi(r) -> [V*phi](G) */
                    vphi_to_G(true);
                    /* add kinetic energy */
                    add_to_hphi(i, ispn__, true);
                }
                /* check if we have to do last wave-function which had no pair */
                first = num_wf_loc - num_wf_loc % 2;
            }
            
            /* if we don't have G-vector reductions, first = 0 and we start a normal loop */
            for (int i = first; i < num_wf_loc; i++) {
                
                /* non-collinear case */
                /* 2x2 Hamiltonian in applied to spinor wave-functions
                 * .--------.--------.   .-----.   .------.
                 * |        |        |   |     |   |      |
                 * | H_{uu} | H_{ud} |   |phi_u|   |hphi_u|
                 * |        |        |   |     |   |      |
                 * .--------.--------. x .-----. = .------.
                 * |        |        |   |     |   |      |
                 * | H_{du} | H_{dd} |   |phi_d|   |hphi_d|
                 * |        |        |   |     |   |      |
                 * .--------.--------.   .-----.   .------.
                 *
                 * hphi_u = H_{uu} phi_u + H_{ud} phi_d
                 * hphi_d = H_{du} phi_u + H_{dd} phi_d
                 *
                 * The following indexing scheme will be used for spin-blocks
                 * .---.---.
                 * | 0 | 2 |
                 * .---.---.
                 * | 3 | 1 |
                 * .---.---.
                 */        
                if (ispn__ == 2) {
                    /* phi_u(G) -> phi_u(r) */
                    phi_to_r(i, 0);
                    /* save phi_u(r) */
                    switch (fft_coarse_.pu()) {
                        case CPU: {
                            fft_coarse_.output(buf_rg_.at<CPU>());
                            break;
                        }
                        case GPU: {
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
                    add_to_hphi(i, 0);
                    /* multiply phi_{u} by V_{du} */
                    mul_by_veff(buf_rg_, 3);
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
                    /* V_{du}(r)phi_{u}(r) -> [V*phi]_{d}(G) */
                    vphi_to_G();
                    /* add to hphi_{d} */
                    add_to_hphi(i, 3);

                    /* for the second spin */

                    /* phi_d(G) -> phi_d(r) */
                    phi_to_r(i, 1);
                    /* save phi_d(r) */
                    switch (fft_coarse_.pu()) {
                        case CPU: {
                            fft_coarse_.output(buf_rg_.at<CPU>());
                            break;
                        }
                        case GPU: {
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
                    add_to_hphi(i, 1);
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
                    add_to_hphi(i, 2);

                } else { /* spin-collinear case */
                    /* phi(G) -> phi(r) */
                    phi_to_r(i, ispn__);
                    /* multiply by effective potential */
                    mul_by_veff(fft_coarse_.buffer(), ispn__);
                    /* V(r)phi(r) -> [V*phi](G) */
                    vphi_to_G();
                    /* add kinetic energy */
                    add_to_hphi(i, ispn__);
                }
            }

            for (int ispn = 0; ispn < hphi__.num_sc(); ispn++) {
                hphi__.pw_coeffs(ispn).remap_backward(param_.processing_unit(), n__, idx0__);
            }
        }

        void apply_h_o(int N__,
                       int n__,
                       Wave_functions& phi__,
                       Wave_functions& hphi__,
                       Wave_functions& ophi__)
        {
            PROFILE("sirius::Local_operator::apply_h_o");

            if (!gkvec_p_) {
                TERMINATE("Local operator is not prepared");
            }

            fft_coarse_.prepare(*gkvec_p_);

            mdarray<double_complex, 1> buf_pw(gkvec_p_->gvec_count_fft());

#ifdef __GPU
            if (fft_coarse_.pu() == GPU) {
                phi__.pw_coeffs(0).copy_to_host(N__, n__);
            }
#endif

            //if (param_->control().print_checksum_) {
            //    auto cs = phi__.checksum_pw(N__, n__, param_->processing_unit());
            //    if (phi__.comm().rank() == 0) {
            //        DUMP("checksum(phi_pw): %18.10f %18.10f", cs.real(), cs.imag());
            //    }
            //}

             phi__.pw_coeffs(0).remap_forward(param_.processing_unit(), n__, N__);
            hphi__.pw_coeffs(0).set_num_extra(CPU, n__, N__);
            ophi__.pw_coeffs(0).set_num_extra(CPU, n__, N__);

            for (int j = 0; j < phi__.pw_coeffs(0).spl_num_col().local_size(); j++) {
                if (fft_coarse_.pu() == CPU) {
                    /* phi(G) -> phi(r) */
                    fft_coarse_.transform<1>(phi__.pw_coeffs(0).extra().at<CPU>(0, j));
                    #pragma omp parallel for schedule(static)
                    for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
                        /* save phi(r) */
                        buf_rg_[ir] = fft_coarse_.buffer(ir);
                        /* multiply by step function */
                        fft_coarse_.buffer(ir) *= theta_.f_rg(ir);
                    }
                    /* phi(r) * Theta(r) -> ophi(G) */
                    fft_coarse_.transform<-1>(ophi__.pw_coeffs(0).extra().at<CPU>(0, j));
                    #pragma omp parallel for schedule(static)
                    for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
                        /* multiply be effective potential, which itself was multiplied by the step function in constructor */
                        fft_coarse_.buffer(ir) = buf_rg_[ir] * veff_vec_[0].f_rg(ir);
                    }
                    /* phi(r) * Theta(r) * V(r) -> ophi(G) */
                    fft_coarse_.transform<-1>(hphi__.pw_coeffs(0).extra().at<CPU>(0, j));
                }
                #ifdef __GPU
                if (fft_coarse_.pu() == GPU) {
                    /* phi(G) -> phi(r) */
                    fft_coarse_.transform<1>(phi__.pw_coeffs(0).extra().at<CPU>(0, j));
                    /* save phi(r) */
                    acc::copy(buf_rg_.at<GPU>(), fft_coarse_.buffer().at<GPU>(), fft_coarse_.local_size());
                    /* multiply by step function */
                    scale_matrix_rows_gpu(fft_coarse_.local_size(), 1, fft_coarse_.buffer().at<GPU>(), theta_.f_rg().at<GPU>());
                    /* phi(r) * Theta(r) -> ophi(G) */
                    fft_coarse_.transform<-1>(ophi__.pw_coeffs(0).extra().at<CPU>(0, j));
                    /* multiply by effective potential */
                    scale_matrix_rows_gpu(fft_coarse_.local_size(), 1, buf_rg_.at<GPU>(), veff_vec_[0].f_rg().at<GPU>());
                    /* copy phi(r) * Theta(r) * V(r) to GPU buffer */
                    acc::copy(fft_coarse_.buffer().at<GPU>(), buf_rg_.at<GPU>(), fft_coarse_.local_size());
                    /* phi(r) * Theta(r) * V(r) -> ophi(G) */
                    fft_coarse_.transform<-1>(hphi__.pw_coeffs(0).extra().at<CPU>(0, j));
                }
                #endif

                /* add kinetic energy */
                for (int x: {0, 1, 2}) {
                    for (int igloc = 0; igloc < gkvec_p_->gvec_count_fft(); igloc++) {
                        /* global index of G-vector */
                        int ig = gkvec_p_->idx_gvec(igloc);
                        /* \hat P phi = phi(G+k) * (G+k), \hat P is momentum operator */ 
                        buf_pw[igloc] = phi__.pw_coeffs(0).extra()(igloc, j) * gkvec_p_->gvec().gkvec_cart(ig)[x];
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
                            #ifdef __GPU
                            /* multiply by step function */
                            scale_matrix_rows_gpu(fft_coarse_.local_size(), 1, fft_coarse_.buffer().at<GPU>(), theta_.f_rg().at<GPU>());
                            #endif
                            break;
                        }
                    }
                    /* transform back to PW domain */
                    fft_coarse_.transform<-1>(&buf_pw[0]);
                    for (int igloc = 0; igloc < gkvec_p_->gvec_count_fft(); igloc++) {
                        int ig = gkvec_p_->idx_gvec(igloc);
                        hphi__.pw_coeffs(0).extra()(igloc, j) += 0.5 * buf_pw[igloc] * gkvec_p_->gvec().gkvec_cart(ig)[x];
                    }
                }
            }

            hphi__.pw_coeffs(0).remap_backward(param_.processing_unit(), n__, N__);
            ophi__.pw_coeffs(0).remap_backward(param_.processing_unit(), n__, N__);

            fft_coarse_.dismiss();

#ifdef __GPU
            if (fft_coarse_.pu() == GPU) {
                hphi__.pw_coeffs(0).copy_to_device(N__, n__);
                ophi__.pw_coeffs(0).copy_to_device(N__, n__);
            }
#endif
            //if (param_->control().print_checksum_) {
            //    auto cs1 = hphi__.checksum_pw(N__, n__, param_->processing_unit());
            //    auto cs2 = ophi__.checksum_pw(N__, n__, param_->processing_unit());
            //    if (phi__.comm().rank() == 0) {
            //        DUMP("checksum(hphi_pw): %18.10f %18.10f", cs1.real(), cs1.imag());
            //        DUMP("checksum(ophi_pw): %18.10f %18.10f", cs2.real(), cs2.imag());
            //    }
            //}
        }

        void apply_o(int N__,
                     int n__,
                     Wave_functions& phi__,
                     Wave_functions& ophi__) const
        {
            PROFILE("sirius::Local_operator::apply_o");

            if (!gkvec_p_) {
                TERMINATE("Local operator is not prepared");
            }

            fft_coarse_.prepare(*gkvec_p_);

#ifdef __GPU
            if (fft_coarse_.pu() == GPU) {
                phi__.pw_coeffs(0).copy_to_host(N__, n__);
            }
#endif
            
            if (param_.control().print_checksum_) {
                auto cs = phi__.checksum_pw(param_.processing_unit(), 0, 0, N__+ n__);
                if (phi__.comm().rank() == 0) {
                    print_checksum("phi_[0, N + n)", cs);
                }
                if (N__ != 0) {
                    auto cs1 = ophi__.checksum_pw(param_.processing_unit(), 0, 0, N__);
                    if (phi__.comm().rank() == 0) {
                        print_checksum("ophi_[0, N)", cs1);
                    }
                }
            }

             phi__.pw_coeffs(0).remap_forward(param_.processing_unit(), n__, N__);
            ophi__.pw_coeffs(0).set_num_extra(CPU, n__, N__);

            for (int j = 0; j < phi__.pw_coeffs(0).spl_num_col().local_size(); j++) {
                if (fft_coarse_.pu() == CPU) {
                    /* phi(G) -> phi(r) */
                    fft_coarse_.transform<1>(phi__.pw_coeffs(0).extra().at<CPU>(0, j));
                    #pragma omp parallel for schedule(static)
                    for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
                        /* multiply by step function */
                        fft_coarse_.buffer(ir) *= theta_.f_rg(ir);
                    }
                    /* phi(r) * Theta(r) -> ophi(G) */
                    fft_coarse_.transform<-1>(ophi__.pw_coeffs(0).extra().at<CPU>(0, j));
                } else {
#ifdef __GPU
                    /* phi(G) -> phi(r) */
                    fft_coarse_.transform<1>(phi__.pw_coeffs(0).extra().at<CPU>(0, j));
                    /* multiply by step function */
                    scale_matrix_rows_gpu(fft_coarse_.local_size(), 1, fft_coarse_.buffer().at<GPU>(), theta_.f_rg().at<GPU>());
                    /* phi(r) * Theta(r) -> ophi(G) */
                    fft_coarse_.transform<-1>(ophi__.pw_coeffs(0).extra().at<CPU>(0, j));
#else
                    TERMINATE_NO_GPU
#endif
                }
            }

            ophi__.pw_coeffs(0).remap_backward(param_.processing_unit(), n__, N__);

            fft_coarse_.dismiss();

#ifdef __GPU
            if (fft_coarse_.pu() == GPU) {
                ophi__.pw_coeffs(0).copy_to_device(N__, n__);
            }
#endif
            
            if (param_.control().print_checksum_) {
                auto cs = ophi__.checksum_pw(param_.processing_unit(), 0, 0, N__ + n__);
                if (phi__.comm().rank() == 0) {
                    print_checksum("ophi_istl_[0, N + n)", cs);
                }
            }
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

            phi__.pw_coeffs(0).remap_forward(param_.processing_unit(), n__, N__);
            for (int i: iv) {
                bphi__[i].pw_coeffs(0).set_num_extra(CPU, n__, N__);
            }

            for (int j = 0; j < phi__.pw_coeffs(0).spl_num_col().local_size(); j++) {
                if (fft_coarse_.pu() == CPU) {
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
                } else {
                    #ifdef __GPU
                    /* phi(G) -> phi(r) */
                    fft_coarse_.transform<1>(phi__.pw_coeffs(0).extra().at<CPU>(0, j));
                    /* multiply by Bz */
                    scale_matrix_rows_gpu(fft_coarse_.local_size(), 1, fft_coarse_.buffer().at<GPU>(), veff_vec_[1].f_rg().at<GPU>());
                    /* phi(r) * Bz(r) -> bphi[0](G) */
                    fft_coarse_.transform<-1>(bphi__[0].pw_coeffs(0).extra().at<CPU>(0, j));
                    #else
                    TERMINATE_NO_GPU
                    #endif
                }
            }

            for (int i: iv) {
                bphi__[i].pw_coeffs(0).remap_backward(param_.processing_unit(), n__, N__);
            }

            fft_coarse_.dismiss();
        }

        inline double v0(int ispn__) const
        {
            return v0_[ispn__];
        }
};

} // namespace

#endif // __LOCAL_OPERATOR_H__
