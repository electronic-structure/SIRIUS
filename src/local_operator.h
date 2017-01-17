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

/** \file local_operator.h
 *   
 *  \brief Contains declaration and implementation of sirius::Local_operator class.
 */

#ifndef __LOCAL_OPERATOR_H__
#define __LOCAL_OPERATOR_H__

#include "periodic_function.h"

#ifdef __GPU
extern "C" void add_pw_ekin_gpu(int num_gvec__,
                                double const* pw_ekin__,
                                cuDoubleComplex const* vphi__,
                                cuDoubleComplex* hphi__);
#endif

namespace sirius {

/// Representation of the local operator.
/** The following functionality is implementated:
 *    - application of the local part of Hamiltonian (kinetic + potential) to the wave-fucntions in the PP-PW case
 *    - application of the interstitial part of H and O in the case of FP-LAPW
 *    - remapping of potential and unit-step functions from fine to coarse mesh of G-vectors
 */
class Local_operator
{
    private:
        /// Coarse-grid FFT driver for this operator
        FFT3D& fft_coarse_;
        
        /// Kinetic energy of G+k plane-waves.
        mdarray<double, 1> pw_ekin_;
        
        /// Effective potential components.
        mdarray<double, 2> veff_vec_;

        mdarray<double_complex, 1> vphi1_;

        mdarray<double_complex, 1> vphi2_;

        mdarray<double, 1> theta_;
        
        mdarray<double_complex, 1> buf_rg_;
        
        /// V(G=0) matrix elements.
        double v0_[2];

    public:
        /// Constructor.
        Local_operator(FFT3D& fft_coarse__)
            : fft_coarse_(fft_coarse__)
        {
        }
        
        /// Map effective potential and magnetic field to a coarse FFT mesh in case of PP-PW.
        /** \param [in] gvec_coarse              G-vectors of the coarse FFT grid.
         *  \param [in] num_mag_dims             Number of magnetic dimensions.
         *  \param [in] effective_potential      \f$ V_{eff}({\bf r}) \f$ on the fine grid FFT grid.
         *  \param [in] effective_magnetic_field \f$ {\bf B}_{eff}({\bf r}) \f$ on the fine FFT grid.
         *
         *  This function should be called prior to the band diagonalziation. In case of GPU execution all
         *  effective fields on the coarse grid will be copied to the device and will remain there until the
         *  dismiss() method is called after band diagonalization.
         */
        inline void prepare(Gvec const& gvec_coarse__,
                            int num_mag_dims__,
                            Periodic_function<double>* effective_potential__,
                            Periodic_function<double>* effective_magnetic_field__[3])
        {
            PROFILE("sirius::Local_operator::prepare");

            /* group effective fields into single vector */
            std::vector<Periodic_function<double>*> veff_vec(num_mag_dims__ + 1);
            veff_vec[0] = effective_potential__;
            for (int j = 0; j < num_mag_dims__; j++) {
                veff_vec[1 + j] = effective_magnetic_field__[j];
            }
            
            /* allocate only once */
            if (!veff_vec_.size()) {
                veff_vec_ = mdarray<double, 2>(fft_coarse_.local_size(), num_mag_dims__ + 1, memory_t::host, "Local_operator::veff_vec_");
            }
        
            /* low-frequency part of PW coefficients */
            std::vector<double_complex> v_pw_coarse(gvec_coarse__.partition().gvec_count_fft());
            /* prepare FFT for transformation */
            fft_coarse_.prepare(gvec_coarse__.partition());
            /* map components of effective potential to a corase grid */
            for (int j = 0; j < num_mag_dims__ + 1; j++) {
                /* loop over low-frequency G-vectors */
                for (int ig = 0; ig < gvec_coarse__.partition().gvec_count_fft(); ig++) {
                    /* G-vector in fractional coordinates */
                    auto G = gvec_coarse__.gvec(ig + gvec_coarse__.partition().gvec_offset_fft());
                    v_pw_coarse[ig] = veff_vec[j]->f_pw(G);
                }
                /* transform to real space */
                fft_coarse_.transform<1>(gvec_coarse__.partition(), &v_pw_coarse[0]);
                /* save V(r) */
                fft_coarse_.output(&veff_vec_(0, j));
            }
            fft_coarse_.dismiss();

            if (num_mag_dims__) {
                for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
                    double v0 = veff_vec_(ir, 0);
                    double v1 = veff_vec_(ir, 1);
                    veff_vec_(ir, 0) = v0 + v1; // v + Bz
                    veff_vec_(ir, 1) = v0 - v1; // v - Bz
                }
            }

            if (num_mag_dims__ == 0) {
                v0_[0] = veff_vec[0]->f_pw(0).real();
            } else {
                v0_[0] = veff_vec[0]->f_pw(0).real() + veff_vec[1]->f_pw(0).real();
                v0_[1] = veff_vec[0]->f_pw(0).real() - veff_vec[1]->f_pw(0).real();
            }
            
            /* copy veff to device */
            #ifdef __GPU
            if (fft_coarse_.hybrid()) {
                veff_vec_.allocate(memory_t::device);
                veff_vec_.copy_to_device();
            }
            #endif
        }

        /// Map effective potential and magnetic field to a coarse FFT mesh in case of FP-LAPW.
        /** \param [in] gvec_coarse              G-vectors of the coarse FFT grid.
         *  \param [in] num_mag_dims             Number of magnetic dimensions.
         *  \param [in] effective_potential      \f$ V_{eff}({\bf r}) \f$ on the fine grid FFT grid.
         *  \param [in] effective_magnetic_field \f$ {\bf B}_{eff}({\bf r}) \f$ on the fine FFT grid.
         *  \param [in] step_function            Unit step function of the LAPW method.
         */
        inline void prepare(Gvec const& gvec_coarse__,
                            int num_mag_dims__,
                            Periodic_function<double>* effective_potential__,
                            Periodic_function<double>* effective_magnetic_field__[3],
                            Step_function const& step_function__)
        {
            PROFILE("sirius::Local_operator::prepare");

            /* group effective fields into single vector */
            std::vector<Periodic_function<double>*> veff_vec(num_mag_dims__ + 1);
            veff_vec[0] = effective_potential__;
            for (int j = 0; j < num_mag_dims__; j++) {
                veff_vec[1 + j] = effective_magnetic_field__[j];
            }

            /* allocate only once */
            if (!veff_vec_.size()) {
                veff_vec_ = mdarray<double, 2>(fft_coarse_.local_size(), num_mag_dims__ + 1, memory_t::host, "Local_operator::veff_vec_");
            }
            
            if (!theta_.size()) {
                theta_ = mdarray<double, 1>(fft_coarse_.local_size(), memory_t::host, "Local_operator::theta_");
            }

            auto& fft_dense = effective_potential__->fft();
            auto& gvec_dense = effective_potential__->gvec();

            mdarray<double_complex, 1> v_pw_fine(gvec_dense.num_gvec());
            /* low-frequency part of PW coefficients */
            std::vector<double_complex> v_pw_coarse(gvec_coarse__.partition().gvec_count_fft());
            /* prepare coarse-grained FFT for transformation */
            fft_coarse_.prepare(gvec_coarse__.partition());
            /* map components of effective potential to a corase grid */
            for (int j = 0; j < num_mag_dims__ + 1; j++) {
                for (int ir = 0; ir < fft_dense.local_size(); ir++) {
                    fft_dense.buffer(ir) = veff_vec[j]->f_rg(ir) * step_function__.theta_r(ir);
                }
                fft_dense.transform<-1>(gvec_dense.partition(), &v_pw_fine[gvec_dense.partition().gvec_offset_fft()]);
                fft_dense.comm().allgather(&v_pw_fine[0], gvec_dense.partition().gvec_offset_fft(),
                                           gvec_dense.partition().gvec_count_fft());
                if (j == 0) {
                    v0_[0] = v_pw_fine[0].real();
                }
                /* loop over low-frequency G-vectors */
                for (int ig = 0; ig < gvec_coarse__.partition().gvec_count_fft(); ig++) {
                    /* G-vector in fractional coordinates */
                    auto G = gvec_coarse__.gvec(ig + gvec_coarse__.partition().gvec_offset_fft());
                    v_pw_coarse[ig] = v_pw_fine[gvec_dense.index_by_gvec(G)];
                }

                fft_coarse_.transform<1>(gvec_coarse__.partition(), &v_pw_coarse[0]);
                fft_coarse_.output(&veff_vec_(0, j));
            }
            
            /* map unit-step function */
            for (int ig = 0; ig < gvec_coarse__.partition().gvec_count_fft(); ig++) {
                /* G-vector in fractional coordinates */
                auto G = gvec_coarse__.gvec(ig + gvec_coarse__.partition().gvec_offset_fft());
                v_pw_coarse[ig] = step_function__.theta_pw(gvec_dense.index_by_gvec(G));
            }
            fft_coarse_.transform<1>(gvec_coarse__.partition(), &v_pw_coarse[0]);
            fft_coarse_.output(&theta_(0));
            /* release FFT driver */ 
            fft_coarse_.dismiss();

            if (!buf_rg_.size()) {
                buf_rg_ = mdarray<double_complex, 1>(fft_coarse_.local_size(), memory_t::host, "Local_operator::buf_rg_");
            }
            #ifdef __GPU
            if (fft_coarse_.hybrid()) {
                veff_vec_.allocate(memory_t::device);
                veff_vec_.copy_to_device();
                theta_.allocate(memory_t::device);
                theta_.copy_to_device();
                buf_rg_.allocate(memory_t::device);
            }
            #endif
        }
        
        /// Prepare the k-point dependent arrays.
        inline void prepare(Gvec const& gkvec__)
        {
            PROFILE("sirius::Local_operator::prepare");

            int ngv_fft = gkvec__.partition().gvec_count_fft();
            
            /* cache kinteic energy of plane-waves */
            if (static_cast<int>(pw_ekin_.size()) < ngv_fft) {
                pw_ekin_ = mdarray<double, 1>(ngv_fft, memory_t::host, "Local_operator::pw_ekin");
            }
            for (int ig_loc = 0; ig_loc < ngv_fft; ig_loc++) {
                /* global index of G-vector */
                int ig = gkvec__.partition().gvec_offset_fft() + ig_loc;
                /* get G+k in Cartesian coordinates */
                auto gv = gkvec__.gkvec_cart(ig);
                pw_ekin_[ig_loc] = 0.5 * (gv * gv);
            }

            if (static_cast<int>(vphi1_.size()) < ngv_fft) {
                vphi1_ = mdarray<double_complex, 1>(ngv_fft, memory_t::host, "Local_operator::vphi1");
            }
            if (gkvec__.reduced() && static_cast<int>(vphi2_.size()) < ngv_fft) {
                vphi2_ = mdarray<double_complex, 1>(ngv_fft, memory_t::host, "Local_operator::vphi2");
            }

            #ifdef __GPU
            if (fft_coarse_.hybrid()) {
                pw_ekin_.allocate(memory_t::device);
                pw_ekin_.copy_to_device();
                if (fft_coarse_.gpu_only()) {
                    vphi1_.allocate(memory_t::device);
                    if (gkvec__.reduced()) {
                        vphi2_.allocate(memory_t::device);
                    }
                }
            }
            #endif
        }

        inline void dismiss()
        {
            #ifdef __GPU
            veff_vec_.deallocate_on_device();
            pw_ekin_.deallocate_on_device();
            vphi1_.deallocate_on_device();
            vphi2_.deallocate_on_device();
            theta_.deallocate_on_device();
            buf_rg_.deallocate_on_device();
            #endif
        }

        void apply_h(int ispn__, wave_functions& hphi__, int idx0__, int n__)
        {
            PROFILE("sirius::Local_operator::apply_h");

            auto& gkp = hphi__.gkvec().partition();
            auto& comm_col = hphi__.gkvec().comm_ortho_fft();

            hphi__.pw_coeffs().remap_forward(gkp.gvec_fft_slab(), comm_col, n__, idx0__);

            int first{0};
            /* if G-vectors are reduced, wave-functions are real and 
             * we can transform two of them at once */
            if (gkp.reduced()) {
                int npairs = hphi__.pw_coeffs().spl_num_col().local_size() / 2;

                for (int i = 0; i < npairs; i++) {
                    /* phi(G) -> phi(r) */
                    fft_coarse_.transform<1>(gkp,
                                             hphi__.pw_coeffs().extra().at<CPU>(0, 2 * i),
                                             hphi__.pw_coeffs().extra().at<CPU>(0, 2 * i + 1));
                    /* multiply by effective potential */
                    if (fft_coarse_.hybrid()) {
                        #ifdef __GPU
                        scale_matrix_rows_gpu(fft_coarse_.local_size(), 1, fft_coarse_.buffer<GPU>(), veff_vec_.at<GPU>(0, ispn__));
                        #else
                        TERMINATE_NO_GPU
                        #endif
                    } else {
                        #pragma omp parallel for
                        for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
                            fft_coarse_.buffer(ir) *= veff_vec_(ir, ispn__);
                        }
                    }
                    /* V(r)phi(r) -> [V*phi](G) */
                    fft_coarse_.transform<-1>(gkp, &vphi1_[0], &vphi2_[0]);

                    /* add kinetic energy */
                    #pragma omp parallel for
                    for (int ig = 0; ig < gkp.gvec_count_fft(); ig++) {
                        hphi__.pw_coeffs().extra()(ig, 2 * i)     = hphi__.pw_coeffs().extra()(ig, 2 * i)     * pw_ekin_[ig] + vphi1_[ig];
                        hphi__.pw_coeffs().extra()(ig, 2 * i + 1) = hphi__.pw_coeffs().extra()(ig, 2 * i + 1) * pw_ekin_[ig] + vphi2_[ig];
                    }
                }
                
                /* check if we have to do last wave-function which had no pair */
                first = (hphi__.pw_coeffs().spl_num_col().local_size() % 2) ? hphi__.pw_coeffs().spl_num_col().local_size() - 1
                                                                            : hphi__.pw_coeffs().spl_num_col().local_size();
            }
            
            /* if we don't have G-vector reductions, first = 0 and we start a normal loop */
            for (int i = first; i < hphi__.pw_coeffs().spl_num_col().local_size(); i++) {
                /* phi(G) -> phi(r) */
                if (fft_coarse_.gpu_only()) {
                    fft_coarse_.transform<1>(gkp, hphi__.pw_coeffs().extra().at<GPU>(0, i));
                } else {
                    fft_coarse_.transform<1>(gkp, hphi__.pw_coeffs().extra().at<CPU>(0, i));
                }
                /* multiply by effective potential */
                if (fft_coarse_.hybrid()) {
                    #ifdef __GPU
                    scale_matrix_rows_gpu(fft_coarse_.local_size(), 1, fft_coarse_.buffer<GPU>(), veff_vec_.at<GPU>(0, ispn__));
                    #else
                    TERMINATE_NO_GPU
                    #endif
                } else {
                    #pragma omp parallel for
                    for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
                        fft_coarse_.buffer(ir) *= veff_vec_(ir, ispn__);
                    }
                }
                /* V(r)phi(r) -> [V*phi](G) */
                if (fft_coarse_.gpu_only()) {
                    fft_coarse_.transform<-1>(gkp, vphi1_.at<GPU>());
                } else {
                    fft_coarse_.transform<-1>(gkp, vphi1_.at<CPU>());
                }
                /* add kinetic energy */
                if (fft_coarse_.gpu_only()) {
                    #ifdef __GPU
                    add_pw_ekin_gpu(gkp.gvec_count_fft(), pw_ekin_.at<GPU>(), vphi1_.at<GPU>(),
                                    hphi__.pw_coeffs().extra().at<GPU>(0, i));
                    #endif
                } else {
                    #pragma omp parallel for
                    for (int ig = 0; ig < gkp.gvec_count_fft(); ig++) {
                        hphi__.pw_coeffs().extra()(ig, i) = hphi__.pw_coeffs().extra()(ig, i) * pw_ekin_[ig] + vphi1_[ig];
                    }
                }
            }

            hphi__.pw_coeffs().remap_backward(gkp.gvec_fft_slab(), comm_col, n__, idx0__);
        }

        void apply_h_o(Gvec_partition const& gkvec_par__,
                       int N__,
                       int n__,
                       wave_functions& phi__,
                       wave_functions& hphi__,
                       wave_functions& ophi__)
        {
            PROFILE("sirius::Local_operator::apply_h_o");

            auto& comm_col = gkvec_par__.gvec().comm_ortho_fft();

            fft_coarse_.prepare(gkvec_par__);

            mdarray<double_complex, 1> buf_pw(gkvec_par__.gvec_count_fft());

            #ifdef __GPU
            if (fft_coarse_.hybrid()) {
                phi__.pw_coeffs().copy_to_host(N__, n__);
            }
            #endif

            #ifdef __PRINT_OBJECT_CHECKSUM
            {
                auto cs = phi__.checksum(N__, n__);
                DUMP("checksum(phi): %18.10f %18.10f", cs.real(), cs.imag());
            }
            #endif

             phi__.pw_coeffs().remap_forward(gkvec_par__.gvec_fft_slab(),  comm_col, n__, N__);
            hphi__.pw_coeffs().set_num_extra(gkvec_par__.gvec_count_fft(), comm_col, n__, N__);
            ophi__.pw_coeffs().set_num_extra(gkvec_par__.gvec_count_fft(), comm_col, n__, N__);

            for (int j = 0; j < phi__.pw_coeffs().spl_num_col().local_size(); j++) {
                if (!fft_coarse_.hybrid()) {
                    /* phi(G) -> phi(r) */
                    fft_coarse_.transform<1>(gkvec_par__, phi__.pw_coeffs().extra().at<CPU>(0, j));
                    #pragma omp parallel for
                    for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
                        /* save phi(r) */
                        buf_rg_[ir] = fft_coarse_.buffer(ir);
                        /* multiply by step function */
                        fft_coarse_.buffer(ir) *= theta_[ir];
                    }
                    /* phi(r) * Theta(r) -> ophi(G) */
                    fft_coarse_.transform<-1>(gkvec_par__, ophi__.pw_coeffs().extra().at<CPU>(0, j));
                    #pragma omp parallel for
                    for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
                        /* multiply be effective potential, which itself was multiplied by the step function in constructor */
                        fft_coarse_.buffer(ir) = buf_rg_[ir] * veff_vec_(ir, 0);
                    }
                    /* phi(r) * Theta(r) * V(r) -> ophi(G) */
                    fft_coarse_.transform<-1>(gkvec_par__, hphi__.pw_coeffs().extra().at<CPU>(0, j));
                }
                #ifdef __GPU
                if (fft_coarse_.hybrid()) {
                    /* phi(G) -> phi(r) */
                    fft_coarse_.transform<1>(gkvec_par__, phi__.pw_coeffs().extra().at<CPU>(0, j));
                    /* save phi(r) */
                    acc::copy(buf_rg_.at<GPU>(), fft_coarse_.buffer<GPU>(), fft_coarse_.local_size());
                    /* multiply by step function */
                    scale_matrix_rows_gpu(fft_coarse_.local_size(), 1, fft_coarse_.buffer<GPU>(), theta_.at<GPU>());
                    /* phi(r) * Theta(r) -> ophi(G) */
                    fft_coarse_.transform<-1>(gkvec_par__, ophi__.pw_coeffs().extra().at<CPU>(0, j));
                    /* multiply by effective potential */
                    scale_matrix_rows_gpu(fft_coarse_.local_size(), 1, buf_rg_.at<GPU>(), veff_vec_.at<GPU>());
                    /* copy phi(r) * Theta(r) * V(r) to GPU buffer */
                    acc::copy(fft_coarse_.buffer<GPU>(), buf_rg_.at<GPU>(), fft_coarse_.local_size());
                    /* phi(r) * Theta(r) * V(r) -> ophi(G) */
                    fft_coarse_.transform<-1>(gkvec_par__, hphi__.pw_coeffs().extra().at<CPU>(0, j));
                }
                #endif

                /* add kinetic energy */
                for (int x: {0, 1, 2}) {
                    for (int igloc = 0; igloc < gkvec_par__.gvec_count_fft(); igloc++) {
                        /* global index of G-vector */
                        int ig = gkvec_par__.gvec_offset_fft() + igloc;
                        /* \hat P phi = phi(G+k) * (G+k), \hat P is momentum operator */ 
                        buf_pw[igloc] = phi__.pw_coeffs().extra()(igloc, j) * gkvec_par__.gvec().gkvec_cart(ig)[x];
                    }
                    /* transform Cartesian component of wave-function gradient to real space */
                    fft_coarse_.transform<1>(gkvec_par__, &buf_pw[0]);
                    if (!fft_coarse_.hybrid()) {
                        #pragma omp parallel for
                        for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
                            /* multiply be step function */
                            fft_coarse_.buffer(ir) *= theta_[ir];
                        }
                    }
                    #ifdef __GPU
                    if (fft_coarse_.hybrid()) {
                        /* multiply by step function */
                        scale_matrix_rows_gpu(fft_coarse_.local_size(), 1, fft_coarse_.buffer<GPU>(), theta_.at<GPU>());
                    }
                    #endif
                    /* transform back to PW domain */
                    fft_coarse_.transform<-1>(gkvec_par__, &buf_pw[0]);
                    for (int igloc = 0; igloc < gkvec_par__.gvec_count_fft(); igloc++) {
                        int ig = gkvec_par__.gvec_offset_fft() + igloc;
                        hphi__.pw_coeffs().extra()(igloc, j) += 0.5 * buf_pw[igloc] * gkvec_par__.gvec().gkvec_cart(ig)[x];
                    }
                }
            }

            hphi__.pw_coeffs().remap_backward(gkvec_par__.gvec_fft_slab(), comm_col, n__, N__);
            ophi__.pw_coeffs().remap_backward(gkvec_par__.gvec_fft_slab(), comm_col, n__, N__);

            fft_coarse_.dismiss();

            #ifdef __GPU
            if (fft_coarse_.hybrid()) {
                hphi__.pw_coeffs().copy_to_device(N__, n__);
                ophi__.pw_coeffs().copy_to_device(N__, n__);
            }
            #endif
        }

        void apply_o(Gvec_partition const& gkvec_par__,
                     int N__,
                     int n__,
                     wave_functions& phi__,
                     wave_functions& ophi__) const
        {
            PROFILE("sirius::Local_operator::apply_o");

            auto& comm_col = gkvec_par__.gvec().comm_ortho_fft();

            fft_coarse_.prepare(gkvec_par__);

            #ifdef __GPU
            if (fft_coarse_.hybrid()) {
                phi__.pw_coeffs().copy_to_host(N__, n__);
            }
            #endif

            #ifdef __PRINT_OBJECT_CHECKSUM
            {
                auto cs = phi__.checksum(N__, n__);
                DUMP("checksum(phi): %18.10f %18.10f", cs.real(), cs.imag());
            }
            #endif

             phi__.pw_coeffs().remap_forward(gkvec_par__.gvec_fft_slab(),  comm_col, n__, N__);
            ophi__.pw_coeffs().set_num_extra(gkvec_par__.gvec_count_fft(), comm_col, n__, N__);

            for (int j = 0; j < phi__.pw_coeffs().spl_num_col().local_size(); j++) {
                if (!fft_coarse_.hybrid()) {
                    /* phi(G) -> phi(r) */
                    fft_coarse_.transform<1>(gkvec_par__, phi__.pw_coeffs().extra().at<CPU>(0, j));
                    #pragma omp parallel for
                    for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
                        /* multiply by step function */
                        fft_coarse_.buffer(ir) *= theta_[ir];
                    }
                    /* phi(r) * Theta(r) -> ophi(G) */
                    fft_coarse_.transform<-1>(gkvec_par__, ophi__.pw_coeffs().extra().at<CPU>(0, j));
                } else {
                    #ifdef __GPU
                    /* phi(G) -> phi(r) */
                    fft_coarse_.transform<1>(gkvec_par__, phi__.pw_coeffs().extra().at<CPU>(0, j));
                    /* multiply by step function */
                    scale_matrix_rows_gpu(fft_coarse_.local_size(), 1, fft_coarse_.buffer<GPU>(), theta_.at<GPU>());
                    /* phi(r) * Theta(r) -> ophi(G) */
                    fft_coarse_.transform<-1>(gkvec_par__, ophi__.pw_coeffs().extra().at<CPU>(0, j));
                    #else
                    TERMINATE_NO_GPU
                    #endif
                }
            }

            ophi__.pw_coeffs().remap_backward(gkvec_par__.gvec_fft_slab(), comm_col, n__, N__);

            fft_coarse_.dismiss();

            #ifdef __GPU
            if (fft_coarse_.hybrid()) {
                ophi__.pw_coeffs().copy_to_device(N__, n__);
            }
            #endif

            #ifdef __PRINT_OBJECT_CHECKSUM
            {
                auto cs2 = ophi__.checksum(N__, n__);
                DUMP("checksum(ophi_istl): %18.10f %18.10f", cs2.real(), cs2.imag());
            }
            #endif
        }
        
        /// Apply magnetic field to the wave-functions.
        /** In case of collinear magnetism only Bz is applied to <tt>phi</tt> and stored in the first component of
         *  <tt>bphi</tt>. In case of non-collinear magnetims Bx-iBy is also applied and stored in the third
         *  component of <tt>bphi</tt>. The second componet of <tt>bphi</tt> is used to store -Bz|phi>. */
        void apply_b(Gvec_partition const& gkvec_par__,
                     int N__,
                     int n__,
                     wave_functions& phi__,
                     std::vector<wave_functions>& bphi__)
        {
            PROFILE("sirius::Local_operator::apply_b");

            auto& comm_col = gkvec_par__.gvec().comm_ortho_fft();

            fft_coarse_.prepare(gkvec_par__);

            #ifdef __GPU
            if (fft_coarse_.hybrid()) {
                phi__.pw_coeffs().copy_to_host(N__, n__);
            }
            #endif

            /* components of H|psi> to which H is applied */
            std::vector<int> iv(1, 0);
            if (bphi__.size() == 3) {
                iv.push_back(2);
            }

            phi__.pw_coeffs().remap_forward(gkvec_par__.gvec_fft_slab(), comm_col, n__, N__);
            for (int i: iv) {
                bphi__[i].pw_coeffs().set_num_extra(gkvec_par__.gvec_count_fft(), comm_col, n__, N__);
            }

            for (int j = 0; j < phi__.pw_coeffs().spl_num_col().local_size(); j++) {
                if (!fft_coarse_.hybrid()) {
                    /* phi(G) -> phi(r) */
                    fft_coarse_.transform<1>(gkvec_par__, phi__.pw_coeffs().extra().at<CPU>(0, j));
                    /* save phi(r) */
                    if (bphi__.size() == 3) {
                        fft_coarse_.output(buf_rg_.at<CPU>());
                    }
                    #pragma omp parallel for
                    for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
                        /* multiply by Bz */
                        fft_coarse_.buffer(ir) *= veff_vec_(ir, 1);
                    }
                    /* phi(r) * Bz(r) -> bphi[0](G) */
                    fft_coarse_.transform<-1>(gkvec_par__, bphi__[0].pw_coeffs().extra().at<CPU>(0, j));
                    /* non-collinear case */
                    if (bphi__.size() == 3) {
                        #pragma omp parallel for
                        for (int ir = 0; ir < fft_coarse_.local_size(); ir++) {
                            /* multiply by Bx-iBy */
                            fft_coarse_.buffer(ir) = buf_rg_[ir] * double_complex(veff_vec_(ir, 2), -veff_vec_(ir, 3));
                        }
                        /* phi(r) * (Bx(r)-iBy(r)) -> bphi[2](G) */
                        fft_coarse_.transform<-1>(gkvec_par__, bphi__[2].pw_coeffs().extra().at<CPU>(0, j));
                    }
                } else {
                    STOP();
                    #ifdef __GPU
                    /* phi(G) -> phi(r) */
                    fft_coarse_.transform<1>(gkvec_par__, phi__.pw_coeffs().extra().at<CPU>(0, j));
                    /* multiply by step function */
                    scale_matrix_rows_gpu(fft_coarse_.local_size(), 1, fft_coarse_.buffer<GPU>(), theta_.at<GPU>());
                    /* phi(r) * Theta(r) -> ophi(G) */
                    fft_coarse_.transform<-1>(gkvec_par__, bphi__[0].pw_coeffs().extra().at<CPU>(0, j));
                    #else
                    TERMINATE_NO_GPU
                    #endif
                }
            }

            for (int i: iv) {
                bphi__[i].pw_coeffs().remap_backward(gkvec_par__.gvec_fft_slab(), comm_col, n__, N__);
                #ifdef __GPU
                if (fft_coarse_.hybrid()) {
                    bphi__[i].pw_coeffs().copy_to_device(N__, n__);
                }
                #endif
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
