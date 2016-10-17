#include "simulation_context.h"

#ifndef __INTERSTITIAL_OPERATOR_H__
#define __INTERSTITIAL_OPERATOR_H__

namespace sirius {

class Interstitial_operator
{
    private:

        FFT3D& fft_;

        Communicator const& comm_col_;
        
        mdarray<double, 1> theta_;
        
        mdarray<double, 1> veff_;

        mdarray<double_complex, 1> buf_rg_;

    public:

        Interstitial_operator(FFT3D& fft__,
                              Gvec const& gvec_coarse__,
                              Communicator const& comm_col__,
                              Periodic_function<double>* effective_potential__,
                              Step_function const& step_function__)
            : fft_(fft__),
              comm_col_(comm_col__)
        {
            auto& fft_dense = effective_potential__->fft();
            auto& gvec_dense = effective_potential__->gvec();

            for (int ir = 0; ir < fft_dense.local_size(); ir++) {
                fft_dense.buffer(ir) = effective_potential__->f_rg(ir) * step_function__.theta_r(ir);
            }

            mdarray<double_complex, 1> fpw(gvec_dense.num_gvec());
            
            fft_dense.prepare(gvec_dense.partition());
            fft_dense.transform<-1>(gvec_dense.partition(), &fpw[gvec_dense.partition().gvec_offset_fft()]);
            fft_dense.comm().allgather(&fpw[0], gvec_dense.partition().gvec_offset_fft(),
                                       gvec_dense.partition().gvec_count_fft());
            fft_dense.dismiss();

            veff_  = mdarray<double, 1>(fft_.local_size(), memory_t::host, "veff_");
            theta_ = mdarray<double, 1>(fft_.local_size(), memory_t::host, "theta_");

            fft_.prepare(gvec_coarse__.partition());
            /* low-frequency part of PW coefficients */
            std::vector<double_complex> veff_pw_coarse(gvec_coarse__.partition().gvec_count_fft());
            std::vector<double_complex> theta_pw_coarse(gvec_coarse__.partition().gvec_count_fft());
            /* loop over low-frequency G-vectors */
            for (int ig = 0; ig < gvec_coarse__.partition().gvec_count_fft(); ig++) {
                /* G-vector in fractional coordinates */
                auto G = gvec_coarse__.gvec(ig + gvec_coarse__.partition().gvec_offset_fft());
                veff_pw_coarse[ig] = fpw[gvec_dense.index_by_gvec(G)];
                theta_pw_coarse[ig] = step_function__.theta_pw(gvec_dense.index_by_gvec(G));
            }
            fft_.transform<1>(gvec_coarse__.partition(), &veff_pw_coarse[0]);
            fft_.output(&veff_[0]);
            fft_.transform<1>(gvec_coarse__.partition(), &theta_pw_coarse[0]);
            fft_.output(&theta_[0]);
            fft_.dismiss();
            buf_rg_ = mdarray<double_complex, 1>(fft_.local_size(), memory_t::host, "buf_rg_");
            #ifdef __GPU
            if (fft_.hybrid()) {
                veff_.allocate(memory_t::device);
                veff_.copy_to_device();
                theta_.allocate(memory_t::device);
                theta_.copy_to_device();
                buf_rg_.allocate(memory_t::device);
            }
            #endif
        }

        void apply(K_point* kp__,
                   int N__,
                   int n__,
                   wave_functions& phi__,
                   wave_functions& hphi__,
                   wave_functions& ophi__)
        {
            PROFILE_WITH_TIMER("sirius::Interstitial_operator::apply");

            fft_.prepare(kp__->gkvec().partition());

            mdarray<double_complex, 1> buf_pw(kp__->gkvec().partition().gvec_count_fft());

            #ifdef __GPU
            if (fft_.hybrid()) {
                phi__.pw_coeffs().copy_to_host(N__, n__);
            }
            #endif

            #ifdef __PRINT_OBJECT_CHECKSUM
            {
                auto cs = phi__.checksum(N__, n__);
                DUMP("checksum(phi): %18.10f %18.10f", cs.real(), cs.imag());
            }
            #endif

             phi__.pw_coeffs().remap_forward(kp__->gkvec().partition().gvec_fft_slab(),  comm_col_, n__, N__);
            hphi__.pw_coeffs().set_num_extra(kp__->gkvec().partition().gvec_count_fft(), comm_col_, n__, N__);
            ophi__.pw_coeffs().set_num_extra(kp__->gkvec().partition().gvec_count_fft(), comm_col_, n__, N__);

            for (int j = 0; j < phi__.pw_coeffs().spl_num_col().local_size(); j++) {
                if (!fft_.hybrid()) {
                    /* phi(G) -> phi(r) */
                    fft_.transform<1>(kp__->gkvec().partition(), phi__.pw_coeffs().extra().at<CPU>(0, j));
                    #pragma omp parallel for
                    for (int ir = 0; ir < fft_.local_size(); ir++) {
                        /* save phi(r) */
                        buf_rg_[ir] = fft_.buffer(ir);
                        /* multiply by step function */
                        fft_.buffer(ir) *= theta_[ir];
                    }
                    /* phi(r) * Theta(r) -> ophi(G) */
                    fft_.transform<-1>(kp__->gkvec().partition(), ophi__.pw_coeffs().extra().at<CPU>(0, j));
                    #pragma omp parallel for
                    for (int ir = 0; ir < fft_.local_size(); ir++) {
                        /* multiply be effective potential, which itself was multiplied by the step function in constructor */
                        fft_.buffer(ir) = buf_rg_[ir] * veff_[ir];
                    }
                    /* phi(r) * Theta(r) * V(r) -> ophi(G) */
                    fft_.transform<-1>(kp__->gkvec().partition(), hphi__.pw_coeffs().extra().at<CPU>(0, j));
                }
                #ifdef __GPU
                if (fft_.hybrid()) {
                    /* phi(G) -> phi(r) */
                    fft_.transform<1>(kp__->gkvec().partition(), phi__.pw_coeffs().extra().at<CPU>(0, j));
                    /* save phi(r) */
                    acc::copy(buf_rg_.at<GPU>(), fft_.buffer<GPU>(), fft_.local_size());
                    /* multiply by step function */
                    scale_matrix_rows_gpu(fft_.local_size(), 1, fft_.buffer<GPU>(), theta_.at<GPU>());
                    /* phi(r) * Theta(r) -> ophi(G) */
                    fft_.transform<-1>(kp__->gkvec().partition(), ophi__.pw_coeffs().extra().at<CPU>(0, j));
                    /* multiply by effective potential */
                    scale_matrix_rows_gpu(fft_.local_size(), 1, buf_rg_.at<GPU>(), veff_.at<GPU>());
                    /* copy to GPU buffer */
                    acc::copy(fft_.buffer<GPU>(), buf_rg_.at<GPU>(), fft_.local_size());
                    /* phi(r) * Theta(r) * V(r) -> ophi(G) */
                    fft_.transform<-1>(kp__->gkvec().partition(), hphi__.pw_coeffs().extra().at<CPU>(0, j));
                }
                #endif

                /* add kinetic energy */
                for (int x: {0, 1, 2}) {
                    for (int igloc = 0; igloc < kp__->gkvec().partition().gvec_count_fft(); igloc++) {
                        /* global index of G-vector */
                        int ig = kp__->gkvec().partition().gvec_offset_fft() + igloc;
                        /* \hat P phi = phi(G+k) * (G+k), \hat P is momentum operator */ 
                        buf_pw[igloc] = phi__.pw_coeffs().extra()(igloc, j) * kp__->gkvec().gkvec_cart(ig)[x];
                    }
                    /* transform Cartesian component of wave-function gradient to real space */
                    fft_.transform<1>(kp__->gkvec().partition(), &buf_pw[0]);
                    if (!fft_.hybrid()) {
                        #pragma omp parallel for
                        for (int ir = 0; ir < fft_.local_size(); ir++) {
                            /* multiply be step function */
                            fft_.buffer(ir) *= theta_[ir];
                        }
                    }
                    #ifdef __GPU
                    if (fft_.hybrid()) {
                        /* multiply by step function */
                        scale_matrix_rows_gpu(fft_.local_size(), 1, fft_.buffer<GPU>(), theta_.at<GPU>());
                    }
                    #endif
                    /* transform back to PW domain */
                    fft_.transform<-1>(kp__->gkvec().partition(), &buf_pw[0]);
                    for (int igloc = 0; igloc < kp__->gkvec().partition().gvec_count_fft(); igloc++) {
                        int ig = kp__->gkvec().partition().gvec_offset_fft() + igloc;
                        hphi__.pw_coeffs().extra()(igloc, j) += 0.5 * buf_pw[igloc] * kp__->gkvec().gkvec_cart(ig)[x];
                    }
                }
            }

            hphi__.pw_coeffs().remap_backward(kp__->gkvec().partition().gvec_fft_slab(), comm_col_, n__, N__);
            ophi__.pw_coeffs().remap_backward(kp__->gkvec().partition().gvec_fft_slab(), comm_col_, n__, N__);

            fft_.dismiss();

            #ifdef __GPU
            if (fft_.hybrid()) {
                hphi__.pw_coeffs().copy_to_device(N__, n__);
                ophi__.pw_coeffs().copy_to_device(N__, n__);
            }
            #endif
        }

        void apply_o(K_point* kp__,
                     int N__,
                     int n__,
                     wave_functions& phi__,
                     wave_functions& ophi__) const
        {
            PROFILE_WITH_TIMER("sirius::Interstitial_operator::apply_o");

            fft_.prepare(kp__->gkvec().partition());

            #ifdef __GPU
            if (fft_.hybrid()) {
                phi__.pw_coeffs().copy_to_host(N__, n__);
            }
            #endif

            #ifdef __PRINT_OBJECT_CHECKSUM
            {
                auto cs = phi__.checksum(N__, n__);
                DUMP("checksum(phi): %18.10f %18.10f", cs.real(), cs.imag());
            }
            #endif

             phi__.pw_coeffs().remap_forward(kp__->gkvec().partition().gvec_fft_slab(),  comm_col_, n__, N__);
            ophi__.pw_coeffs().set_num_extra(kp__->gkvec().partition().gvec_count_fft(), comm_col_, n__, N__);

            for (int j = 0; j < phi__.pw_coeffs().spl_num_col().local_size(); j++) {
                if (!fft_.hybrid()) {
                    /* phi(G) -> phi(r) */
                    fft_.transform<1>(kp__->gkvec().partition(), phi__.pw_coeffs().extra().at<CPU>(0, j));
                    #pragma omp parallel for
                    for (int ir = 0; ir < fft_.local_size(); ir++) {
                        /* multiply by step function */
                        fft_.buffer(ir) *= theta_[ir];
                    }
                    /* phi(r) * Theta(r) -> ophi(G) */
                    fft_.transform<-1>(kp__->gkvec().partition(), ophi__.pw_coeffs().extra().at<CPU>(0, j));
                }
                #ifdef __GPU
                if (fft_.hybrid()) {
                    /* phi(G) -> phi(r) */
                    fft_.transform<1>(kp__->gkvec().partition(), phi__.pw_coeffs().extra().at<CPU>(0, j));
                    /* multiply by step function */
                    scale_matrix_rows_gpu(fft_.local_size(), 1, fft_.buffer<GPU>(), theta_.at<GPU>());
                    /* phi(r) * Theta(r) -> ophi(G) */
                    fft_.transform<-1>(kp__->gkvec().partition(), ophi__.pw_coeffs().extra().at<CPU>(0, j));
                }
                #endif
            }

            ophi__.pw_coeffs().remap_backward(kp__->gkvec().partition().gvec_fft_slab(), comm_col_, n__, N__);

            fft_.dismiss();

            #ifdef __GPU
            if (fft_.hybrid()) {
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
};

}

#endif // __INTERSTITIAL_OPERATOR_H__
