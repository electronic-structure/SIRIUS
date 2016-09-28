#include "simulation_context.h"

#ifndef __INTERSTITIAL_OPERATOR_H__
#define __INTERSTITIAL_OPERATOR_H__

namespace sirius {

class Interstitial_operator
{
    private:

        Simulation_context& ctx_;

        FFT3D& fft_;

        Communicator const& comm_col_;
        
        mdarray<double, 1> theta_;
        
        mdarray<double, 1> veff_;

        mdarray<double_complex, 1> buf_rg_;

    public:

        Interstitial_operator(Simulation_context& ctx__,
                              Periodic_function<double>* effective_potential__)
            : ctx_(ctx__),
              fft_(ctx__.fft()),
              comm_col_(ctx__.mpi_grid_fft().communicator(1 << 1))
 
        {
            veff_  = mdarray<double, 1>(fft_.local_size(), memory_t::host, "veff_");
            theta_ = mdarray<double, 1>(fft_.local_size(), memory_t::host, "theta_");
            for (int ir = 0; ir < fft_.local_size(); ir++) {
                veff_[ir]  = effective_potential__->f_rg(ir);
                theta_[ir] = ctx__.step_function().theta_r(ir);
            }
            buf_rg_ = mdarray<double_complex, 1>(fft_.local_size(), memory_t::host, "buf_rg_");
            #ifdef __GPU
            if (ctx__.processing_unit() == GPU) {
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
            fft_.prepare(kp__->gkvec().partition());

            mdarray<double_complex, 1> buf_pw(kp__->gkvec().partition().gvec_count_fft());

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
                if (ctx_.processing_unit() == CPU) {
                    /* phi(G) -> phi(r) */
                    fft_.transform<1>(kp__->gkvec().partition(), phi__.pw_coeffs().extra().at<CPU>(0, j));
                    #pragma omp parallel for
                    for (int ir = 0; ir < fft_.local_size(); ir++) {
                        /* multiply by step function */
                        fft_.buffer(ir) *= theta_[ir];
                        /* save phi(r) * Theta(r) */
                        buf_rg_[ir] = fft_.buffer(ir);
                    }
                    /* phi(r) * Theta(r) -> ophi(G) */
                    fft_.transform<-1>(kp__->gkvec().partition(), ophi__.pw_coeffs().extra().at<CPU>(0, j));
                    #pragma omp parallel for
                    for (int ir = 0; ir < fft_.local_size(); ir++) {
                        /* multiply be effective potential */
                        fft_.buffer(ir) = buf_rg_[ir] * veff_[ir];
                    }
                    /* phi(r) * Theta(r) * V(r) -> ophi(G) */
                    fft_.transform<-1>(kp__->gkvec().partition(), hphi__.pw_coeffs().extra().at<CPU>(0, j));
                }
                #ifdef __GPU
                if (ctx_.processing_unit() == GPU) {
                    /* phi(G) -> phi(r) */
                    fft_.transform<1>(kp__->gkvec().partition(), phi__.pw_coeffs().extra().at<CPU>(0, j));
                    /* multiply by step function */
                    scale_matrix_rows_gpu(fft_.local_size(), 1, fft_.buffer<GPU>(), theta_.at<GPU>());
                    /* save phi(r) * Theta(r) */
                    acc::copy(buf_rg_.at<GPU>(), fft_.buffer<GPU>(), fft_.local_size());
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
                    if (ctx_.processing_unit() == CPU) {
                        #pragma omp parallel for
                        for (int ir = 0; ir < fft_.local_size(); ir++) {
                            /* multiply be step function */
                            fft_.buffer(ir) *= theta_[ir];
                        }
                    }
                    #ifdef __GPU
                    if (ctx_.processing_unit() == GPU) {
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
            if (ctx_.processing_unit() == GPU) {
                hphi__.pw_coeffs().copy_to_device(N__, n__);
                ophi__.pw_coeffs().copy_to_device(N__, n__);
            }
            #endif
        }
};

}

#endif // __INTERSTITIAL_OPERATOR_H__
