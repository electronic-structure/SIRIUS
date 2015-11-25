#ifndef __HLOC_OPERATOR_H__
#define __HLOC_OPERATOR_H__

#include "wave_functions.h"

namespace sirius {

class Hloc_operator
{
    private:

        Simulation_context const& ctx_;

        Gvec const& gkvec_;

        std::vector<double> pw_ekin_;

        std::vector<double> effective_potential_;

    public:

        Hloc_operator(Simulation_context const& ctx__,
                      Gvec const& gkvec__,
                      std::vector<double> pw_ekin__,
                      std::vector<double> effective_potential__) 
            : ctx_(ctx__),
              gkvec_(gkvec__),
              pw_ekin_(pw_ekin__),
              effective_potential_(effective_potential__)
        {
        }
        
        // phi is always on CPU
        // hphi is always on CPU
        void apply(Wave_functions& phi__, Wave_functions& hphi__, int idx0__, int n__)
        {
            PROFILE();

            Timer t("sirius::Hloc_operator::apply");

            phi__.swap_forward(idx0__, n__);
            
            /* save omp_nested flag */
            int nested = omp_get_nested();
            omp_set_nested(1);
            #pragma omp parallel num_threads(ctx_.num_fft_threads())
            {
                int thread_id = omp_get_thread_num();

                #pragma omp for schedule(dynamic, 1)
                for (int i = 0; i < phi__.spl_num_swapped().local_size(); i++)
                {
                    //double t1 = omp_get_wtime();
                    //if (thread_id == ctx_.gpu_thread_id() && parameters_.processing_unit() == GPU)
                    //{
                    //    #ifdef __GPU
                    //    STOP();
                    //    ///* copy phi to GPU */
                    //    //cuda_copy_to_device(pw_buf.at<GPU>(), phi__.at<CPU>(0, i), kp__->num_gkvec() * sizeof(double_complex));

                    //    ///* set PW coefficients into proper positions inside FFT buffer */
                    //    //ctx_.fft_coarse(thread_id)->input_on_device(kp__->num_gkvec(), fft_index.at<GPU>(), pw_buf.at<GPU>());

                    //    ///* phi(G) *= Ekin(G) */
                    //    //scale_matrix_rows_gpu(kp__->num_gkvec(), 1, pw_buf.at<GPU>(), pw_ekin.at<GPU>());
                    //    //
                    //    ///* execute FFT */
                    //    ////ctx_.fft_coarse(thread_id)->transform(1);
                    //    //STOP();
                    //    //
                    //    ///* multiply by potential */
                    //    //scale_matrix_rows_gpu(ctx_.fft_coarse(thread_id)->local_size(), 1,
                    //    //                      ctx_.fft_coarse(thread_id)->buffer<GPU>(), veff.at<GPU>());
                    //    //
                    //    ///* transform back */
                    //    ////ctx_.fft_coarse(thread_id)->transform(-1);
                    //    //STOP();
                    //    //
                    //    ///* phi(G) += fft_buffer(G) */
                    //    //ctx_.fft_coarse(thread_id)->output_on_device(kp__->num_gkvec(), fft_index.at<GPU>(), pw_buf.at<GPU>(), 1.0);
                    //    //
                    //    ///* copy final hphi to CPU */
                    //    //cuda_copy_to_host(hphi__.at<CPU>(0, i), pw_buf.at<GPU>(), kp__->num_gkvec() * sizeof(double_complex));
                    //    #endif
                    //}
                    //else
                    //{
                        /* phi(G) -> phi(r) */
                        ctx_.fft_coarse(thread_id)->transform<1>(gkvec_, phi__[i]);
                        /* multiply by effective potential */
                        for (int ir = 0; ir < ctx_.fft_coarse(thread_id)->local_size(); ir++)
                            ctx_.fft_coarse(thread_id)->buffer(ir) *= effective_potential_[ir];
                        /* V(r)phi(r) -> [V*phi](G) */
                        ctx_.fft_coarse(thread_id)->transform<-1>(gkvec_, hphi__[i]);

                        //if (in_place)
                        //{
                        //    STOP();
                        //    /* psi(G) -> 0.5 * |G|^2 * psi(G) */
                        //    //for (int igk = 0; igk < kp__->num_gkvec(); igk++) hphi__(igk, i) *= pw_ekin__[igk];
                        //    //ctx_.fft_coarse(thread_id)->output(kp__->num_gkvec(), kp__->gkvec_coarse().index_map(), &hphi__(0, i), 1.0);
                        //    //STOP();
                        //}
                        //else
                        //{
                            for (int igk_loc = 0; igk_loc < gkvec_.num_gvec_fft(); igk_loc++)
                            {
                                int igk = gkvec_.offset_gvec_fft() + igk_loc;
                                hphi__[i][igk_loc] += phi__[i][igk_loc] * pw_ekin_[igk];
                            }
                        //}
                    //}
                    //timers(thread_id) += (omp_get_wtime() - t1);
                    //timer_counts(thread_id)++;
                }
            }
            /* restore the nested flag */
            omp_set_nested(nested);

            hphi__.swap_backward(idx0__, n__);
        }
};

};

#endif
