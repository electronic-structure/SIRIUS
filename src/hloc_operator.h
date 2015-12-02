#ifndef __HLOC_OPERATOR_H__
#define __HLOC_OPERATOR_H__

#include "wave_functions.h"

namespace sirius {

class Hloc_operator
{
    private:

        //Simulation_context const& ctx_;
        FFT3D_context& fft_ctx_;

        Gvec const& gkvec_;

        std::vector<double> const& pw_ekin_;

        //std::vector<double> const& effective_potential_;
        mdarray<double, 1> veff_;

        mdarray<double_complex, 2> vphi_;

    public:

        Hloc_operator(FFT3D_context& fft_ctx__,
                      Gvec const& gkvec__,
                      std::vector<double> const& pw_ekin__,
                      std::vector<double> const& effective_potential__) 
            : fft_ctx_(fft_ctx__),
              gkvec_(gkvec__),
              pw_ekin_(pw_ekin__)
        {
            veff_ = mdarray<double, 1>(const_cast<double*>(&effective_potential__[0]), effective_potential__.size(), "veff_");
            #ifdef __GPU
            if (fft_ctx_.pu() == GPU)
            {
                veff_.allocate_on_device();
                veff_.copy_to_device();
            }
            #endif
            vphi_ = mdarray<double_complex, 2>(gkvec__.num_gvec_fft(), fft_ctx_.num_fft_streams());
        }
        
        void apply(Wave_functions& hphi__, int idx0__, int n__)
        {
            PROFILE();

            Timer t("sirius::Hloc_operator::apply");

            hphi__.swap_forward(idx0__, n__);
            
            /* save omp_nested flag */
            int nested = omp_get_nested();
            omp_set_nested(1);
            #pragma omp parallel num_threads(fft_ctx_.num_fft_streams())
            {
                int thread_id = omp_get_thread_num();

                #pragma omp for schedule(dynamic, 1)
                for (int i = 0; i < hphi__.spl_num_swapped().local_size(); i++)
                {
                    /* phi(G) -> phi(r) */
                    fft_ctx_.fft(thread_id)->transform<1>(gkvec_, hphi__[i]);
                    /* multiply by effective potential */
                    if (fft_ctx_.fft(thread_id)->hybrid())
                    {
                        #ifdef __GPU
                        scale_matrix_rows_gpu(fft_ctx_.fft(thread_id)->local_size(), 1,
                                              fft_ctx_.fft(thread_id)->buffer<GPU>(), veff_.at<GPU>());

                        #else
                        TERMINATE_NO_GPU
                        #endif
                    }
                    else
                    {
                        for (int ir = 0; ir < fft_ctx_.fft(thread_id)->local_size(); ir++)
                            fft_ctx_.fft(thread_id)->buffer(ir) *= veff_[ir];
                    }
                    /* V(r)phi(r) -> [V*phi](G) */
                    fft_ctx_.fft(thread_id)->transform<-1>(gkvec_, &vphi_(0, thread_id));

                    /* add kinetic energy */
                    for (int igk_loc = 0; igk_loc < gkvec_.num_gvec_fft(); igk_loc++)
                    {
                        int igk = gkvec_.offset_gvec_fft() + igk_loc;
                        hphi__[i][igk_loc] = hphi__[i][igk_loc] * pw_ekin_[igk] + vphi_(igk_loc, thread_id);
                    }
                }
            }
            /* restore the nested flag */
            omp_set_nested(nested);

            hphi__.swap_backward(idx0__, n__);
        }
};

};

#endif
