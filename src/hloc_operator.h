#ifndef __HLOC_OPERATOR_H__
#define __HLOC_OPERATOR_H__

#include "wave_functions.h"
#include "periodic_function.h"

namespace sirius {

class Hloc_operator
{
    private:

        //Simulation_context const& ctx_;
        FFT3D_context& fft_ctx_;

        Gvec const& gkvec_;

        std::vector<double> pw_ekin_;

        mdarray<double, 1> veff_;

        mdarray<double, 2> veff_vec_;

        mdarray<double_complex, 2> vphi_;

        double v0_[2];

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

        /** \param [in] fft_ctx FFT context of the coarse grid used to apply effective field.
         *  \param [in] gvec G-vectors of the coarse FFT grid.
         *  \param [in] gkvec G-vectors of the wave-functions.
         */
        Hloc_operator(FFT3D_context& fft_ctx__,
                      Gvec const& gvec__,
                      Gvec const& gkvec__,
                      int num_mag_dims__,
                      Periodic_function<double>* effective_potential__,
                      Periodic_function<double>* effective_magnetic_field__[3]) 
            : fft_ctx_(fft_ctx__),
              gkvec_(gkvec__)
        {
            pw_ekin_ = std::vector<double>(gkvec_.num_gvec_fft());
            for (int ig_loc = 0; ig_loc < gkvec_.num_gvec_fft(); ig_loc++)
            {
                /* global index of G-vector */
                int ig = gkvec_.offset_gvec_fft() + ig_loc;
                /* get G+k in Cartesian coordinates */
                auto gv = gkvec_.cart_shifted(ig);
                pw_ekin_[ig_loc] = 0.5 * (gv * gv);
            }
            
            /* group effective fields into single vector */
            std::vector<Periodic_function<double>*> veff_vec(num_mag_dims__ + 1);
            veff_vec[0] = effective_potential__;
            for (int j = 0; j < num_mag_dims__; j++) veff_vec[1 + j] = effective_magnetic_field__[j];

            veff_vec_ = mdarray<double, 2>(fft_ctx_.fft()->local_size(), num_mag_dims__ + 1);

            /* map components of effective potential to a corase grid */
            for (int j = 0; j < num_mag_dims__ + 1; j++)
            {
                auto& gv = veff_vec[j]->gvec();
                std::vector<double_complex> v_pw_coarse(gvec__.num_gvec_fft());

                for (int ig = 0; ig < gvec__.num_gvec_fft(); ig++)
                {
                    auto G = gvec__[ig + gvec__.offset_gvec_fft()];
                    v_pw_coarse[ig] = veff_vec[j]->f_pw(gv.index_by_gvec(G));
                }
                fft_ctx_.fft()->transform<1>(gvec__, &v_pw_coarse[0]);
                fft_ctx_.fft()->output(&veff_vec_(0, j));

            }

            if (num_mag_dims__)
            {
                for (int ir = 0; ir < fft_ctx_.fft()->local_size(); ir++)
                {
                    double v0 = veff_vec_(ir, 0);
                    double v1 = veff_vec_(ir, 1);
                    veff_vec_(ir, 0) = v0 + v1; // v + Bz
                    veff_vec_(ir, 1) = v0 - v1; // v - Bz
                }
            }

            if (num_mag_dims__ == 0)
            {
                v0_[0] = veff_vec[0]->f_pw(0).real();
            }
            else
            {
                v0_[0] = veff_vec[0]->f_pw(0).real() + veff_vec[1]->f_pw(0).real();
                v0_[1] = veff_vec[0]->f_pw(0).real() - veff_vec[1]->f_pw(0).real();
            }

            vphi_ = mdarray<double_complex, 2>(gkvec__.num_gvec_fft(), fft_ctx_.num_fft_streams());
        }
        
        void apply(Wave_functions<false>& hphi__, int idx0__, int n__)
        {
            PROFILE_WITH_TIMER("sirius::Hloc_operator::apply");

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

        void apply(int ispn__, Wave_functions<false>& hphi__, int idx0__, int n__)
        {
            PROFILE_WITH_TIMER("sirius::Hloc_operator::apply");

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
                        STOP();
                        scale_matrix_rows_gpu(fft_ctx_.fft(thread_id)->local_size(), 1,
                                              fft_ctx_.fft(thread_id)->buffer<GPU>(), veff_.at<GPU>());

                        #else
                        TERMINATE_NO_GPU
                        #endif
                    }
                    else
                    {
                        for (int ir = 0; ir < fft_ctx_.fft(thread_id)->local_size(); ir++)
                            fft_ctx_.fft(thread_id)->buffer(ir) *= veff_vec_(ir, ispn__);
                    }
                    /* V(r)phi(r) -> [V*phi](G) */
                    fft_ctx_.fft(thread_id)->transform<-1>(gkvec_, &vphi_(0, thread_id));

                    /* add kinetic energy */
                    for (int ig = 0; ig < gkvec_.num_gvec_fft(); ig++)
                    {
                        hphi__[i][ig] = hphi__[i][ig] * pw_ekin_[ig] + vphi_(ig, thread_id);
                    }
                }
            }
            /* restore the nested flag */
            omp_set_nested(nested);

            hphi__.swap_backward(idx0__, n__);
        }

        inline double v0(int ispn__)
        {
            return v0_[ispn__];
        }
};

};

#endif
