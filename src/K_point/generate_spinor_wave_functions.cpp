#include "k_point.h"

namespace sirius {

void K_point::generate_spinor_wave_functions()
{
    PROFILE_WITH_TIMER("sirius::K_point::generate_spinor_wave_functions");

    double_complex alpha(1, 0);
    double_complex beta(0, 0);

    if (use_second_variation) 
    {
        if (!ctx_.need_sv()) {
            /* copy eigen-states and exit */
            spinor_wave_functions(0).copy_from(fv_states(), 0, ctx_.num_fv_states());
            return;
        }

        int nfv = ctx_.num_fv_states();
        int nbnd = (ctx_.num_mag_dims() == 3) ? ctx_.num_bands() : nfv;

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            int s, o;

            if (ctx_.num_mag_dims() == 3) {
                /* in case of non-collinear magnetism sv_eigen_vectors is a single 2Nx2N matrix */
                s = 0;
                o = ispn * nfv; /* offset for spin up is 0, for spin dn is nfv */
            } else { 
                /* sv_eigen_vectors is composed of two NxN matrices */
                s = ispn;
                o = 0;
            }
            ///* multiply consecutively up and dn blocks */
            //linalg<CPU>::gemm(0, 0, wf_size(), nbnd, nfv, double_complex(1, 0), fv_states<true>().prime(), 0, 0,
            //                  sv_eigen_vectors_[s], o, 0, double_complex(0, 0), spinor_wave_functions<true>(ispn).prime(), 0, 0);
            spinor_wave_functions(ispn).transform_from(fv_states(), nfv, sv_eigen_vectors_[s], o, nbnd);
        }
 
        ///* serial version */
        //if (num_ranks() == 1)
        //{
        //    //spinor_wave_functions_.zero();
        //    if (ctx_.processing_unit() == GPU)
        //    {
        //        #ifdef __GPU
        //        //fv_states_.allocate_on_device();
        //        //fv_states_.copy_to_device();
        //        #endif
        //    }
 
        //    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
        //    {
        //        if (ctx_.num_mag_dims() != 3)
        //        {
        //            if (ctx_.processing_unit() == GPU)
        //            {
        //                #ifdef __GPU
        //                STOP();
        //                //sv_eigen_vectors_[ispn].allocate_on_device();
        //                //sv_eigen_vectors_[ispn].copy_to_device();
        //                //spinor_wave_functions_[ispn].allocate_on_device();
 
        //                //linalg<GPU>::gemm(0, 0, wf_size(), nfv, nfv, &alpha, fv_states_.at<GPU>(), fv_states_.ld(), 
        //                //                  sv_eigen_vectors_[ispn].at<GPU>(), sv_eigen_vectors_[ispn].ld(),
        //                //                  &beta, spinor_wave_functions_[ispn].at<GPU>(), spinor_wave_functions_[ispn].ld());
 
        //                //sv_eigen_vectors_[ispn].deallocate_on_device();
        //                //spinor_wave_functions_[ispn].copy_to_host();
        //                //spinor_wave_functions_[ispn].deallocate_on_device();
        //                #else
        //                TERMINATE_NO_GPU
        //                #endif
        //            }
        //        }
        //    }
        //    if (ctx_.processing_unit() == GPU)
        //    {
        //        #ifdef __GPU
        //        //fv_states_.deallocate_on_device();
        //        #endif
        //    }
        //}
    } else {
        TERMINATE_NOT_IMPLEMENTED;
    }
}

};
