#include <thread>
#include <mutex>
#include "band.h"
#include "debug.hpp"

namespace sirius {

/** \param [in] phi Input wave-functions [storage: CPU].
 *  \param [out] hphi Result of application of operator to the wave-functions [storage: CPU].
 */
void Band::apply_h_local_serial(K_point* kp__,
                                std::vector<double> const& effective_potential__,
                                std::vector<double> const& pw_ekin__,
                                int num_phi__,
                                matrix<double_complex> const& phi__,
                                matrix<double_complex>& hphi__)
{
    PROFILE();

    Timer t("sirius::Band::apply_h_local_serial");

    assert(phi__.size(0) == (size_t)kp__->num_gkvec() && hphi__.size(0) == (size_t)kp__->num_gkvec());
    assert(phi__.size(1) >= (size_t)num_phi__ && hphi__.size(1) >= (size_t)num_phi__);

    bool in_place = (&phi__ == &hphi__);

    #ifdef __GPU
    mdarray<int, 1> fft_index;
    mdarray<double_complex, 1> pw_buf;
    mdarray<double, 1> veff;
    mdarray<double, 1> pw_ekin;
    if (parameters_.processing_unit() == GPU && ctx_.gpu_thread_id() >= 0)
    {
        ctx_.fft_coarse(ctx_.gpu_thread_id())->allocate_on_device();
        /* move fft index to GPU */
        fft_index = mdarray<int, 1>(const_cast<int*>(kp__->gkvec_coarse().index_map()), kp__->num_gkvec());
        fft_index.allocate_on_device();
        fft_index.copy_to_device();

        /* allocate space for plane-wave expansion coefficients */
        pw_buf = mdarray<double_complex, 1>(nullptr, kp__->num_gkvec()); 
        pw_buf.allocate_on_device();
        
        /* copy effective potential to GPU */
        veff = mdarray<double, 1>(const_cast<double*>(&effective_potential__[0]), ctx_.fft_coarse(0)->local_size());
        veff.allocate_on_device();
        veff.copy_to_device();
        
        /* copy kinetic energy to GPU */
        pw_ekin = mdarray<double, 1>(const_cast<double*>(&pw_ekin__[0]), kp__->num_gkvec());
        pw_ekin.allocate_on_device();
        pw_ekin.copy_to_device();

    }
    #endif

    mdarray<double, 1> timers(ctx_.num_fft_threads());
    timers.zero();
    mdarray<int, 1> timer_counts(ctx_.num_fft_threads());
    timer_counts.zero();

    /* save omp_nested flag */
    int nested = omp_get_nested();
    omp_set_nested(1);
    #pragma omp parallel num_threads(ctx_.num_fft_threads())
    {
        int thread_id = omp_get_thread_num();

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < num_phi__; i++)
        {
            double t1 = omp_get_wtime();
            if (thread_id == ctx_.gpu_thread_id() && parameters_.processing_unit() == GPU)
            {
                #ifdef __GPU
                /* copy phi to GPU */
                cuda_copy_to_device(pw_buf.at<GPU>(), phi__.at<CPU>(0, i), kp__->num_gkvec() * sizeof(double_complex));

                /* set PW coefficients into proper positions inside FFT buffer */
                ctx_.fft_coarse(thread_id)->input_on_device(kp__->num_gkvec(), fft_index.at<GPU>(), pw_buf.at<GPU>());

                /* phi(G) *= Ekin(G) */
                scale_matrix_rows_gpu(kp__->num_gkvec(), 1, pw_buf.at<GPU>(), pw_ekin.at<GPU>());
                
                /* execute FFT */
                ctx_.fft_coarse(thread_id)->transform(1);
                
                /* multiply by potential */
                scale_matrix_rows_gpu(ctx_.fft_coarse(thread_id)->local_size(), 1,
                                      ctx_.fft_coarse(thread_id)->buffer<GPU>(), veff.at<GPU>());
                
                /* transform back */
                ctx_.fft_coarse(thread_id)->transform(-1);
                
                /* phi(G) += fft_buffer(G) */
                ctx_.fft_coarse(thread_id)->output_on_device(kp__->num_gkvec(), fft_index.at<GPU>(), pw_buf.at<GPU>(), 1.0);
                
                /* copy final hphi to CPU */
                cuda_copy_to_host(hphi__.at<CPU>(0, i), pw_buf.at<GPU>(), kp__->num_gkvec() * sizeof(double_complex));
                #endif
            }
            else
            {
                ctx_.fft_coarse(thread_id)->input(kp__->num_gkvec(), kp__->gkvec_coarse().index_map(), &phi__(0, i));
                /* phi(G) -> phi(r) */
                ctx_.fft_coarse(thread_id)->transform(1, kp__->gkvec_coarse().z_sticks_coord());
                /* multiply by effective potential */
                for (int ir = 0; ir < ctx_.fft_coarse(thread_id)->size(); ir++) ctx_.fft_coarse(thread_id)->buffer(ir) *= effective_potential__[ir];
                /* V(r)phi(r) -> [V*phi](G) */
                ctx_.fft_coarse(thread_id)->transform(-1, kp__->gkvec_coarse().z_sticks_coord());

                if (in_place)
                {
                    /* psi(G) -> 0.5 * |G|^2 * psi(G) */
                    for (int igk = 0; igk < kp__->num_gkvec(); igk++) hphi__(igk, i) *= pw_ekin__[igk];
                    ctx_.fft_coarse(thread_id)->output(kp__->num_gkvec(), kp__->gkvec_coarse().index_map(), &hphi__(0, i), 1.0);
                }
                else
                {
                    ctx_.fft_coarse(thread_id)->output(kp__->num_gkvec(), kp__->gkvec_coarse().index_map(), &hphi__(0, i));
                    for (int igk = 0; igk < kp__->num_gkvec(); igk++) hphi__(igk, i) += phi__(igk, i) * pw_ekin__[igk];
                }
            }
            timers(thread_id) += (omp_get_wtime() - t1);
            timer_counts(thread_id)++;
        }
    }
    /* restore the nested flag */
    omp_set_nested(nested);

    //== if (kp__->comm().rank() == 0)
    //== {
    //==     std::cout << "---------------------------------" << std::endl;
    //==     std::cout << "thread_id  | fft       | perf    " << std::endl;
    //==     std::cout << "---------------------------------" << std::endl;
    //==     for (int i = 0; i < ctx_.num_fft_threads(); i++)
    //==     {
    //==         printf("   %2i      | %8.4f  | %8.2f\n", i, timers(i), (timer_counts(i) == 0) ? 0 : timer_counts(i) / timers(i));
    //==     }
    //==     std::cout << "---------------------------------" << std::endl;
    //== }

    #ifdef __GPU
    if (parameters_.processing_unit() == GPU && ctx_.gpu_thread_id() >= 0)
    {
        ctx_.fft_coarse(ctx_.gpu_thread_id())->deallocate_on_device();
    }
    #endif
}

};

