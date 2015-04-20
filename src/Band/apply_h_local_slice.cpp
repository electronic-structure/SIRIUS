#include <thread>
#include <mutex>
#include "band.h"
#include "debug.hpp"

namespace sirius {

/** \param [in] phi Input wave-functions [storage: CPU].
 *  \param [out] hphi Result of application of operator to the wave-functions [storage: CPU].
 */
void Band::apply_h_local_slice(K_point* kp__,
                               std::vector<double> const& effective_potential__,
                               std::vector<double> const& pw_ekin__,
                               int num_phi__,
                               matrix<double_complex> const& phi__,
                               matrix<double_complex>& hphi__)
{
    LOG_FUNC_BEGIN();

    Timer t("sirius::Band::apply_h_local_slice");

    assert(phi__.size(0) == (size_t)kp__->num_gkvec() && hphi__.size(0) == (size_t)kp__->num_gkvec());
    assert(phi__.size(1) >= (size_t)num_phi__ && hphi__.size(1) >= (size_t)num_phi__);

    bool in_place = (&phi__ == &hphi__);

    auto pu = parameters_.processing_unit();

    auto fft = parameters_.fft_coarse();
    #ifdef _GPU_
    FFT3D<GPU>* fft_gpu = parameters_.fft_gpu_coarse();
    #endif

    int num_fft_threads = -1;
    switch (pu)
    {
        case CPU:
        {
            num_fft_threads = fft->num_fft_threads();
            break;
        }
        case GPU:
        {
            num_fft_threads = std::min(fft->num_fft_threads() + 1, Platform::max_num_threads());
            break;
        }
    }

    std::vector<std::thread> fft_threads;

    /* index of the wave-function */
    int idx_phi = 0;
    std::mutex idx_phi_mutex;

    int count_fft_cpu = 0;
    #ifdef _GPU_
    int count_fft_gpu = 0;
    #endif

    mdarray<double, 1> timers(Platform::max_num_threads());
    timers.zero();
    
    for (int thread_id = 0; thread_id < num_fft_threads; thread_id++)
    {
        if (thread_id == num_fft_threads - 1 && num_fft_threads > 1 && pu == GPU)
        {
            #ifdef _GPU_
            fft_threads.push_back(std::thread([thread_id, num_phi__, &idx_phi, &idx_phi_mutex, &fft_gpu, kp__, &phi__, 
                                               &hphi__, &effective_potential__, &pw_ekin__, &count_fft_gpu, &timers]()
            {
                Timer t("sirius::Band::apply_h_local_slice|gpu");
                timers(thread_id) = -omp_get_wtime();
                
                /* move fft index to GPU */
                mdarray<int, 1> fft_index(const_cast<int*>(kp__->fft_index_coarse()), kp__->num_gkvec());
                fft_index.allocate_on_device();
                fft_index.copy_to_device();

                /* allocate work area array */
                mdarray<char, 1> work_area(nullptr, fft_gpu->work_area_size());
                work_area.allocate_on_device();
                fft_gpu->set_work_area_ptr(work_area.at<GPU>());
                
                /* allocate space for plane-wave expansion coefficients */
                mdarray<double_complex, 2> pw_buf(nullptr, kp__->num_gkvec(), fft_gpu->num_fft()); 
                pw_buf.allocate_on_device();
                
                /* allocate space for FFT buffers */
                matrix<double_complex> fft_buf(nullptr, fft_gpu->size(), fft_gpu->num_fft()); 
                fft_buf.allocate_on_device();
                
                mdarray<double, 1> veff_gpu((double*)&effective_potential__[0], fft_gpu->size());
                veff_gpu.allocate_on_device();
                veff_gpu.copy_to_device();

                mdarray<double, 1> pw_ekin_gpu((double*)&pw_ekin__[0], kp__->num_gkvec());
                pw_ekin_gpu.allocate_on_device();
                pw_ekin_gpu.copy_to_device();

                Timer t1("sirius::Band::apply_h_local_slice|gpu_loop");
                bool done = false;
                while (!done)
                {
                    /* increment the band index */
                    idx_phi_mutex.lock();
                    int i = idx_phi;
                    if (idx_phi + fft_gpu->num_fft() > num_phi__) 
                    {
                        done = true;
                    }
                    else
                    {
                        idx_phi += fft_gpu->num_fft();
                        count_fft_gpu += fft_gpu->num_fft();
                    }
                    idx_phi_mutex.unlock();

                    if (!done)
                    {
                        int size_of_panel = int(kp__->num_gkvec() * fft_gpu->num_fft() * sizeof(double_complex));

                        /* copy phi to GPU */
                        cuda_copy_to_device(pw_buf.at<GPU>(), phi__.at<CPU>(0, i), size_of_panel);

                        /* set PW coefficients into proper positions inside FFT buffer */
                        fft_gpu->batch_load(kp__->num_gkvec(), fft_index.at<GPU>(), pw_buf.at<GPU>(), fft_buf.at<GPU>());

                        /* phi(G) *= Ekin(G) */
                        scale_matrix_rows_gpu(kp__->num_gkvec(), fft_gpu->num_fft(), pw_buf.at<GPU>(), pw_ekin_gpu.at<GPU>());
                        
                        /* execute batch FFT */
                        fft_gpu->transform(1, fft_buf.at<GPU>());
                        
                        /* multiply by potential */
                        scale_matrix_rows_gpu(fft_gpu->size(), fft_gpu->num_fft(), fft_buf.at<GPU>(), veff_gpu.at<GPU>());
                        
                        /* transform back */
                        fft_gpu->transform(-1, fft_buf.at<GPU>());
                        
                        /* phi(G) += fft_buffer(G) */
                        fft_gpu->batch_unload(kp__->num_gkvec(), fft_index.at<GPU>(), fft_buf.at<GPU>(),
                                              pw_buf.at<GPU>(), 1.0);
                        
                        /* copy final hphi to CPU */
                        cuda_copy_to_host(hphi__.at<CPU>(0, i), pw_buf.at<GPU>(), size_of_panel);
                    }
                }
                timers(thread_id) += omp_get_wtime();
            }));
            #else
            TERMINATE_NO_GPU
            #endif
        }
        else
        {
            fft_threads.push_back(std::thread([thread_id, num_phi__, &idx_phi, &idx_phi_mutex, &fft, kp__, &phi__, 
                                               &hphi__, &effective_potential__, &pw_ekin__, &count_fft_cpu, in_place, &timers]()
            {
                timers(thread_id) = -omp_get_wtime();
                bool done = false;
                while (!done)
                {
                    /* increment the band index */
                    idx_phi_mutex.lock();
                    int i = idx_phi;
                    if (idx_phi + 1 > num_phi__) 
                    {
                        done = true;
                    }
                    else
                    {
                        idx_phi++;
                        count_fft_cpu++;
                    }
                    idx_phi_mutex.unlock();
                
                    if (!done)
                    {
                        fft->input(kp__->num_gkvec(), kp__->fft_index_coarse(), &phi__(0, i), thread_id);
                        /* phi(G) -> phi(r) */
                        fft->transform(1, thread_id);
                        /* multiply by effective potential */
                        for (int ir = 0; ir < fft->size(); ir++) fft->buffer(ir, thread_id) *= effective_potential__[ir];
                        /* V(r)phi(r) -> [V*phi](G) */
                        fft->transform(-1, thread_id);

                        if (in_place)
                        {
                            for (int igk = 0; igk < kp__->num_gkvec(); igk++) hphi__(igk, i) *= pw_ekin__[igk];
                            fft->output(kp__->num_gkvec(), kp__->fft_index_coarse(), &hphi__(0, i), thread_id, 1.0);
                        }
                        else
                        {
                            fft->output(kp__->num_gkvec(), kp__->fft_index_coarse(), &hphi__(0, i), thread_id);
                            for (int igk = 0; igk < kp__->num_gkvec(); igk++) hphi__(igk, i) += phi__(igk, i) * pw_ekin__[igk];
                        }
                    }
                }
                timers(thread_id) += omp_get_wtime();
            }));
        }
    }
    for (auto& thread: fft_threads) thread.join();

    //== if (kp__->comm().rank() == 0)
    //== {
    //==     std::cout << "---------------------" << std::endl;
    //==     std::cout << "thread_id  | fft     " << std::endl;
    //==     std::cout << "---------------------" << std::endl;
    //==     for (int i = 0; i < num_fft_threads; i++)
    //==     {
    //==         printf("   %2i      | %8.4f  \n", i, timers(i));
    //==     }
    //==     std::cout << "---------------------" << std::endl;
    //== }
    
    //== if (kp__->comm().rank() == 0) DUMP("CPU / GPU fft count : %i %i", count_fft_cpu, count_fft_gpu);

    LOG_FUNC_END();
}

};

