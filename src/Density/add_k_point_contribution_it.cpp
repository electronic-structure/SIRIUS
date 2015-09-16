#include <thread>
#include <mutex>
#include "density.h"

namespace sirius {

#ifdef __GPU

extern "C" void update_it_density_matrix_1_gpu(int fft_size, 
                                               int ispin,
                                               cuDoubleComplex const* psi_it, 
                                               double const* wt, 
                                               double* it_density_matrix);
#endif

void Density::add_k_point_contribution_it(K_point* kp__, occupied_bands_descriptor const& occupied_bands__)
{
    PROFILE();

    Timer t("sirius::Density::add_k_point_contribution_it");
    
    if (occupied_bands__.idx_bnd_loc.size() == 0) return;
    
    /* index of the occupied bands */
    int idx_band = 0;
    std::mutex idx_band_mutex;

    int num_fft_threads = -1;
    switch (parameters_.processing_unit())
    {
        case CPU:
        {
            num_fft_threads = parameters_.num_fft_threads();
            break;
        }
        case GPU:
        {
            num_fft_threads = std::min(parameters_.num_fft_threads() + 1, Platform::max_num_threads());
            break;
        }
    }

    mdarray<double, 3> it_density_matrix(fft_->size(), parameters_.num_mag_dims() + 1, num_fft_threads);
    it_density_matrix.zero();
    
    #ifdef __GPU
    mdarray<double, 2> it_density_matrix_gpu;
    /* last thread is doing cuFFT */
    if (parameters_.processing_unit() == GPU && num_fft_threads > 1)
    {
        it_density_matrix_gpu = mdarray<double, 2>(&it_density_matrix(0, 0, num_fft_threads - 1), fft_->size(), parameters_.num_mag_dims() + 1);
        it_density_matrix_gpu.allocate_on_device();
        it_density_matrix_gpu.zero_on_device();
    }
    auto fft_gpu = ctx_.fft_gpu();
    if (fft_gpu->num_fft() != 1) TERMINATE("Current implementation requires batch size of 1");
    #endif

    std::vector<std::thread> fft_threads;

    auto fft = ctx_.fft();
    int num_spins = parameters_.num_spins();
    int num_mag_dims = parameters_.num_mag_dims();
    int num_fv_states = parameters_.num_fv_states();
    double omega = unit_cell_.omega();

    for (int thread_id = 0; thread_id < num_fft_threads; thread_id++)
    {
        if (thread_id == (num_fft_threads - 1) && num_fft_threads > 1 && parameters_.processing_unit() == GPU)
        {
            #ifdef __GPU
            fft_threads.push_back(std::thread([thread_id, kp__, fft_gpu, &idx_band, &idx_band_mutex, num_spins, num_mag_dims,
                                               num_fv_states, omega, &occupied_bands__, &it_density_matrix_gpu]()
            {
                Timer t("sirius::Density::add_k_point_contribution_it|gpu");

                int wf_pw_offset = kp__->wf_pw_offset();
                
                /* move fft index to GPU */
                mdarray<int, 1> fft_index(const_cast<int*>(kp__->gkvec().index_map()), kp__->num_gkvec());
                fft_index.allocate_on_device();
                fft_index.copy_to_device();

                int nfft_max = fft_gpu->num_fft();
 
                /* allocate work area array */
                mdarray<char, 1> work_area(nullptr, fft_gpu->work_area_size());
                work_area.allocate_on_device();
                fft_gpu->set_work_area_ptr(work_area.at<GPU>());
                
                /* allocate space for plane-wave expansion coefficients */
                mdarray<double_complex, 2> psi_pw_gpu(nullptr, kp__->num_gkvec(), nfft_max); 
                psi_pw_gpu.allocate_on_device();
                
                /* allocate space for spinor components */
                mdarray<double_complex, 3> psi_it_gpu(nullptr, fft_gpu->size(), nfft_max, num_spins);
                psi_it_gpu.allocate_on_device();
                
                /* allocate space for weights */
                mdarray<double, 1> w(nfft_max);
                w.allocate_on_device();

                bool done = false;

                while (!done)
                {
                    idx_band_mutex.lock();
                    int i = idx_band;
                    if (idx_band + nfft_max > occupied_bands__.num_occupied_bands_local())
                    {
                        done = true;
                    }
                    else
                    {
                        idx_band += nfft_max;
                    }
                    idx_band_mutex.unlock();

                    if (!done)
                    {
                        int j    = occupied_bands__.idx_bnd_glob[i];
                        int jloc = occupied_bands__.idx_bnd_loc[i];
                        if (num_mag_dims == 3)
                        {
                            TERMINATE("this should be implemented");
                        }
                        else
                        {
                            int ispn = (j < num_fv_states) ? 0 : 1;
                            w(0) = occupied_bands__.weight[i] / omega;

                            mdarray<double_complex, 1>(kp__->spinor_wave_functions(ispn).at<CPU>(wf_pw_offset, jloc),
                                                       psi_pw_gpu.at<GPU>(), kp__->num_gkvec()).copy_to_device();
                            w.copy_to_device();
                            fft_gpu->batch_load(kp__->num_gkvec(), fft_index.at<GPU>(), psi_pw_gpu.at<GPU>(), 
                                                psi_it_gpu.at<GPU>(0, 0, ispn));
                            fft_gpu->transform(1, psi_it_gpu.at<GPU>(0, 0, ispn));
                            
                            update_it_density_matrix_1_gpu(fft_gpu->size(), ispn, psi_it_gpu.at<GPU>(), w.at<GPU>(),
                                                           it_density_matrix_gpu.at<GPU>(0, 0));
                        }


                        //==for (int ispn = 0; ispn < num_spins; ispn++)
                        //=={
                        //==    /* copy PW coefficients to GPU */
                        //==    for (int j = 0; j < nfft_max; j++)
                        //==    {
                        //==        w(j) = occupied_bands[i + j].second / omega;

                        //==        // TODO: use mdarray wrapper for this
                        //==        cublas_set_vector(kp->num_gkvec(), sizeof(double_complex), 
                        //==                          &kp->spinor_wave_function(wf_pw_offset, occupied_bands[i + j].first, ispn), 1, 
                        //==                          psi_pw_gpu.at<GPU>(0, j), 1);
                        //==    }
                        //==    w.copy_to_device();
                        //==    
                        //==    /* set PW coefficients into proper positions inside FFT buffer */
                        //==    fft_gpu->batch_load(kp->num_gkvec(), fft_index.at<GPU>(), psi_pw_gpu.at<GPU>(0, 0), 
                        //==                        psi_it_gpu.at<GPU>(0, 0, ispn));

                        //==    /* execute batch FFT */
                        //==    fft_gpu->transform(1, psi_it_gpu.at<GPU>(0, 0, ispn));
                        //==}

                        //==update_it_density_matrix_gpu(fft_gpu->size(), nfft_max, num_spins, num_mag_dims, 
                        //==                             psi_it_gpu.at<GPU>(), w.at<GPU>(),
                        //==                             it_density_matrix_gpu.at<GPU>(0, 0));
                    }
                }
            }));
            #else
            TERMINATE_NO_GPU
            #endif
        }
        else
        {
            fft_threads.push_back(std::thread([thread_id, kp__, fft, &idx_band, &idx_band_mutex, num_spins, num_mag_dims,
                                               num_fv_states, omega, &occupied_bands__, &it_density_matrix]()
            {
                bool done = false;

                int wf_pw_offset = kp__->wf_pw_offset();
                
                mdarray<double_complex, 2> psi_it(fft->size(), num_spins);

                while (!done)
                {
                    /* increment the band index */
                    idx_band_mutex.lock();
                    int i = idx_band;
                    if (idx_band >= occupied_bands__.num_occupied_bands_local())
                    {
                        done = true;
                    }
                    else
                    {
                        idx_band++;
                    }
                    idx_band_mutex.unlock();

                    if (!done)
                    {
                        int j    = occupied_bands__.idx_bnd_glob[i];
                        int jloc = occupied_bands__.idx_bnd_loc[i];
                        double w = occupied_bands__.weight[i] / omega;

                        if (num_mag_dims == 3)
                        {   
                            /* transform both components of the spinor state */
                            for (int ispn = 0; ispn < num_spins; ispn++)
                            {
                                fft->input(kp__->num_gkvec(), kp__->gkvec().index_map(), 
                                           kp__->spinor_wave_functions(ispn).at<CPU>(wf_pw_offset, jloc), thread_id);
                                fft->transform(1, kp__->gkvec().z_sticks_coord(), thread_id);
                                fft->output(&psi_it(0, ispn), thread_id);
                            }
                            for (int ir = 0; ir < fft->size(); ir++)
                            {
                                auto z0 = psi_it(ir, 0) * std::conj(psi_it(ir, 0)) * w;
                                auto z1 = psi_it(ir, 1) * std::conj(psi_it(ir, 1)) * w;
                                auto z2 = psi_it(ir, 0) * std::conj(psi_it(ir, 1)) * w;
                                it_density_matrix(ir, 0, thread_id) += std::real(z0);
                                it_density_matrix(ir, 1, thread_id) += std::real(z1);
                                it_density_matrix(ir, 2, thread_id) += 2.0 * std::real(z2);
                                it_density_matrix(ir, 3, thread_id) -= 2.0 * std::imag(z2);
                            }
                        }
                        else
                        {
                            /* transform only single compopnent */
                            int ispn = (j < num_fv_states) ? 0 : 1;
                            fft->input(kp__->num_gkvec(), kp__->gkvec().index_map(), 
                                       kp__->spinor_wave_functions(ispn).at<CPU>(wf_pw_offset, jloc), thread_id);
                            fft->transform(1, kp__->gkvec().z_sticks_coord(), thread_id);
                            fft->output(&psi_it(0, ispn), thread_id);

                            for (int ir = 0; ir < fft->size(); ir++)
                                it_density_matrix(ir, ispn, thread_id) += std::real(psi_it(ir, ispn) * std::conj(psi_it(ir, ispn))) * w;

                        }
                    }
                }
            }));
        }
    }

    for (auto& thread: fft_threads) thread.join();

    if (idx_band != (int)occupied_bands__.idx_bnd_loc.size()) 
    {
        std::stringstream s;
        s << "not all FFTs are executed" << std::endl
          << " number of wave-functions : " << occupied_bands__.idx_bnd_loc.size() << ", number of executed FFTs : " << idx_band;
        error_local(__FILE__, __LINE__, s);
    }

    #ifdef __GPU
    if (parameters_.processing_unit() == GPU && num_fft_threads > 1)
    {
        it_density_matrix_gpu.copy_to_host();
        it_density_matrix_gpu.deallocate_on_device();
    }
    #endif

    /* switch from real density matrix to density and magnetization */
    switch (parameters_.num_mag_dims())
    {
        case 3:
        {
            for (int i = 0; i < num_fft_threads; i++)
            {
                for (int ir = 0; ir < fft_->size(); ir++)
                {
                    magnetization_[1]->f_it(ir) += it_density_matrix(ir, 2, i);
                    magnetization_[2]->f_it(ir) += it_density_matrix(ir, 3, i);
                }
            }
        }
        case 1:
        {
            for (int i = 0; i < num_fft_threads; i++)
            {
                for (int ir = 0; ir < fft_->size(); ir++)
                {
                    rho_->f_it(ir) += (it_density_matrix(ir, 0, i) + it_density_matrix(ir, 1, i));
                    magnetization_[0]->f_it(ir) += (it_density_matrix(ir, 0, i) - it_density_matrix(ir, 1, i));
                }
            }
            break;
        }
        case 0:
        {
            for (int i = 0; i < num_fft_threads; i++)
            {
                for (int ir = 0; ir < fft_->size(); ir++) rho_->f_it(ir) += it_density_matrix(ir, 0, i);
            }
        }
    }
}


void Density::add_k_point_contribution_it_pfft(K_point* kp__, occupied_bands_descriptor const& occupied_bands__)
{
    PROFILE();

    Timer t("sirius::Density::add_k_point_contribution_it_pfft");
    
    if (occupied_bands__.idx_bnd_loc.size() == 0) return;
    
    mdarray<double, 2> it_density_matrix(fft_->local_size(), parameters_.num_mag_dims() + 1);
    it_density_matrix.zero();
    
    int num_spins     = parameters_.num_spins();
    int num_mag_dims  = parameters_.num_mag_dims();
    int num_fv_states = parameters_.num_fv_states();
    double omega      = unit_cell_.omega();
    int wf_pw_offset  = kp__->wf_pw_offset();
    
    mdarray<double_complex, 2> psi_it(fft_->local_size(), num_spins);

    splindex<block> spl_gkvec(kp__->num_gkvec(), kp__->num_ranks_row(), kp__->rank_row());

    assert(kp__->spinor_wave_functions(0).num_rows_local() == (int)spl_gkvec.local_size());

    auto a2a = kp__->comm_row().map_alltoall(spl_gkvec.counts(), kp__->gkvec().counts());
    std::vector<double_complex> buf(kp__->gkvec().num_gvec_loc());

    for (int i = 0; i < occupied_bands__.num_occupied_bands_local(); i++)
    {
        int j    = occupied_bands__.idx_bnd_glob[i];
        int jloc = occupied_bands__.idx_bnd_loc[i];
        double w = occupied_bands__.weight[i] / omega;

        if (num_mag_dims == 3)
        {
            STOP();
            ///* transform both components of the spinor state */
            //for (int ispn = 0; ispn < num_spins; ispn++)
            //{
            //    fft->input(kp__->num_gkvec(), kp__->gkvec().index_map(), 
            //               kp__->spinor_wave_functions(ispn).at<CPU>(wf_pw_offset, jloc), thread_id);
            //    fft->transform(1, thread_id);
            //    fft->output(&psi_it(0, ispn), thread_id);
            //}
            //for (int ir = 0; ir < fft->size(); ir++)
            //{
            //    auto z0 = psi_it(ir, 0) * std::conj(psi_it(ir, 0)) * w;
            //    auto z1 = psi_it(ir, 1) * std::conj(psi_it(ir, 1)) * w;
            //    auto z2 = psi_it(ir, 0) * std::conj(psi_it(ir, 1)) * w;
            //    it_density_matrix(ir, 0, thread_id) += std::real(z0);
            //    it_density_matrix(ir, 1, thread_id) += std::real(z1);
            //    it_density_matrix(ir, 2, thread_id) += 2.0 * std::real(z2);
            //    it_density_matrix(ir, 3, thread_id) -= 2.0 * std::imag(z2);
            //}
        }
        else
        {
            /* transform only single compopnent */
            int ispn = (j < num_fv_states) ? 0 : 1;
            
            Timer t1("fft|comm");
            kp__->comm_row().alltoall(kp__->spinor_wave_functions(ispn).at<CPU>(wf_pw_offset, jloc),
                                      &a2a.sendcounts[0], &a2a.sdispls[0],
                                      &buf[0],
                                      &a2a.recvcounts[0], &a2a.rdispls[0]);
            t1.stop();
            fft_->input(kp__->gkvec().num_gvec_loc(), kp__->gkvec().index_map(), &buf[0]);
            fft_->transform(1, kp__->gkvec().z_sticks_coord());
            fft_->output(&psi_it(0, ispn));
            
            #pragma omp parallel for schedule(static)
            for (int ir = 0; ir < fft_->local_size(); ir++)
            {
                double_complex z = psi_it(ir, ispn);
                it_density_matrix(ir, ispn) += w * (std::pow(z.real(), 2) + std::pow(z.imag(), 2));
            }
        }
    }

    /* switch from real density matrix to density and magnetization */
    switch (parameters_.num_mag_dims())
    {
        case 3:
        {
            for (int ir = 0; ir < fft_->local_size(); ir++)
            {
                magnetization_[1]->f_it(ir) += it_density_matrix(ir, 2);
                magnetization_[2]->f_it(ir) += it_density_matrix(ir, 3);
            }
        }
        case 1:
        {
            for (int ir = 0; ir < fft_->local_size(); ir++)
            {
                rho_->f_it(ir) += (it_density_matrix(ir, 0) + it_density_matrix(ir, 1));
                magnetization_[0]->f_it(ir) += (it_density_matrix(ir, 0) - it_density_matrix(ir, 1));
            }
            break;
        }
        case 0:
        {
            for (int ir = 0; ir < fft_->local_size(); ir++) rho_->f_it(ir) += it_density_matrix(ir, 0);
        }
    }
}

};
