#include <thread>
#include <mutex>
#include "density.h"

namespace sirius {

#ifdef __GPU
extern "C" void update_it_density_matrix_1_gpu(int fft_size, 
                                               int ispin,
                                               cuDoubleComplex const* psi_it, 
                                               double wt, 
                                               double* it_density_matrix);
#endif

void Density::add_k_point_contribution_it(K_point* kp__, occupied_bands_descriptor const& occupied_bands__)
{
    PROFILE();

    Timer t("sirius::Density::add_k_point_contribution_it");
    
    if (occupied_bands__.idx_bnd_loc.size() == 0) return;
    
    int num_spins = parameters_.num_spins();
    int num_mag_dims = parameters_.num_mag_dims();
    int num_fv_states = parameters_.num_fv_states();
    double omega = unit_cell_.omega();

    mdarray<double, 3> it_density_matrix(ctx_.fft(0)->size(), parameters_.num_mag_dims() + 1, ctx_.num_fft_threads());
    it_density_matrix.zero();

    #ifdef __GPU
    mdarray<int, 1> fft_index;
    mdarray<double_complex, 1> pw_buf;
    mdarray<double, 2> it_density_matrix_gpu;
    if (parameters_.processing_unit() == GPU && ctx_.gpu_thread_id() >= 0)
    {
        ctx_.fft(ctx_.gpu_thread_id())->allocate_on_device();
        /* move fft index to GPU */
        fft_index = mdarray<int, 1>(const_cast<int*>(kp__->gkvec().index_map()), kp__->num_gkvec());
        fft_index.allocate_on_device();
        fft_index.copy_to_device();

        /* allocate space for plane-wave expansion coefficients */
        pw_buf = mdarray<double_complex, 1>(nullptr, kp__->num_gkvec()); 
        pw_buf.allocate_on_device();

        /* density on GPU */
        it_density_matrix_gpu = mdarray<double, 2>(&it_density_matrix(0, 0, ctx_.gpu_thread_id()), ctx_.fft(0)->size(), parameters_.num_mag_dims() + 1);
        it_density_matrix_gpu.allocate_on_device();
        it_density_matrix_gpu.zero_on_device();
    }
    #endif
    mdarray<double_complex, 2> psi_it;

    mdarray<double, 1> timers(ctx_.num_fft_threads());
    timers.zero();
    mdarray<int, 1> timer_counts(ctx_.num_fft_threads());
    timer_counts.zero();

    /* save omp_nested flag */
    int nested = omp_get_nested();
    omp_set_nested(1);

    int wf_pw_offset = kp__->wf_pw_offset();
    double t0 = -Utils::current_time();
    #pragma omp parallel num_threads(ctx_.num_fft_threads())
    {
        int thread_id = omp_get_thread_num();
        if (num_mag_dims == 3) psi_it = mdarray<double_complex, 2>(ctx_.fft(thread_id)->size(), num_spins);

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < occupied_bands__.num_occupied_bands_local(); i++)
        {
            int j    = occupied_bands__.idx_bnd_glob[i];
            int jloc = occupied_bands__.idx_bnd_loc[i];
            double w = occupied_bands__.weight[i] / omega;
            double t1 = omp_get_wtime();

            if (thread_id == ctx_.gpu_thread_id() && parameters_.processing_unit() == GPU)
            {
                #ifdef __GPU
                if (num_mag_dims == 3)
                {
                    TERMINATE("this should be implemented");
                }
                else
                {
                    int ispn = (j < num_fv_states) ? 0 : 1;
                    
                    /* copy pw coefficients to GPU */
                    mdarray<double_complex, 1>(kp__->spinor_wave_functions(ispn).at<CPU>(wf_pw_offset, jloc),
                                               pw_buf.at<GPU>(), kp__->num_gkvec()).copy_to_device();
                    
                    ctx_.fft(thread_id)->input_on_device(kp__->num_gkvec(), fft_index.at<GPU>(), pw_buf.at<GPU>());
                    ctx_.fft(thread_id)->transform(1);
                    
                    update_it_density_matrix_1_gpu(ctx_.fft(thread_id)->size(), ispn, ctx_.fft(thread_id)->buffer<GPU>(), w,
                                                   it_density_matrix_gpu.at<GPU>());
                }
                #endif
            }
            else
            {
                if (num_mag_dims == 3)
                {   
                    /* transform both components of the spinor state */
                    for (int ispn = 0; ispn < num_spins; ispn++)
                    {
                        ctx_.fft(thread_id)->input(kp__->num_gkvec(), kp__->gkvec().index_map(), 
                                                   kp__->spinor_wave_functions(ispn).at<CPU>(wf_pw_offset, jloc));
                        ctx_.fft(thread_id)->transform(1, kp__->gkvec().z_sticks_coord());
                        ctx_.fft(thread_id)->output(&psi_it(0, ispn));
                    }
                    for (int ir = 0; ir < ctx_.fft(thread_id)->size(); ir++)
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
                    ctx_.fft(thread_id)->input(kp__->num_gkvec(), kp__->gkvec().index_map(), 
                                               kp__->spinor_wave_functions(ispn).at<CPU>(wf_pw_offset, jloc));
                    ctx_.fft(thread_id)->transform(1, kp__->gkvec().z_sticks_coord());
                    
                    #pragma omp parallel for schedule(static) num_threads(ctx_.fft(thread_id)->num_fft_workers())
                    for (int ir = 0; ir < ctx_.fft(thread_id)->size(); ir++)
                    {
                        auto z = ctx_.fft(thread_id)->buffer(ir);
                        it_density_matrix(ir, ispn, thread_id) += w * (std::pow(z.real(), 2) + std::pow(z.imag(), 2));
                    }
                }
            }
            timers(thread_id) += (omp_get_wtime() - t1);
            timer_counts(thread_id)++;
        }
    }
    t0 += Utils::current_time();

    /* restore the nested flag */
    omp_set_nested(nested);

    #ifdef __GPU
    if (parameters_.processing_unit() == GPU && ctx_.gpu_thread_id() >= 0)
    {
        it_density_matrix_gpu.copy_to_host();
        it_density_matrix_gpu.deallocate_on_device();
        ctx_.fft(ctx_.gpu_thread_id())->deallocate_on_device();
    }
    #endif
    
    double t1 = -Utils::current_time();
    /* switch from real density matrix to density and magnetization */
    switch (parameters_.num_mag_dims())
    {
        case 3:
        {
            for (int i = 0; i < ctx_.num_fft_threads(); i++)
            {
                for (int ir = 0; ir < ctx_.fft(0)->size(); ir++)
                {
                    magnetization_[1]->f_it(ir) += it_density_matrix(ir, 2, i);
                    magnetization_[2]->f_it(ir) += it_density_matrix(ir, 3, i);
                }
            }
        }
        case 1:
        {
            for (int i = 0; i < ctx_.num_fft_threads(); i++)
            {
                for (int ir = 0; ir < ctx_.fft(0)->size(); ir++)
                {
                    rho_->f_it(ir) += (it_density_matrix(ir, 0, i) + it_density_matrix(ir, 1, i));
                    magnetization_[0]->f_it(ir) += (it_density_matrix(ir, 0, i) - it_density_matrix(ir, 1, i));
                }
            }
            break;
        }
        case 0:
        {
            #pragma omp parallel
            {
                for (int i = 0; i < ctx_.num_fft_threads(); i++)
                {
                    #pragma omp for schedule(static)
                    for (int ir = 0; ir < ctx_.fft(0)->size(); ir++) rho_->f_it(ir) += it_density_matrix(ir, 0, i);
                }
            }
        }
    }
    t1 += Utils::current_time();

    if (kp__->comm().rank() == 0)
    {
        std::cout << "---------------------------------" << std::endl;
        std::cout << "thread_id  | fft       | perf    " << std::endl;
        std::cout << "---------------------------------" << std::endl;
        for (int i = 0; i < ctx_.num_fft_threads(); i++)
        {
            printf("   %2i      | %8.4f  | %8.2f\n", i, timers(i), (timer_counts(i) == 0) ? 0 : timer_counts(i) / timers(i));
        }
        std::cout << "---------------------------------" << std::endl;
        std::cout << "final reduction: " << t1 << " sec" << std::endl;
        std::cout << "main summation of " << occupied_bands__.num_occupied_bands_local() << " bands is done in " << t0 << "sec." << std::endl;
    }
}


void Density::add_k_point_contribution_it_pfft(K_point* kp__, occupied_bands_descriptor const& occupied_bands__)
{
    PROFILE();

    Timer t("sirius::Density::add_k_point_contribution_it_pfft");
    STOP();
    
    //if (occupied_bands__.idx_bnd_loc.size() == 0) return;
    //
    //mdarray<double, 2> it_density_matrix(fft_->local_size(), parameters_.num_mag_dims() + 1);
    //it_density_matrix.zero();
    //
    //int num_spins     = parameters_.num_spins();
    //int num_mag_dims  = parameters_.num_mag_dims();
    //int num_fv_states = parameters_.num_fv_states();
    //double omega      = unit_cell_.omega();
    //int wf_pw_offset  = kp__->wf_pw_offset();
    //
    //mdarray<double_complex, 2> psi_it(fft_->local_size(), num_spins);

    //splindex<block> spl_gkvec(kp__->num_gkvec(), kp__->num_ranks_row(), kp__->rank_row());

    //assert(kp__->spinor_wave_functions(0).num_rows_local() == (int)spl_gkvec.local_size());

    //auto a2a = kp__->comm_row().map_alltoall(spl_gkvec.counts(), kp__->gkvec().counts());
    //std::vector<double_complex> buf(kp__->gkvec().num_gvec_loc());

    //for (int i = 0; i < occupied_bands__.num_occupied_bands_local(); i++)
    //{
    //    int j    = occupied_bands__.idx_bnd_glob[i];
    //    int jloc = occupied_bands__.idx_bnd_loc[i];
    //    double w = occupied_bands__.weight[i] / omega;

    //    if (num_mag_dims == 3)
    //    {
    //        STOP();
    //        ///* transform both components of the spinor state */
    //        //for (int ispn = 0; ispn < num_spins; ispn++)
    //        //{
    //        //    fft->input(kp__->num_gkvec(), kp__->gkvec().index_map(), 
    //        //               kp__->spinor_wave_functions(ispn).at<CPU>(wf_pw_offset, jloc), thread_id);
    //        //    fft->transform(1, thread_id);
    //        //    fft->output(&psi_it(0, ispn), thread_id);
    //        //}
    //        //for (int ir = 0; ir < fft->size(); ir++)
    //        //{
    //        //    auto z0 = psi_it(ir, 0) * std::conj(psi_it(ir, 0)) * w;
    //        //    auto z1 = psi_it(ir, 1) * std::conj(psi_it(ir, 1)) * w;
    //        //    auto z2 = psi_it(ir, 0) * std::conj(psi_it(ir, 1)) * w;
    //        //    it_density_matrix(ir, 0, thread_id) += std::real(z0);
    //        //    it_density_matrix(ir, 1, thread_id) += std::real(z1);
    //        //    it_density_matrix(ir, 2, thread_id) += 2.0 * std::real(z2);
    //        //    it_density_matrix(ir, 3, thread_id) -= 2.0 * std::imag(z2);
    //        //}
    //    }
    //    else
    //    {
    //        /* transform only single compopnent */
    //        int ispn = (j < num_fv_states) ? 0 : 1;
    //        
    //        Timer t1("fft|comm");
    //        kp__->comm_row().alltoall(kp__->spinor_wave_functions(ispn).at<CPU>(wf_pw_offset, jloc),
    //                                  &a2a.sendcounts[0], &a2a.sdispls[0],
    //                                  &buf[0],
    //                                  &a2a.recvcounts[0], &a2a.rdispls[0]);
    //        t1.stop();
    //        fft_->input(kp__->gkvec().num_gvec_loc(), kp__->gkvec().index_map(), &buf[0]);
    //        fft_->transform(1, kp__->gkvec().z_sticks_coord());
    //        fft_->output(&psi_it(0, ispn));
    //        
    //        #pragma omp parallel for schedule(static)
    //        for (int ir = 0; ir < fft_->local_size(); ir++)
    //        {
    //            double_complex z = psi_it(ir, ispn);
    //            it_density_matrix(ir, ispn) += w * (std::pow(z.real(), 2) + std::pow(z.imag(), 2));
    //        }
    //    }
    //}

    ///* switch from real density matrix to density and magnetization */
    //switch (parameters_.num_mag_dims())
    //{
    //    case 3:
    //    {
    //        for (int ir = 0; ir < fft_->local_size(); ir++)
    //        {
    //            magnetization_[1]->f_it(ir) += it_density_matrix(ir, 2);
    //            magnetization_[2]->f_it(ir) += it_density_matrix(ir, 3);
    //        }
    //    }
    //    case 1:
    //    {
    //        for (int ir = 0; ir < fft_->local_size(); ir++)
    //        {
    //            rho_->f_it(ir) += (it_density_matrix(ir, 0) + it_density_matrix(ir, 1));
    //            magnetization_[0]->f_it(ir) += (it_density_matrix(ir, 0) - it_density_matrix(ir, 1));
    //        }
    //        break;
    //    }
    //    case 0:
    //    {
    //        for (int ir = 0; ir < fft_->local_size(); ir++) rho_->f_it(ir) += it_density_matrix(ir, 0);
    //    }
    //}
}

};
