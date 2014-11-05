#include <thread>
#include <mutex>
#include "band.h"

namespace sirius {

void Band::apply_h_local_slice(K_point* kp__,
                               std::vector<double> const& effective_potential__,
                               std::vector<double> const& pw_ekin__,
                               int num_phi__,
                               matrix<double_complex> const& phi__,
                               matrix<double_complex>& hphi__)
{
    Timer t("sirius::Band::apply_h_local_slice");

    assert(phi__.size(0) == (size_t)kp__->num_gkvec() && hphi__.size(0) == (size_t)kp__->num_gkvec());
    assert(phi__.size(1) >= (size_t)num_phi__ && hphi__.size(1) >= (size_t)num_phi__);

    bool in_place = (&phi__ == &hphi__);

    auto pu = parameters_.processing_unit();

    auto fft = parameters_.reciprocal_lattice()->fft_coarse();
    #ifdef _GPU_
    FFT3D<GPU>* fft_gpu = parameters_.reciprocal_lattice()->fft_gpu_coarse();
    #endif

    int num_fft_threads = -1;
    switch (pu)
    {
        case CPU:
        {
            #ifdef _FFTW_THREADED_
            num_fft_threads = 1;
            #else
            num_fft_threads = Platform::num_fft_threads();
            #endif
            break;
        }
        case GPU:
        {
            #ifdef _FFTW_THREADED_
            num_fft_threads = 2;
            #else
            num_fft_threads = std::min(Platform::num_fft_threads() + 1, Platform::max_num_threads());
            #endif
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
    
    for (int thread_id = 0; thread_id < num_fft_threads; thread_id++)
    {
        if (thread_id == num_fft_threads - 1 && num_fft_threads > 1 && pu == GPU)
        {
            #ifdef _GPU_
            fft_threads.push_back(std::thread([num_phi__, &idx_phi, &idx_phi_mutex, &fft_gpu, kp__, &phi__, 
                                               &hphi__, &effective_potential__, &pw_ekin__, &count_fft_gpu]()
            {
                Timer t("sirius::Band::apply_h_local_slice|gpu");
                
                /* move fft index to GPU */
                mdarray<int, 1> fft_index(kp__->fft_index_coarse(), kp__->num_gkvec());
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
            }));
            #else
            TERMINATE_NO_GPU
            #endif
        }
        else
        {
            fft_threads.push_back(std::thread([thread_id, num_phi__, &idx_phi, &idx_phi_mutex, &fft, kp__, &phi__, 
                                               &hphi__, &effective_potential__, &pw_ekin__, &count_fft_cpu, in_place]()
            {
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
            }));
        }
    }
    for (auto& thread: fft_threads) thread.join();

    //std::cout << "CPU / GPU fft count : " << count_fft_cpu << " " << count_fft_gpu << std::endl;
}

#ifdef _SCALAPACK_
/** \param [in] phi Input wave-functions [storage: CPU || GPU].
 *  \param [out] op_phi Result of application of operator to the wave-functions [storage: CPU || GPU] 
 */
void Band::add_non_local_contribution_parallel(K_point* kp__,
                                               int N__,
                                               int n__,
                                               dmatrix<double_complex>& phi__,
                                               dmatrix<double_complex>& op_phi__, 
                                               matrix<double_complex>& beta_gk__,
                                               mdarray<int, 1> const& packed_mtrx_offset__,
                                               mdarray<double_complex, 1>& op_mtrx_packed__,
                                               double_complex alpha)
{
    log_function_enter(__func__);
    Timer t("sirius::Band::add_non_local_contribution_parallel", kp__->comm());

    /* beginning of the band index */
    splindex<block_cyclic> s0(N__,       kp__->num_ranks_col(), kp__->rank_col(), blacs_grid_.cyclic_block_size());
    /* end of the band index */
    splindex<block_cyclic> s1(N__ + n__, kp__->num_ranks_col(), kp__->rank_col(), blacs_grid_.cyclic_block_size());

    /* local number of states to which Hamiltonian has to be applied */
    int nloc = static_cast<int>(s1.local_size() - s0.local_size());

    if (!nloc) return;

    auto uc = parameters_.unit_cell();

    /* allocate space for <beta|phi> array */
    int nbf_max = uc->max_mt_basis_size() * uc->beta_chunk(0).num_atoms_;
    mdarray<double_complex, 1> beta_phi_tmp(nbf_max * nloc);

    /* result of atom-block-diagonal operator multiplied by <beta|phi> */
    matrix<double_complex> tmp(nbf_max, nloc);

    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU)
    {
        beta_phi_tmp.allocate_on_device();
        tmp.allocate_on_device();
    }
    #endif

    for (int ib = 0; ib < uc->num_beta_chunks(); ib++)
    {
        /* number of beta-projectors in the current chunk */
        int nbeta =  uc->beta_chunk(ib).num_beta_;
        int natoms = uc->beta_chunk(ib).num_atoms_;

        /* wrapper for <beta|phi> with required dimensions */
        matrix<double_complex> beta_phi;
        switch (parameters_.processing_unit())
        {
            case CPU:
            {
                beta_phi = matrix<double_complex>(beta_phi_tmp.at<CPU>(), nbeta, nloc);
                break;
            }
            case GPU:
            {
                beta_phi = matrix<double_complex>(beta_phi_tmp.at<CPU>(), beta_phi_tmp.at<GPU>(), nbeta, nloc);
                break;
            }
        }

        //Timer t1("sirius::Band::add_non_local_contribution_parallel|beta_phi", kp__->comm_row());
        kp__->generate_beta_gk(natoms, uc->beta_chunk(ib).atom_pos_, uc->beta_chunk(ib).desc_, beta_gk__);
        kp__->generate_beta_phi(nbeta, phi__.panel(), nloc, (int)s0.local_size(), beta_gk__, beta_phi);
        //double tval = t1.stop();

        //if (verbosity_level >= 6 && kp__->comm().rank() == 0)
        //{
        //    printf("<beta|phi> effective zgemm with M, N, K: %6i %6i %6i, %12.4f sec, %12.4f GFlops/node\n",
        //           nbeta, nloc, kp__->num_gkvec(),
        //           tval, 8e-9 * nbeta * nloc * kp__->num_gkvec() / tval / kp__->num_ranks_row());
        //}

        kp__->add_non_local_contribution(natoms, nbeta, uc->beta_chunk(ib).desc_, beta_gk__, op_mtrx_packed__,
                                         packed_mtrx_offset__, beta_phi, op_phi__.panel(), nloc, (int)s0.local_size(),
                                         alpha, tmp);
    }
    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU) cuda_device_synchronize();
    #endif

    //#ifdef _GPU_
    //if (parameters_.processing_unit() == GPU)
    //{
    //    cuda_copy_to_host(phi__.at<CPU>(), phi__.at<GPU>(), kp__->num_gkvec_row() * nloc * sizeof(double_complex));
    //}
    //#endif

    log_function_exit(__func__);
}

/** \param [in] phi Input wave-function [storage: CPU].
 *  \param [out] hphi Wave-function multiplied by local Hamiltonian [storage: CPU] 
 */
void Band::apply_h_local_parallel(K_point* kp__,
                                  std::vector<double> const& effective_potential__,
                                  std::vector<double> const& pw_ekin__,
                                  int N__,
                                  int n__,
                                  dmatrix<double_complex>& phi__,
                                  dmatrix<double_complex>& hphi__)
{
    log_function_enter(__func__);
    Timer t("sirius::Band::apply_h_local_parallel", kp__->comm());

    splindex<block_cyclic> s0(N__,       kp__->num_ranks_col(), kp__->rank_col(), blacs_grid_.cyclic_block_size());
    splindex<block_cyclic> s1(N__ + n__, kp__->num_ranks_col(), kp__->rank_col(), blacs_grid_.cyclic_block_size());

    splindex<block> sub_spl_n(s1.local_size() - s0.local_size(), kp__->num_ranks_row(), kp__->rank_row());
    
    int nphi = (int)sub_spl_n.local_size();

    memcpy(&hphi__(0, s0.local_size()), &phi__(0, s0.local_size()), 
           kp__->num_gkvec_row() * (s1.local_size() - s0.local_size()) * sizeof(double_complex));
    
    hphi__.gather(n__, N__);
    apply_h_local_slice(kp__, effective_potential__, pw_ekin__, nphi, hphi__.slice(), hphi__.slice());
    hphi__.scatter(n__, N__);

    log_function_exit(__func__);
}

/** \param [in] phi Input wave-function [storage: CPU && GPU].
 *  \param [out] hphi Wave-function multiplied by local Hamiltonian [storage: CPU || GPU] 
 */
void Band::apply_h_parallel(K_point* kp__,
                            std::vector<double> const& effective_potential__,
                            std::vector<double> const& pw_ekin__,
                            int N__,
                            int n__,
                            dmatrix<double_complex>& phi__,
                            dmatrix<double_complex>& hphi__,
                            matrix<double_complex>& beta_gk__,
                            mdarray<int, 1> const& packed_mtrx_offset__,
                            mdarray<double_complex, 1>& d_mtrx_packed__)
{
    log_function_enter(__func__);
    Timer t("sirius::Band::apply_h_parallel", kp__->comm());

    /* beginning of the band index */
    splindex<block_cyclic> s0(N__,       kp__->num_ranks_col(), kp__->rank_col(), blacs_grid_.cyclic_block_size());
    /* end of the band index */
    splindex<block_cyclic> s1(N__ + n__, kp__->num_ranks_col(), kp__->rank_col(), blacs_grid_.cyclic_block_size());

    /* local number of states to which Hamiltonian has to be applied */
    int nloc = static_cast<int>(s1.local_size() - s0.local_size());

    if (!nloc) return;

    /* apply local part of Hamiltonian */
    apply_h_local_parallel(kp__, effective_potential__, pw_ekin__, N__, n__, phi__, hphi__);

    if (parameters_.processing_unit() == GPU)
    {
        #ifdef _GPU_
        hphi__.copy_cols_to_device(N__, N__ + n__);
        #endif
    }

    add_non_local_contribution_parallel(kp__, N__, n__, phi__, hphi__, beta_gk__, packed_mtrx_offset__,
                                        d_mtrx_packed__, double_complex(1, 0));
    log_function_exit(__func__);
}

/** On input phi is both in host and device memory.
 *  On output hphi and ophi are in host or device memory depending on processing unit.
 */
void Band::apply_h_o_parallel(K_point* kp__,
                              std::vector<double> const& effective_potential__,
                              std::vector<double> const& pw_ekin__,
                              int N__,
                              int n__,
                              dmatrix<double_complex>& phi__,
                              dmatrix<double_complex>& hphi__,
                              dmatrix<double_complex>& ophi__,
                              matrix<double_complex>& beta_gk__,
                              mdarray<int, 1>& packed_mtrx_offset__,
                              mdarray<double_complex, 1>& d_mtrx_packed__,
                              mdarray<double_complex, 1>& q_mtrx_packed__)
{
    log_function_enter(__func__);
    Timer t("sirius::Band::apply_h_o_parallel", kp__->comm());

    /* beginning of the band index */
    splindex<block_cyclic> s0(N__,       kp__->num_ranks_col(), kp__->rank_col(), blacs_grid_.cyclic_block_size());
    /* end of the band index */
    splindex<block_cyclic> s1(N__ + n__, kp__->num_ranks_col(), kp__->rank_col(), blacs_grid_.cyclic_block_size());

    /* local number of states to which Hamiltonian has to be applied */
    int nloc = static_cast<int>(s1.local_size() - s0.local_size());

    if (!nloc) return;

    auto uc = parameters_.unit_cell();

    /* apply local part of Hamiltonian */
    apply_h_local_parallel(kp__, effective_potential__, pw_ekin__, N__, n__, phi__, hphi__);

    if (parameters_.processing_unit() == CPU)
    {
        /* set intial ophi */
        memcpy(&ophi__(0, s0.local_size()), &phi__(0, s0.local_size()), kp__->num_gkvec_row() * nloc * sizeof(double_complex));
    }

    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU)
    {
        /* copy hphi do device */
        cuda_copy_to_device(hphi__.at<GPU>(0, s0.local_size()), hphi__.at<CPU>(0, s0.local_size()), 
                            kp__->num_gkvec_row() * nloc * sizeof(double_complex));

        /* set intial ophi */
        cuda_copy_device_to_device(ophi__.at<GPU>(0, s0.local_size()), phi__.at<GPU>(0, s0.local_size()), 
                                   kp__->num_gkvec_row() * nloc * sizeof(double_complex));
    }
    #endif

    /* allocate space for <beta|phi> array */
    int nbf_max = uc->max_mt_basis_size() * uc->beta_chunk(0).num_atoms_;
    mdarray<double_complex, 1> beta_phi_tmp(nbf_max * nloc);

    /* work space (result of Q or D multiplied by <beta|phi>) */
    matrix<double_complex> work(nbf_max, nloc);

    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU)
    {
        beta_phi_tmp.allocate_on_device();
        work.allocate_on_device();
    }
    #endif

    for (int ib = 0; ib < uc->num_beta_chunks(); ib++)
    {
        /* number of beta-projectors in the current chunk */
        int nbeta =  uc->beta_chunk(ib).num_beta_;
        int natoms = uc->beta_chunk(ib).num_atoms_;

        /* wrapper for <beta|phi> with required dimensions */
        matrix<double_complex> beta_phi;
        switch (parameters_.processing_unit())
        {
            case CPU:
            {
                beta_phi = matrix<double_complex>(beta_phi_tmp.at<CPU>(), nbeta, nloc);
                break;
            }
            case GPU:
            {
                beta_phi = matrix<double_complex>(beta_phi_tmp.at<CPU>(), beta_phi_tmp.at<GPU>(), nbeta, nloc);
                break;
            }
        }

        kp__->generate_beta_gk(natoms, uc->beta_chunk(ib).atom_pos_, uc->beta_chunk(ib).desc_, beta_gk__);
        
        kp__->generate_beta_phi(nbeta, phi__.panel(), nloc, (int)s0.local_size(), beta_gk__, beta_phi);

        kp__->add_non_local_contribution(natoms, nbeta, uc->beta_chunk(ib).desc_, beta_gk__, d_mtrx_packed__,
                                         packed_mtrx_offset__, beta_phi, hphi__.panel(), nloc, (int)s0.local_size(),
                                         complex_one, work);
        
        kp__->add_non_local_contribution(natoms, nbeta, uc->beta_chunk(ib).desc_, beta_gk__, q_mtrx_packed__,
                                         packed_mtrx_offset__, beta_phi, ophi__.panel(), nloc, (int)s0.local_size(),
                                         complex_one, work);
    }

    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU) cuda_device_synchronize();
    #endif
    log_function_exit(__func__);
}

#endif // _SCALAPACK_


};
