#include <thread>
#include <mutex>
#include "band.h"

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
 *  \param [out] op_phi Result of application of operator to the wave-functions [storage: CPU || GPU].
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

    log_function_exit(__func__);
}

void Band::add_non_local_contribution_parallel(K_point* kp__,
                                               dmatrix<double_complex>& phi__,
                                               dmatrix<double_complex>& op_phi__,
                                               dmatrix<double_complex>& op__,
                                               double_complex alpha)
{
    auto uc = parameters_.unit_cell();
    int num_bands = parameters_.num_fv_states();

    /* <\beta_{\xi}^{\alpha}|\phi_j> */
    dmatrix<double_complex> beta_phi(uc->mt_basis_size(), num_bands, kp__->blacs_grid());
    /* compute <beta|phi> */
    linalg<CPU>::gemm(2, 0, uc->mt_basis_size(), num_bands, kp__->num_gkvec(), complex_one, 
                      kp__->beta_pw_panel(), phi__, complex_zero, beta_phi);

    dmatrix<double_complex> tmp(uc->mt_basis_size(), num_bands, kp__->blacs_grid());
    linalg<CPU>::gemm(0, 0, uc->mt_basis_size(), num_bands, uc->mt_basis_size(), complex_one,
                      op__, beta_phi, complex_zero, tmp);

    linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands, uc->mt_basis_size(), alpha,
                      kp__->beta_pw_panel(), tmp, complex_one, op_phi__);
}


/** \param [in] phi Input wave-function [storage: CPU].
 *  \param [out] hphi Wave-function multiplied by local Hamiltonian [storage: CPU].
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

// TODO: fix the arguments
//== void Band::set_fv_h_o_parallel_simple(int N__,
//==                                       int n__,
//==                                       K_point* kp__,
//==                                       std::vector<double> const& veff_it_coarse__,
//==                                       std::vector<double> const& pw_ekin__,
//==                                       dmatrix<double_complex>& phi__,
//==                                       dmatrix<double_complex>& hphi__,
//==                                       dmatrix<double_complex>& ophi__,
//==                                       dmatrix<double_complex>& h__,
//==                                       dmatrix<double_complex>& o__,
//==                                       dmatrix<double_complex>& h_old__,
//==                                       dmatrix<double_complex>& o_old__)
//== {
//==     Timer t("sirius::Band::set_fv_h_o_parallel_simple", kp__->comm());
//== 
//==     splindex<block_cyclic> s0_col(N__,       kp__->num_ranks_col(), kp__->rank_col(), blacs_grid_.cyclic_block_size());
//==     splindex<block_cyclic> s1_col(N__ + n__, kp__->num_ranks_col(), kp__->rank_col(), blacs_grid_.cyclic_block_size());
//==     splindex<block_cyclic> s0_row(N__,       kp__->num_ranks_row(), kp__->rank_row(), blacs_grid_.cyclic_block_size());
//==     splindex<block_cyclic> s1_row(N__ + n__, kp__->num_ranks_row(), kp__->rank_row(), blacs_grid_.cyclic_block_size());
//== 
//==     /* copy old Hamiltonian and overlap */
//==     for (int i = 0; i < (int)s0_col.local_size(); i++)
//==     {
//==         memcpy(&h__(0, i), &h_old__(0, i), s0_row.local_size() * sizeof(double_complex));
//==         memcpy(&o__(0, i), &o_old__(0, i), s0_row.local_size() * sizeof(double_complex));
//==     }
//== 
//==     /* apply Hamiltonian and overlap operators to the new basis functions */
//==     apply_h_o_parallel(kp__, veff_it_coarse__, pw_ekin__, N__, n__, phi__, hphi__, ophi__);
//==     
//==     Timer t2("sirius::Band::set_fv_h_o_uspp_cpu_parallel_simple|zgemm", _global_timer_);
//==     /* <{phi,res}|H|res> */
//==     linalg<CPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec(), complex_one, phi__, 0, 0, hphi__, 0, N__, complex_zero, h__, 0, N__);
//==     /* <{phi,res}|O|res> */
//==     linalg<CPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec(), complex_one, phi__, 0, 0, ophi__, 0, N__, complex_zero, o__, 0, N__);
//==     double tval = t2.stop();
//== 
//==     if (verbosity_level >= 6 && kp__->comm().rank() == 0)
//==     {
//==         printf("pzgemm #4&5 with M, N, K: %6i %6i %6i, offset in B&C: %6i, %12.4f sec, %12.4f GFlops/node\n",
//==                N__ + n__, n__, kp__->num_gkvec(), N__,
//==                tval, 2 * 8e-9 * (N__ + n__) * n__ * kp__->num_gkvec() / tval / kp__->num_ranks());
//==     }
//==     
//==     /* restore the bottom block of the matrix */
//==     if (N__ != 0)
//==     {
//==         linalg<CPU>::tranc(n__, N__, h__, 0, N__, h__, N__, 0);
//==         linalg<CPU>::tranc(n__, N__, o__, 0, N__, o__, N__, 0);
//==     }
//== 
//==     /* save Hamiltonian and overlap */
//==     for (int i = 0; i < (int)s1_col.local_size(); i++)
//==     {
//==         memcpy(&h_old__(0, i), &h__(0, i), s1_row.local_size() * sizeof(double_complex));
//==         memcpy(&o_old__(0, i), &o__(0, i), s1_row.local_size() * sizeof(double_complex));
//==     }
//== }

void Band::set_fv_h_o_parallel(int N__,
                               int n__,
                               K_point* kp__,
                               std::vector<double>& veff_it_coarse__,
                               std::vector<double>& pw_ekin__,
                               dmatrix<double_complex>& phi__,
                               dmatrix<double_complex>& hphi__,
                               dmatrix<double_complex>& ophi__,
                               dmatrix<double_complex>& h__,
                               dmatrix<double_complex>& o__,
                               dmatrix<double_complex>& h_old__,
                               dmatrix<double_complex>& o_old__,
                               mdarray<double_complex, 2>& kappa__,
                               mdarray<int, 1>& packed_mtrx_offset__,
                               mdarray<double_complex, 1>& d_mtrx_packed__,
                               mdarray<double_complex, 1>& q_mtrx_packed__)
{
    log_function_enter(__func__);
    Timer t("sirius::Band::set_fv_h_o_uspp_gpu_parallel_v3", kp__->comm());
    
    bool with_overlap = (parameters_.esm_type() == ultrasoft_pseudopotential);

    splindex<block_cyclic> s0_col(N__,       kp__->num_ranks_col(), kp__->rank_col(), blacs_grid_.cyclic_block_size());
    splindex<block_cyclic> s1_col(N__ + n__, kp__->num_ranks_col(), kp__->rank_col(), blacs_grid_.cyclic_block_size());
    splindex<block_cyclic> s0_row(N__,       kp__->num_ranks_row(), kp__->rank_row(), blacs_grid_.cyclic_block_size());
    splindex<block_cyclic> s1_row(N__ + n__, kp__->num_ranks_row(), kp__->rank_row(), blacs_grid_.cyclic_block_size());

    /* copy old Hamiltonian and overlap */
    for (int i = 0; i < (int)s0_col.local_size(); i++)
    {
        memcpy(&h__(0, i), &h_old__(0, i), s0_row.local_size() * sizeof(double_complex));
        memcpy(&o__(0, i), &o_old__(0, i), s0_row.local_size() * sizeof(double_complex));
    }

    /* apply Hamiltonian and overlap operators to the new basis functions */
    if (with_overlap)
    {
        apply_h_o_parallel(kp__, veff_it_coarse__, pw_ekin__, N__, n__, phi__, hphi__, ophi__,
                           kappa__, packed_mtrx_offset__, d_mtrx_packed__, q_mtrx_packed__);
    }
    else
    {
        apply_h_parallel(kp__, veff_it_coarse__, pw_ekin__, N__, n__, phi__, hphi__, kappa__,
                         packed_mtrx_offset__, d_mtrx_packed__);
    }
    
    #if defined(_GPU_) && !defined(_GPU_DIRECT_)
    if (parameters_.processing_unit() == GPU)
    {
        size_t panel_size = kp__->num_gkvec_row() * (s1_col.local_size() - s0_col.local_size()) * sizeof(double_complex);
        cuda_copy_to_host(hphi__.at<CPU>(0, s0_col.local_size()), hphi__.at<GPU>(0, s0_col.local_size()), panel_size);
        if (with_overlap) cuda_copy_to_host(ophi__.at<CPU>(0, s0_col.local_size()), ophi__.at<GPU>(0, s0_col.local_size()), panel_size);
    }
    #endif

    int max_num_phi_new = 0;
    for (int icol = 0; icol < kp__->num_ranks_col(); icol++)
        max_num_phi_new = std::max(max_num_phi_new, (int)(s1_col.local_size(icol) - s0_col.local_size(icol)));
    
    int num_phi = (int)s1_col.local_size();

    mdarray<double_complex, 3> hphi_tmp, ophi_tmp;
    switch (parameters_.processing_unit())
    {
        case CPU:
        {
            hphi_tmp = mdarray<double_complex, 3>(kappa__.at<CPU>(0, 0),
                                                  kp__->num_gkvec_row(), max_num_phi_new, 2);
            ophi_tmp = mdarray<double_complex, 3>(kappa__.at<CPU>(0,  2 * max_num_phi_new),
                                                  kp__->num_gkvec_row(), max_num_phi_new, 2);
            break;
        }
        case GPU:
        {
            hphi_tmp = mdarray<double_complex, 3>(kappa__.at<CPU>(0, 0),
                                                  kappa__.at<GPU>(0, 0), 
                                                  kp__->num_gkvec_row(), max_num_phi_new, 2);
            ophi_tmp = mdarray<double_complex, 3>(kappa__.at<CPU>(0,  2 * max_num_phi_new), 
                                                  kappa__.at<GPU>(0,  2 * max_num_phi_new), 
                                                  kp__->num_gkvec_row(), max_num_phi_new, 2);
            break;
        }
    }
    
    mdarray<double_complex, 3> h_tmp(num_phi, max_num_phi_new, 2);
    mdarray<double_complex, 3> o_tmp(num_phi, max_num_phi_new, 2);

    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU)
    {
        h_tmp.allocate_on_device();
        o_tmp.allocate_on_device();
    }
    #endif

    std::array<std::atomic_bool, 2> lock_hphi;
    std::array<std::atomic_bool, 2> lock_ophi;
    std::array<std::atomic_bool, 2> lock_h;
    std::array<std::atomic_bool, 2> lock_o;
    for (int i = 0; i < 2; i++)
    {
        lock_hphi[i].store(false);
        lock_ophi[i].store(false);
        lock_h[i].store(false);
        lock_o[i].store(false);
    }
   
    Timer t1("sirius::Band::set_fv_h_o_uspp_gpu_parallel_v3|zgemm_eff", kp__->comm());

    auto pu = parameters_.processing_unit();

    auto bcast_column = [kp__, &s0_col, &s1_col, pu]
                        (int icol, dmatrix<double_complex>& mtrx, mdarray<double_complex, 3>& mtrx_tmp) -> void
    {
        Timer t("sirius::bcast_column");

        #ifdef _GPU_
        #ifdef _GPU_DIRECT_
        bool gpu_direct = true;
        #else
        bool gpu_direct = false;
        #endif
        #endif
 
        int nloc = (int)(s1_col.local_size(icol) - s0_col.local_size(icol));
        size_t panel_size = kp__->num_gkvec_row() * nloc * sizeof(double_complex);

        if (!nloc) return;
        
        if (pu == CPU)
        {
            if (kp__->rank_col() == icol)
                memcpy(mtrx_tmp.at<CPU>(0, 0, icol % 2), mtrx.at<CPU>(0, s0_col.local_size(icol)), panel_size);
            kp__->comm_col().bcast(mtrx_tmp.at<CPU>(0, 0, icol % 2), kp__->num_gkvec_row() * nloc, icol);
        }
        if (pu == GPU)
        {
            #ifdef _GPU_
            if (gpu_direct)
            {
                if (kp__->rank_col() == icol)
                    cuda_copy_device_to_device(mtrx_tmp.at<GPU>(0, 0, icol % 2), mtrx.at<GPU>(0, s0_col.local_size(icol)), panel_size);
                kp__->comm_col().bcast(mtrx_tmp.at<GPU>(0, 0, icol % 2), kp__->num_gkvec_row() * nloc, icol);
            } 
            else
            {
                if (kp__->rank_col() == icol)
                    memcpy(mtrx_tmp.at<CPU>(0, 0, icol % 2), mtrx.at<CPU>(0, s0_col.local_size(icol)), panel_size);
                kp__->comm_col().bcast(mtrx_tmp.at<CPU>(0, 0, icol % 2), kp__->num_gkvec_row() * nloc, icol);
                cuda_copy_to_device(mtrx_tmp.at<GPU>(0, 0, icol % 2), mtrx_tmp.at<CPU>(0, 0, icol % 2), panel_size);
            }
            #else
            TERMINATE_NO_GPU
            #endif
        }
    };

    bcast_column(0, hphi__, hphi_tmp);
    if (with_overlap)
    {
        bcast_column(0, ophi__, ophi_tmp);
    }
    else
    {
        bcast_column(0, phi__, ophi_tmp);
    }
    lock_hphi[0].store(true);
    lock_ophi[0].store(true);

    int nthread = omp_get_max_threads();
    if (nthread > 1) omp_set_num_threads(nthread - 1);

    /* crate communication thread */
    std::thread comm_thread([kp__, &s0_col, &s1_col, &s0_row, &s1_row, &lock_hphi, &lock_ophi, &lock_h, &lock_o, 
                             &hphi__, &ophi__, &hphi_tmp, &ophi_tmp, &h_tmp, &o_tmp, &h__, &o__, bcast_column, 
                             with_overlap, &phi__]()
    {
        int num_phi = (int)s1_col.local_size();
    
        for (int icol = 0; icol < kp__->num_ranks_col(); icol++)
        {
            /* local number of new basis functions */
            int nloc = (int)(s1_col.local_size(icol) - s0_col.local_size(icol));
            
            Timer t1("sirius::Band::set_fv_h_o_uspp_gpu_parallel_v3|comm_thread|1");
            /* broadcast next column */
            if (icol + 1 < kp__->num_ranks_col())
            {
                while (lock_hphi[(icol + 1) % 2].load());
                bcast_column(icol + 1, hphi__, hphi_tmp);
                lock_hphi[(icol + 1) % 2].store(true);
                
                while (lock_ophi[(icol + 1) % 2].load());
                if (with_overlap)
                {
                    bcast_column(icol + 1, ophi__, ophi_tmp);
                }
                else
                {
                    bcast_column(icol + 1, phi__, ophi_tmp);
                }
                lock_ophi[(icol + 1) % 2].store(true);
            }
            t1.stop();
    
            Timer t2("sirius::Band::set_fv_h_o_uspp_gpu_parallel_v3|comm_thread|2");
            if (nloc > 0)
            {
                while (!lock_h[icol % 2].load());
                kp__->comm_row().allreduce(&h_tmp(0, 0, icol % 2), num_phi * nloc);
    
                for (int j = 0; j < nloc; j++)
                {
                    int idx_hphi_glob = (int)s1_col.global_index(s0_col.local_size(icol) + j, icol);
                    auto p = s1_row.location(idx_hphi_glob);
                    if (p.second == kp__->rank_row())
                    {
                        for (int i = 0; i < num_phi; i++)
                        {
                            h__(p.first, i) = conj(h_tmp(i, j, icol % 2));
                        }
                    }
                }
                /* remove lock from h buffer */
                lock_h[icol % 2].store(false);
    
                while (!lock_o[icol % 2].load());
                kp__->comm_row().allreduce(&o_tmp(0, 0, icol % 2), num_phi * nloc);
    
                for (int j = 0; j < nloc; j++)
                {
                    int idx_hphi_glob = (int)s1_col.global_index(s0_col.local_size(icol) + j, icol);
                    auto p = s1_row.location(idx_hphi_glob);
                    if (p.second == kp__->rank_row())
                    {
                        for (int i = 0; i < num_phi; i++)
                        {
                            o__(p.first, i) = conj(o_tmp(i, j, icol % 2));
                        }
                    }
                }
                /* remove lock from o buffer */
                lock_o[icol % 2].store(false);
            }
            t2.stop();
        }
    });

    for (int icol = 0; icol < kp__->num_ranks_col(); icol++)
    {
        int n = (int)(s1_col.local_size(icol) - s0_col.local_size(icol));

        /* wait for broadcast of this column */
        while (!lock_hphi[icol % 2].load());
        /* wait for unlock of h buffer */
        while (lock_h[icol % 2].load());

        if (n > 0)
        {
            Timer t2("sirius::Band::set_fv_h_o_uspp_gpu_parallel_v3|zgemm_loc");
            if (pu == GPU)
            {
                #ifdef _GPU_
                linalg<GPU>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(), phi__.at<GPU>(), phi__.ld(),
                                  hphi_tmp.at<GPU>(0, 0, icol % 2), hphi_tmp.ld(), h_tmp.at<GPU>(0, 0, icol % 2), h_tmp.ld());
                cuda_copy_to_host(h_tmp.at<CPU>(0, 0, icol % 2), h_tmp.at<GPU>(0, 0, icol % 2), num_phi * n * sizeof(double_complex));
                #else
                TERMINATE_NO_GPU
                #endif
            }
            if (pu == CPU)
            {
                linalg<CPU>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(), phi__.at<CPU>(), phi__.ld(),
                                  hphi_tmp.at<CPU>(0, 0, icol % 2), hphi_tmp.ld(), h_tmp.at<CPU>(0, 0, icol % 2), h_tmp.ld());
            }
            lock_h[icol % 2].store(true);
            lock_hphi[icol % 2].store(false);
        }
            
        while (!lock_ophi[icol % 2].load());
        while (lock_o[icol % 2].load());
        if (n > 0)
        {
            Timer t2("sirius::Band::set_fv_h_o_uspp_gpu_parallel_v3|zgemm_loc");
            if (pu == GPU)
            {
                #ifdef _GPU_
                linalg<GPU>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(), phi__.at<GPU>(), phi__.ld(),
                                  ophi_tmp.at<GPU>(0, 0, icol % 2), ophi_tmp.ld(), o_tmp.at<GPU>(0, 0, icol % 2), o_tmp.ld());
                cuda_copy_to_host(o_tmp.at<CPU>(0, 0, icol % 2), o_tmp.at<GPU>(0, 0, icol % 2), num_phi * n * sizeof(double_complex));
                #else
                TERMINATE_NO_GPU
                #endif
            }
            if (pu == CPU)
            {
                linalg<CPU>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(), phi__.at<CPU>(), phi__.ld(),
                                  ophi_tmp.at<CPU>(0, 0, icol % 2), ophi_tmp.ld(), o_tmp.at<CPU>(0, 0, icol % 2), o_tmp.ld());
            }
            lock_o[icol % 2].store(true);
            lock_ophi[icol % 2].store(false);
        }
    }
    comm_thread.join();
    omp_set_num_threads(nthread);

    double tval = t1.stop();

    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    {
        printf("effective zgemm #4&5 with M, N, K: %6i %6i %6i,                        %12.4f sec, %12.4f GFlops/node\n",
               N__ + n__, n__, kp__->num_gkvec(),
               tval, 2 * 8e-9 * (N__ + n__) * n__ * kp__->num_gkvec() / tval / kp__->num_ranks());
    }

    /* restore right block of the matrix */
    if (N__ != 0)
    {
        linalg<CPU>::tranc(N__, n__, h__, N__, 0, h__, 0, N__);
        linalg<CPU>::tranc(N__, n__, o__, N__, 0, o__, 0, N__);
    }

    /* save Hamiltonian and overlap */
    for (int i = 0; i < (int)s1_col.local_size(); i++)
    {
        memcpy(&h_old__(0, i), &h__(0, i), s1_row.local_size() * sizeof(double_complex));
        memcpy(&o_old__(0, i), &o__(0, i), s1_row.local_size() * sizeof(double_complex));
    }
    
    log_function_exit(__func__);
}
#endif // _SCALAPACK_

void Band::apply_h_o_serial(K_point* kp__, 
                            std::vector<double> const& effective_potential__, 
                            std::vector<double> const& pw_ekin__, 
                            int N__,
                            int n__,
                            matrix<double_complex>& phi__,
                            matrix<double_complex>& hphi__,
                            matrix<double_complex>& ophi__,
                            mdarray<int, 1>& packed_mtrx_offset__,
                            mdarray<double_complex, 1>& d_mtrx_packed__,
                            mdarray<double_complex, 1>& q_mtrx_packed__)
{
    Timer t("sirius::Band::apply_h_o_serial", kp__->comm());

    auto uc = parameters_.unit_cell();

    matrix<double_complex> phi, hphi, ophi;

    switch (parameters_.processing_unit())
    {
        case CPU:
        {
            phi =  matrix<double_complex>( phi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
            hphi = matrix<double_complex>(hphi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
            ophi = matrix<double_complex>(ophi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
            break;
        }
        case GPU:
        {
            phi =  matrix<double_complex>( phi__.at<CPU>(0, N__),  phi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
            hphi = matrix<double_complex>(hphi__.at<CPU>(0, N__), hphi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
            ophi = matrix<double_complex>(ophi__.at<CPU>(0, N__), ophi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
            break;
        }
    }
    
    /* apply local part of Hamiltonian */
    apply_h_local_slice(kp__, effective_potential__, pw_ekin__, n__, phi, hphi);
    
    if (parameters_.processing_unit() == CPU)
    {
        /* set intial ophi */
        phi >> ophi;
    }

    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU)
    {
        /* copy hphi do device */
        hphi.copy_to_device();

        /* set intial ophi */
        cuda_copy_device_to_device(ophi.at<GPU>(), phi.at<GPU>(), kp__->num_gkvec_row() * n__ * sizeof(double_complex));
    }
    #endif

    /* <\beta_{\xi}^{\alpha}|\phi_j> */
    matrix<double_complex> beta_phi(uc->mt_lo_basis_size(), n__);
    
    /* Q or D multiplied by <\beta_{\xi}^{\alpha}|\phi_j> */
    matrix<double_complex> work(uc->mt_lo_basis_size(), n__);

    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU)
    {
        beta_phi.allocate_on_device();
        work.allocate_on_device();
    }
    #endif

    kp__->generate_beta_phi(uc->mt_lo_basis_size(), phi, n__, 0, kp__->beta_pw_panel().panel(), beta_phi);

    kp__->add_non_local_contribution(uc->num_atoms(), uc->mt_lo_basis_size(), uc->beta_chunk(0).desc_,
                                     kp__->beta_pw_panel().panel(), d_mtrx_packed__, packed_mtrx_offset__, beta_phi,
                                     hphi, n__, 0, complex_one, work);
        
    kp__->add_non_local_contribution(uc->num_atoms(), uc->mt_lo_basis_size(), uc->beta_chunk(0).desc_,
                                     kp__->beta_pw_panel().panel(), q_mtrx_packed__, packed_mtrx_offset__, beta_phi,
                                     ophi, n__, 0, complex_one, work);
    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU) cuda_device_synchronize();
    #endif
}

};
