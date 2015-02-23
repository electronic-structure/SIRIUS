#include <thread>
#include <mutex>
#include "band.h"
#include "debug.hpp"

namespace sirius {

#ifdef _GPU_
extern "C" void compute_residuals_gpu(int num_gkvec_row,
                                      int num_res_local,
                                      int* res_idx,
                                      double* eval,
                                      double_complex const* hpsi,
                                      double_complex const* opsi,
                                      double_complex* res,
                                      double* res_norm);

extern "C" void apply_preconditioner_gpu(int num_gkvec_row,
                                         int num_res_local,
                                         int* res_idx,
                                         double* eval,
                                         double const* h_diag,
                                         double const* o_diag,
                                         double_complex* res,
                                         double* res_norm);

extern "C" void normalize_residuals_gpu(int num_gkvec_row,
                                        int num_res_local,
                                        int* res_idx,
                                        double* norm2,
                                        double_complex* res);
#endif


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
    Timer t("sirius::Band::add_non_local_contribution_parallel");

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
        //    printf("<beta|phi> effective zgemm with M, N, K: %6i %6i %6i, %12.4f sec, %12.4f GFlops/rank\n",
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
    //auto uc = parameters_.unit_cell();
    //int num_bands = parameters_.num_fv_states();

    STOP();

    ///* <\beta_{\xi}^{\alpha}|\phi_j> */
    //dmatrix<double_complex> beta_phi(uc->mt_basis_size(), num_bands, kp__->blacs_grid());
    ///* compute <beta|phi> */
    //linalg<CPU>::gemm(2, 0, uc->mt_basis_size(), num_bands, kp__->num_gkvec(), complex_one, 
    //                  kp__->beta_gk_panel(), phi__, complex_zero, beta_phi);

    //dmatrix<double_complex> tmp(uc->mt_basis_size(), num_bands, kp__->blacs_grid());
    //linalg<CPU>::gemm(0, 0, uc->mt_basis_size(), num_bands, uc->mt_basis_size(), complex_one,
    //                  op__, beta_phi, complex_zero, tmp);

    //linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands, uc->mt_basis_size(), alpha,
    //                  kp__->beta_gk_panel(), tmp, complex_one, op_phi__);
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
    Timer t("sirius::Band::apply_h_local_parallel", kp__->comm_row());

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
    Timer t("sirius::Band::apply_h_parallel", kp__->comm_row());

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

void Band::apply_h_o_fast_parallel(K_point* kp__,
                                   std::vector<double> const& effective_potential__,
                                   std::vector<double> const& pw_ekin__,
                                   int N__,
                                   int n__,
                                   matrix<double_complex>& phi_slice__,
                                   matrix<double_complex>& phi_slab__,
                                   matrix<double_complex>& hphi_slab__,
                                   matrix<double_complex>& ophi_slab__,
                                   mdarray<int, 1>& packed_mtrx_offset__,
                                   mdarray<double_complex, 1>& d_mtrx_packed__,
                                   mdarray<double_complex, 1>& q_mtrx_packed__,
                                   mdarray<double_complex, 1>& kappa__)
{
    LOG_FUNC_BEGIN();

    Timer t("sirius::Band::apply_h_o_fast_parallel", kp__->comm());

    splindex<block> spl_phi(n__, kp__->comm().size(), kp__->comm().rank());

    kp__->collect_all_gkvec(spl_phi, &phi_slab__(0, N__), &phi_slice__(0, 0));
    
    if (spl_phi.local_size())
        apply_h_local_slice(kp__, effective_potential__, pw_ekin__, (int)spl_phi.local_size(), phi_slice__, phi_slice__);

    kp__->collect_all_bands(spl_phi, &phi_slice__(0, 0),  &hphi_slab__(0, N__));

    auto uc = parameters_.unit_cell();

    if (parameters_.processing_unit() == CPU)
    {
        /* set intial ophi */
        memcpy(&ophi_slab__(0, N__), &phi_slab__(0, N__), kp__->num_gkvec_loc() * n__ * sizeof(double_complex));
    }

    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU)
    {
        /* copy hphi do device */
        cuda_copy_to_device(hphi_slab__.at<GPU>(0, N__), hphi_slab__.at<CPU>(0, N__),
                            kp__->num_gkvec_loc() * n__ * sizeof(double_complex));

        /* set intial ophi */
        cuda_copy_device_to_device(ophi_slab__.at<GPU>(0, N__), phi_slab__.at<GPU>(0, N__), 
                                   kp__->num_gkvec_loc() * n__ * sizeof(double_complex));
    }
    #endif

    int offs = 0;
    for (int ib = 0; ib < uc->num_beta_chunks(); ib++)
    {
        /* number of beta-projectors in the current chunk */
        int nbeta =  uc->beta_chunk(ib).num_beta_;
        int natoms = uc->beta_chunk(ib).num_atoms_;

        /* wrapper for <beta|phi> with required dimensions */
        matrix<double_complex> beta_gk;
        matrix<double_complex> beta_phi;
        matrix<double_complex> work;
        switch (parameters_.processing_unit())
        {
            case CPU:
            {
                beta_phi = matrix<double_complex>(kappa__.at<CPU>(),            nbeta, n__);
                work     = matrix<double_complex>(kappa__.at<CPU>(nbeta * n__), nbeta, n__);
                beta_gk  = matrix<double_complex>(kp__->beta_gk().at<CPU>(0, offs), kp__->num_gkvec_loc(), nbeta);
                break;
            }
            case GPU:
            {
                #ifdef _GPU_
                beta_phi = matrix<double_complex>(kappa__.at<CPU>(),            kappa__.at<GPU>(),            nbeta, n__);
                work     = matrix<double_complex>(kappa__.at<CPU>(nbeta * n__), kappa__.at<GPU>(nbeta * n__), nbeta, n__);
                beta_gk  = matrix<double_complex>(kp__->beta_gk().at<CPU>(0, offs), kappa__.at<GPU>(2 * nbeta * n__), kp__->num_gkvec_loc(), nbeta);
                beta_gk.copy_to_device();
                #endif
                break;
            }
        }

        kp__->generate_beta_phi(nbeta, phi_slab__, n__, N__, beta_gk, beta_phi);

        kp__->add_non_local_contribution(natoms, nbeta, uc->beta_chunk(ib).desc_, beta_gk, d_mtrx_packed__,
                                         packed_mtrx_offset__, beta_phi, hphi_slab__, n__, N__, complex_one, work);
        
        kp__->add_non_local_contribution(natoms, nbeta, uc->beta_chunk(ib).desc_, beta_gk, q_mtrx_packed__,
                                         packed_mtrx_offset__, beta_phi, ophi_slab__, n__, N__, complex_one, work);
        
        offs += nbeta;
    }

    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU) cuda_device_synchronize();
    #endif

    LOG_FUNC_END();
}

/** \param [in] phi Input wave-function [storage: CPU && GPU].
 *  \param [out] hphi Wave-function multiplied by local Hamiltonian [storage: CPU || GPU].
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
    LOG_FUNC_BEGIN();
    Timer t("sirius::Band::apply_h_o_parallel", kp__->comm_row());

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
    int nbmax = 0;
    for (int ib = 0; ib < uc->num_beta_chunks(); ib++) nbmax = std::max(nbmax, uc->beta_chunk(ib).num_beta_);
    mdarray<double_complex, 1> beta_phi_tmp(nbmax * nloc);

    /* work space (result of Q or D multiplied by <beta|phi>) */
    matrix<double_complex> work(nbmax, nloc);

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
    LOG_FUNC_END();
}

void Band::set_fv_h_o_parallel_simple(int N__,
                                      int n__,
                                      K_point* kp__,
                                      std::vector<double> const& veff_it_coarse__,
                                      std::vector<double> const& pw_ekin__,
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
    Timer t("sirius::Band::set_fv_h_o_parallel_simple", kp__->comm());

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
    apply_h_o_parallel(kp__, veff_it_coarse__, pw_ekin__, N__, n__, phi__, hphi__, ophi__, kappa__, 
                       packed_mtrx_offset__, d_mtrx_packed__, q_mtrx_packed__);
    
    Timer t2("sirius::Band::set_fv_h_o_parallel_simple|zgemm", kp__->comm());
    /* <{phi,res}|H|res> */
    linalg<CPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec(), complex_one, phi__, 0, 0, hphi__, 0, N__, complex_zero, h__, 0, N__);
    /* <{phi,res}|O|res> */
    linalg<CPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec(), complex_one, phi__, 0, 0, ophi__, 0, N__, complex_zero, o__, 0, N__);
    double tval = t2.stop();

    if (verbosity_level >= 10 && kp__->comm().rank() == 0)
    {
        printf("pzgemm #4&5 with M, N, K: %6i %6i %6i, offset in B&C: %6i, %12.4f sec, %12.4f GFlops/rank\n",
               N__ + n__, n__, kp__->num_gkvec(), N__,
               tval, 2 * 8e-9 * (N__ + n__) * n__ * kp__->num_gkvec() / tval / kp__->num_ranks());
    }
    
    /* restore the bottom block of the matrix */
    if (N__ != 0)
    {
        linalg<CPU>::tranc(n__, N__, h__, 0, N__, h__, N__, 0);
        linalg<CPU>::tranc(n__, N__, o__, 0, N__, o__, N__, 0);
    }

    /* save Hamiltonian and overlap */
    for (int i = 0; i < (int)s1_col.local_size(); i++)
    {
        memcpy(&h_old__(0, i), &h__(0, i), s1_row.local_size() * sizeof(double_complex));
        memcpy(&o_old__(0, i), &o__(0, i), s1_row.local_size() * sizeof(double_complex));
    }
}

void Band::set_fv_h_o_fast_parallel(int N__,
                                    int n__,
                                    K_point* kp__,
                                    matrix<double_complex>& phi_slab__,
                                    matrix<double_complex>& hphi_slab__,
                                    matrix<double_complex>& ophi_slab__,
                                    dmatrix<double_complex>& h__,
                                    dmatrix<double_complex>& o__,
                                    dmatrix<double_complex>& h_old__,
                                    dmatrix<double_complex>& o_old__,
                                    mdarray<double_complex, 1>& kappa__)
{
    LOG_FUNC_BEGIN();

    Timer t("sirius::Band::set_fv_h_o_fast_parallel", kp__->comm());

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

    matrix<double_complex> tmp;
    switch (parameters_.processing_unit())
    {
        case CPU:
        {
            tmp = matrix<double_complex>(kappa__.at<CPU>(), N__ + n__, n__);
            break;
        }
        case GPU:
        {
            #ifdef _GPU_
            tmp = matrix<double_complex>(kappa__.at<CPU>(), kappa__.at<GPU>(), N__ + n__, n__);
            #endif
            break;
        }
    }
    
    int col_offs = (int)s0_col.local_size();
    Timer t2("sirius::Band::set_fv_h_o_fast_parallel|zgemm_eff", kp__->comm());
    switch (parameters_.processing_unit())
    {
        case CPU:
        {
            linalg<CPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec_loc(), phi_slab__.at<CPU>(), phi_slab__.ld(), 
                              hphi_slab__.at<CPU>(0, N__), hphi_slab__.ld(), tmp.at<CPU>(), tmp.ld());
            break;
        }
        case GPU:
        {
            #ifdef _GPU_
            linalg<GPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec_loc(), phi_slab__.at<GPU>(), phi_slab__.ld(),
                              hphi_slab__.at<GPU>(0, N__), hphi_slab__.ld(), tmp.at<GPU>(), tmp.ld());
            tmp.copy_to_host();
            #endif
            break;
        }
    }
    kp__->comm().allreduce(tmp.at<CPU>(), (int)tmp.size());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)(s1_col.local_size() - col_offs); i++)
    {
        for (int j = 0; j < (int)s1_row.local_size(); j++)
        {
            h__(j, col_offs + i) = tmp(s1_row[j], s1_col[col_offs + i] - N__);
        }
    }

    switch (parameters_.processing_unit())
    {
        case CPU:
        {
            linalg<CPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec_loc(), phi_slab__.at<CPU>(), phi_slab__.ld(), 
                              ophi_slab__.at<CPU>(0, N__), ophi_slab__.ld(), tmp.at<CPU>(), tmp.ld());
            break;
        }
        case GPU:
        {
            #ifdef _GPU_
            linalg<GPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec_loc(), phi_slab__.at<GPU>(), phi_slab__.ld(),
                              ophi_slab__.at<GPU>(0, N__), ophi_slab__.ld(), tmp.at<GPU>(), tmp.ld());
            tmp.copy_to_host();
            #endif
            break;
        }
    }
    kp__->comm().allreduce(tmp.at<CPU>(), (int)tmp.size());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)(s1_col.local_size() - col_offs); i++)
    {
        for (int j = 0; j < (int)s1_row.local_size(); j++)
        {
            o__(j, col_offs + i) = tmp(s1_row[j], s1_col[col_offs + i] - N__);
        }
    }
    double tval = t2.stop();

    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    {
        printf("effective zgemm with M, N, K: %6i %6i %6i for H and O: %12.4f sec, %12.4f GFlops/rank\n",
               N__ + n__, n__, kp__->num_gkvec(), tval,
               2 * 8e-9 * (N__ + n__) * n__ * kp__->num_gkvec() / tval / kp__->num_ranks());
    }
    
    /* restore the bottom block of the matrix */
    if (N__ != 0)
    {
        linalg<CPU>::tranc(n__, N__, h__, 0, N__, h__, N__, 0);
        linalg<CPU>::tranc(n__, N__, o__, 0, N__, o__, N__, 0);
    }

    /* save Hamiltonian and overlap */
    for (int i = 0; i < (int)s1_col.local_size(); i++)
    {
        memcpy(&h_old__(0, i), &h__(0, i), s1_row.local_size() * sizeof(double_complex));
        memcpy(&o_old__(0, i), &o__(0, i), s1_row.local_size() * sizeof(double_complex));
    }

    LOG_FUNC_END();
}

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
    Timer t("sirius::Band::set_fv_h_o_parallel", kp__->comm());
    
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

    assert(max_num_phi_new * 4 <= (int)kappa__.size(1));

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
   
    Timer t1("sirius::Band::set_fv_h_o_parallel|zgemm_eff", kp__->comm());

    auto pu = parameters_.processing_unit();

    //== struct col_to_row_scatter
    //== {
    //==     /* index (in h and o) of each vector for each row rank */
    //==     std::vector< std::vector<int> > ivec_row;

    //==     /* index in (|hphi> or |ophi>) of each vector for each row rank */
    //==     std::vector< std::vector<int> > idx_col;

    //==     /* offsets in the send buffer */
    //==     std::vector<int> offset;
    //== };

    //== std::vector<col_to_row_scatter> scatter_desc(kp__->num_ranks_col());


    //== for (int icol = 0; icol < kp__->num_ranks_col(); icol++)
    //== {
    //==     scatter_desc[icol].ivec_row = std::vector< std::vector<int> >(kp__->num_ranks_row());
    //==     scatter_desc[icol].idx_col = std::vector< std::vector<int> >(kp__->num_ranks_row());
    //==     
    //==     /* local number of new basis functions */
    //==     int nloc = (int)(s1_col.local_size(icol) - s0_col.local_size(icol));

    //==     for (int j = 0; j < nloc; j++)
    //==     {
    //==         int idx_hphi_glob = (int)s1_col.global_index(s0_col.local_size(icol) + j, icol);
    //==         auto p = s1_row.location(idx_hphi_glob);

    //==         scatter_desc[icol].ivec_row[p.second].push_back((int)p.first);
    //==         scatter_desc[icol].idx_col[p.second].push_back((int)(s0_col.local_size(icol) + j));
    //==     }
    //==     scatter_desc[icol].offset = std::vector<int>(kp__->num_ranks_row());
    //==     int i = 0;
    //==     for (int j = 0; j < kp__->num_ranks_row(); j++)
    //==     {
    //==          scatter_desc[icol].offset[j] = i;
    //==          i += (int)scatter_desc[icol].idx_col.size();
    //==     }
    //== }

    
    /* auxiliary function to broadcast vectors (|hphi> or |ophi>) from a column icol
     *
     *    column ranks 
     *    0   1   2   3  ...
     *  +---+---+---+---+---
     *  |   | <-+-*-+-> |
     *  +---+---+---+---+---
     *  |   | <-+-*-+-> |
     *  +---+---+---+---+---
     *  |   | <-+-*-+-> |
     *  .................
     */
    auto bcast_column = [kp__, &s0_col, &s1_col, pu]
                        (int icol,
                         dmatrix<double_complex>& mtrx,
                         mdarray<double_complex, 3>& mtrx_tmp,
                         std::array<std::atomic_bool, 2>& lock_tmp) -> void
    {
        Timer t("sirius::Band::set_fv_h_o_parallel|bcast_column");

        #ifdef _GPU_
        #ifdef _GPU_DIRECT_
        bool gpu_direct = true;
        #else
        bool gpu_direct = false;
        #endif
        #endif

        /* wait for unlocking of this buffer */
        while (lock_tmp[icol % 2].load());
        
        /* number of vectors to broadcast */
        int nloc = (int)(s1_col.local_size(icol) - s0_col.local_size(icol));

        /* return if there is nothing to do */
        if (!nloc) return;
        
        /* total size of the panel to broadcast */
        size_t panel_size = kp__->num_gkvec_row() * nloc * sizeof(double_complex);
        
        if (pu == CPU)
        {
            if (kp__->rank_col() == icol)
            {
                /* this rank copies content of the vectors into temporary buffer */
                memcpy(mtrx_tmp.at<CPU>(0, 0, icol % 2), mtrx.at<CPU>(0, s0_col.local_size(icol)), panel_size);
            }
            /* buffer is broadcasted between columns */
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

        /* lock temporary buffer; it can't be used for the next broadcast until <phi|tmp> is computed */
        lock_tmp[icol % 2].store(true);
    };

    /* broadcast |hphi> from the first column */
    bcast_column(0, hphi__, hphi_tmp, lock_hphi);

    /* same for |ophi> or |phi> */
    if (with_overlap)
    {
        bcast_column(0, ophi__, ophi_tmp, lock_ophi);
    }
    else
    {
        bcast_column(0, phi__, ophi_tmp, lock_ophi);
    }

    /* get maximum number of threads */
    int nthread = omp_get_max_threads();

    /* one thread will be doing communication, others will do local zgemm */
    if (nthread > 1) omp_set_num_threads(nthread - 1);

    /* create communication thread */
    std::thread comm_thread([kp__, &s0_col, &s1_col, &s0_row, &s1_row, &lock_hphi, &lock_ophi, &lock_h, &lock_o, 
                             &hphi__, &ophi__, &hphi_tmp, &ophi_tmp, &h_tmp, &o_tmp, &h__, &o__, bcast_column, 
                             with_overlap, &phi__]()
    {
        int num_phi = (int)s1_col.local_size();
    
        for (int icol = 0; icol < kp__->num_ranks_col(); icol++)
        {
            /* local number of new basis functions */
            int nloc = (int)(s1_col.local_size(icol) - s0_col.local_size(icol));
            
            Timer t1("sirius::Band::set_fv_h_o_parallel|comm_thread|1");
            /* broadcast next column */
            if (icol + 1 < kp__->num_ranks_col())
            {
                bcast_column(icol + 1, hphi__, hphi_tmp, lock_hphi);
                
                if (with_overlap)
                {
                    bcast_column(icol + 1, ophi__, ophi_tmp, lock_ophi);
                }
                else
                {
                    bcast_column(icol + 1, phi__, ophi_tmp, lock_ophi);
                }
            }
            t1.stop();
    
            Timer t2("sirius::Band::set_fv_h_o_parallel|comm_thread|2");
            if (nloc > 0)
            {
                /* wait for locking of h-buffer which happens after a local zgemm;
                 * when this is done the reduction between rows can be performed 
                 */
                while (!lock_h[icol % 2].load());
                kp__->comm_row().allreduce(&h_tmp(0, 0, icol % 2), num_phi * nloc);
                
                /* cycle through the local fraction of new basis functions */
                for (int j = 0; j < nloc; j++)
                {
                    /* compute global index of hphi by local index and column rank */
                    int idx_hphi_glob = (int)s1_col.global_index(s0_col.local_size(icol) + j, icol);
                    /* and now compute row rank and local index */
                    auto p = s1_row.location(idx_hphi_glob);
                    /* check if this rank stores <hphi_tmp_j|phi_i> */ 
                    if (p.second == kp__->rank_row())
                    {
                        for (int i = 0; i < num_phi; i++)
                        {
                            h__(p.first, i) = conj(h_tmp(i, j, icol % 2));
                        }
                    }
                }
                /* when the reduction is done and the result is saved in h__, h-buffer can be unlocked */
                lock_h[icol % 2].store(false);
                
                /* the same with o-buffer */
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

        if (n > 0)
        {
            /* wait for broadcast of this column */
            while (!lock_hphi[icol % 2].load());
            /* wait for unlock of h buffer */
            while (lock_h[icol % 2].load());

            Timer t2("sirius::Band::set_fv_h_o_parallel|zgemm_loc");
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
                /* compute <phi|hphi_tmp> */
                linalg<CPU>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(), phi__.at<CPU>(), phi__.ld(),
                                  hphi_tmp.at<CPU>(0, 0, icol % 2), hphi_tmp.ld(), h_tmp.at<CPU>(0, 0, icol % 2), h_tmp.ld());
            }
            lock_h[icol % 2].store(true);
            lock_hphi[icol % 2].store(false);
        }
            
        if (n > 0)
        {
            while (!lock_ophi[icol % 2].load());
            while (lock_o[icol % 2].load());

            Timer t2("sirius::Band::set_fv_h_o_parallel|zgemm_loc");
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
        printf("effective zgemm #4&5 with M, N, K: %6i %6i %6i,                        %12.4f sec, %12.4f GFlops/rank\n",
               N__ + n__, n__, kp__->num_gkvec(),
               tval, 2 * 8e-9 * (N__ + n__) * n__ * kp__->num_gkvec() / tval / kp__->num_ranks());
    }

    /* restore right block of the matrix */
    if (N__ != 0)
    {
        Timer t1("sirius::Band::set_fv_h_o_parallel|transpose");
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

void Band::precondition_and_normalize_residuals_parallel(int num_bands__,
                                                         K_point* kp__,
                                                         std::vector<double>& eval__,
                                                         dmatrix<double_complex>& hpsi__,
                                                         dmatrix<double_complex>& opsi__,
                                                         dmatrix<double_complex>& res__,
                                                         std::vector<double>& h_diag__,
                                                         std::vector<double>& o_diag__,
                                                         std::vector<double>& res_norm__)

{
    splindex<block_cyclic> spl_num_bands_col(num_bands__, kp__->num_ranks_col(), kp__->rank_col(),
                                             blacs_grid_.cyclic_block_size());

    memset(&res_norm__[0], 0, num_bands__ * sizeof(double));
    /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} and norm squared */
    #pragma omp parallel for
    for (int i = 0; i < (int)spl_num_bands_col.local_size(); i++)
    {
        int ires = (int)spl_num_bands_col[i];
        double norm2 = 0;
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) 
        {
            res__(igk_row, i) = hpsi__(igk_row, i) - eval__[ires] * opsi__(igk_row, i);
            norm2 += real(conj(res__(igk_row, i)) * res__(igk_row, i));
        }
        res_norm__[ires] = norm2;
    }
    kp__->comm().allreduce(res_norm__);
    
    /* compute norm */
    for (int i = 0; i < num_bands__; i++) res_norm__[i] = std::sqrt(res_norm__[i]);
    
    /* apply preconditioner */
    #pragma omp parallel for
    for (int i = 0; i < (int)spl_num_bands_col.local_size(); i++)
    {
        int ires = (int)spl_num_bands_col[i];
    
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
        {
            double p = h_diag__[igk_row] - eval__[ires] * o_diag__[igk_row];

            p *= 2; // QE formula is in Ry; here we convert to Ha
            p = 0.25 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
            res__(igk_row, i) /= p;
        }
    }
    
    std::vector<double> norm2(num_bands__, 0);
    /* Normalize new basis functions */
    #pragma omp parallel for
    for (int i = 0; i < (int)spl_num_bands_col.local_size(); i++)
    {
        int ires = (int)spl_num_bands_col[i];
        double d = 0;
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) 
            d += real(conj(res__(igk_row, i)) * res__(igk_row, i));
        norm2[ires] = d;
    }
    kp__->comm().allreduce(norm2);
    #pragma omp parallel for
    for (int i = 0; i < (int)spl_num_bands_col.local_size(); i++)
    {
        int ires = (int)spl_num_bands_col[i];
        double d = 1.0 / std::sqrt(norm2[ires]);
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) res__(igk_row, i) *= d;
    }
}

void Band::residuals_parallel_simple(int N__,
                                     int num_bands__,
                                     K_point* kp__,
                                     std::vector<double>& eval__,
                                     dmatrix<double_complex>& evec__,
                                     dmatrix<double_complex>& hphi__,
                                     dmatrix<double_complex>& ophi__,
                                     dmatrix<double_complex>& hpsi__,
                                     dmatrix<double_complex>& opsi__,
                                     dmatrix<double_complex>& res__,
                                     std::vector<double>& h_diag__,
                                     std::vector<double>& o_diag__,
                                     std::vector<double>& res_norm__)
{
    Timer t("sirius::Band::residuals_parallel_simple");
    
    Timer t2("sirius::Band::residuals_parallel_simple|zgemm");
    /* compute H\Psi_{i} = H\phi_{mu} * Z_{mu, i} */
    linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, complex_one, hphi__, evec__, complex_zero, hpsi__);
    /* compute O\Psi_{i} = O\phi_{mu} * Z_{mu, i} */
    linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, complex_one, ophi__, evec__, complex_zero, opsi__);
    double tval = t2.stop();

    if (verbosity_level >= 10 && kp__->comm().rank() == 0)
    {
        printf("pzgemm #6&7 with M, N, K: %6i %6i %6i,                        %12.4f sec, %12.4f GFlops/rank\n",
               kp__->num_gkvec(), num_bands__, N__,
               tval, 2 * 8e-9 * kp__->num_gkvec() * num_bands__ * N__ / tval / kp__->num_ranks());
    }
    
    precondition_and_normalize_residuals_parallel(num_bands__, kp__, eval__, hpsi__, opsi__, res__, h_diag__, o_diag__, res_norm__);
}

void Band::residuals_fast_parallel(int N__,
                                   int num_bands__,
                                   K_point* kp__,
                                   std::vector<double>& eval__,
                                   matrix<double_complex>& evec__,
                                   matrix<double_complex>& hphi__,
                                   matrix<double_complex>& ophi__,
                                   matrix<double_complex>& hpsi__,
                                   matrix<double_complex>& opsi__,
                                   matrix<double_complex>& res__,
                                   std::vector<double>& h_diag__,
                                   std::vector<double>& o_diag__,
                                   std::vector<double>& res_norm__,
                                   mdarray<double_complex, 1>& kappa__)
{
    LOG_FUNC_BEGIN();

    Timer t("sirius::Band::residuals_fast_parallel", kp__->comm());

    int num_gkvec_loc = kp__->num_gkvec_loc();
    
    Timer t2("sirius::Band::residuals_fast_parallel|zgemm");
    if (parameters_.processing_unit() == CPU)
    {
        /* compute H\Psi_{i} = H\phi_{mu} * Z_{mu, i} */
        linalg<CPU>::gemm(0, 0, num_gkvec_loc, num_bands__, N__, hphi__, evec__, hpsi__);
        /* compute O\Psi_{i} = O\phi_{mu} * Z_{mu, i} */
        linalg<CPU>::gemm(0, 0, num_gkvec_loc, num_bands__, N__, ophi__, evec__, opsi__);
    }

    if (parameters_.processing_unit() == GPU)
    {
        #ifdef _GPU_
        linalg<GPU>::gemm(0, 0, num_gkvec_loc, num_bands__, N__, hphi__.at<GPU>(), hphi__.ld(),
                          evec__.at<GPU>(), evec__.ld(), hpsi__.at<GPU>(), hpsi__.ld());

        cublas_get_matrix(num_gkvec_loc, num_bands__, sizeof(double_complex), hpsi__.at<GPU>(), hpsi__.ld(),
                          hpsi__.at<CPU>(), hpsi__.ld());

        linalg<GPU>::gemm(0, 0, num_gkvec_loc, num_bands__, N__, ophi__.at<GPU>(), ophi__.ld(),
                          evec__.at<GPU>(), evec__.ld(), opsi__.at<GPU>(), opsi__.ld());

        cublas_get_matrix(num_gkvec_loc, num_bands__, sizeof(double_complex), opsi__.at<GPU>(), opsi__.ld(),
                          opsi__.at<CPU>(), opsi__.ld());
        #endif
    }
    double tval = t2.stop();


    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    {
        printf("effective zgemm with M, N, K: %6i %6i %6i for hpsi and opsi: %12.4f sec, %12.4f GFlops/rank\n",
               kp__->num_gkvec(), num_bands__, N__, tval,
               2 * 8e-9 * kp__->num_gkvec() * num_bands__ * N__ / tval / kp__->num_ranks());
    }
    
    memset(&res_norm__[0], 0, num_bands__ * sizeof(double));
    /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} and norm squared */
    #pragma omp parallel for
    for (int i = 0; i < num_bands__; i++)
    {
        double norm2 = 0;
        for (int igk = 0; igk < num_gkvec_loc; igk++) 
        {
            res__(igk, i) = hpsi__(igk, i) - eval__[i] * opsi__(igk, i);
            norm2 += real(conj(res__(igk, i)) * res__(igk, i));
        }
        res_norm__[i] = norm2;
    }
    kp__->comm().allreduce(res_norm__);
    
    /* compute norm */
    for (int i = 0; i < num_bands__; i++) res_norm__[i] = std::sqrt(res_norm__[i]);

    /* apply preconditioner */
    #pragma omp parallel for
    for (int i = 0; i < num_bands__; i++)
    {
        for (int igk = 0; igk < num_gkvec_loc; igk++)
        {
            double p = h_diag__[igk] - eval__[i] * o_diag__[igk];

            p *= 2; // QE formula is in Ry; here we convert to Ha
            p = 0.25 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
            res__(igk, i) /= p;
        }
    }
    
    std::vector<double> norm2(num_bands__, 0);
    /* Normalize new basis functions */
    #pragma omp parallel for
    for (int i = 0; i < num_bands__; i++)
    {
        double d = 0;
        for (int igk = 0; igk < num_gkvec_loc; igk++) 
            d += real(conj(res__(igk, i)) * res__(igk, i));
        norm2[i] = d;
    }
    kp__->comm().allreduce(norm2);
    #pragma omp parallel for
    for (int i = 0; i < num_bands__; i++)
    {
        double d = 1.0 / std::sqrt(norm2[i]);
        for (int igk = 0; igk < num_gkvec_loc; igk++) res__(igk, i) *= d;
    }

    LOG_FUNC_END();
}

void Band::residuals_parallel(int N__,
                              int num_bands__,
                              K_point* kp__,
                              std::vector<double>& eval__,
                              dmatrix<double_complex>& evec__,
                              dmatrix<double_complex>& hphi__,
                              dmatrix<double_complex>& ophi__,
                              dmatrix<double_complex>& hpsi__,
                              dmatrix<double_complex>& opsi__,
                              dmatrix<double_complex>& res__,
                              std::vector<double>& h_diag__,
                              std::vector<double>& o_diag__,
                              std::vector<double>& res_norm__,
                              mdarray<double_complex, 2>& kappa__)
{
    log_function_enter(__func__);
    Timer t("sirius::Band::residuals_parallel", kp__->comm());

    Timer t1("sirius::Band::residuals_parallel|zgemm_eff", kp__->comm());

    auto pu = parameters_.processing_unit();

    splindex<block_cyclic> spl_num_bands_col(num_bands__, kp__->num_ranks_col(), kp__->rank_col(),
                                             blacs_grid_.cyclic_block_size());
    splindex<block_cyclic> spl_num_bands_row(num_bands__, kp__->num_ranks_row(), kp__->rank_row(),
                                             blacs_grid_.cyclic_block_size());
    
    /* transpose matrix of eigen-vectors;
     * row index of evec_t runs over bands, column index runs over basis functions 
     */ 
    dmatrix<double_complex> evec_t(num_bands__, N__, kp__->blacs_grid());
    linalg<CPU>::tranu(num_bands__, N__, evec__, 0, 0, evec_t, 0, 0);
    
    /* local number of basis function |phi> */
    int num_phi_loc = evec_t.num_cols_local();

    mdarray<double_complex, 3> evec_tmp(num_phi_loc, spl_num_bands_col.local_size(0), 2);
    #ifdef _GPU_
    if (pu == GPU) evec_tmp.allocate_on_device();
    #endif
    
    std::array<std::atomic_bool, 2> lock_evec_tmp;
    std::atomic_bool lock_hpsi_tmp;
    std::atomic_bool lock_opsi_tmp;
    for (int i = 0; i < 2; i++) lock_evec_tmp[i].store(false);
    lock_hpsi_tmp.store(false);
    lock_opsi_tmp.store(false);

    /* maximum local number of bands */
    int num_bnd_max = (int)spl_num_bands_col.local_size(0);

    matrix<double_complex> hpsi_tmp, opsi_tmp;

    switch (parameters_.processing_unit())
    {
        case CPU:
        {
            hpsi_tmp = matrix<double_complex>(kappa__.at<CPU>(0, 0),           kp__->num_gkvec_row(), num_bnd_max);
            opsi_tmp = matrix<double_complex>(kappa__.at<CPU>(0, num_bnd_max), kp__->num_gkvec_row(), num_bnd_max);
            break;
        }
        case GPU:
        {
            hpsi_tmp = matrix<double_complex>(kappa__.at<CPU>(0, 0),           kappa__.at<GPU>(0, 0), 
                                              kp__->num_gkvec_row(), num_bnd_max);
            opsi_tmp = matrix<double_complex>(kappa__.at<CPU>(0, num_bnd_max), kappa__.at<GPU>(0, num_bnd_max),
                                              kp__->num_gkvec_row(), num_bnd_max);
            break;
        }
    }
    
    /* get eigen-vectors for a specific column */
    auto get_evec = [kp__, &spl_num_bands_col, &spl_num_bands_row, &evec_t, &evec_tmp, num_phi_loc, pu]
                    (int icol) -> void 
    {
        /* number of bands for this column */
        int num_bands_of_col = (int)spl_num_bands_col.local_size(icol);
        
        /* zero temporary buffer */
        memset(&evec_tmp(0, 0, icol % 2), 0, num_phi_loc * num_bands_of_col * sizeof(double_complex));

        /* loop over local fraction of bands */
        for (int i = 0; i < num_bands_of_col; i++)
        {
            /* global index of band */
            int iglob = (int)spl_num_bands_col.global_index(i, icol);

            /* location of the global band index in the row of evec_t */
            auto p = spl_num_bands_row.location(iglob); 
            
            /* pick elements of evec_t from row ranks */
            if (p.second == kp__->rank_row())
            {
                for (int j = 0; j < num_phi_loc; j++) evec_tmp(j, i, icol % 2) = evec_t(p.first, j);
            }
        }

        /* reduce evec_tmp; now it contains fraction of the expansion coefficients of bands for the given column
         * over the basis function local to this column 
         */
        kp__->comm_row().allreduce(&evec_tmp(0, 0, icol % 2), num_phi_loc * num_bands_of_col);
        #ifdef _GPU_
        if (pu == GPU)
        {
            /* send evec to gpu */
            cuda_copy_to_device(evec_tmp.at<GPU>(0, 0, icol % 2), evec_tmp.at<CPU>(0, 0, icol % 2), 
                                num_phi_loc * num_bands_of_col * sizeof(double_complex));
        }
        #endif
    };

    /* get evec for first column */
    get_evec(0);
    lock_evec_tmp[0].store(true);
    
    /* communication thread */
    std::thread comm_thread([kp__, &lock_evec_tmp, &lock_hpsi_tmp, &hpsi_tmp, &hpsi__, 
                             &lock_opsi_tmp, &opsi_tmp, &opsi__, &spl_num_bands_col, get_evec, pu]()
    {
        #ifdef _GPU_
        #ifdef _GPU_DIRECT_
        // gpu-direct is not working at the moment
        bool gpu_direct = false;
        #else
        bool gpu_direct = false;
        #endif
        #endif

        for (int icol = 0; icol < kp__->num_ranks_col(); icol++)
        {
            int num_bands_of_col = (int)spl_num_bands_col.local_size(icol);
            if (icol + 1 < kp__->num_ranks_col())
            {
                while (lock_evec_tmp[(icol + 1) % 2].load());
                get_evec(icol + 1);
                lock_evec_tmp[(icol + 1) % 2].store(true);
            }
            
            while (!lock_hpsi_tmp.load());
            switch (pu)
            {
                case CPU:
                {
                    kp__->comm_col().reduce(hpsi_tmp.at<CPU>(), hpsi__.at<CPU>(), kp__->num_gkvec_row() * num_bands_of_col, icol);
                    break;
                }
                case GPU:
                {
                    #ifdef _GPU_
                    if (gpu_direct)
                    {
                        kp__->comm_col().reduce(hpsi_tmp.at<GPU>(), hpsi__.at<GPU>(), kp__->num_gkvec_row() * num_bands_of_col, icol);
                    } 
                    else
                    {
                        cuda_copy_to_host(hpsi_tmp.at<CPU>(), hpsi_tmp.at<GPU>(), kp__->num_gkvec_row() * num_bands_of_col * sizeof(double_complex));
                        kp__->comm_col().reduce(hpsi_tmp.at<CPU>(), hpsi__.at<CPU>(), kp__->num_gkvec_row() * num_bands_of_col, icol);
                        if (icol == kp__->rank_col())
                            cuda_copy_to_device(hpsi__.at<GPU>(), hpsi__.at<CPU>(), kp__->num_gkvec_row() * num_bands_of_col * sizeof(double_complex));
                    }
                    #endif
                    break;
                }
            }
            lock_hpsi_tmp.store(false);

            while (!lock_opsi_tmp.load());
            switch (pu)
            {
                case CPU:
                {
                    kp__->comm_col().reduce(opsi_tmp.at<CPU>(), opsi__.at<CPU>(), kp__->num_gkvec_row() * num_bands_of_col, icol);
                    break;
                }
                case GPU:
                {
                    #ifdef _GPU_
                    if (gpu_direct)
                    {
                        kp__->comm_col().reduce(opsi_tmp.at<GPU>(), opsi__.at<GPU>(), kp__->num_gkvec_row() * num_bands_of_col, icol);
                    }
                    else
                    {
                        cuda_copy_to_host(opsi_tmp.at<CPU>(), opsi_tmp.at<GPU>(), kp__->num_gkvec_row() * num_bands_of_col * sizeof(double_complex));
                        kp__->comm_col().reduce(opsi_tmp.at<CPU>(), opsi__.at<CPU>(), kp__->num_gkvec_row() * num_bands_of_col, icol);
                        if (icol == kp__->rank_col())
                            cuda_copy_to_device(opsi__.at<GPU>(), opsi__.at<CPU>(), kp__->num_gkvec_row() * num_bands_of_col * sizeof(double_complex));
                    }
                    #endif
                    break;
                }
            }
            lock_opsi_tmp.store(false);
        }
    });

    int nthread = omp_get_max_threads();
    if (nthread > 1) omp_set_num_threads(nthread - 1);

    for (int rank_col = 0; rank_col < kp__->num_ranks_col(); rank_col++)
    {
        int num_bands_of_rank = (int)spl_num_bands_col.local_size(rank_col);
        
        while (!lock_evec_tmp[rank_col % 2].load());
        
        while (lock_hpsi_tmp.load());
        switch (pu)
        {
            case CPU:
            {
                linalg<CPU>::gemm(0, 0, kp__->num_gkvec_row(), num_bands_of_rank, num_phi_loc, 
                                  hphi__.at<CPU>(), hphi__.ld(), evec_tmp.at<CPU>(0, 0, rank_col % 2), evec_tmp.ld(), 
                                  hpsi_tmp.at<CPU>(), hpsi_tmp.ld());
                break;
            }
            case GPU:
            {
                #ifdef _GPU_
                linalg<GPU>::gemm(0, 0, kp__->num_gkvec_row(), num_bands_of_rank, num_phi_loc, 
                                  hphi__.at<GPU>(), hphi__.ld(), evec_tmp.at<GPU>(0, 0, rank_col % 2), evec_tmp.ld(), 
                                  hpsi_tmp.at<GPU>(), hpsi_tmp.ld());
                cuda_device_synchronize();
                break;
                #endif
            }
        }
        lock_hpsi_tmp.store(true);
       
        while (lock_opsi_tmp.load());
        switch (pu)
        {
            case CPU:
            {
                linalg<CPU>::gemm(0, 0, kp__->num_gkvec_row(), num_bands_of_rank, num_phi_loc, 
                                  ophi__.at<CPU>(), ophi__.ld(), evec_tmp.at<CPU>(0, 0, rank_col % 2), evec_tmp.ld(), 
                                  opsi_tmp.at<CPU>(), opsi_tmp.ld());
                break;
            }
            case GPU:
            {
                #ifdef _GPU_
                linalg<GPU>::gemm(0, 0, kp__->num_gkvec_row(), num_bands_of_rank, num_phi_loc, 
                                  ophi__.at<GPU>(), ophi__.ld(), evec_tmp.at<GPU>(0, 0, rank_col % 2), evec_tmp.ld(), 
                                  opsi_tmp.at<GPU>(), opsi_tmp.ld());
                cuda_device_synchronize();
                break;
                #endif
            }
        }
        lock_opsi_tmp.store(true);

        lock_evec_tmp[rank_col % 2].store(false);
    }
    comm_thread.join();
    
    omp_set_num_threads(nthread);

    double tval = t1.stop();

    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    {
        printf("effective zgemm #6&7 with M, N, K: %6i %6i %6i,                        %12.4f sec, %12.4f GFlops/rank\n",
               kp__->num_gkvec(), num_bands__, N__,
               tval, 2 * 8e-9 * kp__->num_gkvec() * num_bands__ * N__ / tval / kp__->num_ranks());
    }

    precondition_and_normalize_residuals_parallel(num_bands__, kp__, eval__, hpsi__, opsi__, res__, h_diag__, o_diag__, res_norm__);

    //memset(&res_norm__[0], 0, num_bands__ * sizeof(double));

    //if (pu == CPU)
    //{
    //    /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} and norm squared */
    //    #pragma omp parallel for
    //    for (int i = 0; i < res__.num_cols_local(); i++)
    //    {
    //        int ires = res__.icol(i);
    //        double norm2 = 0;
    //        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) 
    //        {
    //            res__(igk_row, i) = hpsi__(igk_row, i) - eval__[ires] * opsi__(igk_row, i);
    //            norm2 += real(conj(res__(igk_row, i)) * res__(igk_row, i));
    //        }
    //        res_norm__[ires] = norm2;
    //    }
    //    kp__->comm().allreduce(res_norm__);
    //    
    //    /* compute norm */
    //    for (int i = 0; i < num_bands__; i++) res_norm__[i] = std::sqrt(res_norm__[i]);
    //    
    //    /* apply preconditioner */
    //    #pragma omp parallel for
    //    for (int i = 0; i < res__.num_cols_local(); i++)
    //    {
    //        int ires = res__.icol(i);
    //    
    //        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
    //        {
    //            double_complex z = h_diag__[igk_row] - eval__[ires] * o_diag__[igk_row];
    //            if (std::abs(z) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
    //            res__(igk_row, i) /= z;
    //        }
    //    }
    //    
    //    std::vector<double> norm2(num_bands__, 0);
    //    /* normalize new basis functions */
    //    #pragma omp parallel for
    //    for (int i = 0; i < res__.num_cols_local(); i++)
    //    {
    //        int ires = res__.icol(i);
    //        double d = 0;
    //        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) 
    //            d += real(conj(res__(igk_row, i)) * res__(igk_row, i));
    //        norm2[ires] = d;
    //    }
    //    kp__->comm().allreduce(norm2);
    //    #pragma omp parallel for
    //    for (int i = 0; i < res__.num_cols_local(); i++)
    //    {
    //        int ires = res__.icol(i);
    //        double d = 1.0 / std::sqrt(norm2[ires]);
    //        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) res__(igk_row, i) *= d;
    //    }
    //}

    //if (pu == GPU)
    //{
    //    #ifdef _GPU_
    //    mdarray<double, 1> res_norm_gpu(&res_norm__[0], num_bands__);
    //    res_norm_gpu.allocate_on_device();
    //    res_norm_gpu.zero_on_device();

    //    mdarray<double, 1> eval_gpu(&eval__[0], num_bands__);
    //    eval_gpu.allocate_on_device();
    //    eval_gpu.copy_to_device();

    //    mdarray<int, 1> res_idx_gpu(res__.num_cols_local());
    //    for (int i = 0; i < res__.num_cols_local(); i++) res_idx_gpu(i) = res__.icol(i);
    //    res_idx_gpu.allocate_on_device();
    //    res_idx_gpu.copy_to_device();

    //    compute_residuals_gpu(kp__->num_gkvec_row(), res__.num_cols_local(), res_idx_gpu.at<GPU>(), eval_gpu.at<GPU>(),
    //                          hpsi__.at<GPU>(), opsi__.at<GPU>(), res__.at<GPU>(), res_norm_gpu.at<GPU>());
    //    res_norm_gpu.copy_to_host();

    //    kp__->comm().allreduce(res_norm__);
    //    
    //    /* compute norm */
    //    for (int i = 0; i < num_bands__; i++) res_norm__[i] = std::sqrt(res_norm__[i]);

    //    mdarray<double, 1> hdiag_gpu(&h_diag__[0], kp__->num_gkvec_row());
    //    hdiag_gpu.allocate_on_device();
    //    hdiag_gpu.copy_to_device();

    //    mdarray<double, 1> odiag_gpu(&o_diag__[0], kp__->num_gkvec_row());
    //    odiag_gpu.allocate_on_device();
    //    odiag_gpu.copy_to_device();

    //    mdarray<double, 1> norm2(num_bands__);
    //    norm2.allocate_on_device();
    //    norm2.zero_on_device();

    //    apply_preconditioner_gpu(kp__->num_gkvec_row(), res__.num_cols_local(), res_idx_gpu.at<GPU>(), eval_gpu.at<GPU>(),
    //                             hdiag_gpu.at<GPU>(), odiag_gpu.at<GPU>(), res__.at<GPU>(), norm2.at<GPU>());
    //    // TODO: test gpudirect here
    //    norm2.copy_to_host();
    //    kp__->comm().allreduce(norm2.at<CPU>(), num_bands__);
    //    norm2.copy_to_device();

    //    normalize_residuals_gpu(kp__->num_gkvec_row(), res__.num_cols_local(), res_idx_gpu.at<GPU>(),
    //                            norm2.at<GPU>(), res__.at<GPU>());
    //    #else
    //    TERMINATE_NO_GPU
    //    #endif
    //}

    log_function_exit(__func__);
}
#endif // _SCALAPACK_

/** \param [in] phi Input wave-functions [storage: CPU && GPU].
 *  \param [out] hphi Hamiltonian, applied to wave-functions [storage: CPU || GPU].
 */
void Band::apply_h_serial(K_point* kp__, 
                          std::vector<double> const& effective_potential__, 
                          std::vector<double> const& pw_ekin__, 
                          int N__,
                          int n__,
                          matrix<double_complex>& phi__,
                          matrix<double_complex>& hphi__,
                          matrix<double_complex>& kappa__,
                          mdarray<int, 1>& packed_mtrx_offset__,
                          mdarray<double_complex, 1>& d_mtrx_packed__)
{
    Timer t("sirius::Band::apply_h_serial");

    auto uc = parameters_.unit_cell();

    matrix<double_complex> phi, hphi;
    
    /* if temporary array is allocated, this would be the only big array on GPU */
    bool economize_gpu_memory = (kappa__.size() != 0);

    if (parameters_.processing_unit() == CPU)
    {
        phi  = matrix<double_complex>( phi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
        hphi = matrix<double_complex>(hphi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
    }
    if (parameters_.processing_unit() == GPU && !economize_gpu_memory)
    {
        phi  = matrix<double_complex>( phi__.at<CPU>(0, N__),  phi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
        hphi = matrix<double_complex>(hphi__.at<CPU>(0, N__), hphi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
    }
    if (parameters_.processing_unit() == GPU && economize_gpu_memory)
    {
        double_complex* gpu_ptr = kappa__.at<GPU>(0, parameters_.unit_cell()->mt_basis_size());
        phi  = matrix<double_complex>( phi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
        hphi = matrix<double_complex>(hphi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
    }
    
    /* apply local part of Hamiltonian */
    apply_h_local_slice(kp__, effective_potential__, pw_ekin__, n__, phi, hphi);
    
    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU && !economize_gpu_memory)
    {
        /* copy hphi do device */
        hphi.copy_to_device();
    }
    #endif

    /* <\beta_{\xi}^{\alpha}|\phi_j> */
    matrix<double_complex> beta_phi(uc->mt_lo_basis_size(), n__);

    /* D multiplied by <\beta_{\xi}^{\alpha}|\phi_j> */
    matrix<double_complex> work(uc->mt_lo_basis_size(), n__);

    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU)
    {
        beta_phi.allocate_on_device();
        work.allocate_on_device();
    }
    #endif

    if (parameters_.processing_unit() == CPU || (parameters_.processing_unit() == GPU && !economize_gpu_memory))
    {
        kp__->generate_beta_phi(uc->mt_lo_basis_size(), phi, n__, 0, kp__->beta_gk(), beta_phi);

        kp__->add_non_local_contribution(uc->num_atoms(), uc->mt_lo_basis_size(), uc->beta_chunk(0).desc_,
                                         kp__->beta_gk(), d_mtrx_packed__, packed_mtrx_offset__, beta_phi,
                                         hphi, n__, 0, complex_one, work);
    }
    else
    {
        #ifdef _GPU_
        kp__->generate_beta_gk(uc->num_atoms(), uc->beta_chunk(0).atom_pos_, uc->beta_chunk(0).desc_, kappa__);
        phi.copy_to_device();
        kp__->generate_beta_phi(uc->mt_lo_basis_size(), phi, n__, 0, kappa__, beta_phi);
        
        hphi.copy_to_device();
        kp__->add_non_local_contribution(uc->num_atoms(), uc->mt_lo_basis_size(), uc->beta_chunk(0).desc_,
                                         kappa__, d_mtrx_packed__, packed_mtrx_offset__, beta_phi,
                                         hphi, n__, 0, complex_one, work);
        hphi.copy_to_host();
        #else
        TERMINATE_NO_GPU
        #endif
    }
    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU) cuda_device_synchronize();
    #endif
}

/** \param [in] phi Input wave-functions [storage: CPU || GPU].
 *  \param [out] op_phi Result of application of operator to the wave-functions [storage: CPU || GPU].
 */
void Band::add_non_local_contribution_serial(K_point* kp__,
                                             int N__,
                                             int n__,
                                             matrix<double_complex>& phi__,
                                             matrix<double_complex>& op_phi__, 
                                             matrix<double_complex>& kappa__,
                                             mdarray<int, 1> const& packed_mtrx_offset__,
                                             mdarray<double_complex, 1>& op_mtrx_packed__,
                                             double_complex alpha)
{
    log_function_enter(__func__);
    Timer t("sirius::Band::add_non_local_contribution_serial");

    auto uc = parameters_.unit_cell();

    matrix<double_complex> phi, op_phi;
    
    /* if temporary array is allocated, this would be the only big array on GPU */
    bool economize_gpu_memory = (kappa__.size() != 0);

    if (parameters_.processing_unit() == CPU)
    {
        phi    = matrix<double_complex>(   phi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
        op_phi = matrix<double_complex>(op_phi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
    }
    if (parameters_.processing_unit() == GPU && !economize_gpu_memory)
    {
        phi    = matrix<double_complex>(   phi__.at<CPU>(0, N__),    phi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
        op_phi = matrix<double_complex>(op_phi__.at<CPU>(0, N__), op_phi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
    }
    if (parameters_.processing_unit() == GPU && economize_gpu_memory)
    {
        double_complex* gpu_ptr = kappa__.at<GPU>(0, parameters_.unit_cell()->mt_basis_size());
        phi    = matrix<double_complex>(   phi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
        op_phi = matrix<double_complex>(op_phi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
    }
    
    /* <\beta_{\xi}^{\alpha}|\phi_j> */
    matrix<double_complex> beta_phi(uc->mt_lo_basis_size(), n__);

    /* operator multiplied by <\beta_{\xi}^{\alpha}|\phi_j> */
    matrix<double_complex> work(uc->mt_lo_basis_size(), n__);

    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU)
    {
        beta_phi.allocate_on_device();
        work.allocate_on_device();
    }
    #endif

    if (parameters_.processing_unit() == CPU || (parameters_.processing_unit() == GPU && !economize_gpu_memory))
    {
        kp__->generate_beta_phi(uc->mt_lo_basis_size(), phi, n__, 0, kp__->beta_gk(), beta_phi);

        kp__->add_non_local_contribution(uc->num_atoms(), uc->mt_lo_basis_size(), uc->beta_chunk(0).desc_,
                                         kp__->beta_gk(), op_mtrx_packed__, packed_mtrx_offset__, beta_phi,
                                         op_phi, n__, 0, alpha, work);
    }
    else
    {
        #ifdef _GPU_
        kp__->generate_beta_gk(uc->num_atoms(), uc->beta_chunk(0).atom_pos_, uc->beta_chunk(0).desc_, kappa__);
        phi.copy_to_device();
        kp__->generate_beta_phi(uc->mt_lo_basis_size(), phi, n__, 0, kappa__, beta_phi);
        
        op_phi.copy_to_device();
        kp__->add_non_local_contribution(uc->num_atoms(), uc->mt_lo_basis_size(), uc->beta_chunk(0).desc_,
                                         kappa__, op_mtrx_packed__, packed_mtrx_offset__, beta_phi,
                                         op_phi, n__, 0, alpha, work);
        op_phi.copy_to_host();
        #else
        TERMINATE_NO_GPU
        #endif
    }
    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU) cuda_device_synchronize();
    #endif

    log_function_exit(__func__);
}


/** \param [in] phi Input wave-functions [storage: CPU && GPU].
 *  \param [out] hphi Hamiltonian, applied to wave-functions [storage: CPU || GPU].
 *  \param [out] ophi Overlap operator, applied to wave-functions [storage: CPU || GPU].
 */
void Band::apply_h_o_serial(K_point* kp__, 
                            std::vector<double> const& effective_potential__, 
                            std::vector<double> const& pw_ekin__, 
                            int N__,
                            int n__,
                            matrix<double_complex>& phi__,
                            matrix<double_complex>& hphi__,
                            matrix<double_complex>& ophi__,
                            matrix<double_complex>& kappa__,
                            mdarray<int, 1>& packed_mtrx_offset__,
                            mdarray<double_complex, 1>& d_mtrx_packed__,
                            mdarray<double_complex, 1>& q_mtrx_packed__)
{
    Timer t("sirius::Band::apply_h_o_serial");

    auto uc = parameters_.unit_cell();

    matrix<double_complex> phi, hphi, ophi;
    
    /* if temporary array is allocated, this would be the only big array on GPU */
    bool economize_gpu_memory = (kappa__.size() != 0);

    if (parameters_.processing_unit() == CPU)
    {
        phi =  matrix<double_complex>( phi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
        hphi = matrix<double_complex>(hphi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
        ophi = matrix<double_complex>(ophi__.at<CPU>(0, N__), kp__->num_gkvec(), n__);
    }
    if (parameters_.processing_unit() == GPU && !economize_gpu_memory)
    {
        phi =  matrix<double_complex>( phi__.at<CPU>(0, N__),  phi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
        hphi = matrix<double_complex>(hphi__.at<CPU>(0, N__), hphi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
        ophi = matrix<double_complex>(ophi__.at<CPU>(0, N__), ophi__.at<GPU>(0, N__), kp__->num_gkvec(), n__);
    }
    if (parameters_.processing_unit() == GPU && economize_gpu_memory)
    {
        double_complex* gpu_ptr = kappa__.at<GPU>(0, parameters_.unit_cell()->mt_basis_size());
        phi =  matrix<double_complex>( phi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
        hphi = matrix<double_complex>(hphi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
        ophi = matrix<double_complex>(ophi__.at<CPU>(0, N__), gpu_ptr, kp__->num_gkvec(), n__);
    }
    
    /* apply local part of Hamiltonian */
    apply_h_local_slice(kp__, effective_potential__, pw_ekin__, n__, phi, hphi);
    
    /* set intial ophi */
    if (parameters_.processing_unit() == CPU || (parameters_.processing_unit() == GPU && economize_gpu_memory)) 
        phi >> ophi;

    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU && !economize_gpu_memory)
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

    if (parameters_.processing_unit() == CPU || (parameters_.processing_unit() == GPU && !economize_gpu_memory))
    {
        /* compute <beta|phi> */
        kp__->generate_beta_phi(uc->mt_lo_basis_size(), phi, n__, 0, kp__->beta_gk(), beta_phi);
       
        /* add |beta>D<beta|phi> to |hphi> */
        kp__->add_non_local_contribution(uc->num_atoms(), uc->mt_lo_basis_size(), uc->beta_chunk(0).desc_,
                                         kp__->beta_gk(), d_mtrx_packed__, packed_mtrx_offset__, beta_phi,
                                         hphi, n__, 0, complex_one, work);
            
        /* add |beta>Q<beta|phi> to |ophi> */
        kp__->add_non_local_contribution(uc->num_atoms(), uc->mt_lo_basis_size(), uc->beta_chunk(0).desc_,
                                         kp__->beta_gk(), q_mtrx_packed__, packed_mtrx_offset__, beta_phi,
                                         ophi, n__, 0, complex_one, work);
    }
    else
    {
        #ifdef _GPU_
        kp__->generate_beta_gk(uc->num_atoms(), uc->beta_chunk(0).atom_pos_, uc->beta_chunk(0).desc_, kappa__);
        phi.copy_to_device();
        kp__->generate_beta_phi(uc->mt_lo_basis_size(), phi, n__, 0, kappa__, beta_phi);
        
        hphi.copy_to_device();
        kp__->add_non_local_contribution(uc->num_atoms(), uc->mt_lo_basis_size(), uc->beta_chunk(0).desc_,
                                         kappa__, d_mtrx_packed__, packed_mtrx_offset__, beta_phi,
                                         hphi, n__, 0, complex_one, work);
        hphi.copy_to_host();
        
        ophi.copy_to_device();    
        kp__->add_non_local_contribution(uc->num_atoms(), uc->mt_lo_basis_size(), uc->beta_chunk(0).desc_,
                                         kappa__, q_mtrx_packed__, packed_mtrx_offset__, beta_phi,
                                         ophi, n__, 0, complex_one, work);
        ophi.copy_to_host();
        #else
        TERMINATE_NO_GPU
        #endif
    }
    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU) cuda_device_synchronize();
    #endif
}

void Band::apply_h_o_real_space_serial(K_point* kp__, 
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
    Timer t("sirius::Band::apply_h_o_real_space_serial", kp__->comm());

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

    auto fft = parameters_.fft_coarse();
    
    Timer t4("sirius::Band::apply_h_o_real_space_serial|phase_fac");
    std::vector<double_complex> k_phase_fac(fft->size());
    /* loop over 3D array (real space) */
    for (int j0 = 0; j0 < fft->size(0); j0++)
    {
        for (int j1 = 0; j1 < fft->size(1); j1++)
        {
            for (int j2 = 0; j2 < fft->size(2); j2++)
            {
                /* get real space fractional coordinate */
                vector3d<double> v0(double(j0) / fft->size(0), double(j1) / fft->size(1), double(j2) / fft->size(2));
                int ir = static_cast<int>(j0 + j1 * fft->size(0) + j2 * fft->size(0) * fft->size(1));
                k_phase_fac[ir] = std::exp(double_complex(0.0, twopi * (kp__->vk() * v0)));
            }
        }
    }

    mdarray<double_complex, 3> T_phase_fac(mdarray_index_descriptor(-1, 1),
                                           mdarray_index_descriptor(-1, 1),
                                           mdarray_index_descriptor(-1, 1));
    for (int t0 = -1; t0 <= 1; t0++)
    {
        for (int t1 = -1; t1 <= 1; t1++)
        {
            for (int t2 = -1; t2 <= 1; t2++)
            {
                vector3d<int> T(t0, t1, t2);
                T_phase_fac(t0, t1, t2) = std::exp(double_complex(0.0, twopi * (kp__->vk() * T)));
            }
        }
    }
    t4.stop();

    int max_num_bands_per_block = std::min(100, n__);
    int num_band_blocks = n__ / max_num_bands_per_block + std::min(1, n__ % max_num_bands_per_block);
    
    splindex<block> spl_bands(n__, num_band_blocks, 0);

    mdarray<double_complex, 2>  phi_r(fft->size(), spl_bands.local_size(0));
    mdarray<double_complex, 2> hphi_r(fft->size(), spl_bands.local_size(0));
    mdarray<double_complex, 2> ophi_r(fft->size(), spl_bands.local_size(0));
    
    mdarray<double, 2> timers(4, Platform::max_num_threads());
    timers.zero();

    mdarray<double_complex, 2> hphi_tmp(parameters_.real_space_prj_->num_points_, spl_bands.local_size(0));
    mdarray<double_complex, 2> ophi_tmp(parameters_.real_space_prj_->num_points_, spl_bands.local_size(0));

    mdarray<double_complex, 3> phi_tmp(parameters_.real_space_prj_->max_num_points_, spl_bands.local_size(0), Platform::max_num_threads());
    /* <\beta_{\xi}^{\alpha}|\phi_j> */
    mdarray<double_complex, 3> beta_phi(uc->max_mt_basis_size(), spl_bands.local_size(0), Platform::max_num_threads());
    /* Q or D multiplied by <\beta_{\xi}^{\alpha}|\phi_j> */
    mdarray<double_complex, 3> d_beta_phi(uc->max_mt_basis_size(), spl_bands.local_size(0), Platform::max_num_threads());
    mdarray<double_complex, 3> q_beta_phi(uc->max_mt_basis_size(), spl_bands.local_size(0), Platform::max_num_threads());
    
    mdarray<double_complex, 3> beta_tmp(parameters_.real_space_prj_->max_num_points_, uc->max_mt_basis_size(), Platform::max_num_threads());

    for (int iblk = 0; iblk < num_band_blocks; iblk++)
    {
        int nbnd = (int)spl_bands.local_size(iblk);

        Timer t0("sirius::Band::apply_h_o_real_space|fft", kp__->comm());
        #pragma omp parallel
        {
            int thread_id = Platform::thread_id();
            #pragma omp for
            for (int ib = 0; ib < nbnd; ib++)
            {
                int i = (int)spl_bands.global_index(ib, iblk);
                fft->input(kp__->num_gkvec(), kp__->fft_index_coarse(), &phi(0, i), thread_id);
                /* phi(G) -> phi(r) */
                fft->transform(1, thread_id);
                fft->output(&phi_r(0, ib), thread_id);

                for (int ir = 0; ir < fft->size(); ir++)
                {
                    /* multiply phi by effective potential */
                    hphi_r(ir, ib) = phi_r(ir, ib) * effective_potential__[ir];
                    /* set intial ophi */
                    ophi_r(ir, ib) = phi_r(ir, ib);
                }
            }
        }
        t0.stop();

        Timer t2("sirius::Band::apply_h_o_real_space|nonloc", kp__->comm());
        #pragma omp parallel
        {
            int thread_id = Platform::thread_id();

            double w1 = std::sqrt(uc->omega()) / fft->size();
            double w2 = std::sqrt(uc->omega());
            #pragma omp for schedule(static, 1)
            for (int ia = 0; ia < uc->num_atoms(); ia++)
            {
                auto& beta_prj = parameters_.real_space_prj_->beta_projectors_[ia];
                int ofs = beta_prj.offset_;
                int npt = beta_prj.num_points_;
                auto type = parameters_.unit_cell()->atom(ia)->type();
                int nbf = type->mt_basis_size();
                double t0 = omp_get_wtime();
                for (int i = 0; i < nbnd; i++)
                {
                    for (int j = 0; j < npt; j++)
                    {
                        int ir = beta_prj.ir_[j];
                        auto T = beta_prj.T_[j];
                        phi_tmp(j, i, thread_id) = phi_r(ir, i) * w1 * conj(T_phase_fac(T[0], T[1], T[2])) * k_phase_fac[ir];
                    }
                }
                timers(0, thread_id) += (omp_get_wtime() - t0);
                
                t0 = omp_get_wtime();
                /* compute <beta|phi> */
                linalg<CPU>::gemm(2, 0, nbf, nbnd, npt,
                                  beta_prj.beta_.at<CPU>(), beta_prj.beta_.ld(),
                                  phi_tmp.at<CPU>(0, 0, thread_id), phi_tmp.ld(), 
                                  beta_phi.at<CPU>(0, 0, thread_id), beta_phi.ld());
                    
                /* compute D * <beta|phi> */
                linalg<CPU>::gemm(0, 0, nbf, nbnd, nbf,
                                  d_mtrx_packed__.at<CPU>(packed_mtrx_offset__(ia)), nbf,
                                  beta_phi.at<CPU>(0, 0, thread_id), beta_phi.ld(),
                                  d_beta_phi.at<CPU>(0, 0, thread_id), d_beta_phi.ld());
                
                /* compute Q * <beta|phi> */
                linalg<CPU>::gemm(0, 0, nbf, nbnd, nbf,
                                  q_mtrx_packed__.at<CPU>(packed_mtrx_offset__(ia)), nbf,
                                  beta_phi.at<CPU>(0, 0, thread_id), beta_phi.ld(),
                                  q_beta_phi.at<CPU>(0, 0, thread_id), q_beta_phi.ld());
                timers(1, thread_id) += (omp_get_wtime() - t0);
                
                t0 = omp_get_wtime();
                for (int xi = 0; xi < nbf; xi++)
                {
                    for (int j = 0; j < npt; j++)
                    {
                        int ir = beta_prj.ir_[j];
                        auto T = beta_prj.T_[j];
                        beta_tmp(j, xi, thread_id) = beta_prj.beta_(j, xi) * w2 * conj(k_phase_fac[ir]) * T_phase_fac(T[0], T[1], T[2]);
                    }
                }
                timers(2, thread_id) += (omp_get_wtime() - t0);
                
                t0 = omp_get_wtime();
                linalg<CPU>::gemm(0, 0, npt, nbnd, nbf,
                                  beta_tmp.at<CPU>(0, 0, thread_id), beta_tmp.ld(),
                                  d_beta_phi.at<CPU>(0, 0, thread_id), d_beta_phi.ld(),
                                  hphi_tmp.at<CPU>(ofs, 0), hphi_tmp.ld());

                linalg<CPU>::gemm(0, 0, npt, nbnd, nbf,
                                  beta_tmp.at<CPU>(0, 0, thread_id), beta_tmp.ld(),
                                  q_beta_phi.at<CPU>(0, 0, thread_id), q_beta_phi.ld(),
                                  ophi_tmp.at<CPU>(ofs, 0), ophi_tmp.ld());
                timers(3, thread_id) += (omp_get_wtime() - t0);
            }
        }
        t2.stop();
        
        Timer t1("sirius::Band::apply_h_o_real_space|add_nonloc", kp__->comm());
        #pragma omp parallel for
        for (int ib = 0; ib < nbnd; ib++)
        {
            for (int ia = 0; ia < uc->num_atoms(); ia++)
            {
                int ofs = parameters_.real_space_prj_->beta_projectors_[ia].offset_;
                for (int j = 0; j < parameters_.real_space_prj_->beta_projectors_[ia].num_points_; j++)
                {
                    int ir = parameters_.real_space_prj_->beta_projectors_[ia].ir_[j];
                    hphi_r(ir, ib) += hphi_tmp(ofs + j, ib);
                    ophi_r(ir, ib) += ophi_tmp(ofs + j, ib);
                }
            }
        }
        t1.stop();
        
        t0.start();
        #pragma omp parallel
        {
            int thread_id = Platform::thread_id();
            #pragma omp for
            for (int ib = 0; ib < nbnd; ib++)
            {
                int i = (int)spl_bands.global_index(ib, iblk);

                fft->input(&hphi_r(0, ib), thread_id);
                fft->transform(-1, thread_id);
                fft->output(kp__->num_gkvec(), kp__->fft_index_coarse(), &hphi(0, i), thread_id);
                for (int igk = 0; igk < kp__->num_gkvec(); igk++) hphi(igk, i) += phi(igk, i) * pw_ekin__[igk];

                fft->input(&ophi_r(0, ib), thread_id);
                fft->transform(-1, thread_id);
                fft->output(kp__->num_gkvec(), kp__->fft_index_coarse(), &ophi(0, i), thread_id);
            }
        }
        t0.stop();
    }

    if (kp__->comm().rank() == 0)
    {
        std::cout << "------------------------------------------------------------" << std::endl;
        std::cout << "thread_id  | load phi  | 1st zgemms | load beta | 2nd zgemms" << std::endl;
        std::cout << "------------------------------------------------------------" << std::endl;
        for (int i = 0; i < Platform::max_num_threads(); i++)
        {
            printf("   %2i      | %8.4f  | %8.4f   | %8.4f  | %8.4f\n", i, timers(0, i), timers(1, i), timers(2, i), timers(3, i));
        }
        std::cout << "------------------------------------------------------------" << std::endl;
    }
}

/** \param [in] phi Input wave-functions [storage: CPU && GPU].
 *  \param [out] hphi Hamiltonian, applied to wave-functions [storage: CPU || GPU].
 *  \param [out] ophi Overlap operator, applied to wave-functions [storage: CPU || GPU].
 */
void Band::set_fv_h_o_serial(K_point* kp__,
                             int N__,
                             int n__,
                             matrix<double_complex>& phi__,
                             matrix<double_complex>& hphi__,
                             matrix<double_complex>& ophi__,
                             matrix<double_complex>& h__,
                             matrix<double_complex>& o__,
                             matrix<double_complex>& h_old__,
                             matrix<double_complex>& o_old__,
                             matrix<double_complex>& kappa__)
{
    Timer t("sirius::Band::set_fv_h_o_serial");

    /* copy old Hamiltonian and overlap */
    for (int i = 0; i < N__; i++)
    {
        memcpy(&h__(0, i), &h_old__(0, i), N__ * sizeof(double_complex));
        memcpy(&o__(0, i), &o_old__(0, i), N__ * sizeof(double_complex));
    }

    //== /* apply Hamiltonian and overlap operators to the new basis functions */
    //== if (true)
    //== {
    //==     apply_h_o_serial(kp__, effective_potential__, pw_ekin__, N__, n__, phi__, hphi__, ophi__, kappa__, packed_mtrx_offset__,
    //==                      d_mtrx_packed__, q_mtrx_packed__);
    //== }
    //== else
    //== {
    //==     apply_h_o_real_space_serial(kp__, effective_potential__, pw_ekin__, N__, n__, phi__, hphi__, ophi__, packed_mtrx_offset__,
    //==                                 d_mtrx_packed__, q_mtrx_packed__);
    //== }
    
    if (parameters_.processing_unit() == CPU)
    {
        /* <{phi,res}|H|res> */
        linalg<CPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec(), &phi__(0, 0), phi__.ld(), &hphi__(0, N__), hphi__.ld(),
                          &h__(0, N__), h__.ld());
        
        /* <{phi,res}|O|res> */
        linalg<CPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec(), &phi__(0, 0), phi__.ld(), &ophi__(0, N__), ophi__.ld(),
                          &o__(0, N__), o__.ld());
    }

    if (parameters_.processing_unit() == GPU)
    {
        bool economize_gpu_memory = (kappa__.size() != 0);
        #ifdef _GPU_
        if (!economize_gpu_memory)
        {
            linalg<GPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec(), phi__.at<GPU>(0, 0), phi__.ld(),
                              hphi__.at<GPU>(0, N__), hphi__.ld(), h__.at<GPU>(0, N__), h__.ld());
            
            linalg<GPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec(), phi__.at<GPU>(0, 0), phi__.ld(),
                              ophi__.at<GPU>(0, N__), ophi__.ld(), o__.at<GPU>(0, N__), o__.ld());
        }
        else
        {
            /* copy phi to device */
            cublas_set_matrix(kp__->num_gkvec(), N__ + n__, sizeof(double_complex), phi__.at<CPU>(), phi__.ld(),
                              kappa__.at<GPU>(0, 0), kappa__.ld());
            /* copy hphi to device */
            cublas_set_matrix(kp__->num_gkvec(), n__, sizeof(double_complex), hphi__.at<CPU>(0, N__), hphi__.ld(),
                              kappa__.at<GPU>(0, N__ + n__), kappa__.ld());
            
            linalg<GPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec(), kappa__.at<GPU>(0, 0), kappa__.ld(),
                              kappa__.at<GPU>(0, N__ + n__), kappa__.ld(), h__.at<GPU>(0, N__), h__.ld());
            
            /* copy ophi to device */
            cublas_set_matrix(kp__->num_gkvec(), n__, sizeof(double_complex), ophi__.at<CPU>(0, N__), ophi__.ld(),
                              kappa__.at<GPU>(0, N__ + n__), kappa__.ld());

            linalg<GPU>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec(), kappa__.at<GPU>(0, 0), kappa__.ld(),
                              kappa__.at<GPU>(0, N__ + n__), kappa__.ld(), o__.at<GPU>(0, N__), o__.ld());

        }
        cublas_get_matrix(N__ + n__, n__, sizeof(double_complex), h__.at<GPU>(0, N__), h__.ld(), h__.at<CPU>(0, N__), h__.ld());
        cublas_get_matrix(N__ + n__, n__, sizeof(double_complex), o__.at<GPU>(0, N__), o__.ld(), o__.at<CPU>(0, N__), o__.ld());
        #else
        TERMINATE_NO_GPU
        #endif
    }
        
    /* save Hamiltonian and overlap */
    for (int i = N__; i < N__ + n__; i++)
    {
        memcpy(&h_old__(0, i), &h__(0, i), (N__ + n__) * sizeof(double_complex));
        memcpy(&o_old__(0, i), &o__(0, i), (N__ + n__) * sizeof(double_complex));
    }
}

void Band::residuals_serial(K_point* kp__,
                            int N__,
                            int num_bands__,
                            std::vector<double>& eval__,
                            matrix<double_complex>& evec__,
                            matrix<double_complex>& hphi__,
                            matrix<double_complex>& ophi__,
                            matrix<double_complex>& hpsi__,
                            matrix<double_complex>& opsi__,
                            matrix<double_complex>& res__,
                            std::vector<double>& h_diag__,
                            std::vector<double>& o_diag__,
                            std::vector<double>& res_norm__,
                            matrix<double_complex>& kappa__)
{
    Timer t("sirius::Band::residuals_serial");

    auto pu = parameters_.processing_unit();
    bool economize_gpu_memory = (kappa__.size() != 0);

    if (pu == CPU)
    {
        /* compute H\Psi_{i} = \sum_{mu} H\phi_{mu} * Z_{mu, i} */
        linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, &hphi__(0, 0), hphi__.ld(), &evec__(0, 0), evec__.ld(), 
                          &hpsi__(0, 0), hpsi__.ld());

        /* compute O\Psi_{i} = \sum_{mu} O\phi_{mu} * Z_{mu, i} */
        linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, &ophi__(0, 0), ophi__.ld(), &evec__(0, 0), evec__.ld(), 
                          &opsi__(0, 0), opsi__.ld());
    }

    if (pu == GPU)
    {
        #ifdef _GPU_
        if (!economize_gpu_memory)
        {
            /* compute H\Psi_{i} = \sum_{mu} H\phi_{mu} * Z_{mu, i} */
            linalg<GPU>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, hphi__.at<GPU>(), hphi__.ld(),
                              evec__.at<GPU>(), evec__.ld(), hpsi__.at<GPU>(), hpsi__.ld());

            /* compute O\Psi_{i} = \sum_{mu} O\phi_{mu} * Z_{mu, i} */
            linalg<GPU>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, ophi__.at<GPU>(), ophi__.ld(),
                              evec__.at<GPU>(), evec__.ld(), opsi__.at<GPU>(), opsi__.ld());
        }
        else
        {
            /* copy hphi to device */
            cublas_set_matrix(kp__->num_gkvec(), N__, sizeof(double_complex), hphi__.at<CPU>(), hphi__.ld(),
                              kappa__.at<GPU>(), kappa__.ld());
            
            /* compute H\Psi_{i} = \sum_{mu} H\phi_{mu} * Z_{mu, i} */
            linalg<GPU>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, kappa__.at<GPU>(), kappa__.ld(),
                              evec__.at<GPU>(), evec__.ld(), kappa__.at<GPU>(0, N__), kappa__.ld());

            cublas_get_matrix(kp__->num_gkvec(), num_bands__, sizeof(double_complex), kappa__.at<GPU>(0, N__), kappa__.ld(),
                              hpsi__.at<CPU>(), hpsi__.ld());
            
            /* copy ophi to device */
            cublas_set_matrix(kp__->num_gkvec(), N__, sizeof(double_complex), ophi__.at<CPU>(), ophi__.ld(),
                              kappa__.at<GPU>(0, num_bands__), kappa__.ld());
            
            /* compute O\Psi_{i} = \sum_{mu} O\phi_{mu} * Z_{mu, i} */
            linalg<GPU>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, kappa__.at<GPU>(0, num_bands__), kappa__.ld(),
                              evec__.at<GPU>(), evec__.ld(), kappa__.at<GPU>(0, 0), kappa__.ld());

            /* kappa(0, 0) contains opsi */
            cublas_get_matrix(kp__->num_gkvec(), num_bands__, sizeof(double_complex), kappa__.at<GPU>(0, 0), kappa__.ld(),
                              opsi__.at<CPU>(), opsi__.ld());
            /* kappa(0, num_bands) contains hpsi */
            cublas_set_matrix(kp__->num_gkvec(), num_bands__, sizeof(double_complex), hpsi__.at<CPU>(), hpsi__.ld(),
                              kappa__.at<GPU>(0, num_bands__), kappa__.ld());
        }
        #else
        TERMINATE_NO_GPU
        #endif
    }

    if (pu == CPU)
    {
        /* compute residuals norm and apply preconditioner */
        #pragma omp parallel for
        for (int i = 0; i < num_bands__; i++)
        {
            double r = 0;
            for (int igk = 0; igk < kp__->num_gkvec(); igk++)
            {
                /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} */
                res__(igk, i) = hpsi__(igk, i) - eval__[i] * opsi__(igk, i);
                r += real(conj(res__(igk, i)) * res__(igk, i));
            }
            res_norm__[i] = std::sqrt(r);
            
            /* apply preconditioner */
            for (int igk = 0; igk < kp__->num_gkvec(); igk++)
            {
                double p = h_diag__[igk] - eval__[i] * o_diag__[igk];

                //if (std::abs(p) < 0.5e-5) p = copysign(0.5e-5, p);

                p *= 2; // QE formula is in Ry; here we convert to Ha
                p = 0.25 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
                res__(igk, i) /= p;
            }
        }

        /* Normalize new basis functions */
        #pragma omp parallel for
        for (int i = 0; i < num_bands__; i++)
        {
            double d = 0;
            for (int igk = 0; igk < kp__->num_gkvec(); igk++) d += real(conj(res__(igk, i)) * res__(igk, i));
            //printf("res: %4i, norm: %18.12f\n", i, std::sqrt(d));
            d = 1.0 / std::sqrt(d);
            for (int igk = 0; igk < kp__->num_gkvec(); igk++) res__(igk, i) *= d;
        }
    }

    if (pu == GPU)
    {
        #ifdef _GPU_
        double_complex* hpsi_ptr;
        double_complex* opsi_ptr;
        double_complex* res_ptr;

        if (economize_gpu_memory)
        {
            hpsi_ptr = kappa__.at<GPU>(0, num_bands__);
            opsi_ptr = kappa__.at<GPU>(0, 0);
            res_ptr = kappa__.at<GPU>(0, 2 * num_bands__);
        }
        else
        {
            hpsi_ptr = hpsi__.at<GPU>();
            opsi_ptr = opsi__.at<GPU>();
            res_ptr = res__.at<GPU>();
        }

        mdarray<double, 1> res_norm_gpu(&res_norm__[0], num_bands__);
        res_norm_gpu.allocate_on_device();
        res_norm_gpu.zero_on_device();

        mdarray<double, 1> eval_gpu(&eval__[0], num_bands__);
        eval_gpu.allocate_on_device();
        eval_gpu.copy_to_device();

        /* global index of residual */
        mdarray<int, 1> res_idx_gpu(num_bands__);
        for (int i = 0; i < num_bands__; i++) res_idx_gpu(i) = i;
        res_idx_gpu.allocate_on_device();
        res_idx_gpu.copy_to_device();

        compute_residuals_gpu(kp__->num_gkvec(), num_bands__, res_idx_gpu.at<GPU>(), eval_gpu.at<GPU>(),
                              hpsi_ptr, opsi_ptr, res_ptr, res_norm_gpu.at<GPU>());
        res_norm_gpu.copy_to_host();

        /* compute norm */
        for (int i = 0; i < num_bands__; i++) res_norm__[i] = std::sqrt(res_norm__[i]);

        mdarray<double, 1> hdiag_gpu(&h_diag__[0], kp__->num_gkvec_row());
        hdiag_gpu.allocate_on_device();
        hdiag_gpu.copy_to_device();

        mdarray<double, 1> odiag_gpu(&o_diag__[0], kp__->num_gkvec_row());
        odiag_gpu.allocate_on_device();
        odiag_gpu.copy_to_device();

        mdarray<double, 1> norm2(num_bands__);
        norm2.allocate_on_device();
        norm2.zero_on_device();

        apply_preconditioner_gpu(kp__->num_gkvec(), num_bands__, res_idx_gpu.at<GPU>(), eval_gpu.at<GPU>(),
                                 hdiag_gpu.at<GPU>(), odiag_gpu.at<GPU>(), res_ptr, norm2.at<GPU>());

        normalize_residuals_gpu(kp__->num_gkvec_row(), num_bands__, res_idx_gpu.at<GPU>(), norm2.at<GPU>(), res_ptr);

        //== if (economize_gpu_memory)
        //== {
        //==     cublas_get_matrix(kp__->num_gkvec(), num_bands__, sizeof(double_complex), res_ptr, kp__->num_gkvec(),
        //==                       res__.at<CPU>(), res__.ld());
        //== }
        #else
        TERMINATE_NO_GPU
        #endif
    }
}

};

