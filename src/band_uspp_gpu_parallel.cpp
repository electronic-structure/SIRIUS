#include <thread>
#include <atomic>
#include "band.h"

namespace sirius {

template <typename T>
T check_sum(matrix<T> const& mtrx, int irow0, int icol0, int nrow, int ncol)
{
    T sum = 0;

    for (int j = 0; j < ncol; j++)
    {
        for (int i = 0; i < nrow; i++) sum += mtrx(irow0 + i, icol0 + j);
    }

    return sum;
}

//== void Band::apply_h_o_uspp_gpu_parallel_v2(K_point* kp__,
//==                                           std::vector<double> const& effective_potential__,
//==                                           std::vector<double>& pw_ekin__,
//==                                           int N__,
//==                                           int n__,
//==                                           dmatrix<double_complex>& phi__,
//==                                           dmatrix<double_complex>& hphi__,
//==                                           dmatrix<double_complex>& ophi__,
//==                                           int num_atoms_in_block__,
//==                                           matrix<double_complex>& kappa__,
//==                                           matrix<double_complex> const& beta_gk_t__,
//==                                           matrix<double>& gkvec_row__,
//==                                           mdarray<int, 1>& packed_mtrx_offset__,
//==                                           mdarray<double_complex, 1>& d_mtrx_packed__,
//==                                           mdarray<double_complex, 1>& q_mtrx_packed__)
//== {
//==     log_function_enter(__func__);
//==     Timer t("sirius::Band::apply_h_o_uspp_gpu_parallel_v2", kp__->comm());
//== 
//==     bool with_overlap = (parameters_.esm_type() == ultrasoft_pseudopotential);
//== 
//==     splindex<block_cyclic> s0(N__,       kp__->num_ranks_col(), kp__->rank_col(), blacs_grid_.cyclic_block_size());
//==     splindex<block_cyclic> s1(N__ + n__, kp__->num_ranks_col(), kp__->rank_col(), blacs_grid_.cyclic_block_size());
//== 
//==     /* local number of states to which Hamiltonian has to be applied */
//==     int nloc = static_cast<int>(s1.local_size() - s0.local_size());
//== 
//==     if (!nloc) return;
//== 
//==     auto uc = parameters_.unit_cell();
//== 
//==     /* apply local part of Hamiltonian */
//==     apply_h_local_parallel(kp__, effective_potential__, pw_ekin__, N__, n__, phi__, hphi__);
//== 
//==     if (parameters_.processing_unit() == CPU && with_overlap)
//==     {
//==         /* set intial ophi */
//==         memcpy(&ophi__(0, s0.local_size()), &phi__(0, s0.local_size()), kp__->num_gkvec_row() * nloc * sizeof(double_complex));
//==     }
//== 
//==     #ifdef _GPU_
//==     if (parameters_.processing_unit() == GPU)
//==     {
//==         /* copy hphi do device */
//==         cuda_copy_to_device(hphi__.at<GPU>(0, s0.local_size()), hphi__.at<CPU>(0, s0.local_size()), 
//==                             kp__->num_gkvec_row() * nloc * sizeof(double_complex));
//== 
//==         /* set intial ophi */
//==         if (with_overlap)
//==         {
//==             cuda_copy_device_to_device(ophi__.at<GPU>(0, s0.local_size()), phi__.at<GPU>(0, s0.local_size()), 
//==                                        kp__->num_gkvec_row() * nloc * sizeof(double_complex));
//==         }
//==     }
//==     #endif
//== 
//==     int num_atom_blocks = uc->num_atoms() / num_atoms_in_block__ + std::min(1, uc->num_atoms() % num_atoms_in_block__);
//== 
//==     splindex<block> atom_blocks(uc->num_atoms(), num_atom_blocks, 0);
//==     
//==     /* allocate space for <beta|phi> array */
//==     int nbf_max = uc->max_mt_basis_size() * num_atoms_in_block__;
//==     mdarray<double_complex, 1> beta_phi_tmp(nbf_max * nloc);
//== 
//==     /* result of Q or D multiplied by <beta|phi> */
//==     matrix<double_complex> tmp(nbf_max, nloc);
//== 
//==     mdarray<int, 2> beta_pw_desc(3, atom_blocks.local_size(0));
//==     mdarray<double, 2> atom_pos(3, atom_blocks.local_size(0));
//== 
//==     #ifdef _GPU_
//==     if (parameters_.processing_unit() == GPU)
//==     {
//==         beta_phi_tmp.allocate_on_device();
//==         tmp.allocate_on_device();
//==         beta_pw_desc.allocate_on_device();
//==         atom_pos.allocate_on_device();
//==     }
//==     #endif
//== 
//==     #ifdef _GPU_
//==     #ifdef _GPU_DIRECT_
//==     // allrecue with gpu-direct is broken at the moment
//==     bool gpu_direct = false;
//==     #else
//==     bool gpu_direct = false;
//==     #endif
//==     #endif
//== 
//==     for (int iab = 0; iab < num_atom_blocks; iab++)
//==     {
//==         int nbf_in_block = 0;
//== 
//==         for (int i = 0; i < (int)atom_blocks.local_size(iab); i++)
//==         {
//==             int ia = (int)atom_blocks.global_index(i, iab);
//==             auto type = uc->atom(ia)->type();
//==             /* atom fractional coordinates */
//==             for (int x = 0; x < 3; x++) atom_pos(x, i) = uc->atom(ia)->position(x);
//==             /* number of beta functions for atom */
//==             beta_pw_desc(0, i) = type->mt_basis_size();
//==             /* offset in beta_pw */
//==             beta_pw_desc(1, i) = nbf_in_block;
//==             /* offset in beta_gk_t */
//==             beta_pw_desc(2, i) = type->offset_lo();
//== 
//==             nbf_in_block += uc->atom(ia)->mt_basis_size();
//==         }
//== 
//==         #ifdef _GPU_
//==         if (parameters_.processing_unit() == GPU)
//==         {
//==             beta_pw_desc.copy_to_device();
//==             atom_pos.copy_to_device();
//==         }
//==         #endif
//== 
//==         /* wrapper for <beta|phi> with required dimensions */
//==         matrix<double_complex> beta_phi;
//==         switch (parameters_.processing_unit())
//==         {
//==             case CPU:
//==             {
//==                 beta_phi = matrix<double_complex>(beta_phi_tmp.at<CPU>(), nbf_in_block, nloc);
//==                 break;
//==             }
//==             case GPU:
//==             {
//==                 beta_phi = matrix<double_complex>(beta_phi_tmp.at<CPU>(), beta_phi_tmp.at<GPU>(), nbf_in_block, nloc);
//==                 break;
//==             }
//==         }
//== 
//==         Timer t1("sirius::Band::apply_h_o_uspp_cpu_parallel_v2|beta_phi", kp__->comm_row());
//==         if (parameters_.processing_unit() == CPU)
//==         {
//==             /* create beta projectors */
//==             #pragma omp parallel
//==             for (int i = 0; i < (int)atom_blocks.local_size(iab); i++)
//==             {
//==                 int ia = (int)atom_blocks.global_index(i, iab);
//==                 auto type = parameters_.unit_cell()->atom(ia)->type();
//==                 #pragma omp for
//==                 for (int xi = 0; xi < type->mt_basis_size(); xi++)
//==                 {
//==                     for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
//==                     {
//==                         kappa__(igk_row, beta_pw_desc(1, i) + xi) = beta_gk_t__(igk_row, beta_pw_desc(2, i) + xi) * 
//==                                                                     conj(kp__->gkvec_phase_factor(igk_row, ia));
//==                     }
//==                 }
//==             }
//==             /* compute <beta|phi> */
//==             linalg<CPU>::gemm(2, 0, nbf_in_block, nloc, kp__->num_gkvec_row(), 
//==                               kappa__.at<CPU>(), kappa__.ld(), 
//==                               phi__.at<CPU>(0, s0.local_size()), phi__.ld(), 
//==                               beta_phi.at<CPU>(), beta_phi.ld());
//==             kp__->comm_row().allreduce(beta_phi.at<CPU>(), (int)beta_phi.size());
//==         }
//==         #ifdef _GPU_
//==         if (parameters_.processing_unit() == GPU)
//==         {
//==             /* create beta projectors directly on GPU */
//==             create_beta_pw_gpu_v2((int)atom_blocks.local_size(iab),
//==                                   kp__->num_gkvec_row(),
//==                                   beta_pw_desc.at<GPU>(),
//==                                   beta_gk_t__.at<GPU>(),
//==                                   gkvec_row__.at<GPU>(),
//==                                   atom_pos.at<GPU>(),
//==                                   kappa__.at<GPU>());
//== 
//==             /* compute <beta|phi> */
//==             linalg<GPU>::gemm(2, 0, nbf_in_block, nloc, kp__->num_gkvec_row(), 
//==                               kappa__.at<GPU>(), kappa__.ld(), 
//==                               phi__.at<GPU>(0, s0.local_size()), phi__.ld(), 
//==                               beta_phi.at<GPU>(), beta_phi.ld());
//==             
//==             if (gpu_direct)
//==             {
//==                 kp__->comm_row().allreduce(beta_phi.at<GPU>(), (int)beta_phi.size());
//==             }
//==             else
//==             {
//==                 beta_phi.copy_to_host();
//==                 kp__->comm_row().allreduce(beta_phi.at<CPU>(), (int)beta_phi.size());
//==                 beta_phi.copy_to_device();
//==             }
//==         }
//==         #endif
//==         double tval = t1.stop();
//== 
//==         if (verbosity_level >= 6 && kp__->comm().rank() == 0)
//==         {
//==             printf("<beta|phi> effective zgemm with M, N, K: %6i %6i %6i, %12.4f sec, %12.4f GFlops/node\n",
//==                    nbf_in_block, nloc, kp__->num_gkvec(),
//==                    tval, 8e-9 * nbf_in_block * nloc * kp__->num_gkvec() / tval / kp__->num_ranks_row());
//==         }
//== 
//==         if (parameters_.processing_unit() == CPU)
//==         {
//==             #pragma omp parallel for
//==             for (int i = 0; i < (int)atom_blocks.local_size(iab); i++)
//==             {
//==                 int ia = (int)atom_blocks.global_index(i, iab);
//==                 int ofs = beta_pw_desc(1, i);
//==                 
//==                 /* number of beta functions for a given atom */
//==                 int nbf = beta_pw_desc(0, i);
//== 
//==                 /* compute D*<beta|phi> */
//==                 linalg<CPU>::gemm(0, 0, nbf, nloc, nbf, d_mtrx_packed__.at<CPU>(packed_mtrx_offset__(ia)), nbf, 
//==                                   beta_phi.at<CPU>(ofs, 0), beta_phi.ld(), tmp.at<CPU>(ofs, 0), tmp.ld());
//== 
//==             }
//==             
//==             /* compute <G+k|beta> * D*<beta|phi> and add to hphi */
//==             linalg<CPU>::gemm(0, 0, kp__->num_gkvec_row(), nloc, nbf_in_block, complex_one,
//==                               kappa__.at<CPU>(), kappa__.ld(), tmp.at<CPU>(), tmp.ld(), complex_one,
//==                               hphi__.at<CPU>(0, s0.local_size()), hphi__.ld());
//==             
//==             if (with_overlap)
//==             {
//==                 #pragma omp parallel for
//==                 for (int i = 0; i < (int)atom_blocks.local_size(iab); i++)
//==                 {
//==                     int ia = (int)atom_blocks.global_index(i, iab);
//==                     int ofs = beta_pw_desc(1, i);
//==                     
//==                     /* number of beta functions for a given atom */
//==                     int nbf = beta_pw_desc(0, i);
//== 
//==                     /* compute Q*<beta|phi> */
//==                     linalg<CPU>::gemm(0, 0, nbf, nloc, nbf, q_mtrx_packed__.at<CPU>(packed_mtrx_offset__(ia)), nbf,
//==                                       beta_phi.at<CPU>(ofs, 0), beta_phi.ld(), tmp.at<CPU>(ofs, 0), tmp.ld());
//==                 }
//== 
//==                 /* compute <G+k|beta> * Q*<beta|phi> and add to ophi */
//==                 linalg<CPU>::gemm(0, 0, kp__->num_gkvec_row(), nloc, nbf_in_block, complex_one,
//==                                   kappa__.at<CPU>(), kappa__.ld(), tmp.at<CPU>(), tmp.ld(), complex_one,
//==                                   ophi__.at<CPU>(0, s0.local_size()), ophi__.ld());
//==             }
//==         }
//== 
//==         #ifdef _GPU_
//==         if (parameters_.processing_unit() == GPU)
//==         {
//==             #pragma omp parallel for
//==             for (int i = 0; i < (int)atom_blocks.local_size(iab); i++)
//==             {
//==                 int ia = (int)atom_blocks.global_index(i, iab);
//==                 int ofs = beta_pw_desc(1, i);
//==                 
//==                 /* number of beta functions for a given atom */
//==                 int nbf = beta_pw_desc(0, i);
//== 
//==                 /* compute D*<beta|phi> */
//==                 linalg<GPU>::gemm(0, 0, nbf, nloc, nbf, d_mtrx_packed__.at<GPU>(packed_mtrx_offset__(ia)), nbf, 
//==                                   beta_phi.at<GPU>(ofs, 0), beta_phi.ld(), tmp.at<GPU>(ofs, 0), tmp.ld(), 
//==                                   Platform::thread_id());
//== 
//==             }
//==             cuda_device_synchronize();
//==             
//==             double_complex alpha = complex_one;
//==             /* compute <G+k|beta> * D*<beta|phi> and add to hphi */
//==             linalg<GPU>::gemm(0, 0, kp__->num_gkvec_row(), nloc, nbf_in_block, &alpha,
//==                               kappa__.at<GPU>(), kappa__.ld(), tmp.at<GPU>(), tmp.ld(), &alpha, 
//==                               hphi__.at<GPU>(0, s0.local_size()), hphi__.ld());
//==             
//==             if (with_overlap)
//==             {
//==                 #pragma omp parallel for
//==                 for (int i = 0; i < (int)atom_blocks.local_size(iab); i++)
//==                 {
//==                     int ia = (int)atom_blocks.global_index(i, iab);
//==                     int ofs = beta_pw_desc(1, i);
//==                     
//==                     /* number of beta functions for a given atom */
//==                     int nbf = beta_pw_desc(0, i);
//== 
//==                     /* compute Q*<beta|phi> */
//==                     linalg<GPU>::gemm(0, 0, nbf, nloc, nbf, q_mtrx_packed__.at<GPU>(packed_mtrx_offset__(ia)), nbf,
//==                                       beta_phi.at<GPU>(ofs, 0), beta_phi.ld(), tmp.at<GPU>(ofs, 0), tmp.ld(), 
//==                                       Platform::thread_id());
//==                 }
//==                 cuda_device_synchronize();
//== 
//==                 /* compute <G+k|beta> * Q*<beta|phi> and add to ophi */
//==                 linalg<GPU>::gemm(0, 0, kp__->num_gkvec_row(), nloc, nbf_in_block, &alpha,
//==                                   kappa__.at<GPU>(), kappa__.ld(), tmp.at<GPU>(), tmp.ld(), &alpha,
//==                                   ophi__.at<GPU>(0, s0.local_size()), ophi__.ld());
//==             }
//==         }
//==         #endif
//==     }
//==     #ifdef _GPU_
//==     if (parameters_.processing_unit() == GPU) cuda_device_synchronize();
//==     #endif
//==     log_function_exit(__func__);
//== }

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
                                         double_complex const* h_diag,
                                         double_complex const* o_diag,
                                         double_complex* res,
                                         double* res_norm);

extern "C" void normalize_residuals_gpu(int num_gkvec_row,
                                        int num_res_local,
                                        int* res_idx,
                                        double* norm2,
                                        double_complex* res);
#endif

#ifdef _SCALAPACK_
void Band::uspp_residuals_gpu_parallel(int N__,
                                       int num_bands__,
                                       K_point* kp__,
                                       std::vector<double>& eval__,
                                       dmatrix<double_complex>& evec__,
                                       dmatrix<double_complex>& hphi__,
                                       dmatrix<double_complex>& ophi__,
                                       dmatrix<double_complex>& hpsi__,
                                       dmatrix<double_complex>& opsi__,
                                       dmatrix<double_complex>& res__,
                                       std::vector<double_complex>& h_diag__,
                                       std::vector<double_complex>& o_diag__,
                                       std::vector<double>& res_norm__,
                                       mdarray<double_complex, 2>& kappa__)
{
    log_function_enter(__func__);
    Timer t("sirius::Band::uspp_residuals_gpu_parallel", kp__->comm());

    Timer t1("sirius::Band::uspp_residuals_gpu_parallel|zgemm_eff", kp__->comm());

    auto pu = parameters_.processing_unit();

    splindex<block_cyclic> spl_num_bands_col(num_bands__, kp__->num_ranks_col(), kp__->rank_col(),
                                             blacs_grid_.cyclic_block_size());
    splindex<block_cyclic> spl_num_bands_row(num_bands__, kp__->num_ranks_row(), kp__->rank_row(),
                                             blacs_grid_.cyclic_block_size());
    
    /* transpose matrix of eigen-vectors */
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

    auto get_evec = [kp__, &spl_num_bands_col, &spl_num_bands_row, &evec_t, &evec_tmp, num_phi_loc, pu]
                    (int icol) -> void 
    {
        int num_bands_of_col = (int)spl_num_bands_col.local_size(icol);
        memset(&evec_tmp(0, 0, icol % 2), 0, num_phi_loc * num_bands_of_col * sizeof(double_complex));
        for (int i = 0; i < num_bands_of_col; i++)
        {
            int iglob = (int)spl_num_bands_col.global_index(i, icol);
            auto p = spl_num_bands_row.location(iglob); 
            
            if (p.second == kp__->rank_row())
            {
                for (int j = 0; j < num_phi_loc; j++) evec_tmp(j, i, icol % 2) = evec_t(p.first, j);
            }
        }
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
        printf("effective zgemm #6&7 with M, N, K: %6i %6i %6i,                        %12.4f sec, %12.4f GFlops/node\n",
               kp__->num_gkvec(), num_bands__, N__,
               tval, 2 * 8e-9 * kp__->num_gkvec() * num_bands__ * N__ / tval / kp__->num_ranks());
    }

    memset(&res_norm__[0], 0, num_bands__ * sizeof(double));

    if (pu == CPU)
    {
        /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} and norm squared */
        #pragma omp parallel for
        for (int i = 0; i < res__.num_cols_local(); i++)
        {
            int ires = res__.icol(i);
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
        for (int i = 0; i < res__.num_cols_local(); i++)
        {
            int ires = res__.icol(i);
        
            for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
            {
                double_complex z = h_diag__[igk_row] - eval__[ires] * o_diag__[igk_row];
                if (std::abs(z) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
                res__(igk_row, i) /= z;
            }
        }
        
        std::vector<double> norm2(num_bands__, 0);
        /* Normalize new basis functions */
        #pragma omp parallel for
        for (int i = 0; i < res__.num_cols_local(); i++)
        {
            int ires = res__.icol(i);
            double d = 0;
            for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) 
                d += real(conj(res__(igk_row, i)) * res__(igk_row, i));
            norm2[ires] = d;
        }
        kp__->comm().allreduce(norm2);
        #pragma omp parallel for
        for (int i = 0; i < res__.num_cols_local(); i++)
        {
            int ires = res__.icol(i);
            double d = 1.0 / std::sqrt(norm2[ires]);
            for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) res__(igk_row, i) *= d;
        }
    }

    if (pu == GPU)
    {
        #ifdef _GPU_
        mdarray<double, 1> res_norm_gpu(&res_norm__[0], num_bands__);
        res_norm_gpu.allocate_on_device();
        res_norm_gpu.zero_on_device();

        mdarray<double, 1> eval_gpu(&eval__[0], num_bands__);
        eval_gpu.allocate_on_device();
        eval_gpu.copy_to_device();

        mdarray<int, 1> res_idx_gpu(res__.num_cols_local());
        for (int i = 0; i < res__.num_cols_local(); i++) res_idx_gpu(i) = res__.icol(i);
        res_idx_gpu.allocate_on_device();
        res_idx_gpu.copy_to_device();

        compute_residuals_gpu(kp__->num_gkvec_row(), res__.num_cols_local(), res_idx_gpu.at<GPU>(), eval_gpu.at<GPU>(),
                              hpsi__.at<GPU>(), opsi__.at<GPU>(), res__.at<GPU>(), res_norm_gpu.at<GPU>());
        res_norm_gpu.copy_to_host();

        kp__->comm().allreduce(res_norm__);
        
        /* compute norm */
        for (int i = 0; i < num_bands__; i++) res_norm__[i] = std::sqrt(res_norm__[i]);

        mdarray<double_complex, 1> hdiag_gpu(&h_diag__[0], kp__->num_gkvec_row());
        hdiag_gpu.allocate_on_device();
        hdiag_gpu.copy_to_device();

        mdarray<double_complex, 1> odiag_gpu(&o_diag__[0], kp__->num_gkvec_row());
        odiag_gpu.allocate_on_device();
        odiag_gpu.copy_to_device();

        mdarray<double, 1> norm2(num_bands__);
        norm2.allocate_on_device();
        norm2.zero_on_device();

        apply_preconditioner_gpu(kp__->num_gkvec_row(), res__.num_cols_local(), res_idx_gpu.at<GPU>(), eval_gpu.at<GPU>(),
                                 hdiag_gpu.at<GPU>(), odiag_gpu.at<GPU>(), res__.at<GPU>(), norm2.at<GPU>());
        // TODO: test gpudirect here
        norm2.copy_to_host();
        kp__->comm().allreduce(norm2.at<CPU>(), num_bands__);
        norm2.copy_to_device();

        normalize_residuals_gpu(kp__->num_gkvec_row(), res__.num_cols_local(), res_idx_gpu.at<GPU>(),
                                norm2.at<GPU>(), res__.at<GPU>());
        #else
        TERMINATE_NO_GPU
        #endif
    }

    log_function_exit(__func__);
}
#endif

void Band::apply_h_ncpp_parallel(K_point* kp__,
                                 std::vector<double> const& effective_potential__,
                                 std::vector<double> const& pw_ekin__,
                                 dmatrix<double_complex>& phi__,
                                 dmatrix<double_complex>& hphi__,
                                 int num_atoms_in_block__,
                                 matrix<double_complex>& kappa__,
                                 matrix<double_complex> const& beta_gk_t__,
                                 matrix<double>& gkvec_row__,
                                 mdarray<int, 1>& packed_mtrx_offset__,
                                 mdarray<double_complex, 1>& d_mtrx_packed__)
{
    log_function_enter(__func__);
    Timer t("sirius::Band::apply_h_ncpp_parallel", kp__->comm());

    /* local number of states to which Hamiltonian has to be applied */
    int nloc = static_cast<int>(kp__->spl_fv_states().local_size());

    if (!nloc) return;

    auto uc = parameters_.unit_cell();

    /* apply local part of Hamiltonian */
    apply_h_local_parallel(kp__, effective_potential__, pw_ekin__, 0, parameters_.num_bands(), phi__, hphi__);

    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU)
    {
        cuda_copy_to_device(phi__.at<GPU>(), phi__.at<CPU>(), kp__->num_gkvec_row() * nloc * sizeof(double_complex));
        /* copy hphi do device */
        cuda_copy_to_device(hphi__.at<GPU>(), hphi__.at<CPU>(), kp__->num_gkvec_row() * nloc * sizeof(double_complex));
    }
    #endif

    int num_atom_blocks = uc->num_atoms() / num_atoms_in_block__ + std::min(1, uc->num_atoms() % num_atoms_in_block__);

    splindex<block> atom_blocks(uc->num_atoms(), num_atom_blocks, 0);
    
    /* allocate space for <beta|phi> array */
    int nbf_max = uc->max_mt_basis_size() * num_atoms_in_block__;
    mdarray<double_complex, 1> beta_phi_tmp(nbf_max * nloc);

    /* result of D multiplied by <beta|phi> */
    matrix<double_complex> tmp(nbf_max, nloc);

    mdarray<int, 2> beta_pw_desc(3, atom_blocks.local_size(0));
    mdarray<double, 2> atom_pos(3, atom_blocks.local_size(0));

    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU)
    {
        beta_phi_tmp.allocate_on_device();
        tmp.allocate_on_device();
        beta_pw_desc.allocate_on_device();
        atom_pos.allocate_on_device();
    }
    #endif

    #ifdef _GPU_
    #ifdef _GPU_DIRECT_
    // allrecue with gpu-direct is broken at the moment
    bool gpu_direct = false;
    #else
    bool gpu_direct = false;
    #endif
    #endif

    for (int iab = 0; iab < num_atom_blocks; iab++)
    {
        int nbf_in_block = 0;

        for (int i = 0; i < (int)atom_blocks.local_size(iab); i++)
        {
            int ia = (int)atom_blocks.global_index(i, iab);
            auto type = uc->atom(ia)->type();
            /* atom fractional coordinates */
            for (int x = 0; x < 3; x++) atom_pos(x, i) = uc->atom(ia)->position(x);
            /* number of beta functions for atom */
            beta_pw_desc(0, i) = type->mt_basis_size();
            /* offset in beta_pw */
            beta_pw_desc(1, i) = nbf_in_block;
            /* offset in beta_gk_t */
            beta_pw_desc(2, i) = type->offset_lo();

            nbf_in_block += uc->atom(ia)->mt_basis_size();
        }

        #ifdef _GPU_
        if (parameters_.processing_unit() == GPU)
        {
            beta_pw_desc.copy_to_device();
            atom_pos.copy_to_device();
        }
        #endif

        /* wrapper for <beta|phi> with required dimensions */
        matrix<double_complex> beta_phi;
        switch (parameters_.processing_unit())
        {
            case CPU:
            {
                beta_phi = matrix<double_complex>(beta_phi_tmp.at<CPU>(), nbf_in_block, nloc);
                break;
            }
            case GPU:
            {
                beta_phi = matrix<double_complex>(beta_phi_tmp.at<CPU>(), beta_phi_tmp.at<GPU>(), nbf_in_block, nloc);
                break;
            }
        }

        Timer t1("sirius::Band::apply_h_ncpp_parallel|beta_phi", kp__->comm_row());
        if (parameters_.processing_unit() == CPU)
        {
            /* create beta projectors */
            #pragma omp parallel
            for (int i = 0; i < (int)atom_blocks.local_size(iab); i++)
            {
                int ia = (int)atom_blocks.global_index(i, iab);
                auto type = parameters_.unit_cell()->atom(ia)->type();
                #pragma omp for
                for (int xi = 0; xi < type->mt_basis_size(); xi++)
                {
                    for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
                    {
                        kappa__(igk_row, beta_pw_desc(1, i) + xi) = beta_gk_t__(igk_row, beta_pw_desc(2, i) + xi) * 
                                                                    conj(kp__->gkvec_phase_factor(igk_row, ia));
                    }
                }
            }
            /* compute <beta|phi> */
            linalg<CPU>::gemm(2, 0, nbf_in_block, nloc, kp__->num_gkvec_row(), 
                              kappa__.at<CPU>(), kappa__.ld(), 
                              phi__.at<CPU>(), phi__.ld(), 
                              beta_phi.at<CPU>(), beta_phi.ld());
            kp__->comm_row().allreduce(beta_phi.at<CPU>(), (int)beta_phi.size());
        }
        #ifdef _GPU_
        if (parameters_.processing_unit() == GPU)
        {
            ///* create beta projectors directly on GPU */
            //create_beta_pw_gpu_v2((int)atom_blocks.local_size(iab),
            //                      kp__->num_gkvec_row(),
            //                      beta_pw_desc.at<GPU>(),
            //                      beta_gk_t__.at<GPU>(),
            //                      gkvec_row__.at<GPU>(),
            //                      atom_pos.at<GPU>(),
            //                      kappa__.at<GPU>());
            STOP(); // and fix

            /* compute <beta|phi> */
            linalg<GPU>::gemm(2, 0, nbf_in_block, nloc, kp__->num_gkvec_row(), 
                              kappa__.at<GPU>(), kappa__.ld(), 
                              phi__.at<GPU>(), phi__.ld(), 
                              beta_phi.at<GPU>(), beta_phi.ld());
            
            if (gpu_direct)
            {
                kp__->comm_row().allreduce(beta_phi.at<GPU>(), (int)beta_phi.size());
            }
            else
            {
                beta_phi.copy_to_host();
                kp__->comm_row().allreduce(beta_phi.at<CPU>(), (int)beta_phi.size());
                beta_phi.copy_to_device();
            }
        }
        #endif
        double tval = t1.stop();

        if (verbosity_level >= 6 && kp__->comm().rank() == 0)
        {
            printf("<beta|phi> effective zgemm with M, N, K: %6i %6i %6i, %12.4f sec, %12.4f GFlops/node\n",
                   nbf_in_block, nloc, kp__->num_gkvec(),
                   tval, 8e-9 * nbf_in_block * nloc * kp__->num_gkvec() / tval / kp__->num_ranks_row());
        }

        if (parameters_.processing_unit() == CPU)
        {
            #pragma omp parallel for
            for (int i = 0; i < (int)atom_blocks.local_size(iab); i++)
            {
                int ia = (int)atom_blocks.global_index(i, iab);
                int ofs = beta_pw_desc(1, i);
                
                /* number of beta functions for a given atom */
                int nbf = beta_pw_desc(0, i);

                /* compute D*<beta|phi> */
                linalg<CPU>::gemm(0, 0, nbf, nloc, nbf, d_mtrx_packed__.at<CPU>(packed_mtrx_offset__(ia)), nbf, 
                                  beta_phi.at<CPU>(ofs, 0), beta_phi.ld(), tmp.at<CPU>(ofs, 0), tmp.ld());

            }
            
            /* compute <G+k|beta> * D*<beta|phi> and add to hphi */
            linalg<CPU>::gemm(0, 0, kp__->num_gkvec_row(), nloc, nbf_in_block, complex_one,
                              kappa__.at<CPU>(), kappa__.ld(), tmp.at<CPU>(), tmp.ld(), complex_one,
                              hphi__.at<CPU>(), hphi__.ld());
        }

        #ifdef _GPU_
        if (parameters_.processing_unit() == GPU)
        {
            #pragma omp parallel for
            for (int i = 0; i < (int)atom_blocks.local_size(iab); i++)
            {
                int ia = (int)atom_blocks.global_index(i, iab);
                int ofs = beta_pw_desc(1, i);
                
                /* number of beta functions for a given atom */
                int nbf = beta_pw_desc(0, i);

                /* compute D*<beta|phi> */
                linalg<GPU>::gemm(0, 0, nbf, nloc, nbf, d_mtrx_packed__.at<GPU>(packed_mtrx_offset__(ia)), nbf, 
                                  beta_phi.at<GPU>(ofs, 0), beta_phi.ld(), tmp.at<GPU>(ofs, 0), tmp.ld(), 
                                  Platform::thread_id());

            }
            cuda_device_synchronize();
            
            double_complex alpha = complex_one;
            /* compute <G+k|beta> * D*<beta|phi> and add to hphi */
            linalg<GPU>::gemm(0, 0, kp__->num_gkvec_row(), nloc, nbf_in_block, &alpha,
                              kappa__.at<GPU>(), kappa__.ld(), tmp.at<GPU>(), tmp.ld(), &alpha, 
                              hphi__.at<GPU>(), hphi__.ld());
        }
        #endif
    }
    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU) cuda_device_synchronize();
    #endif

    #if defined(_GPU_)
    if (parameters_.processing_unit() == GPU)
    {
        cuda_copy_to_host(hphi__.at<CPU>(), hphi__.at<GPU>(), kp__->num_gkvec_row() * nloc * sizeof(double_complex));
    }
    #endif

    log_function_exit(__func__);
}

//== void Band::set_fv_h_o_ncpp_parallel(K_point* kp__,
//==                                     dmatrix<double_complex>& phi__,
//==                                     dmatrix<double_complex>& hphi__,
//==                                     dmatrix<double_complex>& h__,
//==                                     dmatrix<double_complex>& o__,
//==                                     mdarray<double_complex, 2>& kappa__)
//== {
//==     log_function_enter(__func__);
//==     Timer t("sirius::Band::set_fv_h_o_ncpp_parallel", kp__->comm());
//==     
//==     splindex<block_cyclic> spl_bands_col(parameters_.num_fv_states(), kp__->num_ranks_col(), kp__->rank_col(), blacs_grid_.cyclic_block_size());
//==     splindex<block_cyclic> spl_bands_row(parameters_.num_fv_states(), kp__->num_ranks_row(), kp__->rank_row(), blacs_grid_.cyclic_block_size());
//== 
//==     /* local number of basis functions for the communication grid column */
//==     int num_phi = (int)spl_bands_col.local_size();
//==     /* maximum number of bais functions */
//==     int max_num_phi = (int)spl_bands_col.local_size(0);
//== 
//==     mdarray<double_complex, 3> phi_tmp;
//==     switch (parameters_.processing_unit())
//==     {
//==         case CPU:
//==         {
//==             phi_tmp = mdarray<double_complex, 3>(kappa__.at<CPU>(), kp__->num_gkvec_row(), max_num_phi, 2);
//==             break;
//==         }
//==         case GPU:
//==         {
//==             phi_tmp = mdarray<double_complex, 3>(kappa__.at<CPU>(), kappa__.at<GPU>(), kp__->num_gkvec_row(), max_num_phi, 2);
//==             break;
//==         }
//==     }
//==     
//==     mdarray<double_complex, 3> h_tmp(num_phi, max_num_phi, 2);
//==     mdarray<double_complex, 3> o_tmp(num_phi, max_num_phi, 2);
//== 
//==     #ifdef _GPU_
//==     if (parameters_.processing_unit() == GPU)
//==     {
//==         h_tmp.allocate_on_device();
//==         o_tmp.allocate_on_device();
//==     }
//==     #endif
//== 
//==     std::array<std::atomic_bool, 2> lock_phi;
//==     std::array<std::atomic_bool, 2> lock_h;
//==     std::array<std::atomic_bool, 2> lock_o;
//==     for (int i = 0; i < 2; i++)
//==     {
//==         lock_phi[i].store(false);
//==         lock_h[i].store(false);
//==         lock_o[i].store(false);
//==     }
//==    
//==     Timer t1("sirius::Band::set_fv_h_o_ncpp_parallel|zgemm_eff", kp__->comm());
//== 
//==     auto pu = parameters_.processing_unit();
//== 
//==     auto bcast_column = [kp__, &spl_bands_col, pu]
//==                         (int icol, dmatrix<double_complex>& mtrx, mdarray<double_complex, 3>& mtrx_tmp) -> void
//==     {
//==         Timer t("sirius::Band::set_fv_h_o_ncpp_parallel|bcast");
//== 
//==         #ifdef _GPU_
//==         #ifdef _GPU_DIRECT_
//==         bool gpu_direct = true;
//==         #else
//==         bool gpu_direct = false;
//==         #endif
//==         #endif
//==  
//==         int nloc = (int)spl_bands_col.local_size(icol);
//==         size_t panel_size = kp__->num_gkvec_row() * nloc * sizeof(double_complex);
//== 
//==         if (!nloc) return;
//==         
//==         if (pu == CPU)
//==         {
//==             if (kp__->rank_col() == icol)
//==                 memcpy(mtrx_tmp.at<CPU>(0, 0, icol % 2), mtrx.at<CPU>(), panel_size);
//==             kp__->comm_col().bcast(mtrx_tmp.at<CPU>(0, 0, icol % 2), kp__->num_gkvec_row() * nloc, icol);
//==         }
//==         if (pu == GPU)
//==         {
//==             #ifdef _GPU_
//==             if (gpu_direct)
//==             {
//==                 if (kp__->rank_col() == icol)
//==                     cuda_copy_device_to_device(mtrx_tmp.at<GPU>(0, 0, icol % 2), mtrx.at<GPU>(), panel_size);
//==                 kp__->comm_col().bcast(mtrx_tmp.at<GPU>(0, 0, icol % 2), kp__->num_gkvec_row() * nloc, icol);
//==             } 
//==             else
//==             {
//==                 if (kp__->rank_col() == icol)
//==                     memcpy(mtrx_tmp.at<CPU>(0, 0, icol % 2), mtrx.at<CPU>(), panel_size);
//==                 kp__->comm_col().bcast(mtrx_tmp.at<CPU>(0, 0, icol % 2), kp__->num_gkvec_row() * nloc, icol);
//==                 cuda_copy_to_device(mtrx_tmp.at<GPU>(0, 0, icol % 2), mtrx_tmp.at<CPU>(0, 0, icol % 2), panel_size);
//==             }
//==             #else
//==             TERMINATE_NO_GPU
//==             #endif
//==         }
//==     };
//== 
//==     bcast_column(0, phi__, phi_tmp);
//==     lock_phi[0].store(true);
//== 
//==     int nthread = omp_get_max_threads();
//==     if (nthread > 1) omp_set_num_threads(nthread - 1);
//== 
//==     /* crate communication thread */
//==     std::thread comm_thread([kp__, num_phi, &spl_bands_col, &spl_bands_row, &lock_phi, &lock_h, &lock_o, 
//==                              &phi__, &hphi__, &phi_tmp, &h_tmp, &o_tmp, &h__, &o__, bcast_column]()
//==     {
//==         for (int icol = 0; icol < kp__->num_ranks_col(); icol++)
//==         {
//==             /* local number of basis functions for the column icol*/
//==             int n = (int)spl_bands_col.local_size(icol);
//==             
//==             /* broadcast next column */
//==             if (icol + 1 < kp__->num_ranks_col())
//==             {
//==                 Timer t1("sirius::Band::set_fv_h_o_ncpp_parallel|comm|1");
//==                 while (lock_phi[(icol + 1) % 2].load());
//==                 bcast_column(icol + 1, phi__, phi_tmp);
//==                 lock_phi[(icol + 1) % 2].store(true);
//==             }
//==     
//==             if (n > 0)
//==             {
//==                 Timer t1("sirius::Band::set_fv_h_o_ncpp_parallel|comm|2");
//==                 while (!lock_h[icol % 2].load());
//==                 kp__->comm_row().allreduce(&h_tmp(0, 0, icol % 2), num_phi * n);
//== 
//==                 for (int j = 0; j < n; j++)
//==                 {
//==                     int idx_glob = (int)spl_bands_col.global_index(j, icol);
//==                     auto p = spl_bands_row.location(idx_glob);
//==                     if (p.second == kp__->rank_row())
//==                     {
//==                         for (int i = 0; i < num_phi; i++)
//==                         {
//==                             h__(p.first, i) = conj(h_tmp(i, j, icol % 2));
//==                         }
//==                     }
//==                 }
//== 
//==                 /* remove lock from h buffer */
//==                 lock_h[icol % 2].store(false);
//==     
//==                 while (!lock_o[icol % 2].load());
//==                 kp__->comm_row().allreduce(&o_tmp(0, 0, icol % 2), num_phi * n);
//==     
//==                 for (int j = 0; j < n; j++)
//==                 {
//==                     int idx_glob = (int)spl_bands_col.global_index(j, icol);
//==                     auto p = spl_bands_row.location(idx_glob);
//==                     if (p.second == kp__->rank_row())
//==                     {
//==                         for (int i = 0; i < num_phi; i++)
//==                         {
//==                             o__(p.first, i) = conj(o_tmp(i, j, icol % 2));
//==                         }
//==                     }
//==                 }
//==                 /* remove lock from o buffer */
//==                 lock_o[icol % 2].store(false);
//==             }
//==         }
//==     });
//== 
//==     for (int icol = 0; icol < kp__->num_ranks_col(); icol++)
//==     {
//==         int n = (int)spl_bands_col.local_size(icol);
//== 
//==         /* wait for broadcast of this column */
//==         while (!lock_phi[icol % 2].load());
//== 
//==         if (n > 0)
//==         {
//==             Timer t1("sirius::Band::set_fv_h_o_ncpp_parallel|zgemm_loc");
//== 
//==             /* wait for unlock of h buffer */
//==             while (lock_h[icol % 2].load());
//==             if (pu == GPU)
//==             {
//==                 #ifdef _GPU_
//==                 blas<GPU>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(), 
//==                                 hphi__.at<GPU>(), hphi__.ld(), 
//==                                 phi_tmp.at<GPU>(0, 0, icol % 2), phi_tmp.ld(),
//==                                 h_tmp.at<GPU>(0, 0, icol % 2), h_tmp.ld());
//== 
//==                 cuda_copy_to_host(h_tmp.at<CPU>(0, 0, icol % 2), h_tmp.at<GPU>(0, 0, icol % 2), 
//==                                   num_phi * n * sizeof(double_complex));
//==                 #else
//==                 TERMINATE_NO_GPU
//==                 #endif
//==             }
//==             if (pu == CPU)
//==             {
//==                 blas<CPU>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(),
//==                                 hphi__.at<CPU>(), hphi__.ld(),
//==                                 phi_tmp.at<CPU>(0, 0, icol % 2), phi_tmp.ld(),
//==                                 h_tmp.at<CPU>(0, 0, icol % 2), h_tmp.ld());
//==             }
//==             lock_h[icol % 2].store(true);
//==             
//==             /* wait for unlock of o buffer */
//==             while (lock_o[icol % 2].load());
//==             if (pu == GPU)
//==             {
//==                 #ifdef _GPU_
//==                 blas<GPU>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(),
//==                                 phi__.at<GPU>(), phi__.ld(),
//==                                 phi_tmp.at<GPU>(0, 0, icol % 2), phi_tmp.ld(), 
//==                                 o_tmp.at<GPU>(0, 0, icol % 2), o_tmp.ld());
//==                 cuda_copy_to_host(o_tmp.at<CPU>(0, 0, icol % 2), o_tmp.at<GPU>(0, 0, icol % 2), n * num_phi * sizeof(double_complex));
//==                 #else
//==                 TERMINATE_NO_GPU
//==                 #endif
//==             }
//==             if (pu == CPU)
//==             {
//==                 blas<CPU>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(),
//==                                 phi__.at<CPU>(), phi__.ld(),
//==                                 phi_tmp.at<CPU>(0, 0, icol % 2), phi_tmp.ld(),
//==                                 o_tmp.at<CPU>(0, 0, icol % 2), o_tmp.ld());
//==             }
//==             lock_o[icol % 2].store(true);
//==         }
//==         /* unloc phi buffer */
//==         lock_phi[icol % 2].store(false);
//==     }
//==     comm_thread.join();
//==     omp_set_num_threads(nthread);
//== 
//==     double tval = t1.stop();
//== 
//==     if (verbosity_level >= 6 && kp__->comm().rank() == 0)
//==     {
//==         int nb = parameters_.num_fv_states();
//==         printf("effective pzgemm with M, N, K: %6i %6i %6i, %12.4f sec, %12.4f GFlops/node\n",
//==                nb, nb, kp__->num_gkvec(), tval, 2 * 8e-9 * nb * nb * kp__->num_gkvec() / tval / kp__->num_ranks());
//==     }
//== 
//==     log_function_exit(__func__);
//== }

void Band::set_fv_h_o_ncpp_parallel(K_point* kp__,
                                    dmatrix<double_complex>& phi__,
                                    dmatrix<double_complex>& hphi__,
                                    dmatrix<double_complex>& h__,
                                    dmatrix<double_complex>& o__,
                                    mdarray<double_complex, 2>& kappa__)
{
    log_function_enter(__func__);
    Timer t("sirius::Band::set_fv_h_o_ncpp_parallel", kp__->comm());
    
    splindex<block_cyclic> spl_bands_col(parameters_.num_fv_states(), kp__->num_ranks_col(), kp__->rank_col(), blacs_grid_.cyclic_block_size());
    splindex<block_cyclic> spl_bands_row(parameters_.num_fv_states(), kp__->num_ranks_row(), kp__->rank_row(), blacs_grid_.cyclic_block_size());

    /* local number of basis functions for the communication grid column */
    int num_phi = (int)spl_bands_col.local_size();
    /* maximum number of bais functions */
    int max_num_phi = (int)spl_bands_col.local_size(0);

    mdarray<double_complex, 3> phi_tmp;
    switch (parameters_.processing_unit())
    {
        case CPU:
        {
            phi_tmp = mdarray<double_complex, 3>(kappa__.at<CPU>(), kp__->num_gkvec_row(), max_num_phi, 2);
            break;
        }
        case GPU:
        {
            phi_tmp = mdarray<double_complex, 3>(kappa__.at<CPU>(), kappa__.at<GPU>(), kp__->num_gkvec_row(), max_num_phi, 2);
            break;
        }
    }
    
    mdarray<double_complex, 3> h_tmp(num_phi, max_num_phi, 2);
    mdarray<double_complex, 3> o_tmp(num_phi, max_num_phi, 2);

    matrix<double_complex> h_slab(num_phi, parameters_.num_fv_states());
    matrix<double_complex> o_slab(num_phi, parameters_.num_fv_states());
    h_slab.zero();
    o_slab.zero();

    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU)
    {
        h_tmp.allocate_on_device();
        o_tmp.allocate_on_device();
    }
    #endif

    std::array<std::atomic_bool, 2> lock_phi;
    std::array<std::atomic_bool, 2> lock_h;
    std::array<std::atomic_bool, 2> lock_o;
    for (int i = 0; i < 2; i++)
    {
        lock_phi[i].store(false);
        lock_h[i].store(false);
        lock_o[i].store(false);
    }
   
    Timer t1("sirius::Band::set_fv_h_o_ncpp_parallel|zgemm_eff", kp__->comm());

    auto pu = parameters_.processing_unit();

    auto bcast_column = [kp__, &spl_bands_col, pu]
                        (int icol, dmatrix<double_complex>& mtrx, mdarray<double_complex, 3>& mtrx_tmp) -> void
    {
        Timer t("sirius::Band::set_fv_h_o_ncpp_parallel|bcast");

        #ifdef _GPU_
        #ifdef _GPU_DIRECT_
        bool gpu_direct = true;
        #else
        bool gpu_direct = false;
        #endif
        #endif
 
        int nloc = (int)spl_bands_col.local_size(icol);
        size_t panel_size = kp__->num_gkvec_row() * nloc * sizeof(double_complex);

        if (!nloc) return;
        
        if (pu == CPU)
        {
            if (kp__->rank_col() == icol)
                memcpy(mtrx_tmp.at<CPU>(0, 0, icol % 2), mtrx.at<CPU>(), panel_size);
            kp__->comm_col().bcast(mtrx_tmp.at<CPU>(0, 0, icol % 2), kp__->num_gkvec_row() * nloc, icol);
        }
        if (pu == GPU)
        {
            #ifdef _GPU_
            if (gpu_direct)
            {
                if (kp__->rank_col() == icol)
                    cuda_copy_device_to_device(mtrx_tmp.at<GPU>(0, 0, icol % 2), mtrx.at<GPU>(), panel_size);
                kp__->comm_col().bcast(mtrx_tmp.at<GPU>(0, 0, icol % 2), kp__->num_gkvec_row() * nloc, icol);
            } 
            else
            {
                if (kp__->rank_col() == icol)
                    memcpy(mtrx_tmp.at<CPU>(0, 0, icol % 2), mtrx.at<CPU>(), panel_size);
                kp__->comm_col().bcast(mtrx_tmp.at<CPU>(0, 0, icol % 2), kp__->num_gkvec_row() * nloc, icol);
                cuda_copy_to_device(mtrx_tmp.at<GPU>(0, 0, icol % 2), mtrx_tmp.at<CPU>(0, 0, icol % 2), panel_size);
            }
            #else
            TERMINATE_NO_GPU
            #endif
        }
    };

    bcast_column(0, phi__, phi_tmp);
    lock_phi[0].store(true);

    int nthread = omp_get_max_threads();
    if (nthread > 1) omp_set_num_threads(nthread - 1);

    /* crate communication thread */
    std::thread comm_thread([kp__, num_phi, &spl_bands_col, &spl_bands_row, &lock_phi, &lock_h, &lock_o, 
                             &phi__, &hphi__, &phi_tmp, &h_tmp, &o_tmp, &h__, &o__, bcast_column, &h_slab, &o_slab]()
    {
        for (int icol = 0; icol < kp__->num_ranks_col(); icol++)
        {
            /* local number of basis functions for the column icol*/
            int n = (int)spl_bands_col.local_size(icol);
            
            /* broadcast next column */
            if (icol + 1 < kp__->num_ranks_col())
            {
                Timer t1("sirius::Band::set_fv_h_o_ncpp_parallel|comm|1");
                while (lock_phi[(icol + 1) % 2].load());
                bcast_column(icol + 1, phi__, phi_tmp);
                lock_phi[(icol + 1) % 2].store(true);
            }
    
            if (n > 0)
            {
                Timer t1("sirius::Band::set_fv_h_o_ncpp_parallel|comm|2");
                while (!lock_h[icol % 2].load());

                for (int j = 0; j < n; j++)
                {
                    int idx_glob = (int)spl_bands_col.global_index(j, icol);
                    for (int i = 0; i < num_phi; i++)
                        h_slab(i, idx_glob) += conj(h_tmp(i, j, icol % 2));
                }
                /* remove lock from h buffer */
                lock_h[icol % 2].store(false);
    
                while (!lock_o[icol % 2].load());
    
                for (int j = 0; j < n; j++)
                {
                    int idx_glob = (int)spl_bands_col.global_index(j, icol);
                    for (int i = 0; i < num_phi; i++)
                        o_slab(i, idx_glob) += conj(o_tmp(i, j, icol % 2));
                }
                /* remove lock from o buffer */
                lock_o[icol % 2].store(false);
            }
        }
    });

    for (int icol = 0; icol < kp__->num_ranks_col(); icol++)
    {
        int n = (int)spl_bands_col.local_size(icol);

        /* wait for broadcast of this column */
        while (!lock_phi[icol % 2].load());

        if (n > 0)
        {
            Timer t1("sirius::Band::set_fv_h_o_ncpp_parallel|zgemm_loc");

            /* wait for unlock of h buffer */
            while (lock_h[icol % 2].load());
            if (pu == GPU)
            {
                #ifdef _GPU_
                linalg<GPU>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(), 
                                  hphi__.at<GPU>(), hphi__.ld(), 
                                  phi_tmp.at<GPU>(0, 0, icol % 2), phi_tmp.ld(),
                                  h_tmp.at<GPU>(0, 0, icol % 2), h_tmp.ld());

                cuda_copy_to_host(h_tmp.at<CPU>(0, 0, icol % 2), h_tmp.at<GPU>(0, 0, icol % 2), 
                                  num_phi * n * sizeof(double_complex));
                #else
                TERMINATE_NO_GPU
                #endif
            }
            if (pu == CPU)
            {
                linalg<CPU>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(),
                                  hphi__.at<CPU>(), hphi__.ld(),
                                  phi_tmp.at<CPU>(0, 0, icol % 2), phi_tmp.ld(),
                                  h_tmp.at<CPU>(0, 0, icol % 2), h_tmp.ld());
            }
            lock_h[icol % 2].store(true);
            
            /* wait for unlock of o buffer */
            while (lock_o[icol % 2].load());
            if (pu == GPU)
            {
                #ifdef _GPU_
                linalg<GPU>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(),
                                  phi__.at<GPU>(), phi__.ld(),
                                  phi_tmp.at<GPU>(0, 0, icol % 2), phi_tmp.ld(), 
                                  o_tmp.at<GPU>(0, 0, icol % 2), o_tmp.ld());
                cuda_copy_to_host(o_tmp.at<CPU>(0, 0, icol % 2), o_tmp.at<GPU>(0, 0, icol % 2), n * num_phi * sizeof(double_complex));
                #else
                TERMINATE_NO_GPU
                #endif
            }
            if (pu == CPU)
            {
                linalg<CPU>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(),
                                  phi__.at<CPU>(), phi__.ld(),
                                  phi_tmp.at<CPU>(0, 0, icol % 2), phi_tmp.ld(),
                                  o_tmp.at<CPU>(0, 0, icol % 2), o_tmp.ld());
            }
            lock_o[icol % 2].store(true);
        }
        /* unloc phi buffer */
        lock_phi[icol % 2].store(false);
    }
    comm_thread.join();
    omp_set_num_threads(nthread);

    kp__->comm_row().allreduce(&h_slab(0, 0), parameters_.num_fv_states() * num_phi);
    kp__->comm_row().allreduce(&o_slab(0, 0), parameters_.num_fv_states() * num_phi);

    for (int j = 0; j < (int)spl_bands_col.local_size(); j++)
    {
        for (int i = 0; i < (int)spl_bands_row.local_size(); i++)
        {
            h__(i, j) = h_slab(j, spl_bands_row[i]);
            o__(i, j) = o_slab(j, spl_bands_row[i]);
        }
    }

    double tval = t1.stop();

    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    {
        int nb = parameters_.num_fv_states();
        printf("effective pzgemm with M, N, K: %6i %6i %6i, %12.4f sec, %12.4f GFlops/node\n",
               nb, nb, kp__->num_gkvec(), tval, 2 * 8e-9 * nb * nb * kp__->num_gkvec() / tval / kp__->num_ranks());
    }

    log_function_exit(__func__);
}

#ifdef _SCALAPACK_
void Band::generate_fv_states_pp(K_point* kp__,
                                 int num_phi__,
                                 dmatrix<double_complex>& evec__,
                                 dmatrix<double_complex>& phi__,
                                 dmatrix<double_complex>& psi__,
                                 matrix<double_complex>& kappa__)
{
    log_function_enter(__func__);
    Timer t("sirius::Band::generate_fv_states_pp", kp__->comm());

    auto pu = parameters_.processing_unit();

    splindex<block_cyclic> spl_num_bands_col(parameters_.num_fv_states(), kp__->num_ranks_col(), kp__->rank_col(),
                                             blacs_grid_.cyclic_block_size());
    splindex<block_cyclic> spl_num_bands_row(parameters_.num_fv_states(), kp__->num_ranks_row(), kp__->rank_row(),
                                             blacs_grid_.cyclic_block_size());
    

    int num_bands = parameters_.num_fv_states();
    /* transpose matrix of eigen-vectors */
    dmatrix<double_complex> evec_t(num_bands, num_phi__, kp__->blacs_grid());
    linalg<CPU>::tranu(num_bands, num_phi__, evec__, 0, 0, evec_t, 0, 0);
    
    /* local number of basis function |phi> */
    int num_phi_loc = evec_t.num_cols_local();
    
    int num_bnd_max = (int)spl_num_bands_col.local_size(0);

    mdarray<double_complex, 3> evec_tmp(num_phi_loc, num_bnd_max, 2);
    #ifdef _GPU_
    if (pu == GPU) evec_tmp.allocate_on_device();
    #endif
    
    std::array<std::atomic_bool, 2> lock_evec_tmp;
    std::atomic_bool lock_psi_tmp;
    for (int i = 0; i < 2; i++) lock_evec_tmp[i].store(false);
    lock_psi_tmp.store(false);

    matrix<double_complex> psi_tmp;

    switch (parameters_.processing_unit())
    {
        case CPU:
        {
            psi_tmp = matrix<double_complex>(kappa__.at<CPU>(), kp__->num_gkvec_row(), num_bnd_max);
            break;
        }
        case GPU:
        {
            psi_tmp = matrix<double_complex>(kappa__.at<CPU>(), kappa__.at<GPU>(), kp__->num_gkvec_row(), num_bnd_max);
            break;
        }
    }

    auto get_evec = [kp__, &spl_num_bands_col, &spl_num_bands_row, &evec_t, &evec_tmp, num_phi_loc, pu]
                    (int icol) -> void 
    {
        int num_bands_of_col = (int)spl_num_bands_col.local_size(icol);
        memset(&evec_tmp(0, 0, icol % 2), 0, num_phi_loc * num_bands_of_col * sizeof(double_complex));
        for (int i = 0; i < num_bands_of_col; i++)
        {
            int iglob = (int)spl_num_bands_col.global_index(i, icol);
            auto p = spl_num_bands_row.location(iglob); 
            
            if (p.second == kp__->rank_row())
            {
                for (int j = 0; j < num_phi_loc; j++) evec_tmp(j, i, icol % 2) = evec_t(p.first, j);
            }
        }
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

    Timer t1("Band::generate_fv_states_pp|zgemm_eff");

    /* get evec for first column */
    get_evec(0);
    lock_evec_tmp[0].store(true);
    
    /* communication thread */
    std::thread comm_thread([kp__, &lock_evec_tmp, &psi__, &lock_psi_tmp, &psi_tmp, &spl_num_bands_col, get_evec, pu]()
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
            
            while (!lock_psi_tmp.load());
            switch (pu)
            {
                case CPU:
                {
                    kp__->comm_col().reduce(psi_tmp.at<CPU>(), psi__.at<CPU>(), kp__->num_gkvec_row() * num_bands_of_col, icol);
                    break;
                }
                case GPU:
                {
                    #ifdef _GPU_
                    if (gpu_direct)
                    {
                        kp__->comm_col().reduce(psi_tmp.at<GPU>(), psi__.at<GPU>(), kp__->num_gkvec_row() * num_bands_of_col, icol);
                    } 
                    else
                    {
                        cuda_copy_to_host(psi_tmp.at<CPU>(), psi_tmp.at<GPU>(), kp__->num_gkvec_row() * num_bands_of_col * sizeof(double_complex));
                        kp__->comm_col().reduce(psi_tmp.at<CPU>(), psi__.at<CPU>(), kp__->num_gkvec_row() * num_bands_of_col, icol);
                        if (icol == kp__->rank_col())
                            cuda_copy_to_device(psi__.at<GPU>(), psi__.at<CPU>(), kp__->num_gkvec_row() * num_bands_of_col * sizeof(double_complex));
                    }
                    #endif
                    break;
                }
            }
            lock_psi_tmp.store(false);
        }
    });

    int nthread = omp_get_max_threads();
    if (nthread > 1) omp_set_num_threads(nthread - 1);

    for (int rank_col = 0; rank_col < kp__->num_ranks_col(); rank_col++)
    {
        int num_bands_of_rank = (int)spl_num_bands_col.local_size(rank_col);
        
        while (!lock_evec_tmp[rank_col % 2].load());
        
        while (lock_psi_tmp.load());
        switch (pu)
        {
            case CPU:
            {
                linalg<CPU>::gemm(0, 0, kp__->num_gkvec_row(), num_bands_of_rank, num_phi_loc, 
                                  phi__.at<CPU>(), phi__.ld(), evec_tmp.at<CPU>(0, 0, rank_col % 2), evec_tmp.ld(), 
                                  psi_tmp.at<CPU>(), psi_tmp.ld());
                break;
            }
            case GPU:
            {
                #ifdef _GPU_
                linalg<GPU>::gemm(0, 0, kp__->num_gkvec_row(), num_bands_of_rank, num_phi_loc, 
                                  phi__.at<GPU>(), phi__.ld(), evec_tmp.at<GPU>(0, 0, rank_col % 2), evec_tmp.ld(), 
                                  psi_tmp.at<GPU>(), psi_tmp.ld());
                cuda_device_synchronize();
                break;
                #endif
            }
        }
        lock_psi_tmp.store(true);
       
        lock_evec_tmp[rank_col % 2].store(false);
    }
    comm_thread.join();
    
    omp_set_num_threads(nthread);

    double tval = t1.stop();

    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    {
        printf("effective zgemm #6&7 with M, N, K: %6i %6i %6i,                        %12.4f sec, %12.4f GFlops/node\n",
               kp__->num_gkvec(), num_bands, num_phi__,
               tval, 2 * 8e-9 * kp__->num_gkvec() * num_bands * num_phi__ / tval / kp__->num_ranks());
    }

    log_function_exit(__func__);
}
#endif

#ifdef _SCALAPACK_
void Band::diag_fv_ncpp_parallel(K_point* kp__,
                                 double v0__,
                                 std::vector<double>& veff_it_coarse__)
{
    log_function_enter(__func__);
    Timer t("sirius::Band::diag_fv_ncpp_parallel", kp__->comm());

    /* cache kinetic energy */
    std::vector<double> pw_ekin = kp__->get_pw_ekin();
    
    /* get diagonal elements for preconditioning */
    std::vector<double_complex> h_diag;
    std::vector<double_complex> o_diag;
    get_h_o_diag<false>(kp__, v0__, pw_ekin, h_diag, o_diag);

    /* short notation for number of target wave-functions */
    int num_bands = parameters_.num_fv_states();

    auto& itso = parameters_.iterative_solver_input_section_;

    int order = itso.num_steps_ + 2;

    std::vector< dmatrix<double_complex> > phi(order);
    for (int i = 0; i < order; i++)
    {
        phi[i] = dmatrix<double_complex>(kp__->num_gkvec(), num_bands, kp__->blacs_grid());
        phi[i].allocate_ata_buffer((int)kp__->spl_fv_states().local_size(0));
    }

    dmatrix<double_complex> hmlt(num_bands, num_bands, kp__->blacs_grid());
    dmatrix<double_complex> ovlp(num_bands, num_bands, kp__->blacs_grid());

    dmatrix<double_complex> evec(num_bands, num_bands, kp__->blacs_grid());
    std::vector<double> eval(num_bands);

    /* alias for wave-functions */
    dmatrix<double_complex>& psi = kp__->fv_states_panel();
    
    /* trial basis functions */
    psi.panel() >> phi[0].panel();

    auto uc = parameters_.unit_cell();

    int num_atoms_in_block = std::min(uc->num_atoms(), 256);
    int num_bands_local = (int)kp__->spl_fv_states().local_size(0);
    int kappa_size = std::max(uc->max_mt_basis_size() * num_atoms_in_block, 4 * num_bands_local);
    /* large temporary array for <G+k|beta>, hphi_tmp, ophi_tmp, hpsi_tmp, opsi_tmp */
    matrix<double_complex> kappa(kp__->num_gkvec_row(), kappa_size);
    if (kp__->comm().rank() == 0)
    {
        printf("size of kappa array: %f GB\n", 16 * double(kappa.size()) / 1073741824);
    }
    
    /* offset in the packed array of on-site matrices */
    mdarray<int, 1> packed_mtrx_offset(uc->num_atoms());
    int packed_mtrx_size = 0;
    for (int ia = 0; ia < uc->num_atoms(); ia++)
    {   
        int nbf = uc->atom(ia)->mt_basis_size();
        packed_mtrx_offset(ia) = packed_mtrx_size;
        packed_mtrx_size += nbf * nbf;
    }
    
    /* pack D matrices */
    mdarray<double_complex, 1> d_mtrx_packed(packed_mtrx_size);

    for (int ia = 0; ia < uc->num_atoms(); ia++)
    {
        int nbf = uc->atom(ia)->mt_basis_size();
        for (int xi2 = 0; xi2 < nbf; xi2++)
        {
            for (int xi1 = 0; xi1 < nbf; xi1++)
            {
                d_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = uc->atom(ia)->d_mtrx(xi1, xi2);
            }
        }
    }
    
    /* copy G+k vectors to device */
    matrix<double> gkvec_row(3, kp__->num_gkvec_row());
    for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
    {
        for (int x = 0; x < 3; x++) gkvec_row(x, igk_row) = kp__->gklo_basis_descriptor_row(igk_row).gkvec[x];
    }

    auto& beta_gk_t = kp__->beta_gk_t();

    if (parameters_.processing_unit() == GPU)
    {
        #ifdef _GPU_
        psi.allocate_on_device();
        for (int i = 0; i < order; i++) phi[i].allocate_on_device();
        kappa.allocate_on_device();
        d_mtrx_packed.allocate_on_device();
        d_mtrx_packed.copy_to_device();
        gkvec_row.allocate_on_device();
        gkvec_row.copy_to_device();
        //beta_gk_t.allocate_on_device();
        //beta_gk_t.copy_to_device();
        #else
        TERMINATE_NO_GPU
        #endif
    }

    /* apply Hamiltonian to the basis functions */
    apply_h_ncpp_parallel(kp__, veff_it_coarse__, pw_ekin, phi[0], phi[1], num_atoms_in_block, 
                          kappa, beta_gk_t, gkvec_row, packed_mtrx_offset, d_mtrx_packed);

    /* compute Rayleight quotients */
    std::vector<double> e0(num_bands, 0.0);
    #pragma omp parallel for schedule(static)
    for (int iloc = 0; iloc < (int)kp__->spl_fv_states().local_size(); iloc++)
    {
        int i = kp__->spl_fv_states(iloc);
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
        {
            e0[i] += real(conj(phi[0](igk_row, iloc)) * phi[1](igk_row, iloc));
        }
    }
    kp__->comm().allreduce(e0);
    
    /* estimate low and upper bounds of the Chebyshev filter */
    double lambda0 = -1e10;
    for (int i = 0; i < num_bands; i++) lambda0 = std::max(lambda0, e0[i]);
    double lambda1 = 0.5 * std::pow(parameters_.gk_cutoff(), 2);

    double r = (lambda1 - lambda0) / 2.0;
    double c = (lambda1 + lambda0) / 2.0;
    
    /* compute \psi_1 = (H\psi_0 - c\psi_0) / r */
    #pragma omp parallel for schedule(static)
    for (int iloc = 0; iloc < (int)kp__->spl_fv_states().local_size(); iloc++)
    {
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
        {
            phi[1](igk_row, iloc) = (phi[1](igk_row, iloc) - phi[0](igk_row, iloc) * c) / r;
        }
    }
    
    /* compute higher polinomial orders */
    for (int k = 2; k < order; k++)
    {
        apply_h_ncpp_parallel(kp__, veff_it_coarse__, pw_ekin, phi[k - 1], phi[k], num_atoms_in_block, 
                              kappa, beta_gk_t, gkvec_row, packed_mtrx_offset, d_mtrx_packed);

        #pragma omp parallel for schedule(static)
        for (int iloc = 0; iloc < (int)kp__->spl_fv_states().local_size(); iloc++)
        {
            for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
            {
                phi[k](igk_row, iloc) = (phi[k](igk_row, iloc) - c * phi[k - 1](igk_row, iloc)) * 2.0 / r - phi[k - 2](igk_row, iloc);
            }
        }
    }
    
    /* apply Hamiltonian to the "filtered" basis functions */
    apply_h_ncpp_parallel(kp__, veff_it_coarse__, pw_ekin, phi[order - 1], phi[0], num_atoms_in_block, 
                          kappa, beta_gk_t, gkvec_row, packed_mtrx_offset, d_mtrx_packed);
    
    set_fv_h_o_ncpp_parallel(kp__, phi[order - 1], phi[0], hmlt, ovlp, kappa);

    //Timer t1("sirius::Band::diag_fv_ncpp_parallel|set_h_o", kp__->comm());
    //blas<CPU>::gemm(2, 0, num_bands, num_bands, kp__->num_gkvec(), complex_one, phi[order - 1], phi[0],         complex_zero, hmlt);
    //blas<CPU>::gemm(2, 0, num_bands, num_bands, kp__->num_gkvec(), complex_one, phi[order - 1], phi[order - 1], complex_zero, ovlp);
    //double tval = t1.stop();

    //if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    //{
    //    printf("2x pzgemm with M, N, K: %6i %6i %6i: %12.4f sec, %12.4f GFlops/rank\n",
    //           num_bands, num_bands, kp__->num_gkvec(),
    //           tval, 2 * 8e-9 * num_bands * num_bands * kp__->num_gkvec() / tval / kp__->num_ranks());
    //}
    
    Timer t2("sirius::Band::diag_fv_ncpp_parallel|gen_evp");
    gen_evp_solver()->solve(num_bands, hmlt.num_rows_local(), hmlt.num_cols_local(), num_bands, 
                            hmlt.ptr(), hmlt.ld(), ovlp.ptr(), ovlp.ld(), 
                            &eval[0], evec.ptr(), evec.ld());
    t2.stop();
        
    if (kp__->comm().rank() == 0)
    {
        printf("eigen-values:\n");
        for (int i = 0; i < std::min(num_bands, 10); i++) printf("%18.12f ", eval[i]);
        printf("\n");
    }

    generate_fv_states_pp(kp__, num_bands, evec, phi[order - 1], psi, kappa);
    
    //Timer t3("sirius::Band::diag_fv_ncpp_parallel|psi");
    ///* recompute wave-functions: \Psi_{i} = \phi_{mu} * Z_{mu, i} */
    //blas<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands, num_bands, complex_one, phi[order - 1], evec, complex_zero, psi); 
    //t3.stop();
    
    #ifdef _GPU_
    if (parameters_.processing_unit() == GPU)
    {
        //beta_gk_t.deallocate_on_device();
        psi.deallocate_on_device();
    }
    #endif

    kp__->set_fv_eigen_values(&eval[0]);
    log_function_exit(__func__);
}
#endif

}
