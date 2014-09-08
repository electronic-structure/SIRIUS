#include <thread>
#include <atomic>
#include "band.h"

namespace sirius {

void bcast_column(Global& parameters__,
                  K_point* kp__, 
                  splindex<block_cyclic>& s0_col__, 
                  splindex<block_cyclic>& s1_col__, 
                  int icol__, 
                  dmatrix<double_complex>& m__, 
                  mdarray<double_complex, 3>& m_tmp__);

void comm_thread_worker(Global& parameters__, 
                        K_point* kp__, 
                        splindex<block_cyclic>& s0_col__, 
                        splindex<block_cyclic>& s1_col__, 
                        splindex<block_cyclic>& s1_row__, 
                        dmatrix<double_complex>& hphi__, 
                        dmatrix<double_complex>& ophi__, 
                        dmatrix<double_complex>& h__,
                        dmatrix<double_complex>& o__,
                        mdarray<double_complex, 3>& hphi_tmp__,
                        mdarray<double_complex, 3>& ophi_tmp__,
                        mdarray<double_complex, 3>& h_tmp__,
                        mdarray<double_complex, 3>& o_tmp__,
                        std::array<std::atomic_bool, 2>& lock_hphi__,
                        std::array<std::atomic_bool, 2>& lock_ophi__,
                        std::array<std::atomic_bool, 2>& lock_h__,
                        std::array<std::atomic_bool, 2>& lock_o__);

#ifdef _GPU_
extern "C" void create_beta_pw_gpu_v2(int num_atoms,
                                      int num_gkvec, 
                                      int* beta_pw_desc,
                                      double_complex* beta_pw_type,
                                      double* gkvec,
                                      double* atom_pos,
                                      double_complex* beta_pw);

void Band::apply_h_o_uspp_gpu_parallel_v2(K_point* kp__,
                                          std::vector<double>& effective_potential__,
                                          std::vector<double>& pw_ekin__,
                                          int N__,
                                          int n__,
                                          dmatrix<double_complex>& phi__,
                                          dmatrix<double_complex>& hphi__,
                                          dmatrix<double_complex>& ophi__)
{
    Timer t("sirius::Band::apply_h_o_uspp_gpu_parallel_v2", _global_timer_);

    splindex<block_cyclic> s0(N__,       kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());
    splindex<block_cyclic> s1(N__ + n__, kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());

    /* local number of states to which Hamiltonian has to be applied */
    int nloc = static_cast<int>(s1.local_size() - s0.local_size());

    if (!nloc) return;

    auto uc = parameters_.unit_cell();

    /* apply local part of Hamiltonian */
    apply_h_local_parallel(kp__, effective_potential__, pw_ekin__, N__, n__, phi__, hphi__);

    #ifdef _GPU_
    cublas_set_matrix(kp__->num_gkvec_row(), nloc, sizeof(double_complex), &hphi__(0, s0.local_size()), hphi__.ld(), 
                      hphi__.ptr_device(0, s0.local_size()), hphi__.ld());
    #endif
    
    #if !defined(_GPU_)
    /* set intial ophi */
    memcpy(&ophi__(0, s0.local_size()), &phi__(0, s0.local_size()), kp__->num_gkvec_row() * nloc * sizeof(double_complex));
    #else
    cuda_copy_device_to_device(ophi__.ptr_device(0, s0.local_size()), phi__.ptr_device(0, s0.local_size()), 
                               kp__->num_gkvec_row() * nloc * sizeof(double_complex));
    #endif

    /* offset in the packed array of on-site matrices */
    mdarray<int, 1> packed_mtrx_offset(uc->num_atoms());
    int packed_mtrx_size = 0;
    for (int ia = 0; ia < uc->num_atoms(); ia++)
    {   
        int nbf = uc->atom(ia)->mt_basis_size();
        packed_mtrx_offset(ia) = packed_mtrx_size;
        packed_mtrx_size += nbf * nbf;
    }
    
    /* pack Q and D matrices and send them to GPU */
    mdarray<double_complex, 1> d_mtrx_packed(packed_mtrx_size);
    mdarray<double_complex, 1> q_mtrx_packed(packed_mtrx_size);

    for (int ia = 0; ia < uc->num_atoms(); ia++)
    {
        int nbf = uc->atom(ia)->mt_basis_size();
        for (int xi2 = 0; xi2 < nbf; xi2++)
        {
            for (int xi1 = 0; xi1 < nbf; xi1++)
            {
                d_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = uc->atom(ia)->d_mtrx(xi1, xi2);
                q_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = uc->atom(ia)->type()->uspp().q_mtrx(xi1, xi2);
            }
        }
    }

    d_mtrx_packed.allocate_on_device();
    d_mtrx_packed.copy_to_device();
    q_mtrx_packed.allocate_on_device();
    q_mtrx_packed.copy_to_device();

    int num_atoms_in_block = 256;
    int num_atom_blocks = uc->num_atoms() / num_atoms_in_block + std::min(1, uc->num_atoms() % num_atoms_in_block);

    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    {
        printf("num_atom_blocks : %i\n", num_atom_blocks);
    }
    
    splindex<block> atom_blocks(uc->num_atoms(), num_atom_blocks, 0);
    
    auto& beta_pw_t = kp__->beta_pw_t();
    beta_pw_t.allocate_on_device();
    beta_pw_t.copy_to_device();

    /* allocate space for <beta|phi> array */
    int nbf_max = uc->max_mt_basis_size() * num_atoms_in_block;
    mdarray<double_complex, 1> beta_phi_tmp(nbf_max * nloc);
    beta_phi_tmp.allocate_on_device();

    /* result of Q or D multiplied by <beta|phi> */
    mdarray<double_complex, 2> tmp(nullptr, nbf_max, nloc);
    tmp.allocate_on_device();

    /* allocate space for beta-projectors */
    mdarray<double_complex, 2> beta_pw(nullptr, kp__->num_gkvec_row(), nbf_max);
    beta_pw.allocate_on_device();
    
    mdarray<int, 2> beta_pw_desc(3, atom_blocks.local_size(0));
    beta_pw_desc.allocate_on_device();

    mdarray<double, 2> atom_pos(3, atom_blocks.local_size(0));
    atom_pos.allocate_on_device();

    /* copy G+k vectors to device */
    mdarray<double, 2> gkvec_row(3, kp__->num_gkvec_row());
    gkvec_row.allocate_on_device();
    for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
    {
        for (int x = 0; x < 3; x++) gkvec_row(x, igk_row) = kp__->gklo_basis_descriptor_row(igk_row).gkvec[x];
    }
    gkvec_row.copy_to_device();

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
            /* offset in beta_pw_t */
            beta_pw_desc(2, i) = type->offset_lo();

            nbf_in_block += uc->atom(ia)->mt_basis_size();
        }
        beta_pw_desc.copy_to_device();
        atom_pos.copy_to_device();
        
        create_beta_pw_gpu_v2((int)atom_blocks.local_size(iab),
                              kp__->num_gkvec_row(),
                              beta_pw_desc.ptr_device(),
                              beta_pw_t.ptr_device(),
                              gkvec_row.ptr_device(),
                              atom_pos.ptr_device(),
                              beta_pw.ptr_device());

        /* wrapper for <beta|phi> with required dimensions */
        mdarray<double_complex, 2> beta_phi(beta_phi_tmp.ptr(), nbf_in_block, nloc);
        beta_phi.set_ptr_device(beta_phi_tmp.ptr_device());

        Timer t1("sirius::Band::apply_h_o_uspp_cpu_parallel_v2|beta_phi", _global_timer_);
        /* compute <beta|phi> */
        blas<gpu>::gemm(2, 0, nbf_in_block, nloc, kp__->num_gkvec_row(), 
                        beta_pw.ptr_device(), beta_pw.ld(), 
                        phi__.ptr_device(0, s0.local_size()), phi__.ld(), 
                        beta_phi.ptr_device(), beta_phi.ld());

        beta_phi.copy_to_host();
        kp__->comm_row().allreduce(beta_phi.ptr(), (int)beta_phi.size());
        beta_phi.copy_to_device();

        double tval = t1.stop();

        if (verbosity_level >= 6 && kp__->comm().rank() == 0)
        {
            printf("<beta|phi> effective zgemm with M, N, K: %6i %6i %6i, %12.4f sec, %12.4f GFlops/node\n",
                   nbf_in_block, nloc, kp__->num_gkvec(),
                   tval, 8e-9 * nbf_in_block * nloc * kp__->num_gkvec() / tval / kp__->num_ranks_row());
        }
       
        #pragma omp parallel for
        for (int i = 0; i < (int)atom_blocks.local_size(iab); i++)
        {
            int ia = (int)atom_blocks.global_index(i, iab);
            int ofs = beta_pw_desc(1, i);
            
            /* number of beta functions for a given atom */
            int nbf = beta_pw_desc(0, i);

            /* compute D*<beta|phi> */
            blas<gpu>::gemm(0, 0, nbf, nloc, nbf, d_mtrx_packed.ptr_device(packed_mtrx_offset(ia)), nbf, 
                            beta_phi.ptr_device(ofs, 0), beta_phi.ld(), tmp.ptr_device(ofs, 0), tmp.ld(), 
                            Platform::thread_id());

        }
        cuda_device_synchronize();
        
        double_complex alpha = complex_one;
        /* compute <G+k|beta> * D*<beta|phi> and add to hphi */
        blas<gpu>::gemm(0, 0, kp__->num_gkvec_row(), nloc, nbf_in_block, &alpha,
                        beta_pw.ptr_device(), beta_pw.ld(), tmp.ptr_device(), tmp.ld(), &alpha, 
                        hphi__.ptr_device(0, s0.local_size()), hphi__.ld());

        #pragma omp parallel for
        for (int i = 0; i < (int)atom_blocks.local_size(iab); i++)
        {
            int ia = (int)atom_blocks.global_index(i, iab);
            int ofs = beta_pw_desc(1, i);
            
            /* number of beta functions for a given atom */
            int nbf = beta_pw_desc(0, i);

            /* compute Q*<beta|phi> */
            blas<gpu>::gemm(0, 0, nbf, nloc, nbf, q_mtrx_packed.ptr_device(packed_mtrx_offset(ia)), nbf,
                            beta_phi.ptr_device(ofs, 0), beta_phi.ld(), tmp.ptr_device(ofs, 0), tmp.ld(), 
                            Platform::thread_id());
        }
        cuda_device_synchronize();

        /* compute <G+k|beta> * Q*<beta|phi> and add to ophi */
        blas<gpu>::gemm(0, 0, kp__->num_gkvec_row(), nloc, nbf_in_block, &alpha,
                        beta_pw.ptr_device(), beta_pw.ld(), tmp.ptr_device(), tmp.ld(), &alpha,
                        ophi__.ptr_device(0, s0.local_size()), ophi__.ld());
    }
    
    kp__->comm().barrier();
}

void Band::set_fv_h_o_uspp_gpu_parallel_v3(int N__,
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
                                           dmatrix<double_complex>& o_old__)
{
    kp__->comm().barrier();
    Timer t("sirius::Band::set_fv_h_o_uspp_cpu_parallel", _global_timer_);
    

    splindex<block_cyclic> s0_col(N__,       kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());
    splindex<block_cyclic> s1_col(N__ + n__, kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());
    splindex<block_cyclic> s0_row(N__,       kp__->num_ranks_row(), kp__->rank_row(), parameters_.cyclic_block_size());
    splindex<block_cyclic> s1_row(N__ + n__, kp__->num_ranks_row(), kp__->rank_row(), parameters_.cyclic_block_size());

    /* copy old Hamiltonian and overlap */
    for (int i = 0; i < (int)s0_col.local_size(); i++)
    {
        memcpy(&h__(0, i), &h_old__(0, i), s0_row.local_size() * sizeof(double_complex));
        memcpy(&o__(0, i), &o_old__(0, i), s0_row.local_size() * sizeof(double_complex));
    }

    #ifdef _GPU_
    cublas_set_matrix(kp__->num_gkvec_row(), (int)(s1_col.local_size() - s0_col.local_size()), sizeof(double_complex),
                      &phi__(0, s0_col.local_size()), phi__.ld(), 
                      phi__.ptr_device(0, s0_col.local_size()), phi__.ld());
    #endif

    /* apply Hamiltonian and overlap operators to the new basis functions */
    apply_h_o_uspp_gpu_parallel_v2(kp__, veff_it_coarse__, pw_ekin__, N__, n__, phi__, hphi__, ophi__);

    #ifdef _GPU_
    cublas_get_matrix(kp__->num_gkvec_row(), (int)(s1_col.local_size() - s0_col.local_size()), sizeof(double_complex), 
                      hphi__.ptr_device(0, s0_col.local_size()), hphi__.ld(),
                      &hphi__(0, s0_col.local_size()), hphi__.ld());
    cublas_get_matrix(kp__->num_gkvec_row(), (int)(s1_col.local_size() - s0_col.local_size()), sizeof(double_complex), 
                      ophi__.ptr_device(0, s0_col.local_size()), ophi__.ld(),
                      &ophi__(0, s0_col.local_size()), ophi__.ld());
    #endif
    int max_num_hphi = 0;
    for (int icol = 0; icol < kp__->num_ranks_col(); icol++)
        max_num_hphi = std::max(max_num_hphi, (int)(s1_col.local_size(icol) - s0_col.local_size(icol)));
    
    int num_phi = (int)s1_col.local_size();
 
    mdarray<double_complex, 3> hphi_tmp(kp__->num_gkvec_row(), max_num_hphi, 2);
    mdarray<double_complex, 3> ophi_tmp(kp__->num_gkvec_row(), max_num_hphi, 2);
    mdarray<double_complex, 3> h_tmp(num_phi, max_num_hphi, 2);
    mdarray<double_complex, 3> o_tmp(num_phi, max_num_hphi, 2);

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
   
    int icol = 0;
    
    Timer t1("sirius::Band::set_fv_h_o_uspp_cpu_parallel|zgemm_eff", _global_timer_);
    
    bcast_column(parameters_, kp__, s0_col, s1_col, icol, hphi__, hphi_tmp);
    bcast_column(parameters_, kp__, s0_col, s1_col, icol, ophi__, ophi_tmp);
    lock_hphi[0].store(true);
    lock_ophi[0].store(true);

    int nthread = omp_get_max_threads();
    if (nthread > 1) omp_set_num_threads(nthread - 1);

    std::thread comm_thread(comm_thread_worker,
                            std::ref(parameters_),
                            kp__,
                            std::ref(s0_col),
                            std::ref(s1_col), 
                            std::ref(s1_row),
                            std::ref(hphi__),
                            std::ref(ophi__),
                            std::ref(h__),
                            std::ref(o__),
                            std::ref(hphi_tmp),
                            std::ref(ophi_tmp),
                            std::ref(h_tmp), 
                            std::ref(o_tmp),
                            std::ref(lock_hphi),
                            std::ref(lock_ophi),
                            std::ref(lock_h),
                            std::ref(lock_o));
    
    for (int icol = 0; icol < kp__->num_ranks_col(); icol++)
    {
        int n = (int)(s1_col.local_size(icol) - s0_col.local_size(icol));

        /* wait for broadcast of this column */
        while (!lock_hphi[icol % 2].load());
        /* wait for unlock of h buffer */
        while (lock_h[icol % 2].load());

        if (n > 0)
        {
            //printf("#5 zgemm for column %i\n", icol);
            Timer t2("sirius::Band::set_fv_h_o_uspp_cpu_parallel|zgemm_loc", _global_timer_);
            blas<cpu>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(), &phi__(0, 0), phi__.ld(),
                            &hphi_tmp(0, 0, icol % 2), hphi_tmp.ld(), &h_tmp(0, 0, icol % 2), h_tmp.ld());
            lock_h[icol % 2].store(true);
            lock_hphi[icol % 2].store(false);
        }
            
        while (!lock_ophi[icol % 2].load());
        while (lock_o[icol % 2].load());
        if (n > 0)
        {
            Timer t2("sirius::Band::set_fv_h_o_uspp_cpu_parallel|zgemm_loc", _global_timer_);
            //printf("#6 zgemm for column %i\n", icol);
            blas<cpu>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(), &phi__(0, 0), phi__.ld(),
                            &ophi_tmp(0, 0, icol % 2), ophi_tmp.ld(), &o_tmp(0, 0, icol % 2), o_tmp.ld());
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
        dmatrix<double_complex>::tranc(N__, n__, h__, N__, 0, h__, 0, N__);
        dmatrix<double_complex>::tranc(N__, n__, o__, N__, 0, o__, 0, N__);
    }

    /* save Hamiltonian and overlap */
    for (int i = 0; i < (int)s1_col.local_size(); i++)
    {
        memcpy(&h_old__(0, i), &h__(0, i), s1_row.local_size() * sizeof(double_complex));
        memcpy(&o_old__(0, i), &o__(0, i), s1_row.local_size() * sizeof(double_complex));
    }

    kp__->comm().barrier();
}

void Band::diag_fv_uspp_gpu_parallel(K_point* kp__,
                                     double v0__,
                                     std::vector<double>& veff_it_coarse__)
{
    log_function_enter(__func__);
    Timer t("sirius::Band::diag_fv_uspp_gpu_parallel", _global_timer_);

    /* cache kinetic energy */
    std::vector<double> pw_ekin = kp__->get_pw_ekin();

    /* get diagonal elements for preconditioning */
    std::vector<double_complex> h_diag;
    std::vector<double_complex> o_diag;
    get_h_o_diag(kp__, v0__, pw_ekin, h_diag, o_diag);

    /* short notation for number of target wave-functions */
    int num_bands = parameters_.num_fv_states();

    auto& itso = parameters_.iterative_solver_input_section_;

    int num_phi = std::min(itso.subspace_size_ * num_bands, kp__->num_gkvec());

    dmatrix<double_complex> phi(kp__->num_gkvec(), num_phi, kp__->blacs_grid());
    dmatrix<double_complex> hphi(kp__->num_gkvec(), num_phi, kp__->blacs_grid());
    dmatrix<double_complex> ophi(kp__->num_gkvec(), num_phi, kp__->blacs_grid());

    #ifdef _GPU_
    phi.allocate_on_device();
    hphi.allocate_on_device();
    ophi.allocate_on_device();
    #endif

    /* current diagonalziation subspace size */
    int N = 0;

    /* number of newly added basis functions */
    int n = num_bands;

    dmatrix<double_complex> hmlt(num_phi, num_phi, kp__->blacs_grid());
    dmatrix<double_complex> ovlp(num_phi, num_phi, kp__->blacs_grid());
    dmatrix<double_complex> hmlt_old(num_phi, num_phi, kp__->blacs_grid());
    dmatrix<double_complex> ovlp_old(num_phi, num_phi, kp__->blacs_grid());
    
    dmatrix<double_complex> evec(num_phi, num_bands, kp__->blacs_grid());
    std::vector<double> eval(num_bands);
    std::vector<double> eval_old(num_bands);

    /* alias for wave-functions */
    dmatrix<double_complex>& psi = kp__->fv_states_panel();
    
    dmatrix<double_complex> hpsi(kp__->num_gkvec(), num_bands, kp__->blacs_grid());
    dmatrix<double_complex> opsi(kp__->num_gkvec(), num_bands, kp__->blacs_grid());
    dmatrix<double_complex> res(kp__->num_gkvec(), num_bands, kp__->blacs_grid());

    /* trial basis functions */
    assert(phi.num_rows_local() == psi.num_rows_local());
    for (int i = 0; i < psi.num_cols_local(); i++) 
        memcpy(&phi(0, i), &psi(0, i), kp__->num_gkvec_row() * sizeof(double_complex));
    //#ifdef _GPU_
    //cublas_set_matrix(kp__->num_gkvec_row(), psi.num_cols_local(), sizeof(double_complex), phi.ptr(), phi.ld(), 
    //                  phi.ptr_device(), phi.ld());
    //#endif

    std::vector<double> res_norm(num_bands);
    std::vector<double> res_rms(num_bands);

    /* start iterative diagonalization */
    for (int k = 0; k < itso.num_steps_; k++)
    {
        /* set H and O for the variational subspace */
        set_fv_h_o_uspp_gpu_parallel_v3(N, n, kp__, veff_it_coarse__, pw_ekin, phi, hphi, ophi, hmlt, ovlp, hmlt_old, ovlp_old);
        
        /* increase size of the variation space */
        N += n;

        if (verbosity_level >= 6 && kp__->comm().rank() == 0)
        {
            printf("iteration : %i, subspace size : %i\n", k, N);
        }

        {
        Timer t2("sirius::Band::diag_fv_uspp_cpu_parallel|solve_gevp");
        eval_old = eval;
        gen_evp_solver()->solve(N, hmlt.num_rows_local(), hmlt.num_cols_local(), num_bands, 
                                hmlt.ptr(), hmlt.ld(), ovlp.ptr(), ovlp.ld(), 
                                &eval[0], evec.ptr(), evec.ld());
        
        //== if (Platform::mpi_rank() == 0)
        //== {
        //==     printf("subspace size : %i, eigen-values:\n", N);
        //==     for (int i = 0; i < std::min(num_bands, 10); i++) printf("%18.12f ", eval[i]);
        //==     printf("\n");
        //== }
        }

        /* don't recompute residuals if we are going to exit on the last iteration */
        std::vector<int> res_list;
        if (k != itso.num_steps_ - 1)
        {
            //uspp_cpu_residuals_parallel_v2(N, num_bands, kp__, eval, evec, hphi, ophi, hpsi, opsi, res, h_diag, o_diag, res_norm);
            uspp_cpu_residuals_parallel(N, num_bands, kp__, eval, evec, hphi, ophi, hpsi, opsi, res, h_diag, o_diag, res_norm);

            for (int i = 0; i < num_bands; i++)
            {
                /* take the residual if it's norm is above the threshold */
                if (kp__->band_occupancy(i) > 1e-12 &&
                    (res_norm[i] > itso.tolerance_ || (res_norm[i] > itso.extra_tolerance_ && n != 0)))
                {
                    res_list.push_back(i);
                }
            }

            /* number of additional basis functions */
            n = (int)res_list.size();
        }

        /* check if we run out of variational space or eigen-vectors are converged or it's a last iteration */
        if (N + n > num_phi || n == 0 || k == (itso.num_steps_ - 1))
        {   
            Timer t3("sirius::Band::diag_fv_uspp_cpu_parallel|update_phi", _global_timer_);

            /* recompute wave-functions: \Psi_{i} = \phi_{mu} * Z_{mu, i} */
            blas<cpu>::gemm(0, 0, kp__->num_gkvec(), num_bands, N, complex_one, phi, evec, complex_zero, psi); 
            
            /* exit loop if the eigen-vectors are converged or this is the last iteration */
            if (n == 0 || k == (itso.num_steps_ - 1))
            {
                if (verbosity_level >= 6 && kp__->comm().rank() == 0)
                {
                    double demax = 0;
                    for (int i = 0; i < num_bands; i++)
                    {
                         if (kp__->band_occupancy(i) > 1e-12) demax = std::max(demax, std::abs(eval_old[i] - eval[i]));
                    }
                    if (k == 0) demax = 0.0;
                    printf("converged in %i iterations with maximum eigen-value error %18.12e\n", k, demax);
                }
                break;
            }

            for (int i = 0; i < psi.num_cols_local(); i++) 
            {
                /* update \phi */
                memcpy(&phi(0, i), &psi(0, i), kp__->num_gkvec_row() * sizeof(double_complex));
                /* update H\phi */
                memcpy(&hphi(0, i), &hpsi(0, i), kp__->num_gkvec_row() * sizeof(double_complex));
                /* update O\phi */
                memcpy(&ophi(0, i), &opsi(0, i), kp__->num_gkvec_row() * sizeof(double_complex));
            }
            STOP();

            /* update H and O matrices. */
            hmlt_old.zero();
            ovlp_old.zero();
            for (int i = 0; i < num_bands; i++)
            {
                hmlt_old.set(i, i, eval[i]);
                ovlp_old.set(i, i, complex_one);
            }
            
            /* set new size of the variational space */
            N = num_bands;
        }
        
        /* expand variational space with extra basis functions */
        for (int i = 0; i < n; i++)
        {
            dmatrix<double_complex>::copy_col(res, res_list[i], phi, N + i);
        }
    }

    kp__->set_fv_eigen_values(&eval[0]);
}
#endif

}
