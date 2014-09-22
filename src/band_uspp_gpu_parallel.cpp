#include <thread>
#include <atomic>
#include "band.h"

namespace sirius {

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
                                          dmatrix<double_complex>& ophi__,
                                          int num_atoms_in_block__,
                                          matrix<double_complex>& kappa__,
                                          matrix<double_complex>& beta_pw_t__,
                                          matrix<double>& gkvec_row__,
                                          mdarray<int, 1>& packed_mtrx_offset__,
                                          mdarray<double_complex, 1>& d_mtrx_packed__,
                                          mdarray<double_complex, 1>& q_mtrx_packed__)
{
    log_function_enter(__func__);
    Timer t("sirius::Band::apply_h_o_uspp_gpu_parallel_v2", kp__->comm());

    splindex<block_cyclic> s0(N__,       kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());
    splindex<block_cyclic> s1(N__ + n__, kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());

    /* local number of states to which Hamiltonian has to be applied */
    int nloc = static_cast<int>(s1.local_size() - s0.local_size());

    if (!nloc) return;

    auto uc = parameters_.unit_cell();

    /* apply local part of Hamiltonian */
    apply_h_local_parallel(kp__, effective_potential__, pw_ekin__, N__, n__, phi__, hphi__);

    if (parameters_.processing_unit() == cpu)
    {
        /* set intial ophi */
        memcpy(&ophi__(0, s0.local_size()), &phi__(0, s0.local_size()), kp__->num_gkvec_row() * nloc * sizeof(double_complex));
    }

    #ifdef _GPU_
    if (parameters_.processing_unit() == gpu)
    {
        /* copy hphi do device */
        cuda_copy_to_device(hphi__.at<gpu>(0, s0.local_size()), hphi__.at<cpu>(0, s0.local_size()), 
                            kp__->num_gkvec_row() * nloc * sizeof(double_complex));

        /* set intial ophi */
        cuda_copy_device_to_device(ophi__.at<gpu>(0, s0.local_size()), phi__.at<gpu>(0, s0.local_size()), 
                                   kp__->num_gkvec_row() * nloc * sizeof(double_complex));
    }
    #endif

    int num_atom_blocks = uc->num_atoms() / num_atoms_in_block__ + std::min(1, uc->num_atoms() % num_atoms_in_block__);

    splindex<block> atom_blocks(uc->num_atoms(), num_atom_blocks, 0);
    
    /* allocate space for <beta|phi> array */
    int nbf_max = uc->max_mt_basis_size() * num_atoms_in_block__;
    mdarray<double_complex, 1> beta_phi_tmp(nbf_max * nloc);

    /* result of Q or D multiplied by <beta|phi> */
    matrix<double_complex> tmp(nbf_max, nloc);

    mdarray<int, 2> beta_pw_desc(3, atom_blocks.local_size(0));
    mdarray<double, 2> atom_pos(3, atom_blocks.local_size(0));

    #ifdef _GPU_
    if (parameters_.processing_unit() == gpu)
    {
        beta_phi_tmp.allocate_on_device();
        tmp.allocate_on_device();
        beta_pw_desc.allocate_on_device();
        atom_pos.allocate_on_device();
    }
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
            /* offset in beta_pw_t */
            beta_pw_desc(2, i) = type->offset_lo();

            nbf_in_block += uc->atom(ia)->mt_basis_size();
        }

        double_complex* beta_phi_gpu_ptr = nullptr;
        #ifdef _GPU_
        if (parameters_.processing_unit() == gpu)
        {
            beta_pw_desc.copy_to_device();
            atom_pos.copy_to_device();
            beta_phi_gpu_ptr = beta_phi_tmp.at<gpu>();
        }
        #endif

        /* wrapper for <beta|phi> with required dimensions */
        matrix<double_complex> beta_phi(beta_phi_tmp.at<cpu>(), beta_phi_gpu_ptr, nbf_in_block, nloc);

        Timer t1("sirius::Band::apply_h_o_uspp_cpu_parallel_v2|beta_phi", kp__->comm_row());
        if (parameters_.processing_unit() == cpu)
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
                        kappa__(igk_row, beta_pw_desc(1, i) + xi) = beta_pw_t__(igk_row, beta_pw_desc(2, i) + xi) * 
                                                                    conj(kp__->gkvec_phase_factor(igk_row, ia));
                    }
                }
            }

            /* compute <beta|phi> */
            blas<cpu>::gemm(2, 0, nbf_in_block, nloc, kp__->num_gkvec_row(), 
                            kappa__.at<cpu>(), kappa__.ld(), 
                            phi__.at<cpu>(0, s0.local_size()), phi__.ld(), 
                            beta_phi.at<cpu>(), beta_phi.ld());
            kp__->comm_row().allreduce(beta_phi.ptr(), (int)beta_phi.size());
        }
        #ifdef _GPU_
        if (parameters_.processing_unit() == gpu)
        {
            create_beta_pw_gpu_v2((int)atom_blocks.local_size(iab),
                                  kp__->num_gkvec_row(),
                                  beta_pw_desc.at<gpu>(),
                                  beta_pw_t__.at<gpu>(),
                                  gkvec_row__.at<gpu>(),
                                  atom_pos.at<gpu>(),
                                  kappa__.at<gpu>());

            /* compute <beta|phi> */
            blas<gpu>::gemm(2, 0, nbf_in_block, nloc, kp__->num_gkvec_row(), 
                            kappa__.at<gpu>(), kappa__.ld(), 
                            phi__.at<gpu>(0, s0.local_size()), phi__.ld(), 
                            beta_phi.at<gpu>(), beta_phi.ld());

            // TODO: GPU direct MUST(!!!) work here but it doesn't. Standalone tests work, but 
            // here the allreduce fails with a wrong result and a next crash somewhere in ELPA comm.
            beta_phi.copy_to_host();
            kp__->comm_row().allreduce(beta_phi.ptr(), (int)beta_phi.size());
            beta_phi.copy_to_device();
        }
        #endif
        double tval = t1.stop();

        if (verbosity_level >= 6 && kp__->comm().rank() == 0)
        {
            printf("<beta|phi> effective zgemm with M, N, K: %6i %6i %6i, %12.4f sec, %12.4f GFlops/node\n",
                   nbf_in_block, nloc, kp__->num_gkvec(),
                   tval, 8e-9 * nbf_in_block * nloc * kp__->num_gkvec() / tval / kp__->num_ranks_row());
        }

        if (parameters_.processing_unit() == cpu)
        {
            #pragma omp parallel for
            for (int i = 0; i < (int)atom_blocks.local_size(iab); i++)
            {
                int ia = (int)atom_blocks.global_index(i, iab);
                int ofs = beta_pw_desc(1, i);
                
                /* number of beta functions for a given atom */
                int nbf = beta_pw_desc(0, i);

                /* compute D*<beta|phi> */
                blas<cpu>::gemm(0, 0, nbf, nloc, nbf, d_mtrx_packed__.at<cpu>(packed_mtrx_offset__(ia)), nbf, 
                                beta_phi.at<cpu>(ofs, 0), beta_phi.ld(), tmp.at<cpu>(ofs, 0), tmp.ld());

            }
            
            /* compute <G+k|beta> * D*<beta|phi> and add to hphi */
            blas<cpu>::gemm(0, 0, kp__->num_gkvec_row(), nloc, nbf_in_block, complex_one,
                            kappa__.at<cpu>(), kappa__.ld(), tmp.at<cpu>(), tmp.ld(), complex_one,
                            hphi__.at<cpu>(0, s0.local_size()), hphi__.ld());

            #pragma omp parallel for
            for (int i = 0; i < (int)atom_blocks.local_size(iab); i++)
            {
                int ia = (int)atom_blocks.global_index(i, iab);
                int ofs = beta_pw_desc(1, i);
                
                /* number of beta functions for a given atom */
                int nbf = beta_pw_desc(0, i);

                /* compute Q*<beta|phi> */
                blas<cpu>::gemm(0, 0, nbf, nloc, nbf, q_mtrx_packed__.at<cpu>(packed_mtrx_offset__(ia)), nbf,
                                beta_phi.at<cpu>(ofs, 0), beta_phi.ld(), tmp.at<cpu>(ofs, 0), tmp.ld());
            }

            /* compute <G+k|beta> * Q*<beta|phi> and add to ophi */
            blas<cpu>::gemm(0, 0, kp__->num_gkvec_row(), nloc, nbf_in_block, complex_one,
                            kappa__.at<cpu>(), kappa__.ld(), tmp.at<cpu>(), tmp.ld(), complex_one,
                            ophi__.at<cpu>(0, s0.local_size()), ophi__.ld());
        }

        #ifdef _GPU_
        if (parameters_.processing_unit() == gpu)
        {
            #pragma omp parallel for
            for (int i = 0; i < (int)atom_blocks.local_size(iab); i++)
            {
                int ia = (int)atom_blocks.global_index(i, iab);
                int ofs = beta_pw_desc(1, i);
                
                /* number of beta functions for a given atom */
                int nbf = beta_pw_desc(0, i);

                /* compute D*<beta|phi> */
                blas<gpu>::gemm(0, 0, nbf, nloc, nbf, d_mtrx_packed__.at<gpu>(packed_mtrx_offset__(ia)), nbf, 
                                beta_phi.at<gpu>(ofs, 0), beta_phi.ld(), tmp.at<gpu>(ofs, 0), tmp.ld(), 
                                Platform::thread_id());

            }
            cuda_device_synchronize();
            
            double_complex alpha = complex_one;
            /* compute <G+k|beta> * D*<beta|phi> and add to hphi */
            blas<gpu>::gemm(0, 0, kp__->num_gkvec_row(), nloc, nbf_in_block, &alpha,
                            kappa__.at<gpu>(), kappa__.ld(), tmp.at<gpu>(), tmp.ld(), &alpha, 
                            hphi__.at<gpu>(0, s0.local_size()), hphi__.ld());

            #pragma omp parallel for
            for (int i = 0; i < (int)atom_blocks.local_size(iab); i++)
            {
                int ia = (int)atom_blocks.global_index(i, iab);
                int ofs = beta_pw_desc(1, i);
                
                /* number of beta functions for a given atom */
                int nbf = beta_pw_desc(0, i);

                /* compute Q*<beta|phi> */
                blas<gpu>::gemm(0, 0, nbf, nloc, nbf, q_mtrx_packed__.at<gpu>(packed_mtrx_offset__(ia)), nbf,
                                beta_phi.at<gpu>(ofs, 0), beta_phi.ld(), tmp.at<gpu>(ofs, 0), tmp.ld(), 
                                Platform::thread_id());
            }
            cuda_device_synchronize();

            /* compute <G+k|beta> * Q*<beta|phi> and add to ophi */
            blas<gpu>::gemm(0, 0, kp__->num_gkvec_row(), nloc, nbf_in_block, &alpha,
                            kappa__.at<gpu>(), kappa__.ld(), tmp.at<gpu>(), tmp.ld(), &alpha,
                            ophi__.at<gpu>(0, s0.local_size()), ophi__.ld());
        }
        #endif
    }
    log_function_exit(__func__);
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
                                           dmatrix<double_complex>& o_old__,
                                           int num_atoms_in_block__,
                                           mdarray<double_complex, 2>& kappa__,
                                           matrix<double_complex>& beta_pw_t__,
                                           matrix<double>& gkvec_row__,
                                           mdarray<int, 1>& packed_mtrx_offset__,
                                           mdarray<double_complex, 1>& d_mtrx_packed__,
                                           mdarray<double_complex, 1>& q_mtrx_packed__)
{
    log_function_enter(__func__);
    Timer t("sirius::Band::set_fv_h_o_uspp_cpu_parallel_v3", kp__->comm());
    
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

    /* apply Hamiltonian and overlap operators to the new basis functions */
    apply_h_o_uspp_gpu_parallel_v2(kp__, veff_it_coarse__, pw_ekin__, N__, n__, phi__, hphi__, ophi__,
                                   num_atoms_in_block__, kappa__, beta_pw_t__, gkvec_row__, packed_mtrx_offset__,
                                   d_mtrx_packed__, q_mtrx_packed__);

    #if defined(_GPU_) && !defined(_GPU_DIRECT_)
    if (parameters_.processing_unit() == gpu)
    {
        size_t panel_size = kp__->num_gkvec_row() * (s1_col.local_size() - s0_col.local_size()) * sizeof(double_complex);
        cuda_copy_to_host(hphi__.at<cpu>(0, s0_col.local_size()), hphi__.at<gpu>(0, s0_col.local_size()), panel_size);
        cuda_copy_to_host(ophi__.at<cpu>(0, s0_col.local_size()), ophi__.at<gpu>(0, s0_col.local_size()), panel_size);
    }
    #endif

    int max_num_hphi = 0;
    for (int icol = 0; icol < kp__->num_ranks_col(); icol++)
        max_num_hphi = std::max(max_num_hphi, (int)(s1_col.local_size(icol) - s0_col.local_size(icol)));
    
    int num_phi = (int)s1_col.local_size();

    double_complex* hphi_ptr_gpu = nullptr;
    double_complex* ophi_ptr_gpu = nullptr;
    #ifdef _GPU_
    if (parameters_.processing_unit() == gpu)
    {
        hphi_ptr_gpu = kappa__.at<gpu>(0, 0);
        ophi_ptr_gpu = kappa__.at<gpu>(0, 2 * max_num_hphi);
    }
    #endif

    mdarray<double_complex, 3> hphi_tmp(kappa__.at<cpu>(0, 0),                hphi_ptr_gpu, kp__->num_gkvec_row(), max_num_hphi, 2);
    mdarray<double_complex, 3> ophi_tmp(kappa__.at<cpu>(0, 2 * max_num_hphi), ophi_ptr_gpu, kp__->num_gkvec_row(), max_num_hphi, 2);

    mdarray<double_complex, 3> h_tmp(num_phi, max_num_hphi, 2);
    mdarray<double_complex, 3> o_tmp(num_phi, max_num_hphi, 2);

    #ifdef _GPU_
    if (parameters_.processing_unit() == gpu)
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
   
    Timer t1("sirius::Band::set_fv_h_o_uspp_cpu_parallel_v3|zgemm_eff", kp__->comm());

    auto pu = parameters_.processing_unit();

    auto bcast_column = [kp__, &s0_col, &s1_col, pu]
                        (int icol, dmatrix<double_complex>& mtrx, mdarray<double_complex, 3>& mtrx_tmp) -> void
    {
        Timer t("sirius::bcast_column");
 
        int nloc = (int)(s1_col.local_size(icol) - s0_col.local_size(icol));
        size_t panel_size = kp__->num_gkvec_row() * nloc * sizeof(double_complex);
        
        if (pu == cpu)
        {
            if (nloc > 0 && kp__->rank_col() == icol)
            {
                memcpy(mtrx_tmp.at<cpu>(0, 0, icol % 2), mtrx.at<cpu>(0, s0_col.local_size(icol)), panel_size);
            }
            kp__->comm_col().bcast(mtrx_tmp.at<cpu>(0, 0, icol % 2), kp__->num_gkvec_row() * nloc, icol);
        }
        if (pu == gpu)
        {
            #ifdef _GPU_
            #ifdef _GPU_DIRECT_
            if (nloc > 0 && kp__->rank_col() == icol)
            {
                cuda_copy_device_to_device(mtrx_tmp.at<gpu>(0, 0, icol % 2), mtrx.at<gpu>(0, s0_col.local_size(icol)), panel_size);
            }
            kp__->comm_col().bcast(mtrx_tmp.at<gpu>(0, 0, icol % 2), kp__->num_gkvec_row() * nloc, icol);
            #else
            if (nloc > 0 && kp__->rank_col() == icol)
            {
                memcpy(mtrx_tmp.at<cpu>(0, 0, icol % 2), mtrx.at<cpu>(0, s0_col.local_size(icol)), panel_size);
            }
            kp__->comm_col().bcast(mtrx_tmp.at<cpu>(0, 0, icol % 2), kp__->num_gkvec_row() * nloc, icol);
            cuda_copy_to_device(mtrx_tmp.at<gpu>(0, 0, icol % 2), mtrx_tmp.at<cpu>(0, 0, icol % 2), panel_size); 
            #endif
            #else
            TERMINATE_NO_GPU
            #endif
        }
    };

    bcast_column(0, hphi__, hphi_tmp);
    bcast_column(0, ophi__, ophi_tmp);
    lock_hphi[0].store(true);
    lock_ophi[0].store(true);

    int nthread = omp_get_max_threads();
    if (nthread > 1) omp_set_num_threads(nthread - 1);

    /* crate communication thread */
    std::thread comm_thread([kp__, &s0_col, &s1_col, &s0_row, &s1_row, &lock_hphi, &lock_ophi, &lock_h, &lock_o, 
                             &hphi__, &ophi__, &hphi_tmp, &ophi_tmp, &h_tmp, &o_tmp, &h__, &o__, bcast_column]()
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
                bcast_column(icol + 1, ophi__, ophi_tmp);
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
            Timer t2("sirius::Band::set_fv_h_o_uspp_cpu_parallel_v3|zgemm_loc");
            if (pu == gpu)
            {
                #ifdef _GPU_
                blas<gpu>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(), phi__.at<gpu>(), phi__.ld(),
                                hphi_tmp.at<gpu>(0, 0, icol % 2), hphi_tmp.ld(), h_tmp.at<gpu>(0, 0, icol % 2), h_tmp.ld());
                h_tmp.copy_to_host();
                #else
                TERMINATE_NO_GPU
                #endif
            }
            if (pu == cpu)
            {
                blas<cpu>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(), phi__.at<cpu>(), phi__.ld(),
                                hphi_tmp.at<cpu>(0, 0, icol % 2), hphi_tmp.ld(), h_tmp.at<cpu>(0, 0, icol % 2), h_tmp.ld());
            }
            lock_h[icol % 2].store(true);
            lock_hphi[icol % 2].store(false);
        }
            
        while (!lock_ophi[icol % 2].load());
        while (lock_o[icol % 2].load());
        if (n > 0)
        {
            Timer t2("sirius::Band::set_fv_h_o_uspp_cpu_parallel_v3|zgemm_loc");
            if (pu == gpu)
            {
                #ifdef _GPU_
                blas<gpu>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(), phi__.at<gpu>(), phi__.ld(),
                                ophi_tmp.at<gpu>(0, 0, icol % 2), ophi_tmp.ld(), o_tmp.at<gpu>(0, 0, icol % 2), o_tmp.ld());
                o_tmp.copy_to_host();
                #else
                TERMINATE_NO_GPU
                #endif
            }
            if (pu == cpu)
            {
                blas<cpu>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(), phi__.at<cpu>(), phi__.ld(),
                                ophi_tmp.at<cpu>(0, 0, icol % 2), ophi_tmp.ld(), o_tmp.at<cpu>(0, 0, icol % 2), o_tmp.ld());
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
        dmatrix<double_complex>::tranc(N__, n__, h__, N__, 0, h__, 0, N__);
        dmatrix<double_complex>::tranc(N__, n__, o__, N__, 0, o__, 0, N__);
    }

    /* save Hamiltonian and overlap */
    for (int i = 0; i < (int)s1_col.local_size(); i++)
    {
        memcpy(&h_old__(0, i), &h__(0, i), s1_row.local_size() * sizeof(double_complex));
        memcpy(&o_old__(0, i), &o__(0, i), s1_row.local_size() * sizeof(double_complex));
    }

    log_function_exit(__func__);
}

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
                                             parameters_.cyclic_block_size());
    splindex<block_cyclic> spl_num_bands_row(num_bands__, kp__->num_ranks_row(), kp__->rank_row(),
                                             parameters_.cyclic_block_size());
    
    /* transpose matrix of eigen-vectors */
    dmatrix<double_complex> evec_t(num_bands__, N__, kp__->blacs_grid());
    dmatrix<double_complex>::tranu(num_bands__, N__, evec__, 0, 0, evec_t, 0, 0);
    
    /* local number of basis function |phi> */
    int num_phi_loc = evec_t.num_cols_local();

    mdarray<double_complex, 3> evec_tmp(num_phi_loc, spl_num_bands_col.local_size(0), 2);
    #ifdef _GPU_
    if (pu == gpu) evec_tmp.allocate_on_device();
    #endif

    std::array<std::atomic_bool, 2> lock_evec_tmp;
    std::atomic_bool lock_hpsi_tmp;
    std::atomic_bool lock_opsi_tmp;
    for (int i = 0; i < 2; i++) lock_evec_tmp[i].store(false);
    lock_hpsi_tmp.store(false);
    lock_opsi_tmp.store(false);

    int num_bnd_max = (int)spl_num_bands_col.local_size(0);

    double_complex* hpsi_ptr_gpu = nullptr;
    double_complex* opsi_ptr_gpu = nullptr;
    #ifdef _GPU_
    if (pu == gpu)
    {
        hpsi_ptr_gpu = kappa__.at<gpu>(0, 0);
        opsi_ptr_gpu = kappa__.at<gpu>(0, num_bnd_max);
    }
    #endif

    matrix<double_complex> hpsi_tmp(kappa__.at<cpu>(0, 0),           hpsi_ptr_gpu, kp__->num_gkvec_row(), num_bnd_max);
    matrix<double_complex> opsi_tmp(kappa__.at<cpu>(0, num_bnd_max), opsi_ptr_gpu, kp__->num_gkvec_row(), num_bnd_max);

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
        if (pu == gpu)
        {
            /* send evec to gpu */
            cuda_copy_to_device(evec_tmp.at<gpu>(0, 0, icol % 2), evec_tmp.at<cpu>(0, 0, icol % 2), 
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
                case cpu:
                {
                    kp__->comm_col().reduce(hpsi_tmp.at<cpu>(), hpsi__.at<cpu>(), kp__->num_gkvec_row() * num_bands_of_col, icol);
                    break;
                }
                case gpu:
                {
                    #ifdef _GPU_
                    #ifdef _GPU_DIRECT_
                    kp__->comm_col().reduce(hpsi_tmp.at<gpu>(), hpsi__.at<gpu>(), kp__->num_gkvec_row() * num_bands_of_col, icol);
                    #else
                    cuda_copy_to_host(hpsi_tmp.at<cpu>(), hpsi_tmp.at<gpu>(), kp__->num_gkvec_row() * num_bands_of_col * sizeof(double_complex));
                    kp__->comm_col().reduce(hpsi_tmp.at<cpu>(), hpsi__.at<cpu>(), kp__->num_gkvec_row() * num_bands_of_col, icol);
                    if (icol == kp__->rank_col())
                    {
                        cuda_copy_to_device(hpsi__.at<gpu>(), hpsi__.at<cpu>(), kp__->num_gkvec_row() * num_bands_of_col * sizeof(double_complex));
                    }
                    #endif
                    #endif
                    break;
                }
            }
            lock_hpsi_tmp.store(false);

            while (!lock_opsi_tmp.load());
            switch (pu)
            {
                case cpu:
                {
                    kp__->comm_col().reduce(opsi_tmp.at<cpu>(), opsi__.at<cpu>(), kp__->num_gkvec_row() * num_bands_of_col, icol);
                    break;
                }
                case gpu:
                {
                    #ifdef _GPU_
                    #ifdef _GPU_DIRECT_
                    kp__->comm_col().reduce(opsi_tmp.at<gpu>(), opsi__.at<gpu>(), kp__->num_gkvec_row() * num_bands_of_col, icol);
                    #else
                    cuda_copy_to_host(opsi_tmp.at<cpu>(), opsi_tmp.at<gpu>(), kp__->num_gkvec_row() * num_bands_of_col * sizeof(double_complex));
                    kp__->comm_col().reduce(opsi_tmp.at<cpu>(), opsi__.at<cpu>(), kp__->num_gkvec_row() * num_bands_of_col, icol);
                    if (icol == kp__->rank_col())
                    {
                        cuda_copy_to_device(opsi__.at<gpu>(), opsi__.at<cpu>(), kp__->num_gkvec_row() * num_bands_of_col * sizeof(double_complex));
                    }
                    #endif
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
            case cpu:
            {
                blas<cpu>::gemm(0, 0, kp__->num_gkvec_row(), num_bands_of_rank, num_phi_loc, 
                                hphi__.at<cpu>(), hphi__.ld(), evec_tmp.at<cpu>(0, 0, rank_col % 2), evec_tmp.ld(), 
                                hpsi_tmp.at<cpu>(), hpsi_tmp.ld());
                break;
            }
            case gpu:
            {
                #ifdef _GPU_
                blas<gpu>::gemm(0, 0, kp__->num_gkvec_row(), num_bands_of_rank, num_phi_loc, 
                                hphi__.at<gpu>(), hphi__.ld(), evec_tmp.at<gpu>(0, 0, rank_col % 2), evec_tmp.ld(), 
                                hpsi_tmp.at<gpu>(), hpsi_tmp.ld());
                cuda_device_synchronize();
                break;
                #endif
            }
        }
        lock_hpsi_tmp.store(true);
       
        while (lock_opsi_tmp.load());
        switch (pu)
        {
            case cpu:
            {
                blas<cpu>::gemm(0, 0, kp__->num_gkvec_row(), num_bands_of_rank, num_phi_loc, 
                                ophi__.at<cpu>(), ophi__.ld(), evec_tmp.at<cpu>(0, 0, rank_col % 2), evec_tmp.ld(), 
                                opsi_tmp.at<cpu>(), opsi_tmp.ld());
                break;
            }
            case gpu:
            {
                #ifdef _GPU_
                blas<gpu>::gemm(0, 0, kp__->num_gkvec_row(), num_bands_of_rank, num_phi_loc, 
                                ophi__.at<gpu>(), ophi__.ld(), evec_tmp.at<gpu>(0, 0, rank_col % 2), evec_tmp.ld(), 
                                opsi_tmp.at<gpu>(), opsi_tmp.ld());
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

    //== #ifdef _GPU_
    //== /* copy hpsi to host memory */
    //== cublas_get_matrix(kp__->num_gkvec_row(), (int)spl_num_bands_col.local_size(), sizeof(double_complex), 
    //==                   hpsi__.at<gpu>(), hpsi__.ld(), hpsi__.ptr(), hpsi__.ld());
    //== /* copy opsi to host memory */
    //== cublas_get_matrix(kp__->num_gkvec_row(), (int)spl_num_bands_col.local_size(), sizeof(double_complex), 
    //==                   opsi__.at<gpu>(), opsi__.ld(), opsi__.ptr(), opsi__.ld());
    //== #endif

    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    {
        printf("effective zgemm #6&7 with M, N, K: %6i %6i %6i,                        %12.4f sec, %12.4f GFlops/node\n",
               kp__->num_gkvec(), num_bands__, N__,
               tval, 2 * 8e-9 * kp__->num_gkvec() * num_bands__ * N__ / tval / kp__->num_ranks());
    }

    memset(&res_norm__[0], 0, num_bands__ * sizeof(double));

    if (pu == cpu)
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

    if (pu == gpu)
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

        compute_residuals_gpu(kp__->num_gkvec_row(), res__.num_cols_local(), res_idx_gpu.at<gpu>(), eval_gpu.at<gpu>(),
                              hpsi__.at<gpu>(), opsi__.at<gpu>(), res__.at<gpu>(), res_norm_gpu.at<gpu>());
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

        apply_preconditioner_gpu(kp__->num_gkvec_row(), res__.num_cols_local(), res_idx_gpu.at<gpu>(), eval_gpu.at<gpu>(),
                                 hdiag_gpu.at<gpu>(), odiag_gpu.at<gpu>(), res__.at<gpu>(), norm2.at<gpu>());
        // TODO: test gpudirect here
        norm2.copy_to_host();
        kp__->comm().allreduce(norm2.at<cpu>(), num_bands__);
        norm2.copy_to_device();

        normalize_residuals_gpu(kp__->num_gkvec_row(), res__.num_cols_local(), res_idx_gpu.at<gpu>(),
                                norm2.at<gpu>(), res__.at<gpu>());
        #else
        TERMINATE_NO_GPU
        #endif
    }

    log_function_exit(__func__);
}

void Band::diag_fv_uspp_gpu_parallel(K_point* kp__,
                                     double v0__,
                                     std::vector<double>& veff_it_coarse__)
{
    log_function_enter(__func__);
    Timer t("sirius::Band::diag_fv_uspp_gpu_parallel", kp__->comm());
    
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
    memcpy(&phi(0, 0), &psi(0, 0), kp__->num_gkvec_row() * psi.num_cols_local() * sizeof(double_complex));

    std::vector<double> res_norm(num_bands);
    std::vector<double> res_rms(num_bands);

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
    
    /* copy G+k vectors to device */
    matrix<double> gkvec_row(3, kp__->num_gkvec_row());
    for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
    {
        for (int x = 0; x < 3; x++) gkvec_row(x, igk_row) = kp__->gklo_basis_descriptor_row(igk_row).gkvec[x];
    }

    auto& beta_pw_t = kp__->beta_pw_t();

    if (parameters_.processing_unit() == gpu)
    {
        #ifdef _GPU_
        phi.allocate_on_device();
        res.allocate_on_device();
        hphi.allocate_on_device();
        ophi.allocate_on_device();
        hpsi.allocate_on_device();
        opsi.allocate_on_device();
        kappa.allocate_on_device();
        d_mtrx_packed.allocate_on_device();
        d_mtrx_packed.copy_to_device();
        q_mtrx_packed.allocate_on_device();
        q_mtrx_packed.copy_to_device();
        gkvec_row.allocate_on_device();
        gkvec_row.copy_to_device();
        beta_pw_t.allocate_on_device();
        beta_pw_t.copy_to_device();
        /* initial phi on GPU */
        cuda_copy_to_device(phi.at<gpu>(), psi.at<cpu>(), kp__->num_gkvec_row() * psi.num_cols_local() * sizeof(double_complex));
        #else
        TERMINATE_NO_GPU
        #endif
    }

    /* current diagonalziation subspace size */
    int N = 0;

    /* number of newly added basis functions */
    int n = num_bands;

    /* start iterative diagonalization */
    for (int k = 0; k < itso.num_steps_; k++)
    {
        /* set H and O for the variational subspace */
        set_fv_h_o_uspp_gpu_parallel_v3(N, n, kp__, veff_it_coarse__, pw_ekin, phi, hphi, ophi, hmlt, ovlp, 
                                        hmlt_old, ovlp_old, num_atoms_in_block, kappa, beta_pw_t, gkvec_row,
                                        packed_mtrx_offset, d_mtrx_packed, q_mtrx_packed);
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
        
        if (kp__->comm().rank() == 0)
        {
            printf("subspace size : %i, eigen-values:\n", N);
            for (int i = 0; i < std::min(num_bands, 10); i++) printf("%18.12f ", eval[i]);
            printf("\n");
        }
        }

        /* don't recompute residuals if we are going to exit on the last iteration */
        std::vector<int> res_list;
        if (k != itso.num_steps_ - 1)
        {
            uspp_residuals_gpu_parallel(N, num_bands, kp__, eval, evec, hphi, ophi, hpsi, opsi, res, h_diag, o_diag, 
                                        res_norm, kappa);

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
            Timer t3("sirius::Band::diag_fv_uspp_cpu_parallel|update_phi");

            #ifdef _GPU_
            if (parameters_.processing_unit() == gpu) phi.copy_cols_to_host(0, N);
            #endif

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

            STOP();
            for (int i = 0; i < psi.num_cols_local(); i++) 
            {
                /* update \phi */
                memcpy(&phi(0, i), &psi(0, i), kp__->num_gkvec_row() * sizeof(double_complex));
                /* update H\phi */
                memcpy(&hphi(0, i), &hpsi(0, i), kp__->num_gkvec_row() * sizeof(double_complex));
                /* update O\phi */
                memcpy(&ophi(0, i), &opsi(0, i), kp__->num_gkvec_row() * sizeof(double_complex));
            }

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
        
        if (parameters_.processing_unit() == cpu)
        {
            /* expand variational space with extra basis functions */
            for (int i = 0; i < n; i++)
            {
                dmatrix<double_complex>::copy_col<cpu>(res, res_list[i], phi, N + i);
            }
        }
        if (parameters_.processing_unit() == gpu)
        {
            #ifdef _GPU_
            #ifdef _GPU_DIRECT_
            /* expand variational space with extra basis functions */
            for (int i = 0; i < n; i++)
            {
                dmatrix<double_complex>::copy_col<gpu>(res, res_list[i], phi, N + i);
            }
            /* copy new phi to CPU */
            phi.copy_cols_to_host(N, N + n);
            #else
            res.data().copy_to_host();
            for (int i = 0; i < n; i++)
            {
                dmatrix<double_complex>::copy_col<cpu>(res, res_list[i], phi, N + i);
            }
            phi.copy_cols_to_device(N, N + n);
            #endif
            #endif
        }
    }
    
    #ifdef _GPU_
    if (parameters_.processing_unit() == gpu) beta_pw_t.deallocate_on_device();
    #endif

    kp__->set_fv_eigen_values(&eval[0]);
    log_function_exit(__func__);
    STOP();
}
#endif

}
