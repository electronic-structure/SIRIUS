// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that 
// the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the 
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file band_uspp.cpp
 *   
 *  \brief Contains ultrasoft pseudopotential -related implementation of sirius::Band class.
 */

#include "band.h"

namespace sirius {

void Band::apply_h_local(K_point* kp, std::vector<double>& effective_potential, std::vector<double>& pw_ekin, 
                         int n, double_complex* phi__, double_complex* hphi__)
{
    Timer t("sirius::Band::apply_h_local");

    mdarray<double_complex, 2> phi(phi__, kp->num_gkvec(), n);
    mdarray<double_complex, 2> hphi(hphi__, kp->num_gkvec(), n);

    auto fft = parameters_.reciprocal_lattice()->fft_coarse();
    
    int num_fft_threads = Platform::num_fft_threads();
    #pragma omp parallel default(shared) num_threads(num_fft_threads)
    {        
        int thread_id = omp_get_thread_num();
        std::vector<double_complex> phi_r(fft->size());
        
        #pragma omp for
        for (int i = 0; i < n; i++)
        {
            fft->input(kp->num_gkvec(), kp->fft_index_coarse(), &phi(0, i), thread_id);
            fft->transform(1, thread_id);
            fft->output(&phi_r[0], thread_id);

            for (int ir = 0; ir < fft->size(); ir++) phi_r[ir] *= effective_potential[ir];

            fft->input(&phi_r[0], thread_id);
            fft->transform(-1, thread_id);
            fft->output(kp->num_gkvec(), kp->fft_index_coarse(), &hphi(0, i), thread_id);
            
            for (int igk = 0; igk < kp->num_gkvec(); igk++) hphi(igk, i) += phi(igk, i) * pw_ekin[igk];
        }
    }
}

#ifdef _GPU_
struct exec_fft_args
{
    int thread_id;
    int num_phi;
    K_point* kp;
    FFT3D<cpu>* fft;
    mdarray<double_complex, 2>* phi;
    mdarray<double_complex, 2>* hphi;
    mdarray<double, 1>* veff;
    mdarray<double_complex, 2>* gamma;
    mdarray<double_complex, 2>* kappa;
};

pthread_mutex_t exec_fft_mutex;
int idxfft;

void* exec_gpu_fft(void* args__)
{
    //== exec_fft_args* args = (exec_fft_args*)args__;

    stop_here

    //== FFT3D<gpu> fft(args->fft->grid_size());

    //== mdarray<int, 1> fft_index_coarse(args->kp->fft_index_coarse(), args->kp->num_gkvec());
    //== fft_index_coarse.allocate_on_device();
    //== fft_index_coarse.copy_to_device();

    //== int nfft_buf = (int)(args->gamma->size() / fft.size());
    //== if (nfft_buf == 0) return NULL; // TODO: fix this

    //== int nfft_max = std::min(fft.num_fft_max(), std::min(args->num_phi / 4, nfft_buf));
   
    //== fft.initialize(nfft_max); 

    //== bool done = false;

    //== while (!done)
    //== {
    //==     pthread_mutex_lock(&exec_fft_mutex);
    //==     int i = idxfft;
    //==     if (idxfft + nfft_max > args->num_phi) 
    //==     {
    //==         done = true;
    //==     }
    //==     else
    //==     {
    //==         idxfft += nfft_max;
    //==     }
    //==     pthread_mutex_unlock(&exec_fft_mutex);

    //==     if (!done)
    //==     {
    //==         cublas_set_matrix(args->kp->num_gkvec(), nfft_max, sizeof(double_complex), &(*args->phi)(0, i), args->phi->ld(), 
    //==                           args->kappa->ptr_device(), args->kappa->ld());
    //==         
    //==         // use gamma as fft buffer
    //==         fft.batch_load(args->kp->num_gkvec(), fft_index_coarse.ptr_device(), args->kappa->ptr_device(), 
    //==                        args->gamma->ptr_device());

    //==         fft.transform(1, args->gamma->ptr_device());
    //==         scale_matrix_rows_gpu(fft.size(), nfft_max, args->gamma->ptr_device(), args->veff->ptr_device());
    //==         fft.transform(-1, args->gamma->ptr_device());

    //==         fft.batch_unload(args->kp->num_gkvec(), fft_index_coarse.ptr_device(), args->gamma->ptr_device(), 
    //==                          args->kappa->ptr_device());

    //==         cublas_get_matrix(args->kp->num_gkvec(), nfft_max, sizeof(double_complex), 
    //==                           args->kappa->ptr_device(), args->kappa->ld(),
    //==                           &(*args->hphi)(0, i), args->hphi->ld());
    //==     }
    //== }

    //== fft.finalize();
    //== 
    return NULL;
}

void* exec_cpu_fft(void* args__)
{
    exec_fft_args* args = (exec_fft_args*)args__;
    
    auto fft = args->fft;

    int thread_id = args->thread_id;

    bool done = false;
    while (!done)
    {
        pthread_mutex_lock(&exec_fft_mutex);
        int i = idxfft;
        if (idxfft + 1 > args->num_phi)
        {
            done = true;
        }
        else
        {
            idxfft++;
        }
        pthread_mutex_unlock(&exec_fft_mutex);

        if (!done)
        {
            fft->input(args->kp->num_gkvec(), args->kp->fft_index_coarse(), &(*args->phi)(0, i), thread_id);
            fft->transform(1, thread_id);

            for (int ir = 0; ir < fft->size(); ir++) fft->buffer(ir, thread_id) *= (*args->veff)(ir);

            fft->transform(-1, thread_id);
            fft->output(args->kp->num_gkvec(), args->kp->fft_index_coarse(), &(*args->hphi)(0, i), thread_id);
        }
    }

    return NULL;
}

void Band::apply_h_local_gpu(K_point* kp, std::vector<double>& effective_potential, std::vector<double>& pw_ekin, 
                             int num_phi, mdarray<double_complex, 2>& gamma, mdarray<double_complex, 2>& kappa,
                             double_complex* phi__, double_complex* hphi__)
{
    Timer t("sirius::Band::apply_h_local_gpu");

    auto fft = parameters_.reciprocal_lattice()->fft_coarse();

    pthread_mutex_init(&exec_fft_mutex, NULL);

    mdarray<double, 1> veff(&effective_potential[0], fft->size());
    veff.allocate_on_device();
    veff.copy_to_device();

    mdarray<double_complex, 2> phi(phi__, kp->num_gkvec(), num_phi);
    mdarray<double_complex, 2> hphi(hphi__, kp->num_gkvec(), num_phi);

    idxfft = 0;

    int num_fft_threads = std::min(Platform::num_fft_threads() + 1, Platform::max_num_threads());
    
    std::vector<pthread_t> pthread_id(num_fft_threads);
    std::vector<exec_fft_args> args(num_fft_threads);

    for (int i = 0; i < num_fft_threads; i++)
    {
        args[i].thread_id = i;
        args[i].num_phi = num_phi;
        args[i].kp = kp;
        args[i].fft = fft;
        args[i].phi = &phi;
        args[i].hphi = &hphi;
        args[i].veff = &veff;
        args[i].gamma = &gamma;
        args[i].kappa = &kappa;
        if (i == 0 && num_fft_threads > 1)
        {
            pthread_create(&pthread_id[i], NULL, exec_gpu_fft, &args[i]);
        }
        else
        {
            pthread_create(&pthread_id[i], NULL, exec_cpu_fft, &args[i]);
        }
    }

    // sync threads
    for (int i = 0; i < num_fft_threads; i++) pthread_join(pthread_id[i], NULL);

    if (idxfft != num_phi) 
    {
        std::stringstream s;
        s << "not all FFTs are executed" << std::endl
          << " number of FFTS : " << num_phi << ", number of executed FFTs : " << idxfft;
        error_local(__FILE__, __LINE__, s);
    }

    pthread_mutex_destroy(&exec_fft_mutex);
    
    #pragma omp parallel for
    for (int i = 0; i < num_phi; i++)
    {
        for (int igk = 0; igk < kp->num_gkvec(); igk++) hphi(igk, i) += phi(igk, i) * pw_ekin[igk];
    }
}
#endif

void Band::get_h_o_diag(K_point* kp__,
                        double v0__,
                        std::vector<double>& pw_ekin__,
                        std::vector<double_complex>& h_diag__,
                        std::vector<double_complex>& o_diag__)
{
    Timer t("sirius::Band::get_h_o_diag");

    h_diag__.resize(kp__->num_gkvec_row());
    o_diag__.resize(kp__->num_gkvec_row());

    auto uc = parameters_.unit_cell();
    
    /* local H contribution */
    for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
    {
        int igk = kp__->gklo_basis_descriptor_row(igk_row).igk;
        h_diag__[igk_row] = pw_ekin__[igk] + v0__;
        o_diag__[igk_row] = 1.0;
    }

    /* non-local H contribution */
    auto& beta_pw_t = kp__->beta_pw_t();
    mdarray<double_complex, 2> beta_pw_tmp(uc->max_mt_basis_size(), kp__->num_gkvec_row());

    for (int iat = 0; iat < uc->num_atom_types(); iat++)
    {
        auto atom_type = uc->atom_type(iat);
        int nbf = atom_type->mt_basis_size();
        mdarray<double_complex, 2> d_sum(nbf, nbf);
        mdarray<double_complex, 2> q_sum(nbf, nbf);
        d_sum.zero();
        q_sum.zero();

        for (int i = 0; i < atom_type->num_atoms(); i++)
        {
            int ia = atom_type->atom_id(i);
        
            for (int xi2 = 0; xi2 < nbf; xi2++)
            {
                for (int xi1 = 0; xi1 < nbf; xi1++)
                {
                    d_sum(xi1, xi2) += uc->atom(ia)->d_mtrx(xi1, xi2);
                    q_sum(xi1, xi2) += uc->atom(ia)->type()->uspp().q_mtrx(xi1, xi2);
                }
            }
        }

        int ofs = uc->atom_type(iat)->offset_lo();
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
        {
            for (int xi = 0; xi < nbf; xi++) beta_pw_tmp(xi, igk_row) = beta_pw_t(igk_row, ofs + xi);
        }

        std::vector< std::pair<int, int> > idx(nbf * nbf);
        for (int xi2 = 0, n = 0; xi2 < nbf; xi2++)
        {
            for (int xi1 = 0; xi1 < nbf; xi1++) idx[n++] = std::pair<int, int>(xi1, xi2);
        }

        #pragma omp parallel for
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
        {
            for (auto& it: idx)
            {
                int xi1 = it.first;
                int xi2 = it.second;
                double_complex z = beta_pw_tmp(xi1, igk_row) * conj(beta_pw_tmp(xi2, igk_row));

                h_diag__[igk_row] += z * d_sum(xi1, xi2);
                o_diag__[igk_row] += z * q_sum(xi1, xi2);
            }
        }
    }
}

//== // memory-conservative implementation
//== void Band::apply_h_o(K_point* kp, Periodic_function<double>* effective_potential, std::vector<double>& pw_ekin, int n,
//==                      double_complex* phi__, double_complex* hphi__, double_complex* ophi__)
//== {
//==     Timer t("sirius::Band::apply_h_o");
//== 
//==     mdarray<double_complex, 2> phi(phi__, kp->num_gkvec(), n);
//==     mdarray<double_complex, 2> hphi(hphi__, kp->num_gkvec(), n);
//==     mdarray<double_complex, 2> ophi(ophi__, kp->num_gkvec(), n);
//==     
//==     // apply local part of Hamiltonian
//==     apply_h_local(kp, effective_potential, pw_ekin, n, phi__, hphi__);
//==    
//==     // set intial ophi
//==     memcpy(ophi__, phi__, kp->num_gkvec() * n * sizeof(double_complex));
//== 
//==     mdarray<double_complex, 2> beta_pw(kp->num_gkvec(), parameters_.unit_cell()->max_mt_basis_size());
//==     mdarray<double_complex, 2> beta_phi(parameters_.unit_cell()->max_mt_basis_size(), n);
//==     mdarray<double_complex, 2> d_beta_phi(parameters_.unit_cell()->max_mt_basis_size(), n);
//==     mdarray<double_complex, 2> q_beta_phi(parameters_.unit_cell()->max_mt_basis_size(), n);
//== 
//==     for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
//==     {   
//==         // number of beta functions for a given atom
//==         int nbf = parameters_.unit_cell()->atom(ia)->type()->mt_basis_size();
//== 
//==         kp->generate_beta_pw(&beta_pw(0, 0), ia);
//== 
//==         // compute <beta|phi>
//==         blas<cpu>::gemm(2, 0, nbf, n, kp->num_gkvec(), &beta_pw(0, 0), beta_pw.ld(), &phi(0, 0), phi.ld(), 
//==                         &beta_phi(0, 0), beta_phi.ld());
//== 
//==         // compute D*<beta|phi>
//==         blas<cpu>::gemm(0, 0, nbf, n, nbf, &parameters_.unit_cell()->atom(ia)->d_mtrx(0, 0), nbf, 
//==                         &beta_phi(0, 0), beta_phi.ld(), &d_beta_phi(0, 0), d_beta_phi.ld());
//==         
//==         // multiply by <G+k|beta> and add to hphi
//==         blas<cpu>::gemm(0, 0, kp->num_gkvec(), n, nbf, double_complex(1, 0), &beta_pw(0, 0), beta_pw.ld(), 
//==                         &d_beta_phi(0, 0), d_beta_phi.ld(), double_complex(1, 0), &hphi(0, 0), hphi.ld());
//==         
//==         // compute Q*<beta|phi>
//==         blas<cpu>::gemm(0, 0, nbf, n, nbf, &parameters_.unit_cell()->atom(ia)->type()->uspp().q_mtrx(0, 0), nbf, 
//==                         &beta_phi(0, 0), beta_phi.ld(), &q_beta_phi(0, 0), q_beta_phi.ld());
//==         
//==         // multiply by <G+k|beta> and add to ophi
//==         blas<cpu>::gemm(0, 0, kp->num_gkvec(), n, nbf, double_complex(1, 0), &beta_pw(0, 0), beta_pw.ld(), 
//==                         &q_beta_phi(0, 0), q_beta_phi.ld(), double_complex(1, 0), &ophi(0, 0), ophi.ld());
//==     }
//== }

// memory-greedy implementation
void Band::apply_h_o_uspp_cpu(K_point* kp__, 
                              std::vector<double>& effective_potential__, 
                              std::vector<double>& pw_ekin__, 
                              int n__,
                              double_complex* phi__, 
                              double_complex* hphi__, 
                              double_complex* ophi__)
{
    Timer t("sirius::Band::apply_h_o", _global_timer_);

    auto uc = parameters_.unit_cell();

    mdarray<double_complex, 2> phi(phi__, kp__->num_gkvec(), n__);
    mdarray<double_complex, 2> hphi(hphi__, kp__->num_gkvec(), n__);
    mdarray<double_complex, 2> ophi(ophi__, kp__->num_gkvec(), n__);
    
    /* apply local part of Hamiltonian */
    apply_h_local(kp__, effective_potential__, pw_ekin__, n__, phi__, hphi__);
   
    /* set intial ophi */
    memcpy(ophi__, phi__, kp__->num_gkvec() * n__ * sizeof(double_complex));

    /* <\beta_{\xi}^{\alpha}|\phi_j> */
    mdarray<double_complex, 2> beta_phi(uc->mt_lo_basis_size(), n__);
    
    /* Q or D multiplied by <\beta_{\xi}^{\alpha}|\phi_j> */
    mdarray<double_complex, 2> tmp(uc->mt_lo_basis_size(), n__);

    Timer t1("sirius::Band::apply_h_o|beta_phi");

    /* compute <beta|phi> */
    blas<cpu>::gemm(2, 0, uc->mt_lo_basis_size(), n__, kp__->num_gkvec(), 
                    kp__->beta_pw_panel().data().ptr(), kp__->num_gkvec(), 
                    &phi(0, 0), phi.ld(), &beta_phi(0, 0), beta_phi.ld());
    t1.stop();
    
    /* compute D*<beta|phi> */
    for (int ia = 0; ia < uc->num_atoms(); ia++)
    {   
        int ofs = uc->atom(ia)->offset_lo();
        /* number of beta functions for a given atom */
        int nbf = uc->atom(ia)->type()->mt_lo_basis_size();
        blas<cpu>::gemm(0, 0, nbf, n__, nbf, &uc->atom(ia)->d_mtrx(0, 0), nbf, 
                        &beta_phi(ofs, 0), beta_phi.ld(), &tmp(ofs, 0), tmp.ld());
    }

    Timer t3("sirius::Band::apply_h_o|beta_D_beta_phi");
    /* compute <G+k|beta> * D*<beta|phi> and add to hphi */
    blas<cpu>::gemm(0, 0, kp__->num_gkvec(), n__, uc->mt_lo_basis_size(), complex_one, 
                    kp__->beta_pw_panel().data().ptr(), kp__->num_gkvec(), 
                    &tmp(0, 0), tmp.ld(), complex_one, &hphi(0, 0), hphi.ld());
    t3.stop();

    /* compute Q*<beta|phi> */
    for (int ia = 0; ia < uc->num_atoms(); ia++)
    {   
        int ofs = uc->atom(ia)->offset_lo();
        /* number of beta functions for a given atom */
        int nbf = uc->atom(ia)->type()->mt_basis_size();
        blas<cpu>::gemm(0, 0, nbf, n__, nbf, &uc->atom(ia)->type()->uspp().q_mtrx(0, 0), nbf, 
                        &beta_phi(ofs, 0), beta_phi.ld(), &tmp(ofs, 0), tmp.ld());
    }

    Timer t5("sirius::Band::apply_h_o|beta_Q_beta_phi");
    /* computr <G+k|beta> * Q*<beta|phi> and add to ophi */
    blas<cpu>::gemm(0, 0, kp__->num_gkvec(), n__, uc->mt_lo_basis_size(), complex_one, 
                    kp__->beta_pw_panel().data().ptr(), kp__->num_gkvec(), 
                    &tmp(0, 0), tmp.ld(), complex_one, &ophi(0, 0), ophi.ld());
    t5.stop();
}

#ifdef _GPU_

// memory-greedy implementation
void Band::apply_h_o_uspp_gpu(K_point* kp, std::vector<double>& effective_potential, std::vector<double>& pw_ekin, int n,
                              mdarray<double_complex, 2>& gamma, mdarray<double_complex, 2>& kappa, 
                              double_complex* phi__, double_complex* hphi__, double_complex* ophi__)
{
    Timer t("sirius::Band::apply_h_o_uspp_gpu");

    mdarray<double_complex, 2> phi(phi__, kp->num_gkvec(), n);
    mdarray<double_complex, 2> hphi(hphi__, kp->num_gkvec(), n);
    mdarray<double_complex, 2> ophi(ophi__, kp->num_gkvec(), n);

    // apply local part of Hamiltonian
    apply_h_local_gpu(kp, effective_potential, pw_ekin, n, gamma, kappa, phi__, hphi__);
    
    // load hphi to the first part of kappa; TODO: apply_h_local_gpu must return hpi on gpu
    cublas_set_matrix(kp->num_gkvec(), n, sizeof(double_complex), hphi.ptr(), hphi.ld(), kappa.ptr_device(0, 0), kappa.ld());

    // load phi to the second part of kappa; this will be the initial ophi
    cublas_set_matrix(kp->num_gkvec(), n, sizeof(double_complex), phi.ptr(), phi.ld(), kappa.ptr_device(0, n), kappa.ld());
    
    
    // offset in the packed array of on-site matrices
    mdarray<int, 1> mtrx_ofs(parameters_.unit_cell()->num_atoms());     
    int packed_mtrx_size = 0;
    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {   
        int nbf = parameters_.unit_cell()->atom(ia)->type()->mt_basis_size();
        mtrx_ofs(ia) = packed_mtrx_size;
        packed_mtrx_size += nbf * nbf;
    }

    // pack D and Q matrices
    mdarray<double_complex, 1> d_mtrx_packed(packed_mtrx_size);
    mdarray<double_complex, 1> q_mtrx_packed(packed_mtrx_size);
    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {
        int nbf = parameters_.unit_cell()->atom(ia)->type()->mt_basis_size();
        for (int xi2 = 0; xi2 < nbf; xi2++)
        {
            for (int xi1 = 0; xi1 < nbf; xi1++)
            {
                d_mtrx_packed(mtrx_ofs(ia) + xi2 * nbf + xi1) = parameters_.unit_cell()->atom(ia)->d_mtrx(xi1, xi2);
                q_mtrx_packed(mtrx_ofs(ia) + xi2 * nbf + xi1) = parameters_.unit_cell()->atom(ia)->type()->uspp().q_mtrx(xi1, xi2);
            }
        }
    }

    kp->beta_pw_t().allocate_on_device();
    kp->beta_pw_t().copy_to_device();

    kp->gkvec().allocate_on_device(); 
    kp->gkvec().copy_to_device();

    parameters_.unit_cell()->atom_pos().allocate_on_device(); 
    parameters_.unit_cell()->atom_pos().copy_to_device();

    parameters_.unit_cell()->beta_t_idx().allocate_on_device(); 
    parameters_.unit_cell()->beta_t_idx().copy_to_device();

    // create <G+k|beta> and store it in gamma matrix
    create_beta_pw_gpu(kp->num_gkvec(), 
                       parameters_.unit_cell()->mt_lo_basis_size(), 
                       parameters_.unit_cell()->beta_t_idx().ptr_device(),
                       kp->beta_pw_t().ptr_device(),
                       kp->gkvec().ptr_device(),
                       parameters_.unit_cell()->atom_pos().ptr_device(),
                       gamma.ptr_device());

    parameters_.unit_cell()->beta_t_idx().deallocate_on_device();
    parameters_.unit_cell()->atom_pos().deallocate_on_device();
    kp->gkvec().deallocate_on_device();
    kp->beta_pw_t().deallocate_on_device();

    // <\beta_{\xi}^{\alpha}|\phi_j>
    mdarray<double_complex, 2> beta_phi(NULL, parameters_.unit_cell()->mt_lo_basis_size(), n);
    beta_phi.allocate_on_device();
    
    blas<gpu>::gemm(2, 0, parameters_.unit_cell()->mt_lo_basis_size(), n, kp->num_gkvec(), gamma.ptr_device(0, 0), gamma.ld(), 
                    kappa.ptr_device(0, n), kappa.ld(), beta_phi.ptr_device(0, 0), beta_phi.ld());

    // Q or D multiplied by <\beta_{\xi}^{\alpha}|\phi_j>
    mdarray<double_complex, 2> tmp(NULL, parameters_.unit_cell()->mt_lo_basis_size(), n);
    tmp.allocate_on_device();

    d_mtrx_packed.allocate_on_device();
    d_mtrx_packed.copy_to_device();
    // compute D*<beta|phi>
    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {   
        int ofs = parameters_.unit_cell()->atom(ia)->offset_lo();
        // number of beta functions for a given atom
        int nbf = parameters_.unit_cell()->atom(ia)->type()->mt_basis_size();
        blas<gpu>::gemm(0, 0, nbf, n, nbf, d_mtrx_packed.ptr_device(mtrx_ofs(ia)), nbf, 
                        beta_phi.ptr_device(ofs, 0), beta_phi.ld(), tmp.ptr_device(ofs, 0), tmp.ld());
    }
    d_mtrx_packed.deallocate_on_device();

    double_complex zone(1, 0);
    // compute <G+k|beta> * D*<beta|phi> and add to hphi
    blas<gpu>::gemm(0, 0, kp->num_gkvec(), n, parameters_.unit_cell()->mt_lo_basis_size(), &zone, gamma.ptr_device(), gamma.ld(), 
                    tmp.ptr_device(), tmp.ld(), &zone, kappa.ptr_device(), kappa.ld());

    q_mtrx_packed.allocate_on_device();
    q_mtrx_packed.copy_to_device();

    // compute Q*<beta|phi>
    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {   
        int ofs = parameters_.unit_cell()->atom(ia)->offset_lo();
        // number of beta functions for a given atom
        int nbf = parameters_.unit_cell()->atom(ia)->type()->mt_basis_size();
        blas<gpu>::gemm(0, 0, nbf, n, nbf, q_mtrx_packed.ptr_device(mtrx_ofs(ia)), nbf, 
                        beta_phi.ptr_device(ofs, 0), beta_phi.ld(), tmp.ptr_device(ofs, 0), tmp.ld());
    }
    q_mtrx_packed.deallocate_on_device();

    // computr <G+k|beta> * Q*<beta|phi> and add to ophi
    blas<gpu>::gemm(0, 0, kp->num_gkvec(), n, parameters_.unit_cell()->mt_lo_basis_size(), &zone, gamma.ptr_device(), gamma.ld(), 
                    tmp.ptr_device(), tmp.ld(), &zone, kappa.ptr_device(0, n), kappa.ld());
    
    kappa.copy_to_host();
    for (int j = 0; j < n; j++)
    {
        for (int igk = 0; igk < kp->num_gkvec(); igk++)
        {
            hphi(igk, j) = kappa(igk, j);
            ophi(igk, j) = kappa(igk, j + n);
        }
    }
    tmp.deallocate_on_device();
    beta_phi.deallocate_on_device();
}

//== extern "C" void create_single_beta_pw_gpu(int num_gkvec, 
//==                                           int num_beta_a, 
//==                                           int beta_a_ofs, 
//==                                           int* beta_t_idx,
//==                                           void* beta_pw_type,
//==                                           double* gkvec,
//==                                           double* atom_pos,
//==                                           void* beta_pw);
//== 
//== extern "C" void generate_beta_phi_gpu(int num_gkvec, 
//==                                       int num_beta, 
//==                                       int num_phi, 
//==                                       int* beta_t_idx, 
//==                                       double* atom_pos,
//==                                       double* gkvec,
//==                                       void* beta_pw_type,
//==                                       void* phi,
//==                                       void* beta_phi);
//== 
//== // memory-safe implementation
//== void Band::apply_h_o_uspp_gpu(K_point* kp, std::vector<double>& effective_potential, std::vector<double>& pw_ekin, int n,
//==                               double_complex* phi__, double_complex* hphi__, double_complex* ophi__)
//== {
//==     Timer t("sirius::Band::apply_h_o_uspp_gpu");
//== 
//==     mdarray<double_complex, 2> phi(phi__, kp->num_gkvec(), n);
//==     mdarray<double_complex, 2> hphi(hphi__, kp->num_gkvec(), n);
//==     mdarray<double_complex, 2> ophi(ophi__, kp->num_gkvec(), n);
//== 
//==     phi.allocate_on_device();
//==     phi.copy_to_device();
//== 
//==     hphi.allocate_on_device();
//==     hphi.zero_on_device();
//== 
//==     ophi.allocate_on_device();
//==     ophi.zero_on_device();
//== 
//==     // apply local part of Hamiltonian
//==     apply_h_local(kp, effective_potential, pw_ekin, n, phi__, hphi__);
//==     
//== 
//==     mdarray<int, 1> mtrx_ofs(parameters_.unit_cell()->num_atoms()); // offset in the packed array of on-site matrices
//==     int packed_mtrx_size = 0;
//==     for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
//==     {   
//==         int nbf = parameters_.unit_cell()->atom(ia)->type()->mt_basis_size();
//==         mtrx_ofs(ia) = packed_mtrx_size;
//==         packed_mtrx_size += nbf * nbf;
//==     }
//== 
//==     mdarray<double_complex, 1> d_mtrx_packed(packed_mtrx_size);
//==     mdarray<double_complex, 1> q_mtrx_packed(packed_mtrx_size);
//== 
//==     for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
//==     {
//==         int nbf = parameters_.unit_cell()->atom(ia)->type()->mt_basis_size();
//==         for (int xi2 = 0; xi2 < nbf; xi2++)
//==         {
//==             for (int xi1 = 0; xi1 < nbf; xi1++)
//==             {
//==                 d_mtrx_packed(mtrx_ofs(ia) + xi2 * nbf + xi1) = parameters_.unit_cell()->atom(ia)->d_mtrx(xi1, xi2);
//==                 q_mtrx_packed(mtrx_ofs(ia) + xi2 * nbf + xi1) = parameters_.unit_cell()->atom(ia)->type()->uspp().q_mtrx(xi1, xi2);
//==             }
//==         }
//==     }
//== 
//==     d_mtrx_packed.allocate_on_device();
//==     d_mtrx_packed.copy_to_device();
//==     q_mtrx_packed.allocate_on_device();
//==     q_mtrx_packed.copy_to_device();
//== 
//==     // <G+k|\beta_{\xi}^{\alpha}>
//==     mdarray<double_complex, 2> beta_pw(NULL, kp->num_gkvec(), parameters_.unit_cell()->max_mt_basis_size());
//==     beta_pw.allocate_on_device();
//== 
//==     // <\beta_{\xi}^{\alpha}|\phi_j>
//==     //== mdarray<double_complex, 2> beta_phi(NULL, parameters_.unit_cell()->max_mt_basis_size(), n);
//==     //== beta_phi.allocate_on_device();
//==     mdarray<double_complex, 2> beta_phi(NULL, parameters_.unit_cell()->num_beta_a(), n);
//==     beta_phi.allocate_on_device();
//== 
//==     // Q or D multiplied by <\beta_{\xi}^{\alpha}|\phi_j>
//==     mdarray<double_complex, 2> tmp(NULL, parameters_.unit_cell()->max_mt_basis_size(), n);
//==     tmp.allocate_on_device();
//== 
//==     kp->beta_pw().allocate_on_device();
//==     kp->beta_pw().copy_to_device();
//== 
//==     kp->gkvec_gpu().allocate_on_device(); 
//==     kp->gkvec_gpu().copy_to_device();
//== 
//==     parameters_.unit_cell()->atom_pos().allocate_on_device(); 
//==     parameters_.unit_cell()->atom_pos().copy_to_device();
//== 
//==     parameters_.unit_cell()->beta_t_idx().allocate_on_device(); 
//==     parameters_.unit_cell()->beta_t_idx().copy_to_device();
//== 
//==     generate_beta_phi_gpu(kp->num_gkvec(), 
//==                           parameters_.unit_cell()->num_beta_a(), 
//==                           n, 
//==                           parameters_.unit_cell()->beta_t_idx().ptr_device(),
//==                           parameters_.unit_cell()->atom_pos().ptr_device(),
//==                           kp->gkvec_gpu().ptr_device(),
//==                           kp->beta_pw().ptr_device(),
//==                           phi.ptr_device(),
//==                           beta_phi.ptr_device());
//== 
//== 
//==     
//==     double_complex zone(1, 0);
//==     for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
//==     {
//==         int ofs = parameters_.unit_cell()->beta_a_ofs(ia);
//==         // number of beta functions for a given atom
//==         int nbf = parameters_.unit_cell()->atom(ia)->type()->mt_basis_size();
//== 
//==         create_single_beta_pw_gpu(kp->num_gkvec(), 
//==                                   nbf,
//==                                   ofs,
//==                                   parameters_.unit_cell()->beta_t_idx().ptr_device(),
//==                                   kp->beta_pw().ptr_device(),
//==                                   kp->gkvec_gpu().ptr_device(),
//==                                   parameters_.unit_cell()->atom_pos().ptr_device(),
//==                                   beta_pw.ptr_device());
//== 
//== 
//==         // compute <beta|phi>
//==         //blas<gpu>::gemm(2, 0, nbf, n, kp->num_gkvec(), beta_pw.ptr_device(), beta_pw.ld(), 
//==         //                phi.ptr_device(), phi.ld(), beta_phi.ptr_device(), beta_phi.ld());
//== 
//== 
//==         // compute D*<beta|phi>
//==         blas<gpu>::gemm(0, 0, nbf, n, nbf, d_mtrx_packed.ptr_device(mtrx_ofs(ia)), nbf, 
//==                         beta_phi.ptr_device(ofs), beta_phi.ld(), tmp.ptr_device(), tmp.ld());
//== 
//==         // multiply by <G+k|beta> and add to hphi
//==         blas<gpu>::gemm(0, 0, kp->num_gkvec(), n, nbf, &zone, beta_pw.ptr_device(), beta_pw.ld(), 
//==                         tmp.ptr_device(), tmp.ld(), &zone, hphi.ptr_device(), hphi.ld());
//== 
//==         // compute Q*<beta|phi>
//==         blas<gpu>::gemm(0, 0, nbf, n, nbf, q_mtrx_packed.ptr_device(mtrx_ofs(ia)), nbf, 
//==                         beta_phi.ptr_device(ofs), beta_phi.ld(), tmp.ptr_device(), tmp.ld());
//==         
//==         // multiply by <G+k|beta> and add to ophi
//==         blas<gpu>::gemm(0, 0, kp->num_gkvec(), n, nbf, &zone, beta_pw.ptr_device(), beta_pw.ld(), 
//==                         tmp.ptr_device(), tmp.ld(), &zone, ophi.ptr_device(), ophi.ld());
//==     }
//== 
//==     parameters_.unit_cell()->beta_t_idx().deallocate_on_device();
//==     parameters_.unit_cell()->atom_pos().deallocate_on_device();
//==     kp->gkvec_gpu().deallocate_on_device();
//==     kp->beta_pw().deallocate_on_device();
//== 
//==     tmp.deallocate_on_device();
//==     beta_phi.deallocate_on_device();
//==     beta_pw.deallocate_on_device();
//== 
//==     q_mtrx_packed.deallocate_on_device();
//==     d_mtrx_packed.deallocate_on_device();
//== 
//== 
//==     ophi.copy_to_host(hphi.ptr_device());
//== 
//==     for (int j = 0; j < n; j++)
//==     {
//==         for (int igk = 0; igk < kp->num_gkvec(); igk++) hphi(igk, j) += ophi(igk, j);
//==     }
//== 
//==     ophi.copy_to_host();
//==     for (int j = 0; j < n; j++)
//==     {
//==         for (int igk = 0; igk < kp->num_gkvec(); igk++) ophi(igk, j) += phi(igk, j);
//==     }
//== 
//==     phi.deallocate_on_device();
//==     hphi.deallocate_on_device();
//==     ophi.deallocate_on_device();
//== }
#endif

void Band::apply_h_local_parallel(K_point* kp__,
                                  std::vector<double>& effective_potential__,
                                  std::vector<double>& pw_ekin__,
                                  int N__,
                                  int n__,
                                  dmatrix<double_complex>& phi__,
                                  dmatrix<double_complex>& hphi__)
{
    Timer t("sirius::Band::apply_h_local_parallel");

    auto fft = parameters_.reciprocal_lattice()->fft_coarse();

    splindex<block_cyclic> s0(N__,       kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());
    splindex<block_cyclic> s1(N__ + n__, kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());

    splindex<block> sub_spl_n(s1.local_size() - s0.local_size(), kp__->num_ranks_row(), kp__->rank_row());

    mdarray<double_complex, 2> phi_slice(kp__->num_gkvec(), sub_spl_n.local_size());
    phi__.gather(n__, N__, phi_slice, parameters_.mpi_grid().communicator(1 << _dim_row_));

    mdarray<double_complex, 2> hphi_slice(kp__->num_gkvec(), sub_spl_n.local_size());

    int num_fft_threads = Platform::num_fft_threads();
    #pragma omp parallel default(shared) num_threads(num_fft_threads)
    {        
        int thread_id = omp_get_thread_num();
        std::vector<double_complex> phi_r(fft->size());
        
        /* loop over local fraction of wave-functions */
        #pragma omp for
        for (int i = 0; i < (int)sub_spl_n.local_size(); i++)
        {
            fft->input(kp__->num_gkvec(), kp__->fft_index_coarse(), &phi_slice(0, i), thread_id);
            fft->transform(1, thread_id);
            fft->output(&phi_r[0], thread_id);

            for (int ir = 0; ir < fft->size(); ir++) phi_r[ir] *= effective_potential__[ir];

            fft->input(&phi_r[0], thread_id);
            fft->transform(-1, thread_id);
            fft->output(kp__->num_gkvec(), kp__->fft_index_coarse(), &hphi_slice(0, i), thread_id);
            
            for (int igk = 0; igk < kp__->num_gkvec(); igk++) hphi_slice(igk, i) += phi_slice(igk, i) * pw_ekin__[igk];
        }
    }

    hphi__.scatter(n__, N__, hphi_slice, parameters_.mpi_grid().communicator(1 << _dim_row_));
}

void Band::apply_h_o_uspp_cpu_parallel(K_point* kp__,
                                       std::vector<double>& effective_potential__,
                                       std::vector<double>& pw_ekin__,
                                       int N__,
                                       int n__,
                                       dmatrix<double_complex>& phi__,
                                       dmatrix<double_complex>& hphi__,
                                       dmatrix<double_complex>& ophi__)
{
    Timer t("sirius::Band::apply_h_o_uspp_cpu_parallel");
    log_function_enter(__func__);
    auto uc = parameters_.unit_cell();

    splindex<block_cyclic> s0(N__,       kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());
    splindex<block_cyclic> s1(N__ + n__, kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());

    int nloc = static_cast<int>(s1.local_size() - s0.local_size());

    /* apply local part of Hamiltonian */
    apply_h_local_parallel(kp__, effective_potential__, pw_ekin__, N__, n__, phi__, hphi__);
    
    /* set intial ophi */
    if (nloc > 0) memcpy(&ophi__(0, s0.local_size()), &phi__(0, s0.local_size()), kp__->num_gkvec_row() * nloc * sizeof(double_complex));

    /* <\beta_{\xi}^{\alpha}|\phi_j> */
    dmatrix<double_complex> beta_phi(uc->mt_basis_size(), n__, parameters_.blacs_context());

    /* Q or D multiplied by <\beta_{\xi}^{\alpha}|\phi_j> */
    dmatrix<double_complex> tmp(uc->mt_basis_size(), n__, parameters_.blacs_context());

    Timer t1("sirius::Band::apply_h_o_uspp_cpu_parallel|beta_phi");
    /* compute <beta|phi> */
    blas<cpu>::gemm(2, 0, uc->mt_basis_size(), n__, kp__->num_gkvec(), complex_one, 
                    kp__->beta_pw_panel(), 0, 0, phi__, 0, N__, complex_zero, beta_phi, 0, 0);
    double tval = t1.stop();

    if (verbosity_level >= 6 && parameters_.mpi_grid().root(1 << _dim_row_ | 1 << _dim_col_))
    {
        printf("pzgemm #1   with M, N, K: %6i %6i %6i,   offset in B: %6i, %12.4f sec, %12.4f GFlops/node\n",
               uc->mt_basis_size(), n__, kp__->num_gkvec(), N__, 
               tval, 8e-9 * uc->mt_basis_size() * n__ * kp__->num_gkvec() / tval / kp__->num_ranks());
    }

    splindex<block> sub_spl_col(beta_phi.spl_col().local_size(), kp__->num_ranks_row(), kp__->rank_row());
    mdarray<double_complex, 2> beta_phi_slice(uc->mt_basis_size(), sub_spl_col.local_size());
    mdarray<double_complex, 2> tmp_slice(uc->mt_basis_size(), sub_spl_col.local_size());

    /* gather slices of full vectors from distributed matrix */
    beta_phi.gather(beta_phi_slice, parameters_.mpi_grid().communicator(1 << _dim_row_));
    
    if (sub_spl_col.local_size() != 0)
    {
        /* compute D*<beta|phi> */
        for (int ia = 0; ia < uc->num_atoms(); ia++)
        {   
            int ofs = uc->atom(ia)->offset_lo();
            /* number of beta functions for a given atom */
            int nbf = uc->atom(ia)->mt_basis_size();
            blas<cpu>::gemm(0, 0, nbf, (int)sub_spl_col.local_size(), nbf, &uc->atom(ia)->d_mtrx(0, 0), nbf, 
                            &beta_phi_slice(ofs, 0), beta_phi_slice.ld(), &tmp_slice(ofs, 0), tmp_slice.ld());
        }
    }
    tmp.scatter(tmp_slice, parameters_.mpi_grid().communicator(1 << _dim_row_));

    Timer t3("sirius::Band::apply_h_o_uspp_cpu_parallel|beta_D_beta_phi");
    /* compute <G+k|beta> * D*<beta|phi> and add to hphi */
    blas<cpu>::gemm(0, 0, kp__->num_gkvec(), n__, uc->mt_basis_size(), complex_one,
                    kp__->beta_pw_panel(), 0, 0, tmp, 0, 0, complex_one, hphi__, 0, N__);
    tval = t3.stop();
     
    //if (verbosity_level >= 6 && parameters_.mpi_grid().root(1 << _dim_row_ | 1 << _dim_col_))
    //{
    //    printf("pzgemm #2 with M, N, K: %i %i %i, offset in C: %i, performance (GFlop/s/node): %f\n",
    //           kp__->num_gkvec(), n__, uc->mt_basis_size(), N__,
    //           8e-9 * kp__->num_gkvec() * n__ * uc->mt_basis_size() / tval / kp__->num_ranks());
    //}

    if (sub_spl_col.local_size() != 0)
    {
        /* compute Q*<beta|phi> */
        for (int ia = 0; ia < uc->num_atoms(); ia++)
        {   
            int ofs = uc->atom(ia)->offset_lo();
            /* number of beta functions for a given atom */
            int nbf = uc->atom(ia)->mt_basis_size();
            blas<cpu>::gemm(0, 0, nbf, (int)sub_spl_col.local_size(), nbf, &uc->atom(ia)->type()->uspp().q_mtrx(0, 0), nbf, 
                            &beta_phi_slice(ofs, 0), beta_phi_slice.ld(), &tmp_slice(ofs, 0), tmp_slice.ld());
        }
    }
    tmp.scatter(tmp_slice, parameters_.mpi_grid().communicator(1 << _dim_row_));

    Timer t5("sirius::Band::apply_h_o_uspp_cpu_parallel|beta_Q_beta_phi");
    /* computr <G+k|beta> * Q*<beta|phi> and add to ophi */
    blas<cpu>::gemm(0, 0, kp__->num_gkvec(), n__, uc->mt_basis_size(), complex_one, 
                    kp__->beta_pw_panel(), 0, 0, tmp, 0, 0, complex_one, ophi__, 0, N__);
    tval += t5.stop();

    if (verbosity_level >= 6 && parameters_.mpi_grid().root(1 << _dim_row_ | 1 << _dim_col_))
    {
        printf("pzgemm #2&3 with M, N, K: %6i %6i %6i,   offset in C: %6i, %12.4f sec, %12.4f GFlops/node\n",
               kp__->num_gkvec(), n__, uc->mt_basis_size(), N__,
               tval, 2 * 8e-9 * kp__->num_gkvec() * n__ * uc->mt_basis_size() / tval / kp__->num_ranks());
    }

    log_function_exit(__func__);
}

void Band::apply_h_o_uspp_cpu_parallel_v2(K_point* kp__,
                                       std::vector<double>& effective_potential__,
                                       std::vector<double>& pw_ekin__,
                                       int N__,
                                       int n__,
                                       dmatrix<double_complex>& phi__,
                                       dmatrix<double_complex>& hphi__,
                                       dmatrix<double_complex>& ophi__)
{
    Timer t("sirius::Band::apply_h_o_uspp_cpu_parallel", _global_timer_);
    log_function_enter(__func__);
    auto uc = parameters_.unit_cell();

    splindex<block_cyclic> s0(N__,       kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());
    splindex<block_cyclic> s1(N__ + n__, kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());

    int nloc = static_cast<int>(s1.local_size() - s0.local_size());

    if (nloc == 0) return;

    /* apply local part of Hamiltonian */
    apply_h_local_parallel(kp__, effective_potential__, pw_ekin__, N__, n__, phi__, hphi__);
    
    /* set intial ophi */
    memcpy(&ophi__(0, s0.local_size()), &phi__(0, s0.local_size()), kp__->num_gkvec_row() * nloc * sizeof(double_complex));



    int num_atoms_in_block = 32;
    int num_atom_blocks = parameters_.unit_cell()->num_atoms() / num_atoms_in_block + 
                          std::min(1, parameters_.unit_cell()->num_atoms() % num_atoms_in_block);

    if (verbosity_level >= 6 && parameters_.mpi_grid().root(1 << _dim_row_ | 1 << _dim_col_))
    {
        printf("num_atom_blocks : %i\n", num_atom_blocks);
    }

    splindex<block> atom_blocks(parameters_.unit_cell()->num_atoms(), num_atom_blocks, 0);
    
    auto& beta_pw_t = kp__->beta_pw_t();

    for (int iab = 0; iab < num_atom_blocks; iab++)
    {
        std::vector<int> bf_offset_in_block(atom_blocks.local_size(iab));
        int nbf_in_block = 0;

        for (int i = 0; i < (int)atom_blocks.local_size(iab); i++)
        {
            int ia = (int)atom_blocks.global_offset(iab) + i;
            bf_offset_in_block[i] = nbf_in_block;
            nbf_in_block += parameters_.unit_cell()->atom(ia)->mt_basis_size();
        }

        Timer t1("sirius::Band::apply_h_o_uspp_cpu_parallel|beta_phi", _global_timer_);

        mdarray<double_complex, 2> beta_phi(nbf_in_block, nloc);
        mdarray<double_complex, 2> tmp(nbf_in_block, nloc);

        // create beta projectors
        mdarray<double_complex, 2> beta_pw(kp__->num_gkvec_row(), nbf_in_block);
        for (int i = 0, n = 0; i < (int)atom_blocks.local_size(iab); i++)
        {
            int ia = (int)atom_blocks.global_offset(iab) + i;
            auto type = parameters_.unit_cell()->atom(ia)->type();
            for (int xi = 0; xi < type->mt_basis_size(); xi++)
            {
                for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
                {
                    beta_pw(igk_row, n) = beta_pw_t(igk_row, type->offset_lo() + xi) * 
                                          conj(kp__->gkvec_phase_factor(igk_row, ia));
                }
                n++;
            }
        }

        /* compute <beta|phi> */
        blas<cpu>::gemm(2, 0, nbf_in_block, nloc, kp__->num_gkvec_row(), 
                        beta_pw.ptr(), beta_pw.ld(), &phi__(0, s0.local_size()), phi__.ld(), beta_phi.ptr(), beta_phi.ld());
        Platform::allreduce(beta_phi.ptr(), (int)beta_phi.size(), parameters_.mpi_grid().communicator(1 << _dim_row_));
        t1.stop();


        for (int i = 0; i < (int)atom_blocks.local_size(iab); i++)
        {
            int ia = (int)atom_blocks.global_offset(iab) + i;
            int ofs = bf_offset_in_block[i];
            
            /* number of beta functions for a given atom */
            int nbf = uc->atom(ia)->mt_basis_size();

            /* compute D*<beta|phi> */
            blas<cpu>::gemm(0, 0, nbf, nloc, nbf, &uc->atom(ia)->d_mtrx(0, 0), nbf, 
                            &beta_phi(ofs, 0), beta_phi.ld(), &tmp(ofs, 0), tmp.ld());
        }

        Timer t3("sirius::Band::apply_h_o_uspp_cpu_parallel|beta_D_beta_phi", _global_timer_);
        /* compute <G+k|beta> * D*<beta|phi> and add to hphi */
        blas<cpu>::gemm(0, 0, kp__->num_gkvec_row(), nloc, nbf_in_block, complex_one,
                        beta_pw.ptr(), beta_pw.ld(), tmp.ptr(), tmp.ld(), complex_one, &hphi__(0, s0.local_size()), hphi__.ld());
        t3.stop();



        for (int i = 0; i < (int)atom_blocks.local_size(iab); i++)
        {
            int ia = (int)atom_blocks.global_offset(iab) + i;
            int ofs = bf_offset_in_block[i];
            
            /* number of beta functions for a given atom */
            int nbf = uc->atom(ia)->mt_basis_size();

            /* compute Q*<beta|phi> */
            blas<cpu>::gemm(0, 0, nbf, nloc, nbf, &uc->atom(ia)->type()->uspp().q_mtrx(0, 0), nbf, 
                            &beta_phi(ofs, 0), beta_phi.ld(), &tmp(ofs, 0), tmp.ld());
        }

        Timer t4("sirius::Band::apply_h_o_uspp_cpu_parallel|beta_Q_beta_phi", _global_timer_);
        /* compute <G+k|beta> * Q*<beta|phi> and add to hphi */
        blas<cpu>::gemm(0, 0, kp__->num_gkvec_row(), nloc, nbf_in_block, complex_one,
                        beta_pw.ptr(), beta_pw.ld(), tmp.ptr(), tmp.ld(), complex_one, &ophi__(0, s0.local_size()), ophi__.ld());
        t4.stop();
    }

    log_function_exit(__func__);
}

void Band::set_fv_h_o_uspp_cpu_parallel(int N__,
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
    Timer t1("sirius::Band::set_fv_h_o_uspp_cpu_parallel", _global_timer_);

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
    apply_h_o_uspp_cpu_parallel_v2(kp__, veff_it_coarse__, pw_ekin__, N__, n__, phi__, hphi__, ophi__);
    
    Timer t2("sirius::Band::set_fv_h_o_uspp_cpu_parallel|zgemm", _global_timer_);
    /* <{phi,res}|H|res> */
    blas<cpu>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec(), complex_one, phi__, 0, 0, hphi__, 0, N__, complex_zero, h__, 0, N__);
    /* <{phi,res}|O|res> */
    blas<cpu>::gemm(2, 0, N__ + n__, n__, kp__->num_gkvec(), complex_one, phi__, 0, 0, ophi__, 0, N__, complex_zero, o__, 0, N__);
    double tval = t2.stop();

    if (verbosity_level >= 6 && parameters_.mpi_grid().root(1 << _dim_row_ | 1 << _dim_col_))
    {
        printf("pzgemm #4&5 with M, N, K: %6i %6i %6i, offset in B&C: %6i, %12.4f sec, %12.4f GFlops/node\n",
               N__ + n__, n__, kp__->num_gkvec(), N__,
               tval, 2 * 8e-9 * (N__ + n__) * n__ * kp__->num_gkvec() / tval / kp__->num_ranks());
    }
    
    /* restore the bottom block of the matrix */
    if (N__ != 0)
    {
        dmatrix<double_complex>::tranc(n__, N__, h__, 0, N__, h__, N__, 0);
        dmatrix<double_complex>::tranc(n__, N__, o__, 0, N__, o__, N__, 0);
    }

    /* save Hamiltonian and overlap */
    for (int i = 0; i < (int)s1_col.local_size(); i++)
    {
        memcpy(&h_old__(0, i), &h__(0, i), s1_row.local_size() * sizeof(double_complex));
        memcpy(&o_old__(0, i), &o__(0, i), s1_row.local_size() * sizeof(double_complex));
    }
}

void Band::uspp_cpu_residuals_parallel(int N__,
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
                                       std::vector<double>& res_norm__)

{
    Timer t("sirius::Band::uspp_cpu_residuals_parallel");
    
    Timer t2("sirius::Band::uspp_cpu_residuals_parallel|zgemm");
    /* Compute H\Psi_{i} = H\phi_{mu} * Z_{mu, i} */
    blas<cpu>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, complex_one, hphi__, evec__, complex_zero, hpsi__);
    /* Compute O\Psi_{i} = O\phi_{mu} * Z_{mu, i} */
    blas<cpu>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, complex_one, ophi__, evec__, complex_zero, opsi__);
    double tval = t2.stop();

    if (verbosity_level >= 6 && parameters_.mpi_grid().root(1 << _dim_row_ | 1 << _dim_col_))
    {
        printf("pzgemm #6&7 with M, N, K: %6i %6i %6i,                        %12.4f sec, %12.4f GFlops/node\n",
               kp__->num_gkvec(), num_bands__, N__,
               tval, 2 * 8e-9 * kp__->num_gkvec() * num_bands__ * N__ / tval / kp__->num_ranks());
    }

    memset(&res_norm__[0], 0, num_bands__ * sizeof(double));
    /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} and norm suared*/
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
    Platform::allreduce(&res_norm__[0], num_bands__, parameters_.mpi_grid().communicator(1 << _dim_row_ | 1 << _dim_col_));
    
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
    Platform::allreduce(&norm2[0], num_bands__, parameters_.mpi_grid().communicator(1 << _dim_row_ | 1 << _dim_col_));
    #pragma omp parallel for
    for (int i = 0; i < res__.num_cols_local(); i++)
    {
        int ires = res__.icol(i);
        double d = 1.0 / std::sqrt(norm2[ires]);
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) res__(igk_row, i) *= d;
    }
}

void Band::diag_fv_uspp_cpu_parallel(K_point* kp__,
                                     double v0__,
                                     std::vector<double>& veff_it_coarse__)
{
    log_function_enter(__func__);
    Timer t("sirius::Band::diag_fv_uspp_cpu_parallel", _global_timer_);

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

    dmatrix<double_complex> phi(kp__->num_gkvec(), num_phi, parameters_.blacs_context());
    dmatrix<double_complex> hphi(kp__->num_gkvec(), num_phi, parameters_.blacs_context());
    dmatrix<double_complex> ophi(kp__->num_gkvec(), num_phi, parameters_.blacs_context());

    /* current diagonalziation subspace size */
    int N = 0;

    /* number of newly added basis functions */
    int n = num_bands;

    dmatrix<double_complex> hmlt(num_phi, num_phi, parameters_.blacs_context());
    dmatrix<double_complex> ovlp(num_phi, num_phi, parameters_.blacs_context());
    dmatrix<double_complex> hmlt_old(num_phi, num_phi, parameters_.blacs_context());
    dmatrix<double_complex> ovlp_old(num_phi, num_phi, parameters_.blacs_context());
    
    dmatrix<double_complex> evec(num_phi, num_bands, parameters_.blacs_context());
    std::vector<double> eval(num_bands);
    std::vector<double> eval_old(num_bands);

    /* alias for wave-functions */
    dmatrix<double_complex>& psi = kp__->fv_states_panel();
    
    dmatrix<double_complex> hpsi(kp__->num_gkvec(), num_bands, parameters_.blacs_context());
    dmatrix<double_complex> opsi(kp__->num_gkvec(), num_bands, parameters_.blacs_context());
    dmatrix<double_complex> res(kp__->num_gkvec(), num_bands, parameters_.blacs_context());

    /* trial basis functions */
    assert(phi.num_rows_local() == psi.num_rows_local());
    for (int i = 0; i < psi.num_cols_local(); i++) memcpy(&phi(0, i), &psi(0, i), kp__->num_gkvec_row() * sizeof(double_complex));

    std::vector<double> res_norm(num_bands);
    std::vector<double> res_rms(num_bands);

    /* start iterative diagonalization */
    for (int k = 0; k < itso.num_steps_; k++)
    {
        /* set H and O for the variational subspace */
        set_fv_h_o_uspp_cpu_parallel(N, n, kp__, veff_it_coarse__, pw_ekin, phi, hphi, ophi, hmlt, ovlp, hmlt_old, ovlp_old);
        
        /* increase size of the variation space */
        N += n;

        {
        Timer t2("sirius::Band::diag_fv_uspp_cpu_parallel|solve_gevp");
        eval_old = eval;
        parameters_.gen_evp_solver()->solve(N, hmlt.num_rows_local(), hmlt.num_cols_local(), num_bands, 
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
            Timer t3("sirius::Band::diag_fv_uspp_cpu_parallel|update_phi");

            /* recompute wave-functions: \Psi_{i} = \phi_{mu} * Z_{mu, i} */
            blas<cpu>::gemm(0, 0, kp__->num_gkvec(), num_bands, N, complex_one, phi, evec, complex_zero, psi); 
            
            /* exit loop if the eigen-vectors are converged or this is the last iteration */
            if (n == 0 || k == (itso.num_steps_ - 1))
            {
                if (verbosity_level >= 6 && parameters_.mpi_grid().root(1 << _dim_row_ | 1 << _dim_col_))
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
        
        /* Expand variational space with extra basis functions */
        for (int i = 0; i < n; i++)
        {
            dmatrix<double_complex>::copy_col(res, res_list[i], phi, N + i, parameters_.mpi_grid().communicator(1 << _dim_col_));
        }
    }

    kp__->set_fv_eigen_values(&eval[0]);
}

void Band::diag_fv_uspp_cpu_serial_v0(K_point* kp__,
                                      std::vector<double>& veff_it_coarse__)
{
    /* cache kinetic energy */
    std::vector<double> pw_ekin = kp__->get_pw_ekin();

    /* short notation for target wave-functions */
    mdarray<double_complex, 2>& psi = kp__->fv_states();

    /* short notation for number of target wave-functions */
    int num_bands = parameters_.num_fv_states();     

    int ngk = kp__->num_gkvec();

    mdarray<double_complex, 2> phi(ngk, ngk);
    mdarray<double_complex, 2> hphi(ngk, ngk);
    mdarray<double_complex, 2> ophi(ngk, ngk);
    
    std::vector<double> eval(ngk);

    phi.zero();
    for (int i = 0; i < ngk; i++) phi(i, i) = complex_one;
    
    apply_h_o_uspp_cpu(kp__, veff_it_coarse__, pw_ekin, ngk, &phi(0, 0), &hphi(0, 0), &ophi(0, 0));
        
    parameters_.gen_evp_solver()->solve(ngk, num_bands, num_bands, num_bands, hphi.ptr(), hphi.ld(), ophi.ptr(), ophi.ld(), 
                                        &eval[0], psi.ptr(), psi.ld());

    kp__->set_fv_eigen_values(&eval[0]);
    kp__->fv_states_panel().scatter(psi, parameters_.mpi_grid().communicator(1 << _dim_row_));
}


void Band::diag_fv_uspp_cpu_serial_v1(K_point* kp__,
                                      double v0__,
                                      std::vector<double>& veff_it_coarse__)
{
    /* cache kinetic energy */
    std::vector<double> pw_ekin = kp__->get_pw_ekin();

    /* short notation for target wave-functions */
    mdarray<double_complex, 2>& psi = kp__->fv_states();

    /* short notation for number of target wave-functions */
    int num_bands = parameters_.num_fv_states();     

    /* get diagonal elements for preconditioning */
    std::vector<double_complex> h_diag;
    std::vector<double_complex> o_diag;
    get_h_o_diag(kp__, v0__, pw_ekin, h_diag, o_diag);

    auto& itso = parameters_.iterative_solver_input_section_;
    
    int num_phi = std::min(itso.subspace_size_ * num_bands, kp__->num_gkvec());

    mdarray<double_complex, 2> phi(kp__->num_gkvec(), num_phi);
    mdarray<double_complex, 2> hphi(kp__->num_gkvec(), num_phi);
    mdarray<double_complex, 2> ophi(kp__->num_gkvec(), num_phi);
    mdarray<double_complex, 2> hpsi(kp__->num_gkvec(), num_bands);
    mdarray<double_complex, 2> opsi(kp__->num_gkvec(), num_bands);

    mdarray<double_complex, 2> hmlt(num_phi, num_phi);
    mdarray<double_complex, 2> ovlp(num_phi, num_phi);
    mdarray<double_complex, 2> hmlt_old(num_phi, num_phi);
    mdarray<double_complex, 2> ovlp_old(num_phi, num_phi);
    mdarray<double_complex, 2> evec(num_phi, num_bands);

    std::vector<double> eval(num_bands);
    
    mdarray<double_complex, 2> res(kp__->num_gkvec(), num_bands); // residuals

    std::vector<double> res_norm(num_bands); // norm of residuals

    int N = 0; // current eigen-value problem size
    int n = num_bands; // number of added residuals

    // trial basis functions
    assert(phi.size(0) == psi.size(0));
    for (int i = 0; i < num_bands; i++) memcpy(&phi(0, i), &psi(0, i), kp__->num_gkvec() * sizeof(double_complex));

    // start iterative diagonalization
    for (int k = 0; k < itso.num_steps_; k++)
    {
        {
            Timer t1("sirius::Band::diag_fv_uspp_cpu|set_gevp");

            // copy old Hamiltonian and overlap
            for (int i = 0; i < N; i++)
            {
                memcpy(&hmlt(0, i), &hmlt_old(0, i), N * sizeof(double_complex));
                memcpy(&ovlp(0, i), &ovlp_old(0, i), N * sizeof(double_complex));
            }

            // apply Hamiltonian and overlap operators to the new basis functions
            apply_h_o_uspp_cpu(kp__, veff_it_coarse__, pw_ekin, n, &phi(0, N), &hphi(0, N), &ophi(0, N));
            
            // <{phi,res}|H|res>
            blas<cpu>::gemm(2, 0, N + n, n, kp__->num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, N), hphi.ld(), &hmlt(0, N), hmlt.ld());
            
            // <{phi,res}|O|res>
            blas<cpu>::gemm(2, 0, N + n, n, kp__->num_gkvec(), &phi(0, 0), phi.ld(), &ophi(0, N), ophi.ld(), &ovlp(0, N), ovlp.ld());
            
            // increase the size of the variation space
            N += n;
            
            // save Hamiltonian and overlap
            for (int i = 0; i < N; i++)
            {
                memcpy(&hmlt_old(0, i), &hmlt(0, i), N * sizeof(double_complex));
                memcpy(&ovlp_old(0, i), &ovlp(0, i), N * sizeof(double_complex));
            }
        }
        
        {
            Timer t2("sirius::Band::diag_fv_uspp_cpu|solve_gevp");
            parameters_.gen_evp_solver()->solve(N, num_bands, num_bands, num_bands, hmlt.ptr(), hmlt.ld(), ovlp.ptr(), ovlp.ld(), 
                                                &eval[0], evec.ptr(), evec.ld());
        }
        
        {
            Timer t3("sirius::Band::diag_fv_uspp_cpu|residuals");
            /* compute H\Psi_{i} = H\phi_{mu} * Z_{mu, i} */
            blas<cpu>::gemm(0, 0, kp__->num_gkvec(), num_bands, N, &hphi(0, 0), hphi.ld(), &evec(0, 0), evec.ld(), 
                            &hpsi(0, 0), hpsi.ld());

            /* compute O\Psi_{i} = O\phi_{mu} * Z_{mu, i} */
            blas<cpu>::gemm(0, 0, kp__->num_gkvec(), num_bands, N, &ophi(0, 0), ophi.ld(), &evec(0, 0), evec.ld(), 
                            &opsi(0, 0), opsi.ld());

            /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} */
            for (int i = 0; i < num_bands; i++)
            {
                for (int igk = 0; igk < kp__->num_gkvec(); igk++) res(igk, i) = hpsi(igk, i) - eval[i] * opsi(igk, i);
            }

            /* compute norm and apply preconditioner */
            #pragma omp parallel for
            for (int i = 0; i < num_bands; i++)
            {
                double r = 0;
                for (int igk = 0; igk < kp__->num_gkvec(); igk++) r += real(conj(res(igk, i)) * res(igk, i));
                res_norm[i] = std::sqrt(r);
                
                /* apply preconditioner */
                for (int igk = 0; igk < kp__->num_gkvec(); igk++)
                {
                    double_complex z = h_diag[igk] - eval[i] * o_diag[igk];
                    if (std::abs(z) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
                    res(igk, i) /= z;
                }
            }

            n = 0;
            for (int i = 0; i < num_bands; i++)
            {
                /* take the residual if it's norm is above the threshold */
                if (kp__->band_occupancy(i) > 1e-12 &&
                    (res_norm[i] > itso.tolerance_ || (res_norm[i] > itso.extra_tolerance_ && n != 0)))
                {
                    /* shift unconverged residuals to the beginning of array */
                    if (n != i) memcpy(&res(0, n), &res(0, i), kp__->num_gkvec() * sizeof(double_complex));
                    n++;
                }
            }

            //std::cout << "Iteration: " << k << ", number of residuals : " << n << std::endl;
            //for (int i = 0; i < std::min(10, num_bands); i++) std::cout << "eval["<<i<<"] = " << eval[i] << std::endl;
        }

        /* check if we run out of variational space or eigen-vectors are converged or it's a last iteration */
        if (N + n > num_phi || n == 0 || k == (itso.num_steps_ - 1))
        {   
            Timer t3("sirius::Band::diag_fv_uspp_cpu|update_phi");
            /* \Psi_{i} = \phi_{mu} * Z_{mu, i} */
            blas<cpu>::gemm(0, 0, kp__->num_gkvec(), num_bands, N, &phi(0, 0), phi.ld(), &evec(0, 0), evec.ld(), 
                            &psi(0, 0), psi.ld());

            if (n == 0 || k == (itso.num_steps_ - 1)) // exit the loop if the eigen-vectors are converged or it's a last iteration
            {
                std::cout << "converged in " << k << " iterations" << std::endl;
                break;
            }
            else // otherwise set Psi as a new trial basis
            {
                hmlt_old.zero();
                ovlp_old.zero();
                for (int i = 0; i < num_bands; i++)
                {
                    hmlt_old(i, i) = eval[i];
                    ovlp_old(i, i) = complex_one;
                }
 
                /* set new basis functions */
                memcpy(hphi.ptr(), hpsi.ptr(), num_bands * kp__->num_gkvec() * sizeof(double_complex));
                memcpy(ophi.ptr(), opsi.ptr(), num_bands * kp__->num_gkvec() * sizeof(double_complex));
                memcpy(phi.ptr(), psi.ptr(), num_bands * kp__->num_gkvec() * sizeof(double_complex));
                N = num_bands;
            }
        }
        /* expand variational subspace with new basis vectors obtatined from residuals */
        memcpy(&phi(0, N), &res(0, 0), n * kp__->num_gkvec() * sizeof(double_complex));
    }

    kp__->set_fv_eigen_values(&eval[0]);
    kp__->fv_states_panel().scatter(psi, parameters_.mpi_grid().communicator(1 << _dim_row_));
}

void Band::diag_fv_uspp_cpu_serial_v2(K_point* kp__,
                                      double v0__,
                                      std::vector<double>& veff_it_coarse__)
{
    /* cache kinetic energy */
    std::vector<double> pw_ekin = kp__->get_pw_ekin();

    /* short notation for target wave-functions */
    mdarray<double_complex, 2>& psi = kp__->fv_states();

    /* short notation for number of target wave-functions */
    int num_bands = parameters_.num_fv_states();     

    auto& itso = parameters_.iterative_solver_input_section_;

    /* get diagonal elements for preconditioning */
    std::vector<double_complex> h_diag;
    std::vector<double_complex> o_diag;
    get_h_o_diag(kp__, v0__, pw_ekin, h_diag, o_diag);
    
    mdarray<double_complex, 2> phi(kp__->num_gkvec(), 2 * num_bands);
    mdarray<double_complex, 2> hphi(kp__->num_gkvec(), 2 * num_bands);
    mdarray<double_complex, 2> ophi(kp__->num_gkvec(), 2 * num_bands);

    mdarray<double_complex, 2> hpsi(kp__->num_gkvec(), num_bands);
    mdarray<double_complex, 2> opsi(kp__->num_gkvec(), num_bands);

    mdarray<double_complex, 2> hmlt(2 * num_bands, 2 * num_bands);
    mdarray<double_complex, 2> ovlp(2 * num_bands, 2 * num_bands);
    mdarray<double_complex, 2> evec(2 * num_bands, num_bands);

    std::vector<double> eval(num_bands);
    
    mdarray<double_complex, 2> res(kp__->num_gkvec(), num_bands); // residuals

    std::vector<double> res_norm(num_bands); // norm of residuals

    int N = 0; // current eigen-value problem size
    int n = num_bands; // number of added residuals

    // trial basis functions
    assert(phi.size(0) == psi.size(0));
    for (int i = 0; i < num_bands; i++) memcpy(&phi(0, i), &psi(0, i), kp__->num_gkvec() * sizeof(double_complex));

    // start iterative diagonalization
    for (int k = 0; k < itso.num_steps_; k++)
    {
        {
            Timer t1("sirius::Band::diag_fv_uspp_cpu|set_gevp");

            if (N != 0)
            {
                hmlt.zero();
                ovlp.zero();
                for (int i = 0; i < N; i++)
                {
                    hmlt(i, i) = eval[i];
                    ovlp(i, i) = complex_one;
                }
            }

            // apply Hamiltonian and overlap operators to the new basis functions
            apply_h_o_uspp_cpu(kp__, veff_it_coarse__, pw_ekin, n, &phi(0, N), &hphi(0, N), &ophi(0, N));
            
            // <{phi,res}|H|res>
            blas<cpu>::gemm(2, 0, N + n, n, kp__->num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, N), hphi.ld(), &hmlt(0, N), hmlt.ld());
            
            // <{phi,res}|O|res>
            blas<cpu>::gemm(2, 0, N + n, n, kp__->num_gkvec(), &phi(0, 0), phi.ld(), &ophi(0, N), ophi.ld(), &ovlp(0, N), ovlp.ld());
            
            // increase the size of the variation space
            N += n;
        }
        
        {
            Timer t2("sirius::Band::diag_fv_uspp_cpu|solve_gevp");
            parameters_.gen_evp_solver()->solve(N, num_bands, num_bands, num_bands, hmlt.ptr(), hmlt.ld(), ovlp.ptr(), ovlp.ld(), 
                                                &eval[0], evec.ptr(), evec.ld());
        }

        Timer t3("sirius::Band::diag_fv_uspp_cpu|residuals");

        blas<cpu>::gemm(0, 0, kp__->num_gkvec(), num_bands, N, &hphi(0, 0), hphi.ld(), &evec(0, 0), evec.ld(), 
                        &hpsi(0, 0), hpsi.ld());

        blas<cpu>::gemm(0, 0, kp__->num_gkvec(), num_bands, N, &ophi(0, 0), ophi.ld(), &evec(0, 0), evec.ld(), 
                        &opsi(0, 0), opsi.ld());

        for (int i = 0; i < num_bands; i++)
        {
            for (int igk = 0; igk < kp__->num_gkvec(); igk++) res(igk, i) = hpsi(igk, i) - eval[i] * opsi(igk, i);
        }

        // compute norm and apply preconditioner
        #pragma omp parallel for
        for (int i = 0; i < num_bands; i++)
        {
            double r = 0;
            for (int igk = 0; igk < kp__->num_gkvec(); igk++) r += real(conj(res(igk, i)) * res(igk, i));
            res_norm[i] = std::sqrt(r);
            
            // apply preconditioner
            for (int igk = 0; igk < kp__->num_gkvec(); igk++)
            {
                double_complex z = h_diag[igk] - eval[i] * o_diag[igk];
                if (abs(z) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
                res(igk, i) /= z;
            }
        }
        n = 0;
        for (int i = 0; i < num_bands; i++)
        {
            /* take the residual if it's norm is above the threshold */
            if (kp__->band_occupancy(i) > 1e-12 &&
                (res_norm[i] > itso.tolerance_ || (res_norm[i] > itso.extra_tolerance_ && n != 0)))
            {
                /* shift unconverged residuals to the beginning of array */
                if (n != i) memcpy(&res(0, n), &res(0, i), kp__->num_gkvec() * sizeof(double_complex));
                n++;
            }
        }

        //std::cout << "Iteration: " << k << ", number of residuals : " << n << std::endl;
        //for (int i = 0; i < std::min(10, num_bands); i++) std::cout << "eval["<<i<<"] = " << eval[i] << std::endl;

        /* recompute wave-functions: \Psi_{i} = \phi_{mu} * Z_{mu, i} */
        blas<cpu>::gemm(0, 0, kp__->num_gkvec(), num_bands, N, &phi(0, 0), phi.ld(), &evec(0, 0), evec.ld(), 
                        &psi(0, 0), psi.ld());
            
        /* exit loop if the eigen-vectors are converged or this is the last iteration */
        if (n == 0 || k == itso.num_steps_ - 1)
        {
            std::cout << "converged in " << k << " iterations" << std::endl;
            break;
        }

        for (int i = 0; i < num_bands; i++)
        {
            /* update \phi */
            memcpy(&phi(0, i), &psi(0, i), kp__->num_gkvec() * sizeof(double_complex));
            /* update H\phi */
            memcpy(&hphi(0, i), &hpsi(0, i), kp__->num_gkvec() * sizeof(double_complex));
            /* update O\phi */
            memcpy(&ophi(0, i), &opsi(0, i), kp__->num_gkvec() * sizeof(double_complex));
        }
            
        /* set new size of the variational space */
        N = num_bands;

        /* expand variational subspace with new basis vectors obtatined from residuals */
        memcpy(&phi(0, N), &res(0, 0), n * kp__->num_gkvec() * sizeof(double_complex));
    }

    kp__->set_fv_eigen_values(&eval[0]);
    kp__->fv_states_panel().scatter(psi, parameters_.mpi_grid().communicator(1 << _dim_row_));
}

void Band::diag_fv_uspp_cpu_serial_v3(K_point* kp__,
                                      double v0__,
                                      std::vector<double>& veff_it_coarse__)
{
    /* cache kinetic energy */
    std::vector<double> pw_ekin = kp__->get_pw_ekin();

    /* short notation for target wave-functions */
    mdarray<double_complex, 2>& psi = kp__->fv_states();

    /* short notation for number of target wave-functions */
    int num_bands = parameters_.num_fv_states();     

    auto& itso = parameters_.iterative_solver_input_section_;

    /* get diagonal elements for preconditioning */
    std::vector<double_complex> h_diag;
    std::vector<double_complex> o_diag;
    get_h_o_diag(kp__, v0__, pw_ekin, h_diag, o_diag);
    
    mdarray<double_complex, 2> phi(kp__->num_gkvec(), 3 * num_bands);
    mdarray<double_complex, 2> hphi(kp__->num_gkvec(), 3 * num_bands);
    mdarray<double_complex, 2> ophi(kp__->num_gkvec(), 3 * num_bands);

    mdarray<double_complex, 2> hpsi(kp__->num_gkvec(), num_bands);
    mdarray<double_complex, 2> opsi(kp__->num_gkvec(), num_bands);

    mdarray<double_complex, 2> hmlt(3 * num_bands, 3 * num_bands);
    mdarray<double_complex, 2> ovlp(3 * num_bands, 3 * num_bands);
    mdarray<double_complex, 2> evec(3 * num_bands, num_bands);

    std::vector<double> eval(num_bands);
    std::vector<double> eval_tmp(num_bands);
    
    //mdarray<double_complex, 2> res(kp__->num_gkvec(), num_bands); // residuals

    std::vector<double> res_norm(num_bands); // norm of residuals

    //int N = 0; // current eigen-value problem size
    //int n = num_bands; // number of added residuals

    // trial basis functions
    assert(phi.size(0) == psi.size(0));
    for (int i = 0; i < num_bands; i++) memcpy(&phi(0, i), &psi(0, i), kp__->num_gkvec() * sizeof(double_complex));

    // start iterative diagonalization
    for (int k = 0; k < itso.num_steps_; k++)
    {
        // apply Hamiltonian and overlap operators to the new basis functions
        apply_h_o_uspp_cpu(kp__, veff_it_coarse__, pw_ekin, num_bands, &phi(0, 0), &hphi(0, 0), &ophi(0, 0));


        for (int i = 0; i < num_bands; i++)
        {
            double a(0), b(0);
            for (int igk = 0; igk < kp__->num_gkvec(); igk++)
            {
                a += real(conj(phi(igk, i)) * hphi(igk, i));
                b += real(conj(phi(igk, i)) * ophi(igk, i));
            }
            eval[i] = a / b;
        }

        for (int i = 0; i < num_bands; i++)
        {
            double r = 0;
            for (int igk = 0; igk < kp__->num_gkvec(); igk++) 
            {
                phi(igk, num_bands + i) = hphi(igk, i) - eval[i] * ophi(igk, i);
                r += std::pow(std::abs(phi(igk, num_bands + i)), 2);

                double_complex z = h_diag[igk] - eval[i] * o_diag[igk];
                if (std::abs(z) > 1e-10) phi(igk, num_bands + i) /= z;
            }
            res_norm[i] = std::sqrt(r);
        }

        int m = (k == 0) ? num_bands : 2 * num_bands;
        
        apply_h_o_uspp_cpu(kp__, veff_it_coarse__, pw_ekin, m, &phi(0, num_bands), &hphi(0, num_bands), &ophi(0, num_bands));

        
//        mdarray<double_complex, 2> zm(num_bands, m);
//
//        blas<cpu>::gemm(2, 0, num_bands, m, kp__->num_gkvec(), &phi(0, 0), phi.ld(), &ophi(0, num_bands), ophi.ld(), 
//                        &zm(0, 0), zm.ld());
//
//        blas<cpu>::gemm(0, 0, kp__->num_gkvec(), m, num_bands, double_complex(-1, 0), &zm(0, 0), zm.ld(), &phi(0, 0), phi.ld(), 
//                        double_complex(1, 0), 
//
        Timer t1("sirius::Band::diag_fv_uspp_cpu_serial_v3:orth");
        int n = 0;
        for (int i = 0; i < m; i++)
        {
            //std::vector<double_complex> z(num_bands + n, complex_zero);
            //blas<cpu>::gemv(2, kp__->num_gkvec(), num_bands + n, complex_one, phi.ptr(), phi.ld(), 
            //                &ophi(0, num_bands + i), 1, complex_zero, &z[0], 1); 

            for (int j = 0; j < num_bands + n; j++)
            {
                double_complex z(0, 0);
                for (int igk = 0; igk < kp__->num_gkvec(); igk++) z += conj(phi(igk, j)) * ophi(igk, num_bands + i);
            
                for (int igk = 0; igk < kp__->num_gkvec(); igk++)
                {
                    phi(igk, num_bands + i) -= z * phi(igk, j);
                    ophi(igk, num_bands + i) -= z * ophi(igk, j);
                    hphi(igk, num_bands + i) -= z * hphi(igk, j);
                }
            }

            double norm = 0;
            for (int igk = 0; igk < kp__->num_gkvec(); igk++)
            {
                norm += real(conj(phi(igk, num_bands + i)) * ophi(igk, num_bands + i));
            }
            norm = std::sqrt(norm);
            if (norm > 1e-5)
            {
                norm = 1 / norm;
                for (int igk = 0; igk < kp__->num_gkvec(); igk++)
                {
                    phi(igk, num_bands + n) = phi(igk, num_bands + i) * norm;
                    ophi(igk, num_bands + n) = ophi(igk, num_bands + i) * norm;
                    hphi(igk, num_bands + n) = hphi(igk, num_bands + i) * norm;
                }
                n++;
            }
        }
        t1.stop();

        int N = num_bands + n;


        std::cout << "Iteration: " << k << ", subspace size : " << N << std::endl;
        //for (int i = 0; i < std::min(10, num_bands); i++) std::cout << "eval["<<i<<"] = " << eval[i] << std::endl;
        //for (int i = 0; i < std::min(10, num_bands); i++) std::cout << "res["<<i<<"] = " << res_norm[i] << std::endl;

        Timer t2("sirius::Band::diag_fv_uspp_cpu_serial_v3:gevp");

        blas<cpu>::gemm(2, 0, N, N, kp__->num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, 0), hphi.ld(), &hmlt(0, 0), hmlt.ld());
          
        blas<cpu>::gemm(2, 0, N, N, kp__->num_gkvec(), &phi(0, 0), phi.ld(), &ophi(0, 0), ophi.ld(), &ovlp(0, 0), ovlp.ld());

        //== {
        //==     mdarray<double_complex, 2> o1(3 * num_bands, 3 * num_bands);
        //==     mdarray<double_complex, 2> ev1(3 * num_bands, 3 * num_bands);
        //==     std::vector<double> e1(3 * num_bands);
        //==     ovlp >> o1;

        //==     parameters_.std_evp_solver()->solve(N, o1.ptr(), o1.ld(), &e1[0], ev1.ptr(), ev1.ld());

        //==     for (int i = 0; i < N; i++) std::cout << "eval_o["<<i<<"] = " << e1[i] << std::endl;
        //== }
        
        parameters_.gen_evp_solver()->solve(N, N, N, num_bands, hmlt.ptr(), hmlt.ld(), ovlp.ptr(), ovlp.ld(), 
                                            &eval_tmp[0], evec.ptr(), evec.ld());
        t2.stop();

        
        Timer t3("sirius::Band::diag_fv_uspp_cpu_serial_v3:update");
        blas<cpu>::gemm(0, 0, kp__->num_gkvec(), num_bands, N, &phi(0, 0), phi.ld(), 
                        &evec(0, 0), evec.ld(), &psi(0, 0), psi.ld());

        blas<cpu>::gemm(0, 0, kp__->num_gkvec(), num_bands, n, &phi(0, num_bands), phi.ld(), 
                        &evec(num_bands, 0), evec.ld(), &phi(0, 0), phi.ld());
        t3.stop();
        for (int j = 0; j < num_bands; j++)
        {
            memcpy(&phi(0, 2 * num_bands + j), &phi(0, j), kp__->num_gkvec() * sizeof(double_complex));
            memcpy(&phi(0, j), &psi(0, j), kp__->num_gkvec() * sizeof(double_complex));
        }
    }

    kp__->set_fv_eigen_values(&eval[0]);
    kp__->fv_states_panel().scatter(psi, parameters_.mpi_grid().communicator(1 << _dim_row_));
}

void Band::diag_fv_uspp_cpu_serial_v4(K_point* kp__,
                                      double v0__,
                                      std::vector<double>& veff_it_coarse__)
{
    /* cache kinetic energy */
    std::vector<double> pw_ekin = kp__->get_pw_ekin();

    /* short notation for target wave-functions */
    mdarray<double_complex, 2>& psi = kp__->fv_states();

    /* short notation for number of target wave-functions */
    int num_bands = parameters_.num_fv_states();     

    auto& itso = parameters_.iterative_solver_input_section_;

    /* get diagonal elements for preconditioning */
    std::vector<double_complex> h_diag;
    std::vector<double_complex> o_diag;
    get_h_o_diag(kp__, v0__, pw_ekin, h_diag, o_diag);
    
    mdarray<double_complex, 2> phi(kp__->num_gkvec(), 2 * num_bands);
    mdarray<double_complex, 2> hphi(kp__->num_gkvec(), 2 * num_bands);
    mdarray<double_complex, 2> ophi(kp__->num_gkvec(), 2 * num_bands);

    mdarray<double_complex, 2> hpsi(kp__->num_gkvec(), num_bands);
    mdarray<double_complex, 2> opsi(kp__->num_gkvec(), num_bands);

    mdarray<double_complex, 2> hmlt(2 * num_bands, 2 * num_bands);
    mdarray<double_complex, 2> ovlp(2 * num_bands, 2 * num_bands);
    mdarray<double_complex, 2> evec(2 * num_bands, num_bands);

    std::vector<double> eval(num_bands);
    std::vector<double> eval_tmp(num_bands);
    
    std::vector<double> res_norm(num_bands); // norm of residuals

    // trial basis functions
    assert(phi.size(0) == psi.size(0));
    for (int i = 0; i < num_bands; i++) memcpy(&phi(0, i), &psi(0, i), kp__->num_gkvec() * sizeof(double_complex));
    
    // start iterative diagonalization
    for (int k = 0; k < itso.num_steps_; k++)
    {
        // apply Hamiltonian and overlap operators to the new basis functions
        if (k == 0) apply_h_o_uspp_cpu(kp__, veff_it_coarse__, pw_ekin, num_bands, &phi(0, 0), &hphi(0, 0), &ophi(0, 0));


        for (int i = 0; i < num_bands; i++)
        {
            double a(0), b(0);
            for (int igk = 0; igk < kp__->num_gkvec(); igk++)
            {
                a += real(conj(phi(igk, i)) * hphi(igk, i));
                b += real(conj(phi(igk, i)) * ophi(igk, i));
            }
            eval[i] = a / b;
        }

        for (int i = 0; i < num_bands; i++)
        {
            double r = 0;
            for (int igk = 0; igk < kp__->num_gkvec(); igk++) 
            {
                phi(igk, num_bands + i) = hphi(igk, i) - eval[i] * ophi(igk, i);
                r += std::pow(std::abs(phi(igk, num_bands + i)), 2);

                double_complex z = h_diag[igk] - eval[i] * o_diag[igk];
                if (std::abs(z) > 1e-10) phi(igk, num_bands + i) /= z;
            }
            res_norm[i] = std::sqrt(r);
        }

        int n = 0;
        for (int i = 0; i < num_bands; i++)
        {
            /* take the residual if it's norm is above the threshold */
            if (kp__->band_occupancy(i) > 1e-12 &&
                (res_norm[i] > itso.tolerance_ || (res_norm[i] > itso.extra_tolerance_ && n != 0)))
            {
                /* shift unconverged residuals to the beginning of array */
                if (n != i) memcpy(&phi(0, num_bands + n), &phi(0, num_bands + i), kp__->num_gkvec() * sizeof(double_complex));
                n++;
            }
        }

        if (n == 0)
        {
            std::cout << "converged in " << k << " iterations" << std::endl;
            break;
        }

        apply_h_o_uspp_cpu(kp__, veff_it_coarse__, pw_ekin, n, &phi(0, num_bands), &hphi(0, num_bands), &ophi(0, num_bands));

        int N = num_bands + n;

        //std::cout << "Iteration: " << k << ", subspace size : " << N << std::endl;
        //for (int i = 0; i < std::min(10, num_bands); i++) std::cout << "eval["<<i<<"] = " << eval[i] << std::endl;
        //for (int i = 0; i < std::min(10, num_bands); i++) std::cout << "res["<<i<<"] = " << res_norm[i] << std::endl;

        Timer t2("sirius::Band::diag_fv_uspp_cpu_serial_v3:gevp");

        blas<cpu>::gemm(2, 0, N, N, kp__->num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, 0), hphi.ld(), &hmlt(0, 0), hmlt.ld());
          
        blas<cpu>::gemm(2, 0, N, N, kp__->num_gkvec(), &phi(0, 0), phi.ld(), &ophi(0, 0), ophi.ld(), &ovlp(0, 0), ovlp.ld());

        //== {
        //==     mdarray<double_complex, 2> o1(3 * num_bands, 3 * num_bands);
        //==     mdarray<double_complex, 2> ev1(3 * num_bands, 3 * num_bands);
        //==     std::vector<double> e1(3 * num_bands);
        //==     ovlp >> o1;

        //==     parameters_.std_evp_solver()->solve(N, o1.ptr(), o1.ld(), &e1[0], ev1.ptr(), ev1.ld());

        //==     for (int i = 0; i < N; i++) std::cout << "eval_o["<<i<<"] = " << e1[i] << std::endl;
        //== }
        
        parameters_.gen_evp_solver()->solve(N, N, N, num_bands, hmlt.ptr(), hmlt.ld(), ovlp.ptr(), ovlp.ld(), 
                                            &eval_tmp[0], evec.ptr(), evec.ld());
        t2.stop();

        
        Timer t3("sirius::Band::diag_fv_uspp_cpu_serial_v3:update");
        
        blas<cpu>::gemm(0, 0, kp__->num_gkvec(), num_bands, N, &hphi(0, 0), hphi.ld(), 
                        &evec(0, 0), evec.ld(), &psi(0, 0), psi.ld());
        for (int j = 0; j < num_bands; j++)
        {
            memcpy(&hphi(0, j), &psi(0, j), kp__->num_gkvec() * sizeof(double_complex));
        }

        blas<cpu>::gemm(0, 0, kp__->num_gkvec(), num_bands, N, &ophi(0, 0), ophi.ld(), 
                        &evec(0, 0), evec.ld(), &psi(0, 0), psi.ld());
        for (int j = 0; j < num_bands; j++)
        {
            memcpy(&ophi(0, j), &psi(0, j), kp__->num_gkvec() * sizeof(double_complex));
        }

        blas<cpu>::gemm(0, 0, kp__->num_gkvec(), num_bands, N, &phi(0, 0), phi.ld(), 
                        &evec(0, 0), evec.ld(), &psi(0, 0), psi.ld());
        for (int j = 0; j < num_bands; j++)
        {
            memcpy(&phi(0, j), &psi(0, j), kp__->num_gkvec() * sizeof(double_complex));
        }
        t3.stop();
    }

    kp__->set_fv_eigen_values(&eval[0]);
    kp__->fv_states_panel().scatter(psi, parameters_.mpi_grid().communicator(1 << _dim_row_));
}



void Band::diag_fv_uspp_cpu(K_point* kp__, 
                            Periodic_function<double>* effective_potential__)
{
    Timer t("sirius::Band::diag_fv_uspp_cpu");

    auto rl = parameters_.reciprocal_lattice();

    /* map effective potential to a corase grid */
    std::vector<double> veff_it_coarse(rl->fft_coarse()->size());
    std::vector<double_complex> veff_pw_coarse(rl->num_gvec_coarse());

    /* take only first num_gvec_coarse plane-wave harmonics; this is enough to apply V_eff to \Psi */
    for (int igc = 0; igc < rl->num_gvec_coarse(); igc++)
    {
        int ig = rl->gvec_index(igc);
        veff_pw_coarse[igc] = effective_potential__->f_pw(ig);
    }
    rl->fft_coarse()->input(rl->num_gvec_coarse(), rl->fft_index_coarse(), &veff_pw_coarse[0]);
    rl->fft_coarse()->transform(1);
    rl->fft_coarse()->output(&veff_it_coarse[0]);

    double v0 = real(effective_potential__->f_pw(0));

    if (parameters_.gen_evp_solver()->parallel())
    {
        diag_fv_uspp_cpu_parallel(kp__, v0, veff_it_coarse);
    }
    else
    {
        switch (parameters_.iterative_solver_input_section_.version_)
        {
            case 0:
            {
                diag_fv_uspp_cpu_serial_v0(kp__, veff_it_coarse);
                break;
            }
            case 1:
            {
                diag_fv_uspp_cpu_serial_v1(kp__, v0, veff_it_coarse);
                break;
            }
            case 2:
            {
                diag_fv_uspp_cpu_serial_v2(kp__, v0, veff_it_coarse);
                break;
            }
            case 3:
            {
                diag_fv_uspp_cpu_serial_v3(kp__, v0, veff_it_coarse);
                break;
            }
            case 4:
            {
                diag_fv_uspp_cpu_serial_v4(kp__, v0, veff_it_coarse);
                break;
            }
        }
    }
}

#ifdef _GPU_
void Band::diag_fv_uspp_gpu(K_point* kp, Periodic_function<double>* effective_potential)
{
    Timer t("sirius::Band::diag_fv_uspp_gpu");

    // map effective potential to a corase grid
    std::vector<double> veff_it_coarse(parameters_.reciprocal_lattice()->fft_coarse()->size());
    std::vector<double_complex> veff_pw_coarse(parameters_.reciprocal_lattice()->num_gvec_coarse());

    // take only first num_gvec_coarse plane-wave harmonics; this is enough to apply V_eff to \Psi
    for (int igc = 0; igc < parameters_.reciprocal_lattice()->num_gvec_coarse(); igc++)
    {
        int ig = parameters_.reciprocal_lattice()->gvec_index(igc);
        veff_pw_coarse[igc] = effective_potential->f_pw(ig);
    }
    parameters_.reciprocal_lattice()->fft_coarse()->input(parameters_.reciprocal_lattice()->num_gvec_coarse(), 
                                                          parameters_.reciprocal_lattice()->fft_index_coarse(),
                                                          &veff_pw_coarse[0]);
    parameters_.reciprocal_lattice()->fft_coarse()->transform(1);
    parameters_.reciprocal_lattice()->fft_coarse()->output(&veff_it_coarse[0]);

    // short notation for target wave-functions
    mdarray<double_complex, 2>& psi = kp->fv_states();

    // short notation for number of target wave-functions
    int num_bands = parameters_.num_fv_states();     

    // cache kinetic energy,
    std::vector<double> pw_ekin = kp->get_pw_ekin();

    // get diagonal elements for preconditioning
    std::vector<double_complex> h_diag;
    std::vector<double_complex> o_diag;
    get_h_o_diag(kp, effective_potential, pw_ekin, h_diag, o_diag);
    
    int max_iter = 10;
    int num_phi = std::min(4 * num_bands, kp->num_gkvec());

    // big array to store one of the beta, phi, hphi or ophi on the GPU
    mdarray<double_complex, 2> gamma(NULL, kp->num_gkvec(), std::max(num_phi, parameters_.unit_cell()->mt_lo_basis_size()));
    gamma.allocate_on_device();

    // small array to store residuals, wave-functions and updates to hphi and ophi
    mdarray<double_complex, 2> kappa(kp->num_gkvec(), 2 * num_bands);
    kappa.allocate_on_device();
    kappa.pin_memory();

    mdarray<double_complex, 2> phi(kp->num_gkvec(), num_phi);
    mdarray<double_complex, 2> hphi(kp->num_gkvec(), num_phi);
    mdarray<double_complex, 2> ophi(kp->num_gkvec(), num_phi);
    
    mdarray<double_complex, 2> hmlt(num_phi, num_phi);
    mdarray<double_complex, 2> ovlp(num_phi, num_phi);
    mdarray<double_complex, 2> hmlt_old(num_phi, num_phi);
    mdarray<double_complex, 2> ovlp_old(num_phi, num_phi);
    
    mdarray<double_complex, 2> evec(num_phi, num_phi);
    evec.allocate_on_device();
    evec.pin_memory();

    std::vector<double> eval(num_bands);
    std::vector<double> eval_old(num_bands, 1e100);
    
    std::vector<double> res_norm(num_bands); // norm of residuals
    std::vector<double> res_rms(num_bands); // RMS of residual
    std::vector<double> res_e(num_bands);
    
    if (parameters_.gen_evp_solver()->type() == ev_magma)
    {
        hmlt.pin_memory();
        ovlp.pin_memory();
    }

    bool convergence_by_energy = false;

    int N = 0; // current eigen-value problem size
    int n = num_bands; // number of added residuals

    // trial basis functions
    assert(phi.size(0) == psi.size(0));
    for (int i = 0; i < num_bands; i++) memcpy(&phi(0, i), &psi(0, i), kp->num_gkvec() * sizeof(double_complex));
    
    // start iterative diagonalization
    for (int k = 0; k < max_iter; k++)
    {
        Timer t1("sirius::Band::diag_fv_uspp_gpu|set_gevp");

        // copy old Hamiltonian and overlap
        for (int i = 0; i < N; i++)
        {
            memcpy(&hmlt(0, i), &hmlt_old(0, i), N * sizeof(double_complex));
            memcpy(&ovlp(0, i), &ovlp_old(0, i), N * sizeof(double_complex));
        }

        // apply Hamiltonian and overlap operators
        apply_h_o_uspp_gpu(kp, veff_it_coarse, pw_ekin, n, gamma, kappa, &phi(0, N), &hphi(0, N), &ophi(0, N));

        // copy all phi to GPU
        cublas_set_matrix(kp->num_gkvec(), n + N, sizeof(double_complex), phi.ptr(), phi.ld(), gamma.ptr_device(), gamma.ld());

        // temporary storage for Hamiltonian and overlap 
        mdarray<double_complex, 2> tmp(NULL, N + n, n);
        tmp.allocate_on_device();

        // compute the Hamiltonian matrix: <phi|H|phi>
        blas<gpu>::gemm(2, 0, N + n, n, kp->num_gkvec(), gamma.ptr_device(), gamma.ld(), 
                        kappa.ptr_device(), kappa.ld(), tmp.ptr_device(), tmp.ld());

        cublas_get_matrix(N + n, n, sizeof(double_complex), tmp.ptr_device(), tmp.ld(), &hmlt(0, N), hmlt.ld());

        // compute overlap matrix <phi|O|phi>
        blas<gpu>::gemm(2, 0, N + n, n, kp->num_gkvec(), gamma.ptr_device(), gamma.ld(), 
                        kappa.ptr_device(0, n), kappa.ld(), tmp.ptr_device(), tmp.ld());

        cublas_get_matrix(N + n, n, sizeof(double_complex), tmp.ptr_device(), tmp.ld(), &ovlp(0, N), ovlp.ld());

        tmp.deallocate_on_device();

        // MAGMA works with lower triangular part
        #ifdef _MAGMA_
        for (int i = 0; i < N; i++)
        {
            for (int j = N; j < N + n; j++)
            {
                hmlt(j, i) = conj(hmlt(i, j));
                ovlp(j, i) = conj(ovlp(i, j));
            }
        }
        #endif

        // increase the size of the variation space
        N += n;

        // save Hamiltonian and overlap
        for (int i = 0; i < N; i++)
        {
            memcpy(&hmlt_old(0, i), &hmlt(0, i), N * sizeof(double_complex));
            memcpy(&ovlp_old(0, i), &ovlp(0, i), N * sizeof(double_complex));
        }
        t1.stop();
        
        Timer t2("sirius::Band::diag_fv_uspp_gpu|solve_gevp");
        parameters_.gen_evp_solver()->solve(N, num_bands, num_bands, num_bands, hmlt.ptr(), hmlt.ld(), ovlp.ptr(), ovlp.ld(), 
                                            &eval[0], evec.ptr(), evec.ld());
        t2.stop();

        Timer t3("sirius::Band::diag_fv_uspp_gpu|residuals");
        /* Quantum Espresso way of estimating basis update: residuals for which |e - e_old| > eps 
           are accepted as the additional basis functions */
        if (convergence_by_energy)
        {
            //== if (k == 0)
            //== {
            //==     n = num_bands;
            //==     memcpy(&hmlt(0, 0), &evec(0, 0), N * n * sizeof(double_complex));
            //== }
            //== else
            //== {
            //==     n = 0;
            //==     // check eigen-values for convergence
            //==     for (int i = 0; i < num_bands; i++)
            //==     {
            //==         if (fabs(eval[i] - eval_old[i]) > parameters_.iterative_solver_tolerance())
            //==         {
            //==             res_e[n] = eval[i];
            //==             
            //==             // use hmlt as a temporary storage for evec
            //==             memcpy(&hmlt(0, n), &evec(0, i), N * sizeof(double_complex));
 
            //==             n++;
            //==         }
            //==         eval_old[i] = eval[i];
            //==     }
            //== }

            //== // if we have unconverged eigen-states
            //== if (n != 0)
            //== {
            //==     // compute residuals
            //==     // 1. O\Psi_{i} = O\phi_{mu} * Z_{mu, i}
            //==     blas<cpu>::gemm(0, 0, kp->num_gkvec(), n, N, &ophi(0, 0), ophi.ld(), &hmlt(0, 0), hmlt.ld(), 
            //==                     &res(0, 0), res.ld());
            //==     // 2. multiply O\Psi_{i} with energy
            //==     for (int i = 0; i < n; i++)
            //==     {
            //==         for (int igk = 0; igk < kp->num_gkvec(); igk++) res(igk, i) *= eval[i];
            //==     }
            //==     // 3. r_{i} = H\Psi_{i} - E_{i}O\Psi_{i}
            //==     blas<cpu>::gemm(0, 0, kp->num_gkvec(), n, N, double_complex(1, 0), &hphi(0, 0), hphi.ld(), 
            //==                     &hmlt(0, 0), hmlt.ld(), double_complex(-1, 0), &res(0, 0), res.ld());

            //==     // apply preconditioner
            //==     #pragma omp parallel for
            //==     for (int i = 0; i < n; i++)
            //==     {
            //==         // apply preconditioner
            //==         for (int igk = 0; igk < kp->num_gkvec(); igk++)
            //==         {
            //==             //double_complex z = h_diag[igk] - res_e[i] * o_diag[igk];
            //==             //if (abs(z) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
            //==             double d = real(h_diag[igk] - res_e[i] * o_diag[igk]);
            //==             if (fabs(d) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
            //==             res(igk, i) /= d;
            //==         }
            //==     }
            //== }
        }
        /* Alternative way to estimate basis update: take residuals with norm > eps */
        else
        {
            cublas_set_matrix(N, N, sizeof(double_complex), evec.ptr(), evec.ld(), evec.ptr_device(), evec.ld());

            // copy all ophi to GPU
            cublas_set_matrix(kp->num_gkvec(), N, sizeof(double_complex), ophi.ptr(), ophi.ld(), gamma.ptr_device(), gamma.ld());
            
            // O\Psi_{i} = O\phi_{mu} * Z_{mu, i}
            blas<gpu>::gemm(0, 0, kp->num_gkvec(), num_bands, N, gamma.ptr_device(), gamma.ld(), 
                            evec.ptr_device(), evec.ld(), kappa.ptr_device(), kappa.ld());
        
            mdarray<double, 1> eval_gpu(&eval[0], num_bands);
            eval_gpu.allocate_on_device();
            eval_gpu.copy_to_device();
            // multiply O\Psi_{i} with energy
            scale_matrix_columns_gpu(kp->num_gkvec(), num_bands, kappa.ptr_device(), eval_gpu.ptr_device());
            eval_gpu.deallocate_on_device();
            
            // copy all hphi to GPU
            cublas_set_matrix(kp->num_gkvec(), N, sizeof(double_complex), hphi.ptr(), hphi.ld(), gamma.ptr_device(), gamma.ld());
            
            double_complex zone(1, 0);
            double_complex mzone(-1, 0);
            // r_{i} = H\Psi_{i} - E_{i}O\Psi_{i}
            blas<gpu>::gemm(0, 0, kp->num_gkvec(), num_bands, N, &zone, gamma.ptr_device(), gamma.ld(), 
                            evec.ptr_device(), evec.ld(), &mzone, kappa.ptr_device(), kappa.ld());
           
            // copy residuals to the host memory
            cublas_get_matrix(kp->num_gkvec(), num_bands, sizeof(double_complex), kappa.ptr_device(), kappa.ld(), 
                              kappa.ptr(), kappa.ld());

            Timer t("sirius::Band::diag_fv_uspp_gpu|residuals|cpu_part");
            // compute norm and apply preconditioner
            #pragma omp parallel for
            for (int i = 0; i < num_bands; i++)
            {
                double r = 0;
                for (int igk = 0; igk < kp->num_gkvec(); igk++) r += real(conj(kappa(igk, i)) * kappa(igk, i));
                res_norm[i] = r;
                res_rms[i] = sqrt(r / kp->num_gkvec());
                
                // apply preconditioner
                for (int igk = 0; igk < kp->num_gkvec(); igk++)
                {
                    double_complex z = h_diag[igk] - eval[i] * o_diag[igk];
                    if (abs(z) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
                    kappa(igk, i) /= z;
                }
            }

            // check which residuals are converged
            n = 0;
            std::vector< std::pair<double, int> > res_rms_sorted;
            for (int i = 0; i < num_bands; i++)
            {
                res_rms_sorted.push_back(std::pair<double, int>(res_rms[i], i));

                // take the residual if it's norm is above the threshold
                if (res_rms[i] > parameters_.iterative_solver_tolerance()) n++;
            }
            
            if (n > 0 && n < num_bands)
            {
                n = std::max(n, (num_bands - 1) / (k + 1));

                std::sort(res_rms_sorted.begin(), res_rms_sorted.end());

                double tol = res_rms_sorted[num_bands - n].first;

                n = 0;
                for (int i = 0; i < num_bands; i++)
                {
                    // take the residual if it's norm is above the threshold
                    if (res_rms[i] > tol) 
                    {
                        // shift unconverged residuals to the beginning of array
                        if (n != i) memcpy(&kappa(0, n), &kappa(0, i), kp->num_gkvec() * sizeof(double_complex));
                        n++;
                    }
                }
            }
            t.stop();
 
            //== n = 0;
            //== for (int i = 0; i < num_bands; i++)
            //== {
            //==     // take the residual if it's norm is above the threshold
            //==     if (res_rms[i] > parameters_.iterative_solver_tolerance()) 
            //==     {
            //==         // shift unconverged residuals to the beginning of array
            //==         if (n != i) memcpy(&res(0, n), &res(0, i), kp->num_gkvec() * sizeof(double_complex));
            //==         n++;
            //==     }
            //== }
        }
        t3.stop();

        //if (Platform::mpi_rank() == 0)
        //{
        //    std::cout << "iteration:" << k << ", current eigen-value size = " << N << ", number of added residuals = " << n << std::endl;
        //    //printf("lower and upper eigen-values : %16.8f %16.8f\n", eval[0], eval[num_bands - 1]);
        //}

        // check if we run out of variational space or eigen-vectors are converged or it's a last iteration
        if (N + n > num_phi || n == 0 || k == (max_iter - 1))
        {   
            Timer t3("sirius::Band::diag_fv_uspp_gpu|update_phi");
            // copy all phi to GPU
            cublas_set_matrix(kp->num_gkvec(), N, sizeof(double_complex), phi.ptr(), phi.ld(), 
                              gamma.ptr_device(), gamma.ld());
            // \Psi_{i} = \phi_{mu} * Z_{mu, i}
            blas<gpu>::gemm(0, 0, kp->num_gkvec(), num_bands, N, gamma.ptr_device(), gamma.ld(), 
                            evec.ptr_device(), evec.ld(), kappa.ptr_device(), kappa.ld());

            cublas_get_matrix(kp->num_gkvec(), num_bands, sizeof(double_complex), 
                              kappa.ptr_device(), kappa.ld(), psi.ptr(), psi.ld());
            t3.stop();

            if (n == 0 || k == (max_iter - 1)) // exit the loop if the eigen-vectors are converged or it's a last iteration
            {
                std::cout << "converged in " << k << " iterations" << std::endl;
                break;
            }
            else // otherwise set \Psi as a new trial basis and update related arrays
            {
                Timer t("sirius::Band::diag_fv_uspp_gpu|update_h_o");

                // temporary storage for Hamiltonian and overlap 
                mdarray<double_complex, 2> tmp(NULL, num_bands, num_bands);
                tmp.allocate_on_device();

                // compute H\Psi
                cublas_set_matrix(kp->num_gkvec(), N, sizeof(double_complex), hphi.ptr(), hphi.ld(), 
                                  gamma.ptr_device(), gamma.ld());

                blas<gpu>::gemm(0, 0, kp->num_gkvec(), num_bands, N, gamma.ptr_device(), gamma.ld(), 
                                evec.ptr_device(), evec.ld(), kappa.ptr_device(0, num_bands), kappa.ld());
                
                // copy H\Psi to host memory
                cublas_get_matrix(kp->num_gkvec(), num_bands, sizeof(double_complex),
                                  kappa.ptr_device(0, num_bands), kappa.ld(), hphi.ptr(), hphi.ld());

                // compute the Hamiltonian matrix: <Psi|H|Psi>
                blas<gpu>::gemm(2, 0, num_bands, num_bands, kp->num_gkvec(), kappa.ptr_device(), kappa.ld(), 
                                kappa.ptr_device(0, num_bands), kappa.ld(), tmp.ptr_device(), tmp.ld());

                // copy Hamiltonian to host
                cublas_get_matrix(num_bands, num_bands, sizeof(double_complex), tmp.ptr_device(), tmp.ld(), 
                                  hmlt_old.ptr(), hmlt_old.ld());
                
                // compute O\Psi
                cublas_set_matrix(kp->num_gkvec(), N, sizeof(double_complex), ophi.ptr(), ophi.ld(), 
                                  gamma.ptr_device(), gamma.ld());

                blas<gpu>::gemm(0, 0, kp->num_gkvec(), num_bands, N, gamma.ptr_device(), gamma.ld(), 
                                evec.ptr_device(), evec.ld(), kappa.ptr_device(0, num_bands), kappa.ld());

                // copy O\Psi to host memory
                cublas_get_matrix(kp->num_gkvec(), num_bands, sizeof(double_complex),
                                  kappa.ptr_device(0, num_bands), kappa.ld(), ophi.ptr(), ophi.ld());

                // compute the overlap matrix: <Psi|O|Psi>
                blas<gpu>::gemm(2, 0, num_bands, num_bands, kp->num_gkvec(), kappa.ptr_device(), kappa.ld(), 
                                kappa.ptr_device(0, num_bands), kappa.ld(), tmp.ptr_device(), tmp.ld());

                // copy overlap matrix to host
                cublas_get_matrix(num_bands, num_bands, sizeof(double_complex), tmp.ptr_device(), tmp.ld(), 
                                  ovlp_old.ptr(), ovlp_old.ld());
             
                // update phi with Psi
                memcpy(phi.ptr(), psi.ptr(), num_bands * kp->num_gkvec() * sizeof(double_complex));

                // new size of eigen-value problem 
                N = num_bands;

                tmp.deallocate_on_device();
            }
        }
        // expand variational space with new preconditioned residuals
        memcpy(&phi(0, N), &kappa(0, 0), n * kp->num_gkvec() * sizeof(double_complex));
    }

    kp->set_fv_eigen_values(&eval[0]);
}
#endif

}
