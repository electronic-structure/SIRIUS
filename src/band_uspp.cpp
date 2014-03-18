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
    exec_fft_args* args = (exec_fft_args*)args__;

    FFT3D<gpu> fft(args->fft->grid_size());

    mdarray<int, 1> fft_index_coarse(args->kp->fft_index_coarse(), args->kp->num_gkvec());
    fft_index_coarse.allocate_on_device();
    fft_index_coarse.copy_to_device();

    int nfft_buf = (int)(args->gamma->size() / fft.size());
    if (nfft_buf == 0) return NULL; // TODO: fix this

    int nfft_max = std::min(fft.num_fft_max(), std::min(args->num_phi / 4, nfft_buf));
   
    fft.initialize(nfft_max); 

    bool done = false;

    while (!done)
    {
        pthread_mutex_lock(&exec_fft_mutex);
        int i = idxfft;
        if (idxfft + nfft_max > args->num_phi) 
        {
            done = true;
        }
        else
        {
            idxfft += nfft_max;
        }
        pthread_mutex_unlock(&exec_fft_mutex);

        if (!done)
        {
            cublas_set_matrix(args->kp->num_gkvec(), nfft_max, sizeof(double_complex), &(*args->phi)(0, i), args->phi->ld(), 
                              args->kappa->ptr_device(), args->kappa->ld());
            
            // use gamma as fft buffer
            fft.batch_load(args->kp->num_gkvec(), fft_index_coarse.ptr_device(), args->kappa->ptr_device(), 
                           args->gamma->ptr_device());

            fft.transform(1, args->gamma->ptr_device());
            scale_matrix_rows_gpu(fft.size(), nfft_max, args->gamma->ptr_device(), args->veff->ptr_device());
            fft.transform(-1, args->gamma->ptr_device());

            fft.batch_unload(args->kp->num_gkvec(), fft_index_coarse.ptr_device(), args->gamma->ptr_device(), 
                             args->kappa->ptr_device());

            cublas_get_matrix(args->kp->num_gkvec(), nfft_max, sizeof(double_complex), 
                              args->kappa->ptr_device(), args->kappa->ld(),
                              &(*args->hphi)(0, i), args->hphi->ld());
        }
    }

    fft.finalize();
    
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

void Band::get_h_o_diag(K_point* kp, Periodic_function<double>* effective_potential, std::vector<double>& pw_ekin, 
                        std::vector<double_complex>& h_diag, std::vector<double_complex>& o_diag)
{
    Timer t("sirius::Band::get_h_o_diag");

    h_diag.resize(kp->num_gkvec());
    o_diag.resize(kp->num_gkvec());
    
    // compute V_{loc}(G=0)
    double v0 = 0;
    for (int ir = 0; ir < fft_->size(); ir++) v0 += effective_potential->f_it<global>(ir);
    v0 /= parameters_.unit_cell()->omega();
    
    for (int igk = 0; igk < kp->num_gkvec(); igk++) h_diag[igk] = pw_ekin[igk] + v0;

    mdarray<double_complex, 2> beta_pw(kp->num_gkvec(), parameters_.unit_cell()->max_mt_basis_size());
    mdarray<double_complex, 2> beta_pw_tmp(parameters_.unit_cell()->max_mt_basis_size(), kp->num_gkvec());
    for (int iat = 0; iat < parameters_.unit_cell()->num_atom_types(); iat++)
    {
        auto atom_type = parameters_.unit_cell()->atom_type(iat);
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
                    d_sum(xi1, xi2) += parameters_.unit_cell()->atom(ia)->d_mtrx(xi1, xi2);
                    q_sum(xi1, xi2) += parameters_.unit_cell()->atom(ia)->type()->uspp().q_mtrx(xi1, xi2);
                }
            }
        }

        //kp->generate_beta_pw(&beta_pw(0, 0), atom_type);
        int ofs = parameters_.unit_cell()->beta_t_ofs(iat);
        for (int igk = 0; igk < kp->num_gkvec(); igk++)
        {
            for (int xi = 0; xi < nbf; xi++) beta_pw_tmp(xi, igk) = kp->beta_pw_t(igk, ofs + xi);
        }

        std::vector< std::pair<int, int> > idx(nbf * nbf);
        int n = 0;
        for (int xi2 = 0; xi2 < nbf; xi2++)
        {
            for (int xi1 = 0; xi1 < nbf; xi1++)
            {
                idx[n++] = std::pair<int, int>(xi1, xi2);
            }
        }

        #pragma omp parallel for
        for (int igk = 0; igk < kp->num_gkvec(); igk++)
        {
            for (int i = 0; i < n; i++)
            {
                int xi1 = idx[i].first;
                int xi2 = idx[i].second;
                double_complex z = beta_pw_tmp(xi1, igk) * conj(beta_pw_tmp(xi2, igk));

                h_diag[igk] += z * d_sum(xi1, xi2);
                o_diag[igk] += z * q_sum(xi1, xi2);
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
void Band::apply_h_o_uspp_cpu(K_point* kp, std::vector<double>& effective_potential, std::vector<double>& pw_ekin, int n,
                              double_complex* phi__, double_complex* hphi__, double_complex* ophi__)
{
    Timer t("sirius::Band::apply_h_o", _global_timer_);

    mdarray<double_complex, 2> phi(phi__, kp->num_gkvec(), n);
    mdarray<double_complex, 2> hphi(hphi__, kp->num_gkvec(), n);
    mdarray<double_complex, 2> ophi(ophi__, kp->num_gkvec(), n);
    
    // apply local part of Hamiltonian
    apply_h_local(kp, effective_potential, pw_ekin, n, phi__, hphi__);
   
    // set intial ophi
    memcpy(ophi__, phi__, kp->num_gkvec() * n * sizeof(double_complex));

    // <\beta_{\xi}^{\alpha}|\phi_j>
    mdarray<double_complex, 2> beta_phi(parameters_.unit_cell()->num_beta_a(), n);
    
    // Q or D multiplied by <\beta_{\xi}^{\alpha}|\phi_j>
    mdarray<double_complex, 2> tmp(parameters_.unit_cell()->num_beta_a(), n);

    Timer t1("sirius::Band::apply_h_o|beta_phi");

    // compute <beta|phi>
    blas<cpu>::gemm(2, 0, parameters_.unit_cell()->num_beta_a(), n, kp->num_gkvec(), &kp->beta_pw_a(0, 0), kp->num_gkvec(), 
                    &phi(0, 0), phi.ld(), &beta_phi(0, 0), beta_phi.ld());
    t1.stop();
    
    // compute D*<beta|phi>
    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {   
        int ofs = parameters_.unit_cell()->beta_a_ofs(ia);
        // number of beta functions for a given atom
        int nbf = parameters_.unit_cell()->atom(ia)->type()->mt_basis_size();
        blas<cpu>::gemm(0, 0, nbf, n, nbf, &parameters_.unit_cell()->atom(ia)->d_mtrx(0, 0), nbf, 
                        &beta_phi(ofs, 0), beta_phi.ld(), &tmp(ofs, 0), tmp.ld());
    }

    Timer t3("sirius::Band::apply_h_o|beta_D_beta_phi");
    // compute <G+k|beta> * D*<beta|phi> and add to hphi
    blas<cpu>::gemm(0, 0, kp->num_gkvec(), n, parameters_.unit_cell()->num_beta_a(), complex_one, 
                    &kp->beta_pw_a(0, 0), kp->num_gkvec(), &tmp(0, 0), tmp.ld(), complex_one, &hphi(0, 0), hphi.ld());
    t3.stop();

    // compute Q*<beta|phi>
    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {   
        int ofs = parameters_.unit_cell()->beta_a_ofs(ia);
        // number of beta functions for a given atom
        int nbf = parameters_.unit_cell()->atom(ia)->type()->mt_basis_size();
        blas<cpu>::gemm(0, 0, nbf, n, nbf, &parameters_.unit_cell()->atom(ia)->type()->uspp().q_mtrx(0, 0), nbf, 
                        &beta_phi(ofs, 0), beta_phi.ld(), &tmp(ofs, 0), tmp.ld());
    }

    Timer t5("sirius::Band::apply_h_o|beta_Q_beta_phi");
    // computr <G+k|beta> * Q*<beta|phi> and add to ophi
    blas<cpu>::gemm(0, 0, kp->num_gkvec(), n, parameters_.unit_cell()->num_beta_a(), complex_one, 
                    &kp->beta_pw_a(0, 0), kp->num_gkvec(), &tmp(0, 0), tmp.ld(), complex_one, &ophi(0, 0), ophi.ld());
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
                       parameters_.unit_cell()->num_beta_a(), 
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
    mdarray<double_complex, 2> beta_phi(NULL, parameters_.unit_cell()->num_beta_a(), n);
    beta_phi.allocate_on_device();
    
    blas<gpu>::gemm(2, 0, parameters_.unit_cell()->num_beta_a(), n, kp->num_gkvec(), gamma.ptr_device(0, 0), gamma.ld(), 
                    kappa.ptr_device(0, n), kappa.ld(), beta_phi.ptr_device(0, 0), beta_phi.ld());

    // Q or D multiplied by <\beta_{\xi}^{\alpha}|\phi_j>
    mdarray<double_complex, 2> tmp(NULL, parameters_.unit_cell()->num_beta_a(), n);
    tmp.allocate_on_device();

    d_mtrx_packed.allocate_on_device();
    d_mtrx_packed.copy_to_device();
    // compute D*<beta|phi>
    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {   
        int ofs = parameters_.unit_cell()->beta_a_ofs(ia);
        // number of beta functions for a given atom
        int nbf = parameters_.unit_cell()->atom(ia)->type()->mt_basis_size();
        blas<gpu>::gemm(0, 0, nbf, n, nbf, d_mtrx_packed.ptr_device(mtrx_ofs(ia)), nbf, 
                        beta_phi.ptr_device(ofs, 0), beta_phi.ld(), tmp.ptr_device(ofs, 0), tmp.ld());
    }
    d_mtrx_packed.deallocate_on_device();

    double_complex zone(1, 0);
    // compute <G+k|beta> * D*<beta|phi> and add to hphi
    blas<gpu>::gemm(0, 0, kp->num_gkvec(), n, parameters_.unit_cell()->num_beta_a(), &zone, gamma.ptr_device(), gamma.ld(), 
                    tmp.ptr_device(), tmp.ld(), &zone, kappa.ptr_device(), kappa.ld());

    q_mtrx_packed.allocate_on_device();
    q_mtrx_packed.copy_to_device();

    // compute Q*<beta|phi>
    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {   
        int ofs = parameters_.unit_cell()->beta_a_ofs(ia);
        // number of beta functions for a given atom
        int nbf = parameters_.unit_cell()->atom(ia)->type()->mt_basis_size();
        blas<gpu>::gemm(0, 0, nbf, n, nbf, q_mtrx_packed.ptr_device(mtrx_ofs(ia)), nbf, 
                        beta_phi.ptr_device(ofs, 0), beta_phi.ld(), tmp.ptr_device(ofs, 0), tmp.ld());
    }
    q_mtrx_packed.deallocate_on_device();

    // computr <G+k|beta> * Q*<beta|phi> and add to ophi
    blas<gpu>::gemm(0, 0, kp->num_gkvec(), n, parameters_.unit_cell()->num_beta_a(), &zone, gamma.ptr_device(), gamma.ld(), 
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

void Band::diag_fv_uspp_cpu(K_point* kp, Periodic_function<double>* effective_potential)
{
    Timer t("sirius::Band::diag_fv_uspp_cpu", _global_timer_);

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

    mdarray<double_complex, 2> phi(kp->num_gkvec(), num_phi);
    mdarray<double_complex, 2> hphi(kp->num_gkvec(), num_phi);
    mdarray<double_complex, 2> ophi(kp->num_gkvec(), num_phi);

    mdarray<double_complex, 2> hmlt(num_phi, num_phi);
    mdarray<double_complex, 2> ovlp(num_phi, num_phi);
    mdarray<double_complex, 2> hmlt_old(num_phi, num_phi);
    mdarray<double_complex, 2> ovlp_old(num_phi, num_phi);
    mdarray<double_complex, 2> evec(num_phi, num_phi);
    std::vector<double> eval(num_bands);
    std::vector<double> eval_old(num_bands, 1e100);
    
    mdarray<double_complex, 2> res(kp->num_gkvec(), num_bands); // residuals

    std::vector<double> res_norm(num_bands); // norm of residuals
    std::vector<double> res_rms(num_bands); // RMS of residual
    std::vector<double> res_e(num_bands);

    bool convergence_by_energy = false;

    int N = 0; // current eigen-value problem size
    int n = num_bands; // number of added residuals

    // trial basis functions
    assert(phi.size(0) == psi.size(0));
    for (int i = 0; i < num_bands; i++) memcpy(&phi(0, i), &psi(0, i), kp->num_gkvec() * sizeof(double_complex));
    
    // start iterative diagonalization
    for (int k = 0; k < max_iter; k++)
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
            apply_h_o_uspp_cpu(kp, veff_it_coarse, pw_ekin, n, &phi(0, N), &hphi(0, N), &ophi(0, N));
            
            // <{phi,res}|H|res>
            blas<cpu>::gemm(2, 0, N + n, n, kp->num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, N), hphi.ld(), &hmlt(0, N), hmlt.ld());
            
            // <{phi,res}|O|res>
            blas<cpu>::gemm(2, 0, N + n, n, kp->num_gkvec(), &phi(0, 0), phi.ld(), &ophi(0, N), ophi.ld(), &ovlp(0, N), ovlp.ld());
            
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

        Timer t3("sirius::Band::diag_fv_uspp_cpu|residuals");
        /* Quantum Espresso way of estimating basis update: residuals for which |e - e_old| > eps 
           are accepted as the additional basis functions */
        if (convergence_by_energy)
        {
            if (k == 0)
            {
                n = num_bands;
                memcpy(&hmlt(0, 0), &evec(0, 0), N * n * sizeof(double_complex));
            }
            else
            {
                n = 0;
                // check eigen-values for convergence
                for (int i = 0; i < num_bands; i++)
                {
                    if (fabs(eval[i] - eval_old[i]) > parameters_.iterative_solver_tolerance())
                    {
                        res_e[n] = eval[i];
                        
                        // use hmlt as a temporary storage for evec
                        memcpy(&hmlt(0, n), &evec(0, i), N * sizeof(double_complex));
 
                        n++;
                    }
                    eval_old[i] = eval[i];
                }
            }

            // if we have unconverged eigen-states
            if (n != 0)
            {
                // compute residuals
                // 1. O\Psi_{i} = O\phi_{mu} * Z_{mu, i}
                blas<cpu>::gemm(0, 0, kp->num_gkvec(), n, N, &ophi(0, 0), ophi.ld(), &hmlt(0, 0), hmlt.ld(), 
                                &res(0, 0), res.ld());
                // 2. multiply O\Psi_{i} with energy
                for (int i = 0; i < n; i++)
                {
                    for (int igk = 0; igk < kp->num_gkvec(); igk++) res(igk, i) *= eval[i];
                }
                // 3. r_{i} = H\Psi_{i} - E_{i}O\Psi_{i}
                blas<cpu>::gemm(0, 0, kp->num_gkvec(), n, N, double_complex(1, 0), &hphi(0, 0), hphi.ld(), 
                                &hmlt(0, 0), hmlt.ld(), double_complex(-1, 0), &res(0, 0), res.ld());

                // apply preconditioner
                #pragma omp parallel for
                for (int i = 0; i < n; i++)
                {
                    // apply preconditioner
                    for (int igk = 0; igk < kp->num_gkvec(); igk++)
                    {
                        //double_complex z = h_diag[igk] - res_e[i] * o_diag[igk];
                        //if (abs(z) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
                        double d = real(h_diag[igk] - res_e[i] * o_diag[igk]);
                        if (fabs(d) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
                        res(igk, i) /= d;
                    }
                }
            }
        }
        /* Alternative way to estimate basis update: take residuals with norm > eps */
        else
        {
            // compute residuals
            // 1. O\Psi_{i} = O\phi_{mu} * Z_{mu, i}
            blas<cpu>::gemm(0, 0, kp->num_gkvec(), num_bands, N, &ophi(0, 0), ophi.ld(), &evec(0, 0), evec.ld(), 
                            &res(0, 0), res.ld());
            // 2. multiply O\Psi_{i} with energy
            for (int i = 0; i < num_bands; i++)
            {
                for (int igk = 0; igk < kp->num_gkvec(); igk++) res(igk, i) *= eval[i];
            }
            // 3. r_{i} = H\Psi_{i} - E_{i}O\Psi_{i}
            blas<cpu>::gemm(0, 0, kp->num_gkvec(), num_bands, N, double_complex(1, 0), &hphi(0, 0), hphi.ld(), 
                            &evec(0, 0), evec.ld(), double_complex(-1, 0), &res(0, 0), res.ld());

            // compute norm and apply preconditioner
            #pragma omp parallel for
            for (int i = 0; i < num_bands; i++)
            {
                double r = 0;
                for (int igk = 0; igk < kp->num_gkvec(); igk++) r += real(conj(res(igk, i)) * res(igk, i));
                res_norm[i] = r;
                res_rms[i] = sqrt(r / kp->num_gkvec());
                
                // apply preconditioner
                for (int igk = 0; igk < kp->num_gkvec(); igk++)
                {
                    double_complex z = h_diag[igk] - eval[i] * o_diag[igk];
                    if (abs(z) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
                    res(igk, i) /= z;
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
                        if (n != i) memcpy(&res(0, n), &res(0, i), kp->num_gkvec() * sizeof(double_complex));
                        n++;
                    }
                }
            }
 
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
            Timer t3("sirius::Band::diag_fv_uspp_cpu|update_phi");
            // \Psi_{i} = \phi_{mu} * Z_{mu, i}
            blas<cpu>::gemm(0, 0, kp->num_gkvec(), num_bands, N, &phi(0, 0), phi.ld(), &evec(0, 0), evec.ld(), 
                            &psi(0, 0), psi.ld());

            if (n == 0 || k == (max_iter - 1)) // exit the loop if the eigen-vectors are converged or it's a last iteration
            {
                std::cout << "converged in " << k << " iterations" << std::endl;
                break;
            }
            else // otherwise set Psi as a new trial basis
            {
                // use phi as a temporary vector to compute H\Psi
                blas<cpu>::gemm(0, 0, kp->num_gkvec(), num_bands, N, &hphi(0, 0), hphi.ld(), 
                                &evec(0, 0), evec.ld(), &phi(0, 0), phi.ld());
                
                // compute the Hamiltonian matrix: <Psi|H|Psi>
                blas<cpu>::gemm(2, 0, num_bands, num_bands, kp->num_gkvec(), &psi(0, 0), psi.ld(), &phi(0, 0), phi.ld(), 
                                &hmlt_old(0, 0), hmlt_old.ld());
             
                memcpy(hphi.ptr(), phi.ptr(), num_bands * kp->num_gkvec() * sizeof(double_complex));

                // use phi as a temporary vector to compute O\Psi
                blas<cpu>::gemm(0, 0, kp->num_gkvec(), num_bands, N, &ophi(0, 0), ophi.ld(), 
                                &evec(0, 0), evec.ld(), &phi(0, 0), phi.ld());
                
                // compute the overlap matrix: <Psi|O|Psi>
                blas<cpu>::gemm(2, 0, num_bands, num_bands, kp->num_gkvec(), &psi(0, 0), psi.ld(), &phi(0, 0), phi.ld(), 
                                &ovlp_old(0, 0), ovlp_old.ld());
            
                memcpy(ophi.ptr(), phi.ptr(), num_bands * kp->num_gkvec() * sizeof(double_complex));
                
                // set new basis functions
                memcpy(phi.ptr(), psi.ptr(), num_bands * kp->num_gkvec() * sizeof(double_complex));
                N = num_bands;
            }
        }
        // expand variational subspace with new basis vectors obtatined from residuals
        memcpy(&phi(0, N), &res(0, 0), n * kp->num_gkvec() * sizeof(double_complex));
    }

    kp->set_fv_eigen_values(&eval[0]);
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
    mdarray<double_complex, 2> gamma(NULL, kp->num_gkvec(), std::max(num_phi, parameters_.unit_cell()->num_beta_a()));
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
