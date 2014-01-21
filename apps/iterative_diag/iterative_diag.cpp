#include <sirius.h>

// TODO: pin memory for magma!!!

using namespace sirius;

void apply_h_cpu(Global& parameters, K_point& kp, int n, std::vector<double>& v_r, complex16* phi__, complex16* hphi__)
{
    Timer t("apply_h_cpu");

    auto fft = parameters.reciprocal_lattice()->fft();

    mdarray<complex16, 2> phi(phi__, kp.num_gkvec(), n);
    mdarray<complex16, 2> hphi(hphi__, kp.num_gkvec(), n);
    
    int num_fft_threads = Platform::num_fft_threads();
    #pragma omp parallel default(shared) num_threads(num_fft_threads)
    {        
        int thread_id = omp_get_thread_num();
        std::vector<complex16> phi_r(fft->size());
        
        #pragma omp for
        for (int i = 0; i < n; i++)
        {
            fft->input(kp.num_gkvec(), kp.fft_index(), &phi(0, i), thread_id);
            fft->transform(1, thread_id);
            fft->output(&phi_r[0], thread_id);

            for (int ir = 0; ir < fft->size(); ir++) phi_r[ir] *= v_r[ir];

            fft->input(&phi_r[0], thread_id);
            fft->transform(-1, thread_id);
            fft->output(kp.num_gkvec(), kp.fft_index(), &hphi(0, i), thread_id);

            for (int ig = 0; ig < kp.num_gkvec(); ig++) hphi(ig, i) += phi(ig, i) * pow(kp.gkvec_cart(ig).length(), 2) / 2.0;
        }
    }
}

struct exec_fft_args
{
    int thread_id;
    Global* parameters;
    K_point* kp;
    int n;
    mdarray<complex16, 2>* phi;
    mdarray<complex16, 2>* hphi;
    mdarray<double, 1>* v_r;
};

pthread_mutex_t exec_fft_mutex;
int idxfft;

#ifdef _GPU_
void* exec_gpu_fft(void* args__)
{
    exec_fft_args* args = (exec_fft_args*)args__;

    FFT3D<gpu> fft(args->parameters->reciprocal_lattice()->fft()->grid_size());

    int nfft_max = fft.num_fft_max();
    
    fft.initialize(nfft_max);

    bool done = false;

    while (!done)
    {
        pthread_mutex_lock(&exec_fft_mutex);
        int i = idxfft;
        if (idxfft + nfft_max > args->n) 
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
            for (int k = 0; k < nfft_max; k++) fft.input(args->kp->num_gkvec(), args->kp->fft_index(), &(*args->phi)(0, i + k), k);
            fft.copy_to_device();
            fft.transform(1);
            scale_matrix_rows_gpu(fft.size(), nfft_max, fft.fft_buffer().get_ptr_device(), args->v_r->get_ptr_device());
            fft.transform(-1);
            fft.copy_to_host();
            
            for (int k = 0; k < nfft_max; k++) fft.output(args->kp->num_gkvec(), args->kp->fft_index(), &(*args->hphi)(0, i + k), k);
        }
    }

    fft.finalize();
    
    return NULL;
}

void* exec_cpu_fft(void* args__)
{
    exec_fft_args* args = (exec_fft_args*)args__;
    
    auto fft = args->parameters->reciprocal_lattice()->fft();

    int thread_id = args->thread_id;

    bool done = false;
    while (!done)
    {
        pthread_mutex_lock(&exec_fft_mutex);
        int i = idxfft;
        if (idxfft + 1 > args->n)
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
            fft->input(args->kp->num_gkvec(), args->kp->fft_index(), &(*args->phi)(0, i), thread_id);
            fft->transform(1, thread_id);

            for (int ir = 0; ir < fft->size(); ir++) fft->output_buffer(ir, thread_id) *= (*args->v_r)(ir);

            fft->input(&fft->output_buffer(0, thread_id), thread_id);
            fft->transform(-1, thread_id);
            fft->output(args->kp->num_gkvec(), args->kp->fft_index(), &(*args->hphi)(0, i), thread_id);
        }
    }

    return NULL;
}

void apply_h_gpu(Global& parameters, K_point& kp, int n, std::vector<double>& v_r__, complex16* phi__, complex16* hphi__)
{
    Timer t("apply_h_gpu");

    pthread_mutex_init(&exec_fft_mutex, NULL);

    mdarray<double, 1> v_r(&v_r__[0], parameters.reciprocal_lattice()->fft()->size());

    v_r.allocate_on_device();
    v_r.copy_to_device();
   
    mdarray<complex16, 2> phi(phi__, kp.num_gkvec(), n);
    mdarray<complex16, 2> hphi(hphi__, kp.num_gkvec(), n);

    idxfft = 0;
    
    int num_fft_threads = std::min(Platform::num_fft_threads(), Platform::max_num_threads() - 1);
    
    std::vector<pthread_t> pthread_id(num_fft_threads + 1);
    std::vector<exec_fft_args> args(num_fft_threads + 1);

    for (int i = 0; i <= num_fft_threads; i++)
    {
        args[i].thread_id = i;
        args[i].parameters = &parameters;
        args[i].kp = &kp;
        args[i].n = n;
        args[i].phi = &phi;
        args[i].hphi = &hphi;
        args[i].v_r = &v_r;
        if (i < num_fft_threads)
        {
            pthread_create(&pthread_id[i], NULL, exec_cpu_fft, &args[i]);
        }
        else
        {
            pthread_create(&pthread_id[i], NULL, exec_gpu_fft, &args[i]);
        }
    }

    // sync threads
    for (int i = 0; i <= num_fft_threads; i++) pthread_join(pthread_id[i], NULL);

    if (idxfft != n) 
    {
        std::stringstream s;
        s << "not all FFTs are executed" << std::endl
          << " number of FFTS : " << n << ", number of executed FFTs : " << idxfft;
        error_local(__FILE__, __LINE__, s);
    }

    pthread_mutex_destroy(&exec_fft_mutex);
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        for (int ig = 0; ig < kp.num_gkvec(); ig++) hphi(ig, i) += phi(ig, i) * pow(kp.gkvec_cart(ig).length(), 2) / 2.0;
    }
}
#endif

//== void apply_p(K_point& kp, mdarray<complex16, 2>& r)
//== {
//==     for (int i = 0; i < r.size(1); i++)
//==     {
//==         // compute kinetic energy of the vector
//==         double ekin = 0;
//==         for (int ig = 0; ig < kp.num_gkvec(); ig++) ekin += real(conj(r(ig, i)) * r(ig, i)) * pow(kp.gkvec_cart(ig).length(), 2) / 2.0;
//== 
//==         // apply the preconditioner
//==         for (int ig = 0; ig < kp.num_gkvec(); ig++)
//==         {
//==             double x = pow(kp.gkvec_cart(ig).length(), 2) / 2 / 1.5 / ekin;
//==             r(ig, i) = r(ig, i) * (27 + 18 * x + 12 * x * x + 8 * x * x * x) / (27 + 18 * x + 12 * x * x + 8 * x * x * x + 16 * x * x * x * x);
//==         }
//==     }
//== }
//== 
//== void check_orth(mdarray<complex16, 2>& f, int num_f)
//== {
//==     for (int i = 0; i < num_f; i++)
//==     {
//==         for (int j = 0; j < num_f; j++)
//==         {
//==             complex16 z(0, 0);
//==             for (int ig = 0; ig < f.size(0); ig++)
//==             {
//==                 z += conj(f(ig, i)) * f(ig, j);
//==             }
//==             if (i == j) z -= 1.0;
//==             if (abs(z) > 1e-10)
//==             {
//==                 std::stringstream s;
//==                 s << "basis is not orthonormal, error : " << abs(z);
//==                 error_local(__FILE__, __LINE__, s);
//==             }
//==         }
//==     }
//== }
//== 
//== void expand_subspace_v2(K_point& kp, int N, int num_bands, mdarray<complex16, 2>& phi, mdarray<complex16, 2>& res)
//== {
//==     Timer t("expand_subspace");
//== 
//==     assert(phi.size(0) == res.size(0));
//==     assert(phi.size(0) == kp.num_gkvec());
//==     memcpy(&phi(0, N), &res(0, 0), num_bands * kp.num_gkvec() * sizeof(complex16));
//== }
//==
//== void check_degeneracy(K_point& kp, int N, int n, mdarray<complex16, 2>& phi, mdarray<complex16, 2>& res)
//== {
//==     Timer t("check_degeneracy");
//== 
//==     std::vector<bool> drop_res(n, false);
//==     // normalize residuals or discard converged
//==     for (int i = 0; i < n; i++)
//==     {
//==         double norm = 0;
//==         for (int ig = 0; ig < kp.num_gkvec(); ig++) norm += real(conj(res(ig, i)) * res(ig, i));
//==         if (norm < 1e-8)
//==         {
//==             drop_res[i] = true;
//==         }
//==         else
//==         {
//==             for (int ig = 0; ig < kp.num_gkvec(); ig++) res(ig, i) /= sqrt(norm);
//==         }
//==     }
//== 
//==     // overlap between new addisional basis vectors and old basis vectors
//==     mdarray<complex16, 2> ovlp(N, n);
//==     blas<cpu>::gemm(2, 0, N, n, kp.num_gkvec(), &phi(0, 0), phi.ld(), &res(0, 0), res.ld(), &ovlp(0, 0), ovlp.ld());
//==     
//==     // project out the the old subspace
//==     blas<cpu>::gemm(0, 0, kp.num_gkvec(), n, N, complex16(-1, 0), &phi(0, 0), phi.ld(), &ovlp(0, 0), ovlp.ld(), 
//==                     complex16(1, 0), &res(0, 0), res.ld());
//==     
//==     // orthogonalize
//==     for (int j = 0; j < n; j++)
//==     {
//==         std::vector<complex16> v(kp.num_gkvec());
//==         memcpy(&v[0], &res(0, j), kp.num_gkvec() * sizeof(complex16));
//==         for (int j1 = 0; j1 < j; j1++)
//==         {
//==             complex16 z(0, 0);
//==             for (int ig = 0; ig < kp.num_gkvec(); ig++) z += conj(res(ig, j1)) * v[ig];
//==             for (int ig = 0; ig < kp.num_gkvec(); ig++) v[ig] -= z * res(ig, j1);
//==         }
//==         double norm = 0;
//==         for (int ig = 0; ig < kp.num_gkvec(); ig++) norm += real(conj(v[ig]) * v[ig]);
//==         if (norm < 1e-10) error_local(__FILE__, __LINE__, "final residual norm is small");
//==         for (int ig = 0; ig < kp.num_gkvec(); ig++) res(ig, j) = v[ig] / sqrt(norm);
//==     }
//== }


void block_davidson_cpu(Global& parameters, K_point& kp, std::vector<double>& v_r, int num_bands, int max_iter,
                        mdarray<complex16, 2>& psi, std::vector<double>& eval_out)
{
    Timer t("block_davidson_cpu");

    auto fft = parameters.reciprocal_lattice()->fft();
    
    double v0 = 0;
    for (int ir = 0; ir < fft->size(); ir++) v0 += v_r[ir];
    v0 /= parameters.unit_cell()->omega();

    int num_phi = num_bands * 5;

    mdarray<complex16, 2> phi(kp.num_gkvec(), num_phi);
    phi.zero();
    mdarray<complex16, 2> hphi(kp.num_gkvec(), num_phi);
    
    // initial basis functions
    for (int i = 0; i < num_bands; i++) phi(i, i) = 1.0; 
    
    mdarray<complex16, 2> hmlt(num_phi, num_phi);
    mdarray<complex16, 2> ovlp(num_phi, num_phi);
    mdarray<complex16, 2> hmlt_old(num_phi, num_phi);
    mdarray<complex16, 2> ovlp_old(num_phi, num_phi);
    mdarray<complex16, 2> evec(num_phi, num_phi);
    std::vector<double> eval(num_phi);
    
    mdarray<complex16, 2> res(kp.num_gkvec(), num_bands);

    std::vector<double> res_norm(num_bands); // norm of residuals

    generalized_evp* gevp = new generalized_evp_lapack(-1.0);

    int N = -1; // intial eigen-value problem size
    int n = 0; // number of added residuals

    for (int k = 0; k < max_iter; k++)
    {
        Timer t1("block_davidson_cpu|setup_evp");
        if (k == 0)
        {
            N = num_bands;

            apply_h_cpu(parameters, kp, num_bands, v_r, &phi(0, 0), &hphi(0, 0));

            // compute the Hamiltonian matrix: <phi|H|phi>
            blas<cpu>::gemm(2, 0, N, N, kp.num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, 0), hphi.ld(), &hmlt(0, 0), hmlt.ld());
       
            // compute overlap matrix <phi|phi>
            ovlp.zero();
            for (int i = 0; i < N; i++) ovlp(i, i) = complex16(1, 0);
        }
        else
        {
            memcpy(&phi(0, N), &res(0, 0), n * kp.num_gkvec() * sizeof(complex16));

            apply_h_cpu(parameters, kp, n, v_r, &phi(0, N), &hphi(0, N));

            // copy old Hamiltonian and overlap
            for (int i = 0; i < N; i++)
            {
                memcpy(&hmlt(0, i), &hmlt_old(0, i), N * sizeof(complex16));
                memcpy(&ovlp(0, i), &ovlp_old(0, i), N * sizeof(complex16));
            }
            
            // <{phi,res}|H|res>
            blas<cpu>::gemm(2, 0, N + n, n, kp.num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, N), hphi.ld(), &hmlt(0, N), hmlt.ld());
            
            // <{phi,res}|res>
            blas<cpu>::gemm(2, 0, N + n, n, kp.num_gkvec(), &phi(0, 0), phi.ld(), &phi(0, N), phi.ld(), &ovlp(0, N), ovlp.ld());

            N += n;
        }

        // save Hamiltonian and overlap
        for (int i = 0; i < N; i++)
        {
            memcpy(&hmlt_old(0, i), &hmlt(0, i), N * sizeof(complex16));
            memcpy(&ovlp_old(0, i), &ovlp(0, i), N * sizeof(complex16));
        }
        t1.stop();
        
        Timer t2("block_davidson_cpu|solve_evp");
        gevp->solve(N, num_bands, hmlt.get_ptr(), hmlt.ld(), ovlp.get_ptr(), ovlp.ld(), &eval[0], 
                    evec.get_ptr(), evec.ld());
        t2.stop();
        
        printf("\n");
        printf("Iteration : %i, subspace size : %i\n", k, N);
        printf("lower and upper eigen-values : %16.8f %16.8f\n", eval[0], eval[num_bands - 1]);

        Timer t3("block_davidson_cpu|residuals");
        // compute residuals
        // 1. \Psi_{i} = \phi_{mu} * Z_{mu, i}
        blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, N, &phi(0, 0), phi.ld(), &evec(0, 0), evec.ld(), &res(0, 0), res.ld());
        // 2. multiply \Psi_{i} with energy
        for (int i = 0; i < num_bands; i++)
        {
            for (int ig = 0; ig < kp.num_gkvec(); ig++) res(ig, i) *= eval[i];
        }
        // 3. r_{i} = H\Psi_{i} - E_{i}\Psi_{i}
        blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, N, complex16(1, 0), &hphi(0, 0), hphi.ld(), &evec(0, 0), evec.ld(), 
                        complex16(-1, 0), &res(0, 0), res.ld());

        // compute norm and apply preconditioner
        #pragma omp parallel for
        for (int i = 0; i < num_bands; i++)
        {
            double r = 0;
            for (int ig = 0; ig < kp.num_gkvec(); ig++) r += real(conj(res(ig, i)) * res(ig, i));
            res_norm[i] = r;
            
            // apply preconditioner
            for (int ig = 0; ig < kp.num_gkvec(); ig++)
            {
                complex16 t = pow(kp.gkvec_cart(ig).length(), 2) / 2.0 + v0 - eval[i];
                if (abs(t) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
                res(ig, i) /= t;
            }
        }

        // check which residuals are converged
        n = 0;
        for (int i = 0; i < num_bands; i++)
        {
            // take the residual if it's norm is above the threshold
            if (res_norm[i] > 1e-5) 
            {
                // shift unconverged residuals to the beginning of array
                if (n != i) memcpy(&res(0, n), &res(0, i), kp.num_gkvec() * sizeof(complex16));
                n++;
            }
        }
        std::cout << "number of non-converged eigen-vectors : " << n << std::endl;
        t3.stop();

        //== check_degeneracy(kp, N, n, phi, res);

        // check if we run out of variational space or eigen-vectors are converged or it's a last iteration
        if (N + n > num_phi || n == 0 || k == (max_iter - 1))
        {   
            Timer t3("block_davidson_cpu|update_phi");
            // \Psi_{i} = \phi_{mu} * Z_{mu, i}
            blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, N, &phi(0, 0), phi.ld(), &evec(0, 0), evec.ld(), 
                            &psi(0, 0), psi.ld());
            t3.stop();

            if (n == 0 || k == (max_iter - 1)) // exit the loop if the eigen-vectors are converged or it's a last iteration
            {
                break;
            }
            else
            {
                blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, N, &hphi(0, 0), hphi.ld(), 
                                &evec(0, 0), evec.ld(), &phi(0, 0), phi.ld());
                
                // compute the Hamiltonian matrix: <phi|H|phi>
                blas<cpu>::gemm(2, 0, num_bands, num_bands, kp.num_gkvec(), &psi(0, 0), psi.ld(), &phi(0, 0), phi.ld(), &hmlt_old(0, 0), hmlt_old.ld());
             
                memcpy(hphi.get_ptr(), phi.get_ptr(), num_bands * kp.num_gkvec() * sizeof(complex16));

                memcpy(phi.get_ptr(), psi.get_ptr(), num_bands * kp.num_gkvec() * sizeof(complex16));
                N = num_bands;
            }
        }
    }

    delete gevp;

    eval_out.resize(num_bands);
    memcpy(&eval_out[0], &eval[0], num_bands * sizeof(double));
}

void block_davidson_gpu(Global& parameters, K_point& kp, std::vector<double>& v_r, int num_bands, int max_iter,
                        mdarray<complex16, 2>& psi, std::vector<double>& eval_out)
{
#ifdef _GPU_
    Timer t("block_davidson_gpu");

    auto fft = parameters.reciprocal_lattice()->fft();
    
    double v0 = 0;
    for (int ir = 0; ir < fft->size(); ir++) v0 += v_r[ir];
    v0 /= parameters.unit_cell()->omega();

    int num_phi = num_bands * 5;

    mdarray<complex16, 2> phi(kp.num_gkvec(), num_phi);
    phi.zero();
    mdarray<complex16, 2> hphi(kp.num_gkvec(), num_phi);
    
    // initial basis functions
    for (int i = 0; i < num_bands; i++) phi(i, i) = 1.0; 
    
    mdarray<complex16, 2> hmlt(num_phi, num_phi);
    mdarray<complex16, 2> ovlp(num_phi, num_phi);
    mdarray<complex16, 2> hmlt_old(num_phi, num_phi);
    mdarray<complex16, 2> ovlp_old(num_phi, num_phi);
    mdarray<complex16, 2> evec(num_phi, num_phi);
    std::vector<double> eval(num_phi);
    
    mdarray<complex16, 2> res(kp.num_gkvec(), num_bands);

    std::vector<double> res_norm(num_bands); // norm of residuals
    
    #ifndef _MAGMA_
    generalized_evp* gevp = new generalized_evp_lapack(-1.0);
    #else
    generalized_evp* gevp = new generalized_evp_magma();
    #endif

    int N = -1; // intial eigen-value problem size
    int n = 0; // number of added residuals

    for (int k = 0; k < max_iter; k++)
    {
        Timer t1("block_davidson_gpu|setup_evp");
        if (k == 0)
        {
            N = num_bands;

            apply_h_gpu(parameters, kp, num_bands, v_r, &phi(0, 0), &hphi(0, 0));

            mdarray<complex16, 2> mtrx_gpu(NULL, N, N);
            mtrx_gpu.allocate_on_device();

            mdarray<complex16, 2> phi_gpu(&phi(0, 0), kp.num_gkvec(), N);
            phi_gpu.allocate_on_device();
            phi_gpu.copy_to_device();
            
            mdarray<complex16, 2> hphi_gpu(&hphi(0, 0), kp.num_gkvec(), N);
            hphi_gpu.allocate_on_device();
            hphi_gpu.copy_to_device();

            complex16 zone(1, 0);
            complex16 zzero(0, 0);
            blas<gpu>::gemm(2, 0, N, N, kp.num_gkvec(), &zone, phi_gpu.get_ptr_device(), phi_gpu.ld(),
                            hphi_gpu.get_ptr_device(), hphi_gpu.ld(), &zzero, mtrx_gpu.get_ptr_device(), mtrx_gpu.ld());
            
            cublas_get_matrix(N, N, sizeof(complex16), mtrx_gpu.get_ptr_device(), mtrx_gpu.ld(), &hmlt(0, 0), hmlt.ld());
       
            // compute overlap matrix <phi|phi>
            ovlp.zero();
            for (int i = 0; i < N; i++) ovlp(i, i) = complex16(1, 0);
        }
        else
        {
            memcpy(&phi(0, N), &res(0, 0), n * kp.num_gkvec() * sizeof(complex16));

            apply_h_gpu(parameters, kp, n, v_r, &phi(0, N), &hphi(0, N));

            // copy old Hamiltonian and overlap
            for (int i = 0; i < N; i++)
            {
                memcpy(&hmlt(0, i), &hmlt_old(0, i), N * sizeof(complex16));
                memcpy(&ovlp(0, i), &ovlp_old(0, i), N * sizeof(complex16));
            }
            
            mdarray<complex16, 2> mtrx_gpu(NULL, N + n, n);
            mtrx_gpu.allocate_on_device();

            mdarray<complex16, 2> phi_gpu(&phi(0, 0), kp.num_gkvec(), N + n);
            phi_gpu.allocate_on_device();
            phi_gpu.copy_to_device();
            
            mdarray<complex16, 2> hphi_gpu(&hphi(0, N), kp.num_gkvec(), n);
            hphi_gpu.allocate_on_device();
            hphi_gpu.copy_to_device();

            complex16 zone(1, 0);
            complex16 zzero(0, 0);
            blas<gpu>::gemm(2, 0, N + n, n, kp.num_gkvec(), &zone, phi_gpu.get_ptr_device(), phi_gpu.ld(),
                            hphi_gpu.get_ptr_device(), hphi_gpu.ld(), &zzero, mtrx_gpu.get_ptr_device(), mtrx_gpu.ld());
            
            cublas_get_matrix(N + n, n, sizeof(complex16), mtrx_gpu.get_ptr_device(), mtrx_gpu.ld(), &hmlt(0, N), hmlt.ld());
            
            blas<gpu>::gemm(2, 0, N + n, n, kp.num_gkvec(), &zone, phi_gpu.get_ptr_device(), phi_gpu.ld(),
                            &phi_gpu.get_ptr_device()[kp.num_gkvec() * N], phi_gpu.ld(), &zzero, 
                            mtrx_gpu.get_ptr_device(), mtrx_gpu.ld());
            
            cublas_get_matrix(N + n, n, sizeof(complex16), mtrx_gpu.get_ptr_device(), mtrx_gpu.ld(), &ovlp(0, N), ovlp.ld());

            mtrx_gpu.deallocate_on_device();
            phi_gpu.deallocate_on_device();
            hphi_gpu.deallocate_on_device();

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

            N += n;
        }
        
        // save Hamiltonian and overlap
        for (int i = 0; i < N; i++)
        {
            memcpy(&hmlt_old(0, i), &hmlt(0, i), N * sizeof(complex16));
            memcpy(&ovlp_old(0, i), &ovlp(0, i), N * sizeof(complex16));
        }
        t1.stop();
        
        Timer t2("block_davidson_gpu|solve_evp");
        gevp->solve(N, num_bands, hmlt.get_ptr(), hmlt.ld(), ovlp.get_ptr(), ovlp.ld(), &eval[0], 
                    evec.get_ptr(), evec.ld());
        t2.stop();

        printf("\n");
        printf("Iteration : %i, subspace size : %i\n", k, N);
        printf("lower and upper eigen-values : %16.8f %16.8f\n", eval[0], eval[num_bands - 1]);
        
        Timer t3("block_davidson_gpu|residuals");
        
        // compute residuals
        // 1. \Psi_{i} = \phi_{mu} * Z_{mu, i}
        // 2. multiply \Psi_{i} with energy
        // 3. r_{i} = H\Psi_{i} - E_{i}\Psi_{i}
            
        // allocate residuals on GPU
        res.allocate_on_device();

        // move eigen vectors to GPU
        mdarray<complex16, 2> evec_gpu(&evec(0, 0), num_phi, N);
        evec_gpu.allocate_on_device();
        evec_gpu.copy_to_device();

        // move phi to gpu
        mdarray<complex16, 2> phi_gpu(&phi(0, 0), kp.num_gkvec(), N);
        phi_gpu.allocate_on_device();
        phi_gpu.copy_to_device();

        // execute first zgemm: \Psi_{i} = \phi_{mu} * Z_{mu, i}
        complex16 zone(1, 0);
        complex16 zzero(0, 0);
        blas<gpu>::gemm(0, 0, kp.num_gkvec(), num_bands, N, &zone, phi_gpu.get_ptr_device(), phi_gpu.ld(), 
                        evec_gpu.get_ptr_device(), evec_gpu.ld(), &zzero, res.get_ptr_device(), res.ld());

        // scale by eigen-values
        mdarray<double, 1> eval_gpu(&eval[0], num_bands);
        eval_gpu.allocate_on_device();
        eval_gpu.copy_to_device();
        scale_matrix_columns_gpu(kp.num_gkvec(), num_bands, res.get_ptr_device(), eval_gpu.get_ptr_device());
        
        // move hphi to gpu, use already allocated phi_gpu
        phi_gpu.set_ptr(hphi.get_ptr());
        phi_gpu.copy_to_device();

        complex16 mzone(-1, 0);
        // execute second zgemm: r_{i} = H\Psi_{i} - E_{i}\Psi_{i}
        blas<gpu>::gemm(0, 0, kp.num_gkvec(), num_bands, N, &zone, phi_gpu.get_ptr_device(), phi_gpu.ld(), 
                        evec_gpu.get_ptr_device(), evec_gpu.ld(), &mzone, res.get_ptr_device(), res.ld());

        // copy residual to host memory
        res.copy_to_host();
        
        // free memory on gpu
        res.deallocate_on_device();
        evec_gpu.deallocate_on_device();
        phi_gpu.deallocate_on_device();
        eval_gpu.deallocate_on_device();

        // compute norm and apply preconditioner
        #pragma omp parallel for
        for (int i = 0; i < num_bands; i++)
        {
            double r = 0;
            for (int ig = 0; ig < kp.num_gkvec(); ig++) r += real(conj(res(ig, i)) * res(ig, i));
            res_norm[i] = r;
            
            // apply preconditioner
            for (int ig = 0; ig < kp.num_gkvec(); ig++)
            {
                complex16 t = pow(kp.gkvec_cart(ig).length(), 2) / 2.0 + v0 - eval[i];
                if (abs(t) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
                res(ig, i) /= t;
            }
        }

        // check which residuals are converged
        n = 0;
        for (int i = 0; i < num_bands; i++)
        {
            // take the residual if it's norm is above the threshold
            if (res_norm[i] > 1e-5) 
            {
                // shift unconverged residuals to the beginning of array
                if (n != i) memcpy(&res(0, n), &res(0, i), kp.num_gkvec() * sizeof(complex16));
                n++;
            }
        }
        std::cout << "number of non-converged eigen-vectors : " << n << std::endl;
        t3.stop();

        //== check_degeneracy(kp, N, n, phi, res);
        
        // check if we run out of variational space or eigen-vectors are converged or it's a last iteration
        if (N + n > num_phi || n == 0 || k == (max_iter - 1))
        {   
            Timer t3("block_davidson_cpu|update_phi");
            // \Psi_{i} = \phi_{mu} * Z_{mu, i}
            blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, N, &phi(0, 0), phi.ld(), &evec(0, 0), evec.ld(), 
                            &psi(0, 0), psi.ld());
            t3.stop();

            if (n == 0 || k == (max_iter - 1)) // exit the loop if the eigen-vectors are converged or it's a last iteration
            {
                break;
            }
            else
            {
                // TODO: to GPU
                blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, N, &hphi(0, 0), hphi.ld(), 
                                &evec(0, 0), evec.ld(), &phi(0, 0), phi.ld());
                
                // compute the Hamiltonian matrix: <phi|H|phi>
                blas<cpu>::gemm(2, 0, num_bands, num_bands, kp.num_gkvec(), &psi(0, 0), psi.ld(), &phi(0, 0), phi.ld(), &hmlt_old(0, 0), hmlt_old.ld());
             
                memcpy(hphi.get_ptr(), phi.get_ptr(), num_bands * kp.num_gkvec() * sizeof(complex16));

                memcpy(phi.get_ptr(), psi.get_ptr(), num_bands * kp.num_gkvec() * sizeof(complex16));
                N = num_bands;
            }
        }
    }

    delete gevp;

    eval_out.resize(num_bands);
    memcpy(&eval_out[0], &eval[0], num_bands * sizeof(double));
#endif
}

//== void sum_rho(Global& parameters, K_point& kp, int num_bands, mdarray<complex16, 2>& psi)
//== {
//==     Timer t("sum_rho");
//== 
//==     std::vector<double> rho(parameters.fft().size());
//== 
//==     int num_fft_threads = Platform::num_fft_threads();
//==     #pragma omp parallel default(shared) num_threads(num_fft_threads)
//==     {
//==         int thread_id = omp_get_thread_num();
//== 
//==         std::vector<double> rho_pt(parameters.fft().size(), 0);
//==         
//==         std::vector<complex16> psi_r(parameters.fft().size());
//== 
//==         #pragma omp for
//==         for (int i = 0; i < num_bands; i++)
//==         {
//==             parameters.fft().input(kp.num_gkvec(), kp.fft_index(), &psi(0, i), thread_id);
//==             parameters.fft().transform(1, thread_id);
//==             parameters.fft().output(&psi_r[0], thread_id);
//==             
//==             double w = 1.0 / parameters.omega();
//==             
//==             for (int ir = 0; ir < parameters.fft().size(); ir++) rho_pt[ir] += real(psi_r[ir] * conj(psi_r[ir])) * w;
//==         }
//== 
//==         #pragma omp critical
//==         for (int ir = 0; ir < parameters.fft().size(); ir++) rho[ir] += rho_pt[ir];
//==     }
//== }

//== void diag_exact(Global& parameters, K_point& kp, std::vector<complex16>& v_pw, int num_bands, std::vector<double>& eval)
//== {
//==     Timer t("diag_exact");
//== 
//==     splindex<block_cyclic> spl_gkvec_row(kp.num_gkvec(), parameters.mpi_grid().dimension_size(_dim_row_), 
//==                                          parameters.mpi_grid().coordinate(_dim_row_), parameters.cyclic_block_size());
//==     splindex<block_cyclic> spl_gkvec_col(kp.num_gkvec(), parameters.mpi_grid().dimension_size(_dim_col_), 
//==                                          parameters.mpi_grid().coordinate(_dim_col_), parameters.cyclic_block_size());
//== 
//==     mdarray<complex16, 2> hmlt(spl_gkvec_row.local_size(), spl_gkvec_col.local_size());
//==     mdarray<complex16, 2> ovlp(spl_gkvec_row.local_size(), spl_gkvec_col.local_size());
//==     mdarray<complex16, 2> evec(spl_gkvec_row.local_size(), spl_gkvec_col.local_size());
//==     eval.resize(num_bands);
//== 
//==     for (int icol_loc = 0; icol_loc < spl_gkvec_col.local_size(); icol_loc++)
//==     {
//==         int icol = spl_gkvec_col[icol_loc];
//==         for (int irow_loc = 0; irow_loc < spl_gkvec_row.local_size(); irow_loc++)
//==         {
//==             int irow = spl_gkvec_row[irow_loc];
//== 
//==             int ig = parameters.index_g12(kp.gvec_index(irow), kp.gvec_index(icol));
//==             hmlt(irow_loc, icol_loc) = v_pw[ig];
//==             ovlp(irow_loc, icol_loc) = 0;
//== 
//==             if (irow == icol) 
//==             {
//==                 hmlt(irow_loc, icol_loc) += pow(kp.gkvec_cart(irow).length(), 2) / 2.0;
//==                 ovlp(irow_loc, icol_loc) = 1;
//==             }
//==         }
//==     }
//== 
//==     generalized_evp* solver = new generalized_evp_scalapack(parameters.cyclic_block_size(),  
//==                                                             parameters.mpi_grid().dimension_size(_dim_row_),
//==                                                             parameters.mpi_grid().dimension_size(_dim_col_),
//==                                                             parameters.blacs_context(), -1.0);
//==     
//==     solver->solve(kp.num_gkvec(), num_bands, hmlt.get_ptr(), hmlt.ld(), ovlp.get_ptr(), ovlp.ld(), &eval[0], 
//==                   evec.get_ptr(), evec.ld());
//==     
//==     delete solver;
//== }



void test_iterative_diag()
{
    int num_bands = 100;
    int max_iter = 8;
    double gk_cutoff = 5;
    double pw_cutoff = 17;

    std::string device("cpu");

    std::string fname("input.json");
    if (Utils::file_exists(fname))
    {
        JSON_tree parser(fname);
        num_bands = parser["num_bands"].get(num_bands);
        max_iter = parser["max_iter"].get(max_iter);
        pw_cutoff = parser["pw_cutoff"].get(pw_cutoff);
        gk_cutoff = parser["gk_cutoff"].get(gk_cutoff);
        device = parser["device"].get(device);
    }

    Global parameters;

    double a0[] = {12.975 * 1.889726125, 0, 0};
    double a1[] = {0, 12.975 * 1.889726125, 0};
    double a2[] = {0, 0, 12.975 * 1.889726125};

    parameters.unit_cell()->set_lattice_vectors(a0, a1, a2);
    parameters.set_pw_cutoff(pw_cutoff);
    parameters.initialize();
    
    double vk[] = {0, 0, 0};
    K_point kp(parameters, vk, 1.0);
    kp.generate_gkvec(gk_cutoff);

    if (Platform::mpi_rank() == 0) std::cout << "num_gkvec = " << kp.num_gkvec() << std::endl;

    // generate some potential in plane-wave domain
    std::vector<complex16> v_pw(parameters.reciprocal_lattice()->num_gvec());
    for (int ig = 0; ig < parameters.reciprocal_lattice()->num_gvec(); ig++) 
        v_pw[ig] = complex16(1.0 / pow(parameters.reciprocal_lattice()->gvec_len(ig) + 1.0, 1), 0.0);
   
    // transform potential to real space
    auto fft = parameters.reciprocal_lattice()->fft();
    std::vector<double> v_r(fft->size());
    fft->input(parameters.reciprocal_lattice()->num_gvec(), parameters.reciprocal_lattice()->fft_index(), &v_pw[0]);
    fft->transform(1);
    for (int i = 0; i < fft->size(); i++)
    {
        if (fabs(imag(fft->output_buffer(i))) > 1e-10) error_local(__FILE__, __LINE__, "potential is complex");
        v_r[i] = real(fft->output_buffer(i));
    }

    mdarray<complex16, 2> psi(kp.num_gkvec(), num_bands);
    std::vector<double> eval;

    //diag_exact(parameters, kp, v_pw, num_bands, eval);

    if (device == "cpu")
    {
        std::cout << "calling CPU version" << std::endl;
        block_davidson_cpu(parameters, kp, v_r, num_bands, max_iter, psi, eval);
    } 
    else if (device == "gpu")
    {
        std::cout << "calling GPU version" << std::endl;
        block_davidson_gpu(parameters, kp, v_r, num_bands, max_iter, psi, eval);
    }
    else
    {
        error_local(__FILE__, __LINE__, "unknown device");
    }

    //== if (Platform::mpi_rank() == 0)
    //== {
    //==     printf("\n");
    //==     printf("Eigen-values:\n");
    //==     for (int i = 0; i < num_bands; i++) printf("i : %3i,  eval : %16.10f\n", i, eval[i]);
    //== }

    //sum_rho(parameters, kp, num_bands, psi);

    parameters.clear();
}

int main(int argn, char** argv)
{
    Platform::initialize(true);

    test_iterative_diag();

    Timer::print();

    Platform::barrier();

    Platform::finalize();
}
