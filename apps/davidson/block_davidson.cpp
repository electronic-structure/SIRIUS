#include <sirius.h>

using namespace sirius;

void apply_h(Global& parameters, K_point& kp, int n, std::vector<complex16>& v_r, complex16* phi__, complex16* hphi__)
{
    Timer t("apply_h");

    mdarray<complex16, 2> phi(phi__, kp.num_gkvec(), n);
    mdarray<complex16, 2> hphi(hphi__, kp.num_gkvec(), n);
    
    int num_fft_threads = Platform::num_fft_threads();
    #pragma omp parallel default(shared) num_threads(num_fft_threads)
    {        
        int thread_id = omp_get_thread_num();
        std::vector<complex16> phi_r(parameters.fft().size());
        
        #pragma omp for
        for (int i = 0; i < n; i++)
        {
            parameters.fft().input(kp.num_gkvec(), kp.fft_index(), &phi(0, i), thread_id);
            parameters.fft().transform(1, thread_id);
            parameters.fft().output(&phi_r[0], thread_id);

            for (int ir = 0; ir < parameters.fft().size(); ir++) phi_r[ir] *= v_r[ir];

            parameters.fft().input(&phi_r[0], thread_id);
            parameters.fft().transform(-1, thread_id);
            parameters.fft().output(kp.num_gkvec(), kp.fft_index(), &hphi(0, i), thread_id);

            for (int ig = 0; ig < kp.num_gkvec(); ig++) hphi(ig, i) += phi(ig, i) * pow(kp.gkvec_cart(ig).length(), 2) / 2.0;
        }
    }
}

void apply_p(K_point& kp, mdarray<complex16, 2>& r)
{
    for (int i = 0; i < r.size(1); i++)
    {
        // compute kinetic energy of the vector
        double ekin = 0;
        for (int ig = 0; ig < kp.num_gkvec(); ig++) ekin += real(conj(r(ig, i)) * r(ig, i)) * pow(kp.gkvec_cart(ig).length(), 2) / 2.0;

        // apply the preconditioner
        for (int ig = 0; ig < kp.num_gkvec(); ig++)
        {
            double x = pow(kp.gkvec_cart(ig).length(), 2) / 2 / 1.5 / ekin;
            r(ig, i) = r(ig, i) * (27 + 18 * x + 12 * x * x + 8 * x * x * x) / (27 + 18 * x + 12 * x * x + 8 * x * x * x + 16 * x * x * x * x);
        }
    }
}

void check_orth(mdarray<complex16, 2>& f, int num_f)
{
    for (int i = 0; i < num_f; i++)
    {
        for (int j = 0; j < num_f; j++)
        {
            complex16 z(0, 0);
            for (int ig = 0; ig < f.size(0); ig++)
            {
                z += conj(f(ig, i)) * f(ig, j);
            }
            if (i == j) z -= 1.0;
            if (abs(z) > 1e-10)
            {
                std::stringstream s;
                s << "basis is not orthonormal, error : " << abs(z);
                error_local(__FILE__, __LINE__, s);
            }
        }
    }
}

void expand_subspace_v2(K_point& kp, int N, int num_bands, mdarray<complex16, 2>& phi, mdarray<complex16, 2>& res)
{
    Timer t("expand_subspace");

    assert(phi.size(0) == res.size(0));
    assert(phi.size(0) == kp.num_gkvec());
    memcpy(&phi(0, N), &res(0, 0), num_bands * kp.num_gkvec() * sizeof(complex16));
}

void check_degeneracy(K_point& kp, int N, int n, mdarray<complex16, 2>& phi, mdarray<complex16, 2>& res)
{
    Timer t("check_degeneracy");

    std::vector<bool> drop_res(n, false);
    // normalize residuals or discard converged
    for (int i = 0; i < n; i++)
    {
        double norm = 0;
        for (int ig = 0; ig < kp.num_gkvec(); ig++) norm += real(conj(res(ig, i)) * res(ig, i));
        if (norm < 1e-8)
        {
            drop_res[i] = true;
        }
        else
        {
            for (int ig = 0; ig < kp.num_gkvec(); ig++) res(ig, i) /= sqrt(norm);
        }
    }

    // overlap between new addisional basis vectors and old basis vectors
    mdarray<complex16, 2> ovlp(N, n);
    blas<cpu>::gemm(2, 0, N, n, kp.num_gkvec(), &phi(0, 0), phi.ld(), &res(0, 0), res.ld(), &ovlp(0, 0), ovlp.ld());
    
    //== ovlp.zero();
    //== for (int i = 0; i < N; i++)
    //== {
    //==     for (int j = 0; j < n; j++) 
    //==     {
    //==         for (int ig = 0; ig < kp.num_gkvec(); ig++) ovlp(i, j) += conj(phi(ig, i)) * res(ig, j);
    //==     }
    //== }

    // project out the the old subspace
    blas<cpu>::gemm(0, 0, kp.num_gkvec(), n, N, complex16(-1, 0), &phi(0, 0), phi.ld(), &ovlp(0, 0), ovlp.ld(), 
                    complex16(1, 0), &res(0, 0), res.ld());
    
    //== for (int j = 0; j < n; j++)
    //== {
    //==     for (int i = 0; i < N; i++)
    //==     {
    //==         for (int ig = 0; ig < kp.num_gkvec(); ig++) res(ig, j) -= ovlp(i, j) * phi(ig, i);
    //==     }
    //== }

    // orthogonalize
    for (int j = 0; j < n; j++)
    {
        std::vector<complex16> v(kp.num_gkvec());
        memcpy(&v[0], &res(0, j), kp.num_gkvec() * sizeof(complex16));
        for (int j1 = 0; j1 < j; j1++)
        {
            complex16 z(0, 0);
            for (int ig = 0; ig < kp.num_gkvec(); ig++) z += conj(res(ig, j1)) * v[ig];
            for (int ig = 0; ig < kp.num_gkvec(); ig++) v[ig] -= z * res(ig, j1);
        }
        double norm = 0;
        for (int ig = 0; ig < kp.num_gkvec(); ig++) norm += real(conj(v[ig]) * v[ig]);
        if (norm < 1e-10) error_local(__FILE__, __LINE__, "final residual norm is small");
        for (int ig = 0; ig < kp.num_gkvec(); ig++) res(ig, j) = v[ig] / sqrt(norm);
    }
}


void diag_davidson_v2(Global& parameters, K_point& kp, std::vector<complex16>& v_pw, int num_bands, int max_iter,
                      mdarray<complex16, 2>& psi, std::vector<double>& eval_out)
{
    Timer t("diag_davidson");

    std::vector<complex16> v_r(parameters.fft().size());
    parameters.fft().input(parameters.num_gvec(), parameters.fft_index(), &v_pw[0]);
    parameters.fft().transform(1);
    parameters.fft().output(&v_r[0]);

    for (int ir = 0; ir < parameters.fft().size(); ir++)
    {
        if (fabs(imag(v_r[ir])) > 1e-10) error_local(__FILE__, __LINE__, "potential is complex");
    }

    int num_phi = num_bands * 5;

    mdarray<complex16, 2> phi(kp.num_gkvec(), num_phi);
    mdarray<complex16, 2> hphi(kp.num_gkvec(), num_phi);
    
    // initial basis functions
    phi.zero();
    for (int i = 0; i < num_bands; i++) phi(i, i) = 1.0;
    // apply Hamiltonian to intial states
    //apply_h(parameters, kp, num_bands, v_r, &phi(0, 0), &hphi(0, 0));
    for (int i = 0; i < num_bands; i++)
    {
        for (int ig = 0; ig < kp.num_gkvec(); ig++)
        {
            hphi(ig, i) = v_pw[parameters.index_g12(kp.gvec_index(ig), kp.gvec_index(i))];
        }
        hphi(i, i) += pow(kp.gkvec_cart(i).length(), 2) / 2.0;
    }
    
    mdarray<complex16, 2> hmlt(num_phi, num_phi);
    mdarray<complex16, 2> ovlp(num_phi, num_phi);
    mdarray<complex16, 2> hmlt_old(num_phi, num_phi);
    mdarray<complex16, 2> ovlp_old(num_phi, num_phi);
    mdarray<complex16, 2> evec(num_phi, num_phi);
    std::vector<double> eval(num_phi);
    
    mdarray<complex16, 2> res(kp.num_gkvec(), num_bands);

    std::vector<double> res_norm(num_bands); // norm of residuals

    generalized_evp* gevp = new generalized_evp_lapack(-1.0);

    int N = num_bands; // intial eigen-value problem size
    int n = 0; // number of added residuals

    bool full_hmlt_update = true;

    for (int k = 0; k < max_iter; k++)
    {
        std::cout << std::endl;
        std::cout << "Iteration : " << k << ", subspace size : " << N << std::endl;
        
        Timer t1("setup_evp");
        if (full_hmlt_update)
        {
            // compute the Hamiltonian matrix: <phi|H|phi>
            blas<cpu>::gemm(2, 0, N, N, kp.num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, 0), hphi.ld(), &hmlt(0, 0), hmlt.ld());
       
            // compute overlap matrix <phi|phi>
            //blas<cpu>::gemm(2, 0, N, N, kp.num_gkvec(), &phi(0, 0), phi.ld(), &phi(0, 0), phi.ld(), &ovlp(0, 0), ovlp.ld());
            ovlp.zero();
            for (int i = 0; i < N; i++) ovlp(i, i) = complex16(1, 0);

            full_hmlt_update = false;
        }
        else
        {
            int M = N - n;
            // copy old Hamiltonian and overlap
            for (int i = 0; i < M; i++)
            {
                memcpy(&hmlt(0, i), &hmlt_old(0, i), M * sizeof(complex16));
                memcpy(&ovlp(0, i), &ovlp_old(0, i), M * sizeof(complex16));
            }
            
            // <{phi,res}|H|res>
            blas<cpu>::gemm(2, 0, N, n, kp.num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, M), hphi.ld(), &hmlt(0, M), hmlt.ld());
            
            // <{phi,res}|res>
            blas<cpu>::gemm(2, 0, N, n, kp.num_gkvec(), &phi(0, 0), phi.ld(), &phi(0, M), phi.ld(), &ovlp(0, M), ovlp.ld());
        }

        // save Hamiltonian and overlap
        for (int i = 0; i < N; i++)
        {
            memcpy(&hmlt_old(0, i), &hmlt(0, i), N * sizeof(complex16));
            memcpy(&ovlp_old(0, i), &ovlp(0, i), N * sizeof(complex16));
        }
        t1.stop();
        
        Timer t2("solve_evp");
        gevp->solve(N, num_bands, hmlt.get_ptr(), hmlt.ld(), ovlp.get_ptr(), ovlp.ld(), &eval[0], 
                    evec.get_ptr(), evec.ld());
        t2.stop();
        
        printf("lower and upper eigen-values : %16.8f %16.8f\n", eval[0], eval[num_bands - 1]);

        Timer t3("residuals");
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
                complex16 t = pow(kp.gkvec_cart(ig).length(), 2) / 2.0 + v_pw[0] - eval[i];
                if (abs(t) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
                res(ig, i) /= t;
            }
        }
        t3.stop();

        // check which residuals are converged
        n = 0;
        for (int i = 0; i < num_bands; i++)
        {
            // take the residual if it's norm is above the threshold, use hphi as temporary storage
            if (res_norm[i] > 1e-5) 
            {
                if (n != i) memcpy(&res(0, n), &res(0, i), kp.num_gkvec() * sizeof(complex16));
                n++;
            }
        }
        std::cout << "number of non-converged eigen-vectors : " << n << std::endl;

        //== check_degeneracy(kp, N, n, phi, res);

        // check if we run out of variational space or eigen-vectors are converged or it's a last iteration
        if (N + n > num_phi || n == 0 || k == (max_iter - 1))
        {   
            Timer t3("update_phi");
            // \Psi_{i} = \phi_{mu} * Z_{mu, i}
            blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, N, &phi(0, 0), phi.ld(), &evec(0, 0), evec.ld(), 
                            &psi(0, 0), psi.ld());
            t3.stop();

            if (n == 0 || k == (max_iter - 1)) // exit the loop if the eigen-vectors are converged or it's a last iteration
            {
                break;
            }
            else // otherwise set psi as a new trial basis
            {
                memcpy(phi.get_ptr(), psi.get_ptr(), num_bands * kp.num_gkvec() * sizeof(complex16));
                apply_h(parameters, kp, num_bands, v_r, &phi(0, 0), &hphi(0, 0));
                N = num_bands;
                full_hmlt_update = true;
            }
        }
        else
        {
            //apply_p(kp, res_active);
        
            // expand variational subspace with new basis vectors obtatined from residuals
            expand_subspace_v2(kp, N, n, phi, res);

            // apply Hamiltonian to the new basis functions
            apply_h(parameters, kp, n, v_r, &phi(0, N), &hphi(0, N));
        
            // increase the size of the variation space
            N += n;
        }
    }

    delete gevp;

    eval_out.resize(num_bands);
    memcpy(&eval_out[0], &eval[0], num_bands * sizeof(double));
}


void diag_davidson_v3(Global& parameters, K_point& kp, std::vector<complex16>& v_pw, int num_bands, int max_iter,
                      mdarray<complex16, 2>& psi, std::vector<double>& eval_out)
{
    Timer t("diag_davidson");

    std::vector<complex16> v_r(parameters.fft().size());
    parameters.fft().input(parameters.num_gvec(), parameters.fft_index(), &v_pw[0]);
    parameters.fft().transform(1);
    parameters.fft().output(&v_r[0]);

    for (int ir = 0; ir < parameters.fft().size(); ir++)
    {
        if (fabs(imag(v_r[ir])) > 1e-10) error_local(__FILE__, __LINE__, "potential is complex");
    }

    int num_phi = num_bands * 5;

    mdarray<complex16, 2> phi(kp.num_gkvec(), num_phi);
    mdarray<complex16, 2> hphi(kp.num_gkvec(), num_phi);
    
    // initial basis functions
    phi.zero();
    for (int i = 0; i < num_bands; i++) phi(i, i) = 1.0;
    // apply Hamiltonian to intial states
    for (int i = 0; i < num_bands; i++)
    {
        for (int ig = 0; ig < kp.num_gkvec(); ig++)
        {
            hphi(ig, i) = v_pw[parameters.index_g12(kp.gvec_index(ig), kp.gvec_index(i))];
        }
        hphi(i, i) += pow(kp.gkvec_cart(i).length(), 2) / 2.0;
    }

    mdarray<complex16, 2> hmlt(num_phi, num_phi);
    mdarray<complex16, 2> ovlp(num_phi, num_phi);
    mdarray<complex16, 2> hmlt_old(num_phi, num_phi);
    mdarray<complex16, 2> ovlp_old(num_phi, num_phi);
    mdarray<complex16, 2> evec(num_phi, num_phi);

    std::vector<double> eval(num_phi);
    std::vector<double> eval_old(num_phi, 1e100);
    
    mdarray<complex16, 2> res(kp.num_gkvec(), num_bands);

    std::vector<double> res_e(num_bands);

    generalized_evp* gevp = new generalized_evp_lapack(-1.0);

    int N = num_bands; // intial eigen-value problem size
    int n = 0; // number of added residuals

    bool full_hmlt_update = true;

    for (int k = 0; k < max_iter; k++)
    {
        std::cout << std::endl;
        std::cout << "Iteration : " << k << ", subspace size : " << N << std::endl;
       
        Timer t1("setup_evp");
        if (full_hmlt_update)
        {
            // compute the Hamiltonian matrix: <phi|H|phi>
            blas<cpu>::gemm(2, 0, N, N, kp.num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, 0), hphi.ld(), &hmlt(0, 0), hmlt.ld());
       
            // initial overlap matrix is identity
            ovlp.zero();
            for (int i = 0; i < N; i++) ovlp(i, i) = complex16(1, 0);

            full_hmlt_update = false;
        }
        else
        {
            int M = N - n;
            // copy old Hamiltonian and overlap
            for (int i = 0; i < M; i++)
            {
                memcpy(&hmlt(0, i), &hmlt_old(0, i), M * sizeof(complex16));
                memcpy(&ovlp(0, i), &ovlp_old(0, i), M * sizeof(complex16));
            }
            // <{phi,res}|H|res>
            blas<cpu>::gemm(2, 0, N, n, kp.num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, M), hphi.ld(), &hmlt(0, M), hmlt.ld());
            
            // <{phi,res}|res>
            blas<cpu>::gemm(2, 0, N, n, kp.num_gkvec(), &phi(0, 0), phi.ld(), &phi(0, M), phi.ld(), &ovlp(0, M), ovlp.ld());
        }

        //== Utils::write_matrix("hmlt.txt", hmlt, N, N, true, true, "%12.6f");
        //== Utils::write_matrix("ovlp.txt", ovlp, N, N, true, true, "%12.6f");

        // save Hamiltonian and overlap
        for (int i = 0; i < N; i++)
        {
            memcpy(&hmlt_old(0, i), &hmlt(0, i), N * sizeof(complex16));
            memcpy(&ovlp_old(0, i), &ovlp(0, i), N * sizeof(complex16));
        }
        t1.stop();
        
        // solve generalized eigen-value problem    
        Timer t2("solve_evp");
        gevp->solve(N, num_bands, hmlt.get_ptr(), hmlt.ld(), ovlp.get_ptr(), ovlp.ld(), &eval[0], 
                    evec.get_ptr(), evec.ld());
        t2.stop();

        //== Utils::write_matrix("evec.txt", evec, N, N, false, true, "%12.6f");

        printf("lower and upper eigen-values : %16.8f %16.8f\n", eval[0], eval[num_bands - 1]);
        
        n = 0;
        // check eigen-values for convergence
        for (int i = 0; i < num_bands; i++)
        {
            if (fabs(eval[i] - eval_old[i]) > 1e-6)
            {
                res_e[n] = eval[i];
                
                // use hmlt as a temporary storage for evec
                memcpy(&hmlt(0, n), &evec(0, i), N * sizeof(complex16));

                n++;
            }
            eval_old[i] = eval[i];
        }

        std::cout << "number of non-converged eigen-vectors : " << n << std::endl;
        
        // if we have unconverged eigen-states
        if (n != 0)
        {
            Timer t3("residuals");
            // compute residuals
            // 1. \Psi_{i} = \phi_{mu} * Z_{mu, i}
            blas<cpu>::gemm(0, 0, kp.num_gkvec(), n, N, &phi(0, 0), phi.ld(), &hmlt(0, 0), hmlt.ld(), &res(0, 0), res.ld());
            // 2. multiply \Psi_{i} with energy
            for (int i = 0; i < n; i++)
            {
                for (int ig = 0; ig < kp.num_gkvec(); ig++) res(ig, i) *= res_e[i];
            }
            // 3. r_{i} = H\Psi_{i} - E_{i}\Psi_{i}
            blas<cpu>::gemm(0, 0, kp.num_gkvec(), n, N, complex16(1, 0), &hphi(0, 0), hphi.ld(), &hmlt(0, 0), hmlt.ld(), 
                            complex16(-1, 0), &res(0, 0), res.ld());
            t3.stop();

                        
            //== Utils::write_matrix("res.txt", res, kp.num_gkvec(), n, false, true, "%12.6f");

            // apply preconditioner
            #pragma omp parallel for
            for (int i = 0; i < n; i++)
            {
                double norm = 0.0;
                // apply preconditioner
                for (int ig = 0; ig < kp.num_gkvec(); ig++)
                {
                    norm += real(conj(res(ig, i)) * res(ig, i));
                    complex16 t = pow(kp.gkvec_cart(ig).length(), 2) / 2.0 + v_pw[0] - res_e[i];
                    if (abs(t) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
                    res(ig, i) /= t;
                }
                //if (norm < 1e-10) error_local(__FILE__, __LINE__, "residual is converged");
            }

            check_degeneracy(kp, N, n, phi, res);
        }

        // check if we run out of variational space or eigen-vectors are converged or it's a last iteration
        if (N + n > num_phi || n == 0 || k == (max_iter - 1))
        {   
            Timer t3("update_phi");
            // \Psi_{i} = \phi_{mu} * Z_{mu, i}
            blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, N, &phi(0, 0), phi.ld(), &evec(0, 0), evec.ld(), 
                            &psi(0, 0), psi.ld());
            t3.stop();

            if (n == 0 || k == (max_iter - 1)) // exit the loop if the eigen-vectors are converged or it's a last iteration
            {
                break;
            }
            else // otherwise set psi as a new trial basis
            {
                memcpy(phi.get_ptr(), psi.get_ptr(), num_bands * kp.num_gkvec() * sizeof(complex16));
                apply_h(parameters, kp, num_bands, v_r, &phi(0, 0), &hphi(0, 0));
                N = num_bands;
                full_hmlt_update = true;
                for (int i = 0; i < num_bands; i++) eval_old[i] = 1e100;
            }
        }
        else
        {
            //apply_p(kp, res_active);
            
            // expand variational subspace with new basis vectors obtatined from residuals
            expand_subspace_v2(kp, N, n, phi, res);

            // apply Hamiltonian to the new basis functions
            apply_h(parameters, kp, n, v_r, &phi(0, N), &hphi(0, N));
            
            // increase the size of the variation space
            N += n;
        }
    }

    delete gevp;

    eval_out.resize(num_bands);
    memcpy(&eval_out[0], &eval[0], num_bands * sizeof(double));
}

void diag_davidson_v3_gpu(Global& parameters, K_point& kp, std::vector<complex16>& v_pw, int num_bands, int max_iter,
                          mdarray<complex16, 2>& psi, std::vector<double>& eval_out)
{
#ifdef _GPU_
    Timer t("diag_davidson");

    std::vector<complex16> v_r(parameters.fft().size());
    parameters.fft().input(parameters.num_gvec(), parameters.fft_index(), &v_pw[0]);
    parameters.fft().transform(1);
    parameters.fft().output(&v_r[0]);

    for (int ir = 0; ir < parameters.fft().size(); ir++)
    {
        if (fabs(imag(v_r[ir])) > 1e-10) error_local(__FILE__, __LINE__, "potential is complex");
    }

    int num_phi = num_bands * 5;

    mdarray<complex16, 2> phi(kp.num_gkvec(), num_phi);
    mdarray<complex16, 2> hphi(kp.num_gkvec(), num_phi);
    
    // initial basis functions
    phi.zero();
    for (int i = 0; i < num_bands; i++) phi(i, i) = 1.0;
    // apply Hamiltonian to intial states
    for (int i = 0; i < num_bands; i++)
    {
        for (int ig = 0; ig < kp.num_gkvec(); ig++)
        {
            hphi(ig, i) = v_pw[parameters.index_g12(kp.gvec_index(ig), kp.gvec_index(i))];
        }
        hphi(i, i) += pow(kp.gkvec_cart(i).length(), 2) / 2.0;
    }

    mdarray<complex16, 2> hmlt(num_phi, num_phi);
    mdarray<complex16, 2> ovlp(num_phi, num_phi);
    mdarray<complex16, 2> hmlt_old(num_phi, num_phi);
    mdarray<complex16, 2> ovlp_old(num_phi, num_phi);
    mdarray<complex16, 2> evec(num_phi, num_phi);

    std::vector<double> eval(num_phi);
    std::vector<double> eval_old(num_phi, 1e100);
    
    mdarray<complex16, 2> res(kp.num_gkvec(), num_bands);

    std::vector<double> res_e(num_bands);

    generalized_evp* gevp = new generalized_evp_lapack(-1.0);

    int N = num_bands; // intial eigen-value problem size
    int n = 0; // number of added residuals

    bool full_hmlt_update = true;

    for (int k = 0; k < max_iter; k++)
    {
        std::cout << std::endl;
        std::cout << "Iteration : " << k << ", subspace size : " << N << std::endl;
       
        Timer t1("setup_evp");
        if (full_hmlt_update)
        {
            // compute the Hamiltonian matrix: <phi|H|phi>
            blas<cpu>::gemm(2, 0, N, N, kp.num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, 0), hphi.ld(), &hmlt(0, 0), hmlt.ld());
       
            // initial overlap matrix is identity
            ovlp.zero();
            for (int i = 0; i < N; i++) ovlp(i, i) = complex16(1, 0);

            full_hmlt_update = false;
        }
        else
        {
            int M = N - n;
            // copy old Hamiltonian and overlap
            for (int i = 0; i < M; i++)
            {
                memcpy(&hmlt(0, i), &hmlt_old(0, i), M * sizeof(complex16));
                memcpy(&ovlp(0, i), &ovlp_old(0, i), M * sizeof(complex16));
            }
            // <{phi,res}|H|res>
            blas<cpu>::gemm(2, 0, N, n, kp.num_gkvec(), &phi(0, 0), phi.ld(), &hphi(0, M), hphi.ld(), &hmlt(0, M), hmlt.ld());
            
            // <{phi,res}|res>
            blas<cpu>::gemm(2, 0, N, n, kp.num_gkvec(), &phi(0, 0), phi.ld(), &phi(0, M), phi.ld(), &ovlp(0, M), ovlp.ld());
        }

        // save Hamiltonian and overlap
        for (int i = 0; i < N; i++)
        {
            memcpy(&hmlt_old(0, i), &hmlt(0, i), N * sizeof(complex16));
            memcpy(&ovlp_old(0, i), &ovlp(0, i), N * sizeof(complex16));
        }
        t1.stop();
        
        // solve generalized eigen-value problem    
        Timer t2("solve_evp");
        gevp->solve(N, num_bands, hmlt.get_ptr(), hmlt.ld(), ovlp.get_ptr(), ovlp.ld(), &eval[0], 
                    evec.get_ptr(), evec.ld());
        t2.stop();

        printf("lower and upper eigen-values : %16.8f %16.8f\n", eval[0], eval[num_bands - 1]);
        
        n = 0;
        // check eigen-values for convergence
        for (int i = 0; i < num_bands; i++)
        {
            if (fabs(eval[i] - eval_old[i]) > 1e-6)
            {
                res_e[n] = eval[i];
                
                // use hmlt as a temporary storage for evec
                memcpy(&hmlt(0, n), &evec(0, i), N * sizeof(complex16));

                n++;
            }
            eval_old[i] = eval[i];
        }

        std::cout << "number of non-converged eigen-vectors : " << n << std::endl;
        
        // compute residuals if we have unconverged eigen-states
        if (n != 0)
        {
            Timer t3("residuals");

            // allocate residuals on GPU
            mdarray<complex16, 2> res_gpu(&res(0, 0), kp.num_gkvec(), n);
            res_gpu.allocate_on_device();

            // move eigen vectors to GPU
            mdarray<complex16, 2> evec_gpu(&hmlt(0, 0), num_phi, n);
            evec_gpu.allocate_on_device();
            evec_gpu.copy_to_device();

            // move phi to gpu
            mdarray<complex16, 2> phi_gpu(&phi(0, 0), kp.num_gkvec(), N);
            phi_gpu.allocate_on_device();
            phi_gpu.copy_to_device();

            // execute first zgemm: \Psi_{i} = \phi_{mu} * Z_{mu, i}
            complex16 zone(1, 0);
            complex16 zzero(0, 0);
            blas<gpu>::gemm(0, 0, kp.num_gkvec(), n, N, &zone, phi_gpu.get_ptr_device(), phi_gpu.ld(), 
                            evec_gpu.get_ptr_device(), evec_gpu.ld(), &zzero, res_gpu.get_ptr_device(), res_gpu.ld());

            // scale by eigen-values
            mdarray<double, 1> res_e_gpu(&res_e[0], n);
            res_e_gpu.allocate_on_device();
            res_e_gpu.copy_to_device();
            scale_matrix_columns_gpu(kp.num_gkvec(), n, res_gpu.get_ptr_device(), res_e_gpu.get_ptr_device());
            
            // move hphi to gpu
            phi_gpu.set_ptr(hphi.get_ptr());
            phi_gpu.copy_to_device();

            complex16 mzone(-1, 0);
            // execute second zgemm: r_{i} = H\Psi_{i} - E_{i}\Psi_{i}
            blas<gpu>::gemm(0, 0, kp.num_gkvec(), n, N, &zone, phi_gpu.get_ptr_device(), phi_gpu.ld(), 
                            evec_gpu.get_ptr_device(), evec_gpu.ld(), &mzone, res_gpu.get_ptr_device(), res_gpu.ld());

            // copy residual to host memory
            res_gpu.copy_to_host();
            
            // free memory on gpu
            res_gpu.deallocate_on_device();
            evec_gpu.deallocate_on_device();
            phi_gpu.deallocate_on_device();
            res_e_gpu.deallocate_on_device();

            t3.stop();

            // apply preconditioner
            #pragma omp parallel for
            for (int i = 0; i < n; i++)
            {
                // apply preconditioner
                for (int ig = 0; ig < kp.num_gkvec(); ig++)
                {
                    complex16 t = pow(kp.gkvec_cart(ig).length(), 2) / 2.0 + v_pw[0] - res_e[i];
                    if (abs(t) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
                    res(ig, i) /= t;
                }
            }
        }

        // check if we run out of variational space or eigen-vectors are converged or it's a last iteration
        if (N + n > num_phi || n == 0 || k == (max_iter - 1))
        {   
            Timer t3("update_phi");
            // \Psi_{i} = \phi_{mu} * Z_{mu, i}
            blas<cpu>::gemm(0, 0, kp.num_gkvec(), num_bands, N, &phi(0, 0), phi.ld(), &evec(0, 0), evec.ld(), 
                            &psi(0, 0), psi.ld());
            t3.stop();

            if (n == 0 || k == (max_iter - 1)) // exit the loop if the eigen-vectors are converged or it's a last iteration
            {
                break;
            }
            else // otherwise set psi as a new trial basis
            {
                memcpy(phi.get_ptr(), psi.get_ptr(), num_bands * kp.num_gkvec() * sizeof(complex16));
                apply_h(parameters, kp, num_bands, v_r, &phi(0, 0), &hphi(0, 0));
                N = num_bands;
                full_hmlt_update = true;
                for (int i = 0; i < num_bands; i++) eval_old[i] = 1e100;
            }
        }
        else
        {
            //apply_p(kp, res_active);
            
            // expand variational subspace with new basis vectors obtatined from residuals
            expand_subspace_v2(kp, N, n, phi, res);

            // apply Hamiltonian to the new basis functions
            apply_h(parameters, kp, n, v_r, &phi(0, N), &hphi(0, N));
            
            // increase the size of the variation space
            N += n;
        }
    }

    delete gevp;

    eval_out.resize(num_bands);
    memcpy(&eval_out[0], &eval[0], num_bands * sizeof(double));
#endif
}

void sum_rho(Global& parameters, K_point& kp, int num_bands, mdarray<complex16, 2>& psi)
{
    Timer t("sum_rho");

    std::vector<double> rho(parameters.fft().size());

    int num_fft_threads = Platform::num_fft_threads();
    #pragma omp parallel default(shared) num_threads(num_fft_threads)
    {
        int thread_id = omp_get_thread_num();

        std::vector<double> rho_pt(parameters.fft().size(), 0);
        
        std::vector<complex16> psi_r(parameters.fft().size());

        #pragma omp for
        for (int i = 0; i < num_bands; i++)
        {
            parameters.fft().input(kp.num_gkvec(), kp.fft_index(), &psi(0, i), thread_id);
            parameters.fft().transform(1, thread_id);
            parameters.fft().output(&psi_r[0], thread_id);
            
            double w = 1.0 / parameters.omega();
            
            for (int ir = 0; ir < parameters.fft().size(); ir++) rho_pt[ir] += real(psi_r[ir] * conj(psi_r[ir])) * w;
        }

        #pragma omp critical
        for (int ir = 0; ir < parameters.fft().size(); ir++) rho[ir] += rho_pt[ir];
    }
}

void diag_exact(Global& parameters, K_point& kp, std::vector<complex16>& v_pw, int num_bands, std::vector<double>& eval)
{
    Timer t("diag_exact");

    splindex<block_cyclic> spl_gkvec_row(kp.num_gkvec(), parameters.mpi_grid().dimension_size(_dim_row_), 
                                         parameters.mpi_grid().coordinate(_dim_row_), parameters.cyclic_block_size());
    splindex<block_cyclic> spl_gkvec_col(kp.num_gkvec(), parameters.mpi_grid().dimension_size(_dim_col_), 
                                         parameters.mpi_grid().coordinate(_dim_col_), parameters.cyclic_block_size());

    mdarray<complex16, 2> hmlt(spl_gkvec_row.local_size(), spl_gkvec_col.local_size());
    mdarray<complex16, 2> ovlp(spl_gkvec_row.local_size(), spl_gkvec_col.local_size());
    mdarray<complex16, 2> evec(spl_gkvec_row.local_size(), spl_gkvec_col.local_size());
    eval.resize(num_bands);

    for (int icol_loc = 0; icol_loc < spl_gkvec_col.local_size(); icol_loc++)
    {
        int icol = spl_gkvec_col[icol_loc];
        for (int irow_loc = 0; irow_loc < spl_gkvec_row.local_size(); irow_loc++)
        {
            int irow = spl_gkvec_row[irow_loc];

            int ig = parameters.index_g12(kp.gvec_index(irow), kp.gvec_index(icol));
            hmlt(irow_loc, icol_loc) = v_pw[ig];
            ovlp(irow_loc, icol_loc) = 0;

            if (irow == icol) 
            {
                hmlt(irow_loc, icol_loc) += pow(kp.gkvec_cart(irow).length(), 2) / 2.0;
                ovlp(irow_loc, icol_loc) = 1;
            }
        }
    }

    generalized_evp* solver = new generalized_evp_scalapack(parameters.cyclic_block_size(),  
                                                            parameters.mpi_grid().dimension_size(_dim_row_),
                                                            parameters.mpi_grid().dimension_size(_dim_col_),
                                                            parameters.blacs_context(), -1.0);
    
    solver->solve(kp.num_gkvec(), num_bands, hmlt.get_ptr(), hmlt.ld(), ovlp.get_ptr(), ovlp.ld(), &eval[0], 
                  evec.get_ptr(), evec.ld());
    
    delete solver;
}



void test_davidson()
{
    int num_bands = 20;
    int max_iter = 8;
    double Ekin = 4;

    std::string device("cpu");

    std::string fname("input.json");
    if (Utils::file_exists(fname))
    {
        JSON_tree parser(fname);
        num_bands = parser["num_bands"].get(num_bands);
        max_iter = parser["max_iter"].get(max_iter);
        Ekin = parser["Ekin"].get(Ekin);
        device = parser["device"].get(device);
    }

    Global parameters;

    double a0[] = {12.975 * 1.889726125, 0, 0};
    double a1[] = {0, 12.975 * 1.889726125, 0};
    double a2[] = {0, 0, 12.975 * 1.889726125};

    parameters.set_lattice_vectors(a0, a1, a2);
    parameters.set_pw_cutoff(2 * sqrt(2 * Ekin) + 0.5);
    parameters.initialize();
    
    double vk[] = {0, 0, 0};
    K_point kp(parameters, vk, 1.0);
    kp.generate_gkvec(sqrt(2 * Ekin));

    if (Platform::mpi_rank() == 0) std::cout << "num_gkvec = " << kp.num_gkvec() << std::endl;

    // generate some potential in plane-wave domain
    std::vector<complex16> v_pw(parameters.num_gvec());
    for (int ig = 0; ig < parameters.num_gvec(); ig++) v_pw[ig] = complex16(1.0 / pow(parameters.gvec_len(ig) + 1.0, 1), 0.0);

    
    mdarray<complex16, 2> psi(kp.num_gkvec(), num_bands);
    std::vector<double> eval;

    //diag_exact(parameters, kp, v_pw, num_bands, eval);

    if (device == "cpu")
    {
        std::cout << "calling CPU version" << std::endl;
        diag_davidson_v2(parameters, kp, v_pw, num_bands, max_iter, psi, eval);
    } 
    else if (device == "gpu")
    {
        std::cout << "calling GPU version" << std::endl;
        diag_davidson_v3_gpu(parameters, kp, v_pw, num_bands, max_iter, psi, eval);
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

    test_davidson();

    Timer::print();

    Platform::barrier();

    Platform::finalize();
}
